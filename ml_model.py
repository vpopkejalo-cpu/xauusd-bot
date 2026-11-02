"""
Модуль машинного обучения (Machine Learning) для предсказания направления XAU/USD.
Использует scikit-learn (логистическую регрессию с масштабированием признаков).
Обеспечивает:
1. Вычисление вероятности успешности сделки (P >= 60% для генерации сигнала).
2. Автоматическое периодическое самообучение (пересчет весов) на обновленных исторических данных.
3. Сохранение и загрузку обученной модели через joblib.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

from config import MODEL_PATH, MIN_PROBABILITY, SL_ATR_MULTIPLIER, TP_ATR_MULTIPLIER
from database import log_training_run

logger = logging.getLogger(__name__)

# Список признаков (features) для обучения модели
FEATURE_COLUMNS = [
    "rsi",
    "dist_ema_21",
    "dist_ema_50",
    "ema_trend",
    "atr_pct",
    "macd",
    "macd_hist",
    "return_1",
    "return_3",
    "return_5"
]

# Классы: 0 - ВНЕ РЫНКА (HOLD), 1 - ПОКУПКА (BUY), 2 - ПРОДАЖА (SELL)
CLASS_HOLD = 0
CLASS_BUY = 1
CLASS_SELL = 2


class TradingMLModel:
    def __init__(self, model_path=MODEL_PATH, min_prob: float = MIN_PROBABILITY):
        self.model_path = model_path
        self.min_prob = min_prob
        self.pipeline: Optional[Pipeline] = None
        self.is_trained = False
        self.samples_count = 0

        # Попытка загрузить сохраненную ранее модель
        self.load_model()

    def load_model(self) -> bool:
        """Загружает сохраненную модель с диска, если файл существует."""
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                self.pipeline = data.get("pipeline")
                self.last_accuracy = data.get("accuracy", 0.0)
                self.samples_count = data.get("samples_count", 0)
                self.is_trained = True
                logger.info(f"ML модель успешно загружена из {self.model_path}. Образцов: {self.samples_count}, Точность: {self.last_accuracy:.2%}")
                return True
            except Exception as e:
                logger.error(f"Не удалось загрузить модель из {self.model_path}: {e}")
                self.is_trained = False
        return False

    def save_model(self, accuracy: float = 0.0, samples_count: int = 0):
        """Сохраняет обученную модель на диск."""
        try:
            data = {
                "pipeline": self.pipeline,
                "accuracy": accuracy,
                "samples_count": samples_count,
                "features": FEATURE_COLUMNS
            }
            joblib.dump(data, self.model_path)
            self.samples_count = samples_count
            logger.info(f"ML модель успешно сохранена в {self.model_path} ({samples_count} образцов)")
        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {e}")

    def prepare_dataset(self, df: pd.DataFrame, horizon: int = 12) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Формирует матрицу признаков X и метки y на основе исхода сделок (достижение TP раньше SL).
        
        horizon: горизонт анализа в свечах (например, 12 свечей по 15 мин = 3 часа).
        """
        data = df.copy()
        
        # Проверяем наличие всех признаков
        missing_cols = [c for c in FEATURE_COLUMNS if c not in data.columns]
        if missing_cols:
            raise ValueError(f"В DataFrame отсутствуют необходимые колонки: {missing_cols}")

        labels = []
        n = len(data)

        closes = data["close"].values
        highs = data["high"].values
        lows = data["low"].values
        atrs = data["atr"].values

        for i in range(n):
            if i + horizon >= n:
                labels.append(CLASS_HOLD)
                continue

            entry = closes[i]
            atr = atrs[i]

            if np.isnan(atr) or atr <= 0:
                labels.append(CLASS_HOLD)
                continue

            # Параметры сделки BUY
            buy_tp = entry + TP_ATR_MULTIPLIER * atr
            buy_sl = entry - SL_ATR_MULTIPLIER * atr

            # Параметры сделки SELL
            sell_tp = entry - TP_ATR_MULTIPLIER * atr
            sell_sl = entry + SL_ATR_MULTIPLIER * atr

            future_highs = highs[i + 1 : i + 1 + horizon]
            future_lows = lows[i + 1 : i + 1 + horizon]

            buy_tp_hit = False
            buy_sl_hit = False
            sell_tp_hit = False
            sell_sl_hit = False

            # Проверяем, что произошло раньше для BUY
            for h, l in zip(future_highs, future_lows):
                if l <= buy_sl:
                    buy_sl_hit = True
                    break
                if h >= buy_tp:
                    buy_tp_hit = True
                    break

            # Проверяем, что произошло раньше для SELL
            for h, l in zip(future_highs, future_lows):
                if h >= sell_sl:
                    sell_sl_hit = True
                    break
                if l <= sell_tp:
                    sell_tp_hit = True
                    break

            if buy_tp_hit and not buy_sl_hit:
                labels.append(CLASS_BUY)
            elif sell_tp_hit and not sell_sl_hit:
                labels.append(CLASS_SELL)
            else:
                labels.append(CLASS_HOLD)

        data["target"] = labels

        # Удаляем строки с NaN в признаках и последние неразмеченные свечи
        valid_mask = ~data[FEATURE_COLUMNS].isna().any(axis=1) & (data.index < data.index[-horizon])
        clean_data = data[valid_mask]

        X = clean_data[FEATURE_COLUMNS]
        y = clean_data["target"]
        return X, y

    def train(self, df: pd.DataFrame) -> bool:
        """
        Обучает модель логистической регрессии с масштабированием признаков (StandardScaler).
        Оценивает точность по временной кросс-валидации (TimeSeriesSplit).
        """
        try:
            X, y = self.prepare_dataset(df)
            if len(X) < 40:
                logger.warning(f"Недостаточно размеченных данных для обучения ({len(X)} образцов). Обучение отложено.")
                return False

            # Проверяем разнообразие классов
            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                logger.warning(f"В выборке представлен только один класс: {unique_classes}. Обучение отложено.")
                return False

            # Создаем Pipeline: масштабирование + логистическая регрессия
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    C=1.0,
                    random_state=42
                ))
            ])

            # Оценка качества на последних данных (out-of-sample)
            tscv = TimeSeriesSplit(n_splits=3)
            accuracies = []
            for train_idx, test_idx in tscv.split(X):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                
                # Обучаем только если есть оба класса
                if len(np.unique(y_tr)) > 1:
                    pipeline.fit(X_tr, y_tr)
                    preds = pipeline.predict(X_te)
                    accuracies.append(accuracy_score(y_te, preds))

            mean_acc = float(np.mean(accuracies)) if accuracies else 0.50

            # Обучаем финальную модель на всех доступных данных
            pipeline.fit(X, y)
            self.pipeline = pipeline
            self.is_trained = True
            self.last_accuracy = mean_acc

            # Сохраняем модель на диск и в лог базы данных
            self.save_model(accuracy=mean_acc, samples_count=len(X))
            log_training_run(samples_count=len(X), accuracy=mean_acc, notes="Автоматическое переобучение")

            logger.info(f"Обучение завершено успешно. Образцов: {len(X)}, Средняя точность CV: {mean_acc:.2%}")
            return True

        except Exception as e:
            logger.error(f"Ошибка при обучении ML модели: {e}", exc_info=True)
            return False

    def predict(self, current_candle_features: pd.Series) -> Dict[str, Any]:
        """
        Принимает вектор признаков текущей свечи.
        Возвращает словарь:
        {
            "action": "BUY" | "SELL" | "HOLD",
            "probability": float (0.0 .. 1.0),
            "reason": str
        }
        """
        if not self.is_trained or self.pipeline is None:
            return {
                "action": "HOLD",
                "probability": 0.0,
                "reason": "Модель еще не обучена (недостаточно истории)"
            }

        try:
            # Извлекаем только нужные фичи
            features_df = pd.DataFrame([current_candle_features[FEATURE_COLUMNS]])
            if features_df.isna().any().any():
                return {
                    "action": "HOLD",
                    "probability": 0.0,
                    "reason": "В векторе признаков присутствуют NaN (период прогрева индикаторов)"
                }

            # Получаем вероятности для всех доступных классов
            classes = self.pipeline.classes_
            probabilities = self.pipeline.predict_proba(features_df)[0]
            prob_map = dict(zip(classes, probabilities))

            buy_prob = prob_map.get(CLASS_BUY, 0.0)
            sell_prob = prob_map.get(CLASS_SELL, 0.0)

            logger.debug(f"ML Прогноз вероятностей: BUY={buy_prob:.2%}, SELL={sell_prob:.2%}, Порог={self.min_prob:.2%}")

            # Проверка порога уверенности
            if buy_prob >= self.min_prob and buy_prob > sell_prob:
                return {
                    "action": "BUY",
                    "probability": buy_prob,
                    "reason": f"Вероятность успеха BUY {buy_prob:.1%} превышает порог {self.min_prob:.1%}"
                }
            elif sell_prob >= self.min_prob and sell_prob > buy_prob:
                return {
                    "action": "SELL",
                    "probability": sell_prob,
                    "reason": f"Вероятность успеха SELL {sell_prob:.1%} превышает порог {self.min_prob:.1%}"
                }
            else:
                max_prob = max(buy_prob, sell_prob)
                return {
                    "action": "HOLD",
                    "probability": max_prob,
                    "reason": f"Уверенность ({max_prob:.1%}) ниже защитного порога {self.min_prob:.1%}"
                }

        except Exception as e:
            logger.error(f"Ошибка при вычислении прогноза модели: {e}")
            return {
                "action": "HOLD",
                "probability": 0.0,
                "reason": f"Исключение при прогнозировании: {e}"
            }
