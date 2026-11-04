"""
Модуль машинного обучения (Machine Learning) с индивидуальными моделями под каждый актив.
Использует scikit-learn (Pipeline: StandardScaler + LogisticRegression с регуляризацией).

Ключевые преимущества:
1. Изолированные модели: каждый из 6 инструментов обучается исключительно на своей истории.
2. Честная разметка датасета по методу Triple Barrier (First-touch):
   Учитывает очередность касания TP и SL. Если цена сначала коснулась SL, а потом развернулась —
   сделка считается убыточной, исключая ложное обучение (Path-dependent validation).
3. Индивидуальные пороги вероятности (P >= 62%–70% в зависимости от волатильности инструмента).
"""

import os
import re
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

from config import MODEL_PATH, MODELS_DIR, MIN_PROBABILITY
from risk_manager import get_asset_profile
from database import log_training_run

logger = logging.getLogger(__name__)

# Расширенный список признаков (features) для обучения моделей
FEATURE_COLUMNS = [
    "rsi",
    "dist_ema_21",
    "dist_ema_50",
    "dist_ema_200",
    "ema_trend",
    "atr_pct",
    "adx",
    "plus_di",
    "minus_di",
    "macd",
    "macd_hist",
    "return_1",
    "return_3",
    "return_5"
]

# Направленные классы: 0 - ПРОДАЖА/ПАДЕНИЕ (SELL), 1 - ПОКУПКА/РОСТ (BUY)
CLASS_SELL = 0
CLASS_BUY = 1


def clean_symbol_for_filename(symbol: str) -> str:
    """Преобразует тикер в безопасное имя файла (например, 'XAU/USD' -> 'XAU_USD')."""
    return re.sub(r"[^A-Za-z0-9_]", "_", symbol.strip())


class TradingMLModel:
    def __init__(self, models_dir: Path = MODELS_DIR, default_min_prob: float = MIN_PROBABILITY):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.default_min_prob = default_min_prob

        # Словари индивидуальных моделей по символам: symbol -> Pipeline
        self.models: Dict[str, Pipeline] = {}
        self.accuracies: Dict[str, float] = {}
        self.samples_counts: Dict[str, int] = {}

        # Общие атрибуты для обратной совместимости
        self.pipeline: Optional[Pipeline] = None
        self.is_trained = False
        self.samples_count = 0
        self.last_accuracy = 0.0

        # Загружаем все сохраненные модели с диска
        self.load_all_models()

    def get_model_path(self, symbol: str) -> Path:
        """Возвращает путь к файлу модели конкретного актива."""
        safe_sym = clean_symbol_for_filename(symbol)
        return self.models_dir / f"model_{safe_sym}.joblib"

    def load_all_models(self) -> int:
        """Загружает все доступные модели из директории models/."""
        loaded_count = 0
        if not self.models_dir.exists():
            return 0

        for file_path in self.models_dir.glob("model_*.joblib"):
            try:
                data = joblib.load(file_path)
                sym = data.get("symbol", "")
                pipe = data.get("pipeline")
                acc = float(data.get("accuracy", 0.0))
                cnt = int(data.get("samples_count", 0))

                if sym and pipe:
                    self.models[sym] = pipe
                    self.accuracies[sym] = acc
                    self.samples_counts[sym] = cnt
                    loaded_count += 1
                    logger.info(f"Загружена модель для {sym}: точность CV {acc:.1%}, образцов {cnt}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить модель {file_path}: {e}")

        # Для совместимости: если есть модель для XAU/USD, делаем ее дефолтной
        if "XAU/USD" in self.models:
            self.pipeline = self.models["XAU/USD"]
            self.is_trained = True
            self.last_accuracy = self.accuracies.get("XAU/USD", 0.0)
            self.samples_count = self.samples_counts.get("XAU/USD", 0)
        elif self.models:
            # Берем первую доступную
            first_sym = next(iter(self.models))
            self.pipeline = self.models[first_sym]
            self.is_trained = True
            self.last_accuracy = self.accuracies.get(first_sym, 0.0)
            self.samples_count = self.samples_counts.get(first_sym, 0)
        elif os.path.exists(MODEL_PATH):
            # Резервная загрузка старого монолитного файла, если есть
            try:
                data = joblib.load(MODEL_PATH)
                self.pipeline = data.get("pipeline")
                self.is_trained = True
                self.last_accuracy = float(data.get("accuracy", 0.0))
                self.samples_count = int(data.get("samples_count", 0))
            except Exception:
                pass

        return loaded_count

    def save_symbol_model(self, symbol: str, pipeline: Pipeline, accuracy: float, samples_count: int):
        """Сохраняет обученную модель конкретного актива на диск."""
        try:
            path = self.get_model_path(symbol)
            data = {
                "symbol": symbol,
                "pipeline": pipeline,
                "accuracy": float(accuracy),
                "samples_count": int(samples_count),
                "features": FEATURE_COLUMNS
            }
            joblib.dump(data, path)

            # Сохраняем также в память
            self.models[symbol] = pipeline
            self.accuracies[symbol] = accuracy
            self.samples_counts[symbol] = samples_count

            # Если это основной актив, обновляем дефолт
            if symbol == "XAU/USD" or not self.is_trained:
                self.pipeline = pipeline
                self.is_trained = True
                self.last_accuracy = accuracy
                self.samples_count = samples_count
                joblib.dump(data, MODEL_PATH)

            logger.info(f"Модель для {symbol} сохранена в {path} ({samples_count} образцов, точность {accuracy:.1%})")
        except Exception as e:
            logger.error(f"Ошибка сохранения модели для {symbol}: {e}")

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        symbol: str = "XAU/USD",
        horizon: int = 15
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Формирует матрицу признаков X и метки y по методу Triple Barrier (First-touch):
        Для каждой свечи i проверяет реальную траекторию цены на горизонте horizon свечей:
        - Если при сценарии BUY цена коснулась уровня TP раньше, чем уровня SL -> победа BUY.
        - Если при сценарии SELL цена коснулась уровня TP раньше, чем уровня SL -> победа SELL.
        - Если оба сценария выбило по стопу или рынок застрял во флэте -> шум отсекается.
        """
        data = df.copy()

        missing_cols = [c for c in FEATURE_COLUMNS if c not in data.columns]
        if missing_cols:
            raise ValueError(f"В DataFrame отсутствуют колонки: {missing_cols}")

        profile = get_asset_profile(symbol)
        sl_mult = profile["sl_atr_mult"]
        tp_mult = profile["tp_atr_mult"]
        min_sl = profile["min_sl_dist"]
        min_tp = profile["min_tp_dist"]

        n = len(data)
        closes = data["close"].values
        highs = data["high"].values
        lows = data["low"].values
        atrs = data.get("atr", data["close"] * 0.002).values

        labels = []
        valid_indices = []

        for i in range(n - horizon):
            entry = closes[i]
            atr = atrs[i]

            if np.isnan(atr) or atr <= 0 or np.isnan(entry):
                continue

            sl_dist = max(atr * sl_mult, min_sl)
            tp_dist = max(atr * tp_mult, min_tp)

            buy_tp = entry + tp_dist
            buy_sl = entry - sl_dist

            sell_tp = entry - tp_dist
            sell_sl = entry + sl_dist

            buy_outcome = None   # "WIN", "LOSS", None
            sell_outcome = None  # "WIN", "LOSS", None

            # Пошаговая проверка движения цены (Path-dependent Triple Barrier)
            for step in range(1, horizon + 1):
                idx = i + step
                h = highs[idx]
                l = lows[idx]

                # Проверка сценария BUY
                if buy_outcome is None:
                    hit_sl = (l <= buy_sl)
                    hit_tp = (h >= buy_tp)
                    if hit_sl and hit_tp:
                        buy_outcome = "LOSS"  # Консервативно: считаем, что выбило стоп
                    elif hit_sl:
                        buy_outcome = "LOSS"
                    elif hit_tp:
                        buy_outcome = "WIN"

                # Проверка сценария SELL
                if sell_outcome is None:
                    hit_sl = (h >= sell_sl)
                    hit_tp = (l <= sell_tp)
                    if hit_sl and hit_tp:
                        sell_outcome = "LOSS"
                    elif hit_sl:
                        sell_outcome = "LOSS"
                    elif hit_tp:
                        sell_outcome = "WIN"

                if buy_outcome is not None and sell_outcome is not None:
                    break

            # Принятие решения по метке
            if buy_outcome == "WIN" and sell_outcome != "WIN":
                labels.append(CLASS_BUY)
                valid_indices.append(i)
            elif sell_outcome == "WIN" and buy_outcome != "WIN":
                labels.append(CLASS_SELL)
                valid_indices.append(i)
            else:
                # Если ни один уровень не взят чисто, проверяем выраженность смещения
                # без касания стопов
                net_change = closes[i + horizon] - entry
                if buy_outcome != "LOSS" and net_change > 0.4 * tp_dist:
                    labels.append(CLASS_BUY)
                    valid_indices.append(i)
                elif sell_outcome != "LOSS" and net_change < -0.4 * tp_dist:
                    labels.append(CLASS_SELL)
                    valid_indices.append(i)
                # Иначе пропускаем свечу: не учим модель на распиле и двойных стопах

        if not valid_indices:
            return pd.DataFrame(), pd.Series(dtype=int)

        clean_data = data.iloc[valid_indices].copy()
        clean_data["target"] = labels

        valid_mask = ~clean_data[FEATURE_COLUMNS].isna().any(axis=1)
        final_data = clean_data[valid_mask]

        X = final_data[FEATURE_COLUMNS]
        y = final_data["target"]
        return X, y

    def train(self, df: pd.DataFrame, symbol: str = "XAU/USD") -> bool:
        """
        Обучает модель для конкретного актива с использованием временной кросс-валидации.
        """
        try:
            X, y = self.prepare_dataset(df, symbol=symbol)
            if len(X) < 35:
                logger.warning(f"[{symbol}] Недостаточно размеченных сделок ({len(X)} образцов). Требуется минимум 35.")
                return False

            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                logger.warning(f"[{symbol}] В выборке представлен только один класс ({unique_classes}). Обучение отложено.")
                return False

            # Создаем Pipeline со стандартизацией и регуляризованной логистической регрессией
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    C=0.8,              # Регуляризация для защиты от переобучения
                    random_state=42
                ))
            ])

            # Оценка на временной кросс-валидации (TimeSeriesSplit)
            n_splits = min(5, max(2, len(X) // 20))
            tscv = TimeSeriesSplit(n_splits=n_splits)
            accuracies = []
            for train_idx, test_idx in tscv.split(X):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

                if len(np.unique(y_tr)) > 1:
                    pipeline.fit(X_tr, y_tr)
                    preds = pipeline.predict(X_te)
                    accuracies.append(accuracy_score(y_te, preds))

            mean_acc = float(np.mean(accuracies)) if accuracies else 0.50

            # Обучаем финальную модель актива на всем массиве
            pipeline.fit(X, y)

            # Сохраняем модель
            self.save_symbol_model(symbol, pipeline, mean_acc, len(X))
            log_training_run(samples_count=len(X), accuracy=mean_acc, notes=f"Индивидуальная модель {symbol}")

            logger.info(f"[{symbol}] Обучение успешно завершено. Образцов: {len(X)}, Точность CV: {mean_acc:.1%}")
            return True

        except Exception as e:
            logger.error(f"[{symbol}] Ошибка при обучении ML модели: {e}", exc_info=True)
            return False

    def predict(
        self,
        symbol_or_candle,
        current_candle_features: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Прогнозирует направление для конкретного актива.
        Поддерживает оба вызова:
        - predict(symbol, candle)
        - predict(candle) — обратная совместимость
        """
        if current_candle_features is None:
            # Вызов predict(candle)
            candle = symbol_or_candle
            symbol = "XAU/USD"
        else:
            # Вызов predict(symbol, candle)
            symbol = str(symbol_or_candle)
            candle = current_candle_features

        profile = get_asset_profile(symbol)
        min_prob = profile.get("min_probability", self.default_min_prob)

        pipeline = self.models.get(symbol, self.pipeline)
        if pipeline is None:
            return {
                "action": "HOLD",
                "probability": 0.0,
                "reason": f"[{symbol}] Индивидуальная модель еще не обучена"
            }

        try:
            features_df = pd.DataFrame([candle[FEATURE_COLUMNS]])
            if features_df.isna().any().any():
                return {
                    "action": "HOLD",
                    "probability": 0.0,
                    "reason": f"[{symbol}] В векторе признаков есть NaN (период прогрева)"
                }

            classes = list(pipeline.classes_)
            probabilities = pipeline.predict_proba(features_df)[0]
            prob_map = dict(zip(classes, probabilities))

            buy_prob = float(prob_map.get(CLASS_BUY, 0.5))
            sell_prob = float(prob_map.get(CLASS_SELL, 0.5))

            logger.debug(f"[{symbol}] ML Прогноз: BUY={buy_prob:.1%}, SELL={sell_prob:.1%}, Порог={min_prob:.1%}")

            # 1. Покупка BUY при превышении индивидуального порога актива
            if buy_prob >= min_prob:
                return {
                    "action": "BUY",
                    "probability": buy_prob,
                    "reason": f"Вероятность роста {buy_prob:.1%} >= порога {min_prob:.1%}"
                }
            # 2. Продажа SELL при превышении индивидуального порога актива
            elif sell_prob >= min_prob:
                return {
                    "action": "SELL",
                    "probability": sell_prob,
                    "reason": f"Вероятность падения {sell_prob:.1%} >= порога {min_prob:.1%}"
                }
            # 3. Нейтральная зона
            else:
                max_prob = float(max(buy_prob, sell_prob))
                return {
                    "action": "HOLD",
                    "probability": max_prob,
                    "reason": f"Уверенность {max_prob:.1%} ниже порога {min_prob:.1%} — нет устойчивого преимущества"
                }

        except Exception as e:
            logger.error(f"[{symbol}] Ошибка при прогнозировании: {e}")
            return {
                "action": "HOLD",
                "probability": 0.0,
                "reason": f"Исключение при прогнозировании: {e}"
            }
