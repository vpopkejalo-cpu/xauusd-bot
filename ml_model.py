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
import io
import hashlib
import hmac
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

from config import MODEL_PATH, MODELS_DIR, MIN_PROBABILITY
from risk_manager import get_asset_profile
from database import log_training_run, save_model_snapshot, get_latest_model_snapshots

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
        self.model_snapshot_ids: Dict[str, int] = {}
        self.model_snapshot_hashes: Dict[str, str] = {}
        self.model_trained_at: Dict[str, Optional[str]] = {}

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
        """Load local caches, then restore authoritative models from the database."""
        if self.models_dir.exists():
            for file_path in self.models_dir.glob("model_*.joblib"):
                try:
                    data = joblib.load(file_path)
                    symbol, pipeline = data.get("symbol", ""), data.get("pipeline")
                    if not symbol or pipeline is None:
                        continue
                    self.models[symbol] = pipeline
                    self.accuracies[symbol] = float(data.get("accuracy", 0.0))
                    self.samples_counts[symbol] = int(data.get("samples_count", 0))
                    self.model_trained_at[symbol] = data.get("trained_at")
                    logger.info("Загружена локальная модель %s: CV %.1f%%, образцов %s", symbol,
                                self.accuracies[symbol] * 100, self.samples_counts[symbol])
                except Exception as exc:
                    logger.warning("Не удалось загрузить модель %s: %s", file_path, exc)

        if not self.models and os.path.exists(MODEL_PATH):
            try:
                data = joblib.load(MODEL_PATH)
                symbol, pipeline = data.get("symbol", "XAU/USD"), data.get("pipeline")
                if pipeline is not None:
                    self.models[symbol] = pipeline
                    self.accuracies[symbol] = float(data.get("accuracy", 0.0))
                    self.samples_counts[symbol] = int(data.get("samples_count", 0))
                    self.model_trained_at[symbol] = data.get("trained_at")
            except Exception as exc:
                logger.warning("Не удалось загрузить legacy ML модель: %s", exc)

        # Render's local filesystem can be ephemeral; the newest DB artifact is authoritative.
        try:
            for snapshot in get_latest_model_snapshots(include_artifact=True):
                artifact = snapshot.get("artifact")
                if not artifact:
                    continue
                expected = str(snapshot.get("artifact_sha256") or "")
                actual = hashlib.sha256(artifact).hexdigest()
                if not expected or not hmac.compare_digest(expected, actual):
                    logger.error("Хэш снимка модели %s не совпал; восстановление пропущено", snapshot.get("symbol"))
                    continue
                data = joblib.load(io.BytesIO(artifact))
                symbol, pipeline = str(snapshot["symbol"]), data.get("pipeline")
                if pipeline is None:
                    continue
                self.models[symbol] = pipeline
                self.accuracies[symbol] = float(snapshot.get("accuracy") or 0.0)
                self.samples_counts[symbol] = int(snapshot.get("samples_count") or 0)
                self.model_snapshot_ids[symbol] = int(snapshot["id"])
                self.model_snapshot_hashes[symbol] = expected
                self.model_trained_at[symbol] = snapshot.get("trained_at")
                logger.info("Восстановлена модель %s из БД: snapshot=%s, CV=%.1f%%, samples=%s",
                            symbol, snapshot["id"], self.accuracies[symbol] * 100, self.samples_counts[symbol])
        except Exception as exc:
            logger.warning("Не удалось восстановить снимки ML из БД: %s", exc)

        # One-time import of old local models that predate database snapshots.
        for symbol, pipeline in list(self.models.items()):
            if not self.model_snapshot_ids.get(symbol):
                try:
                    self.save_symbol_model(symbol, pipeline, self.accuracies.get(symbol, 0.0),
                                           self.samples_counts.get(symbol, 0),
                                           {"snapshot_reason": "legacy_model_imported_on_startup"},
                                           self.model_trained_at.get(symbol))
                except Exception as exc:
                    logger.error("Не удалось сохранить старую модель %s в БД: %s", symbol, exc)

        if self.models:
            default_symbol = "XAU/USD" if "XAU/USD" in self.models else next(iter(self.models))
            self.pipeline = self.models[default_symbol]
            self.is_trained = True
            self.last_accuracy = self.accuracies.get(default_symbol, 0.0)
            self.samples_count = self.samples_counts.get(default_symbol, 0)
        return len(self.models)

    @staticmethod
    def _weight_snapshot(pipeline: Pipeline) -> Dict[str, Any]:
        scaler = pipeline.named_steps.get("scaler")
        classifier = pipeline.named_steps.get("classifier")
        if scaler is None or classifier is None or not hasattr(classifier, "coef_"):
            raise ValueError("Ожидаются StandardScaler и classifier с coef_")
        means = np.asarray(scaler.mean_, dtype=float)
        scales = np.asarray(scaler.scale_, dtype=float)
        coefficients = np.asarray(classifier.coef_, dtype=float)
        intercepts = np.asarray(classifier.intercept_, dtype=float)
        return {
            "class_order": [int(value) for value in classifier.classes_],
            "class_names": {"0": "SELL", "1": "BUY"},
            "features": list(FEATURE_COLUMNS),
            "standard_scaler": {
                "mean": means.tolist(), "scale": scales.tolist(),
                "variance": np.asarray(scaler.var_, dtype=float).tolist(),
            },
            "standardized_coefficients": coefficients.tolist(),
            "standardized_intercepts": intercepts.tolist(),
            "raw_feature_coefficients": (coefficients / scales).tolist(),
            "raw_intercepts": (intercepts - np.sum(coefficients * means / scales, axis=1)).tolist(),
            "interpretation": "Для двух классов положительный коэффициент смещает log-odds к BUY, отрицательный — к SELL; стандартизированный коэффициент показывает эффект сдвига признака на одно стандартное отклонение.",
        }

    def get_model_snapshot_reference(self, symbol: str) -> Dict[str, Any]:
        return {"snapshot_id": self.model_snapshot_ids.get(symbol),
                "sha256": self.model_snapshot_hashes.get(symbol),
                "trained_at": self.model_trained_at.get(symbol)}

    def save_symbol_model(self, symbol: str, pipeline: Pipeline, accuracy: float, samples_count: int,
                          training_metadata: Optional[Dict[str, Any]] = None,
                          trained_at: Optional[str] = None) -> int:
        """Write model bytes and inspectable parameters to DB before treating it as saved."""
        metadata = dict(training_metadata or {})
        data = {"symbol": symbol, "pipeline": pipeline, "accuracy": float(accuracy),
                "samples_count": int(samples_count), "features": FEATURE_COLUMNS,
                "trained_at": trained_at, "metadata": metadata}
        buffer = io.BytesIO()
        joblib.dump(data, buffer, compress=3)
        artifact = buffer.getvalue()
        artifact_hash = hashlib.sha256(artifact).hexdigest()
        snapshot_id = save_model_snapshot(
            symbol, trained_at, accuracy, samples_count, FEATURE_COLUMNS,
            self._weight_snapshot(pipeline), metadata, artifact_hash, artifact,
        )

        data.update({"snapshot_id": snapshot_id, "artifact_sha256": artifact_hash})
        try:
            joblib.dump(data, self.get_model_path(symbol), compress=3)
        except Exception as exc:
            logger.warning("Снимок %s сохранён в БД, локальный кеш не записан: %s", symbol, exc)
        self.models[symbol] = pipeline
        self.accuracies[symbol] = float(accuracy)
        self.samples_counts[symbol] = int(samples_count)
        self.model_snapshot_ids[symbol] = snapshot_id
        self.model_snapshot_hashes[symbol] = artifact_hash
        self.model_trained_at[symbol] = trained_at
        if symbol == "XAU/USD" or not self.is_trained:
            self.pipeline, self.is_trained = pipeline, True
            self.last_accuracy, self.samples_count = float(accuracy), int(samples_count)
            try:
                joblib.dump(data, MODEL_PATH, compress=3)
            except Exception as exc:
                logger.warning("Снимок %s сохранён в БД, legacy-кеш не записан: %s", symbol, exc)
        logger.info("Снимок модели %s сохранён в БД: id=%s sha256=%s CV=%.1f%% samples=%s",
                    symbol, snapshot_id, artifact_hash, float(accuracy) * 100, samples_count)
        return snapshot_id

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

            training_metadata = {
                "algorithm": "StandardScaler + LogisticRegression",
                "feature_columns": list(FEATURE_COLUMNS),
                "training_start": str(X.index.min()) if len(X.index) else None,
                "training_end": str(X.index.max()) if len(X.index) else None,
                "training_rows": int(len(X)),
                "class_counts": {str(int(k)): int(v) for k, v in y.value_counts().items()},
                "class_names": {"0": "SELL", "1": "BUY"},
                "cross_validation": {
                    "method": "TimeSeriesSplit",
                    "folds": int(n_splits),
                    "fold_accuracies": [float(value) for value in accuracies],
                    "mean_accuracy": float(mean_acc),
                },
                "hyperparameters": {
                    "C": 0.8, "class_weight": "balanced", "max_iter": 1000,
                    "random_state": 42,
                },
            }

            # Commit the reproducible model snapshot before reporting training success.
            trained_at = datetime.now(timezone.utc).isoformat()
            snapshot_id = self.save_symbol_model(
                symbol, pipeline, mean_acc, len(X), training_metadata=training_metadata,
                trained_at=trained_at,
            )
            log_training_run(
                samples_count=len(X), accuracy=mean_acc,
                notes=f"Индивидуальная модель {symbol}; snapshot_id={snapshot_id}; CV folds={n_splits}",
            )

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

            scaler = pipeline.named_steps.get("scaler")
            clf = pipeline.named_steps.get("classifier")

            # Защита от аномальных выбросов: клиппируем значения признаков диапазоном [-3.0, +3.0] сигмы
            if scaler is not None and clf is not None and hasattr(clf, "classes_"):
                scaled_matrix = np.clip(scaler.transform(features_df), -3.0, 3.0)
                classes = list(clf.classes_)
                probabilities = clf.predict_proba(scaled_matrix)[0]
            else:
                scaled_matrix = None
                classes = list(pipeline.classes_)
                probabilities = pipeline.predict_proba(features_df)[0]

            prob_map = dict(zip(classes, probabilities))
            buy_prob = float(prob_map.get(CLASS_BUY, 0.5))
            sell_prob = float(prob_map.get(CLASS_SELL, 0.5))

            # Калибровочный фильтр надежности модели
            acc = float(self.accuracies.get(symbol, self.last_accuracy))
            model_reliable = (acc >= 0.52)
            # Если модель показала слабую точность CV (<52%), ужесточаем порог входа до 72%
            effective_min_prob = min_prob if model_reliable else max(min_prob, 0.72)

            logger.debug(f"[{symbol}] ML Прогноз: BUY={buy_prob:.1%}, SELL={sell_prob:.1%}, Порог={effective_min_prob:.1%} (CV: {acc:.1%})")

            # Извлекаем веса модели и вклад каждого признака (Feature Attribution)
            feature_diagnostics: Dict[str, Any] = {}
            try:
                if scaler is not None and clf is not None and hasattr(clf, "coef_"):
                    scaled_vals = scaled_matrix[0] if scaled_matrix is not None else np.clip(scaler.transform(features_df)[0], -3.0, 3.0)
                    coefs = clf.coef_[0]
                    intercept = float(clf.intercept_[0]) if hasattr(clf, "intercept_") else 0.0
                    contributions = coefs * scaled_vals

                    feat_list = []
                    for col_name, w, s_val, cont in zip(FEATURE_COLUMNS, coefs, scaled_vals, contributions):
                        raw_v = float(candle[col_name]) if pd.notna(candle[col_name]) else 0.0
                        feat_list.append({
                            "name": col_name,
                            "raw_value": round(raw_v, 4),
                            "scaled_value": round(float(s_val), 4),
                            "weight": round(float(w), 4),
                            "contribution": round(float(cont), 4),
                            "bias": "BUY" if cont > 0 else "SELL"
                        })

                    bullish_feats = sorted([f for f in feat_list if f["contribution"] > 0], key=lambda x: x["contribution"], reverse=True)
                    bearish_feats = sorted([f for f in feat_list if f["contribution"] < 0], key=lambda x: x["contribution"])
                    primary_driver = sorted(feat_list, key=lambda x: abs(x["contribution"]), reverse=True)[0] if feat_list else None

                    feature_diagnostics = {
                        "model_accuracy": acc,
                        "model_reliable": model_reliable,
                        "samples_count": int(self.samples_counts.get(symbol, self.samples_count)),
                        "buy_probability": round(buy_prob, 4),
                        "sell_probability": round(sell_prob, 4),
                        "min_probability": round(effective_min_prob, 4),
                        "intercept": round(intercept, 4),
                        "features": feat_list,
                        "top_bullish": [f["name"] for f in bullish_feats[:3]],
                        "top_bearish": [f["name"] for f in bearish_feats[:3]],
                        "primary_driver": primary_driver["name"] if primary_driver else ""
                    }
            except Exception as ex_diag:
                logger.debug(f"[{symbol}] Ошибка извлечения весов признаков: {ex_diag}")

            # 1. Покупка BUY при превышении индивидуального порога актива
            if buy_prob >= effective_min_prob:
                reliability_note = "" if model_reliable else f" [Повышенный порог {effective_min_prob:.1%}, CV={acc:.1%}]"
                return {
                    "action": "BUY",
                    "probability": buy_prob,
                    "reason": f"Вероятность роста {buy_prob:.1%} >= порога {effective_min_prob:.1%}{reliability_note}",
                    "diagnostics": feature_diagnostics,
                    "model_reliable": model_reliable
                }
            # 2. Продажа SELL при превышении индивидуального порога актива
            elif sell_prob >= effective_min_prob:
                reliability_note = "" if model_reliable else f" [Повышенный порог {effective_min_prob:.1%}, CV={acc:.1%}]"
                return {
                    "action": "SELL",
                    "probability": sell_prob,
                    "reason": f"Вероятность падения {sell_prob:.1%} >= порога {effective_min_prob:.1%}{reliability_note}",
                    "diagnostics": feature_diagnostics,
                    "model_reliable": model_reliable
                }
            # 3. Нейтральная зона
            else:
                max_prob = float(max(buy_prob, sell_prob))
                return {
                    "action": "HOLD",
                    "probability": max_prob,
                    "reason": f"Уверенность {max_prob:.1%} ниже порога {effective_min_prob:.1%} — нет устойчивого преимущества",
                    "diagnostics": feature_diagnostics,
                    "model_reliable": model_reliable
                }

        except Exception as e:
            logger.error(f"[{symbol}] Ошибка при прогнозировании: {e}")
            return {
                "action": "HOLD",
                "probability": 0.0,
                "reason": f"Исключение при прогнозировании: {e}"
            }
