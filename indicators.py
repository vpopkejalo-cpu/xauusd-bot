"""
Модуль расчета технических индикаторов для XAU/USD.
Использует библиотеку pandas_ta для стандартизированного и точного расчета:
- RSI (Relative Strength Index)
- EMA (Экспоненциальные скользящие средние: 9, 21, 50)
- ATR (Average True Range) для динамических TP / SL
- MACD (Moving Average Convergence Divergence)
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Проверяем доступность pandas_ta
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    logger.warning("Библиотека pandas_ta не найдена. Будут использованы резервные математические формулы.")


def calculate_indicators(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """
    Рассчитывает набор стандартных индикаторов и добавляет их в DataFrame.
    Возвращает копию DataFrame с добавленными колонками индикаторов.
    """
    if df.empty or len(df) < 50:
        logger.warning(f"Недостаточно данных для расчета индикаторов (всего {len(df)} свечей, требуется минимум 50)")
        return df

    data = df.copy()

    if HAS_PANDAS_TA:
        try:
            # RSI 14
            data["rsi"] = data.ta.rsi(close="close", length=14)

            # Скользящие средние EMA 9, 21, 50
            data["ema_9"] = data.ta.ema(close="close", length=9)
            data["ema_21"] = data.ta.ema(close="close", length=21)
            data["ema_50"] = data.ta.ema(close="close", length=50)

            # ATR 14
            data["atr"] = data.ta.atr(high="high", low="low", close="close", length=atr_period)

            # MACD (12, 26, 9)
            macd_df = data.ta.macd(close="close", fast=12, slow=26, signal=9)
            if macd_df is not None and not macd_df.empty:
                data["macd"] = macd_df.iloc[:, 0]
                data["macd_hist"] = macd_df.iloc[:, 1]
                data["macd_signal"] = macd_df.iloc[:, 2]

        except Exception as e:
            logger.error(f"Ошибка при расчете через pandas_ta: {e}. Переключаемся на базовый расчет.")
            _calculate_indicators_fallback(data, atr_period)
    else:
        _calculate_indicators_fallback(data, atr_period)

    # Относительные фичи для машинного обучения (независимые от абсолютной цены золота)
    # Нормализованный ATR в процентах от цены
    if "atr" in data.columns and "close" in data.columns:
        data["atr_pct"] = (data["atr"] / data["close"]) * 100

    # Расстояние цены от EMA
    if "ema_21" in data.columns:
        data["dist_ema_21"] = (data["close"] - data["ema_21"]) / data["close"] * 100
    if "ema_50" in data.columns:
        data["dist_ema_50"] = (data["close"] - data["ema_50"]) / data["close"] * 100

    # Отношение быстрой EMA к медленной (тренд)
    if "ema_9" in data.columns and "ema_21" in data.columns:
        data["ema_trend"] = (data["ema_9"] - data["ema_21"]) / data["close"] * 100

    # Изменение цены за последние 1, 3, 5 свечей (Momentum)
    data["return_1"] = data["close"].pct_change(1) * 100
    data["return_3"] = data["close"].pct_change(3) * 100
    data["return_5"] = data["close"].pct_change(5) * 100

    return data


def _calculate_indicators_fallback(data: pd.DataFrame, atr_period: int = 14):
    """
    Резервный стандартный расчет на чистом pandas/numpy в случае проблем со сторонней библиотекой.
    """
    close = data["close"]
    high = data["high"]
    low = data["low"]

    # EMA
    data["ema_9"] = close.ewm(span=9, adjust=False).mean()
    data["ema_21"] = close.ewm(span=21, adjust=False).mean()
    data["ema_50"] = close.ewm(span=50, adjust=False).mean()

    # RSI 14 (Wilder's RSI)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["rsi"] = 100 - (100 / (1 + rs))
    data["rsi"] = data["rsi"].fillna(50)

    # ATR 14
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data["atr"] = tr.rolling(window=atr_period).mean()

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]
