"""
Модуль расчета технических индикаторов для активов торгового бота.
Использует высокоскоростной встроенный векторный движок на чистом pandas/numpy:
- RSI (Relative Strength Index, формула Уайлдера)
- EMA (Экспоненциальные скользящие средние: 9, 21, 50)
- ATR (Average True Range) для динамических TP / SL
- MACD (Moving Average Convergence Divergence: 12, 26, 9)
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def calculate_indicators(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """
    Рассчитывает стандартный набор финансовых индикаторов и фичей для ML.
    Возвращает копию DataFrame с добавленными колонками индикаторов.
    """
    if df.empty or len(df) < 50:
        logger.warning(f"Недостаточно данных для расчета индикаторов (всего {len(df)} свечей, требуется минимум 50)")
        return df

    data = df.copy()
    close = data["close"]
    high = data["high"]
    low = data["low"]

    # 1. Экспоненциальные скользящие средние (EMA 9, 21, 50)
    data["ema_9"] = close.ewm(span=9, adjust=False).mean()
    data["ema_21"] = close.ewm(span=21, adjust=False).mean()
    data["ema_50"] = close.ewm(span=50, adjust=False).mean()

    # 2. Индекс относительной силы RSI 14 (Wilder's RSI)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["rsi"] = 100 - (100 / (1 + rs))
    data["rsi"] = data["rsi"].fillna(50)

    # 3. Средний истинный диапазон ATR 14
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data["atr"] = tr.rolling(window=atr_period).mean()

    # 4. Схождение/расхождение скользящих средних MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    # 5. Относительные фичи для машинного обучения (независимые от номинальной цены)
    # Нормализованный ATR в процентах от цены
    if "atr" in data.columns and "close" in data.columns:
        data["atr_pct"] = (data["atr"] / data["close"]) * 100

    # Расстояние цены от EMA (в процентах)
    if "ema_21" in data.columns:
        data["dist_ema_21"] = (data["close"] - data["ema_21"]) / data["close"] * 100
    if "ema_50" in data.columns:
        data["dist_ema_50"] = (data["close"] - data["ema_50"]) / data["close"] * 100

    # Отношение быстрой EMA к медленной (тренд)
    if "ema_9" in data.columns and "ema_21" in data.columns:
        data["ema_trend"] = (data["ema_9"] - data["ema_21"]) / data["close"] * 100

    # Импульс (Momentum) за последние 1, 3, 5 свечей
    data["return_1"] = data["close"].pct_change(1) * 100
    data["return_3"] = data["close"].pct_change(3) * 100
    data["return_5"] = data["close"].pct_change(5) * 100

    return data
