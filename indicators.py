"""
Модуль расчета технических индикаторов для активов торгового бота.
Использует высокоскоростной векторный движок на чистом pandas/numpy:
- RSI (Relative Strength Index, формула Уайлдера)
- EMA (Экспоненциальные скользящие средние: 9, 21, 50, 200)
- ATR (Average True Range) для динамических TP / SL
- MACD (Moving Average Convergence Divergence: 12, 26, 9)
- ADX (Average Directional Index: ADX, +DI, -DI) для фильтрации боковика (флэта)
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def calculate_indicators(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """
    Рассчитывает расширенный набор финансовых индикаторов и признаков для ML.
    Возвращает копию DataFrame с добавленными колонками индикаторов.
    """
    if df.empty or len(df) < 50:
        logger.warning(f"Недостаточно данных для расчета индикаторов (всего {len(df)} свечей, требуется минимум 50)")
        return df

    data = df.copy()
    close = data["close"]
    high = data["high"]
    low = data["low"]

    # 1. Экспоненциальные скользящие средние (EMA 9, 21, 50, 200)
    data["ema_9"] = close.ewm(span=9, adjust=False).mean()
    data["ema_21"] = close.ewm(span=21, adjust=False).mean()
    data["ema_50"] = close.ewm(span=50, adjust=False).mean()
    data["ema_200"] = close.ewm(span=200, adjust=False).mean()

    # 2. Индекс относительной силы RSI 14 (Wilder's RSI) и его дельта (Hook)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["rsi"] = 100 - (100 / (1 + rs))
    data["rsi"] = data["rsi"].fillna(50)
    data["rsi_diff"] = data["rsi"].diff().fillna(0.0)

    # 3. Средний истинный диапазон ATR 14
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data["atr"] = tr.rolling(window=atr_period).mean()

    # 4. ADX (Average Directional Index) — Индикатор силы тренда / фильтр флэта
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Экспоненциальное сглаживание по формуле Уайлдера
    tr_smooth = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm, index=data.index).ewm(alpha=1.0 / 14.0, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=data.index).ewm(alpha=1.0 / 14.0, adjust=False).mean()

    plus_di = 100.0 * (plus_dm_s / tr_smooth.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm_s / tr_smooth.replace(0, np.nan))
    di_sum = plus_di + minus_di
    dx = 100.0 * ((plus_di - minus_di).abs() / di_sum.replace(0, np.nan))
    adx = dx.ewm(alpha=1.0 / 14.0, adjust=False).mean().fillna(20.0)

    data["plus_di"] = plus_di.fillna(0.0)
    data["minus_di"] = minus_di.fillna(0.0)
    data["adx"] = adx

    # 5. Схождение/расхождение скользящих средних MACD (12, 26, 9)
    # Нормализуем в % от цены для сопоставимости разных активов в ML
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["macd"] = (ema12 - ema26) / close * 100
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    # 6. Относительные фичи для машинного обучения (независимые от номинальной цены)
    if "atr" in data.columns and "close" in data.columns:
        data["atr_pct"] = (data["atr"] / data["close"]) * 100

    # Расстояние цены от EMA (в процентах)
    if "ema_21" in data.columns:
        data["dist_ema_21"] = (data["close"] - data["ema_21"]) / data["close"] * 100
    if "ema_50" in data.columns:
        data["dist_ema_50"] = (data["close"] - data["ema_50"]) / data["close"] * 100
    if "ema_200" in data.columns:
        data["dist_ema_200"] = (data["close"] - data["ema_200"]) / data["close"] * 100

    # Отношение быстрой EMA к медленной (тренд)
    if "ema_9" in data.columns and "ema_21" in data.columns:
        data["ema_trend"] = (data["ema_9"] - data["ema_21"]) / data["close"] * 100

    # 6. Импульс (Momentum) за последние 1, 3, 5 свечей
    data["return_1"] = data["close"].pct_change(1) * 100
    data["return_3"] = data["close"].pct_change(3) * 100
    data["return_5"] = data["close"].pct_change(5) * 100

    # 7. Старший таймфрейм (Higher Timeframe Proxy H1/H4 из 5m свечей)
    # 12 свечей = 1 час. 48 свечей = 4 часа. 144 свечи = 12 часов.
    htf_fast = close.ewm(span=48, adjust=False).mean()
    htf_slow = close.ewm(span=144, adjust=False).mean()
    data["htf_trend"] = np.where(htf_fast > htf_slow, 1.0, -1.0)
    data["htf_bias"] = (htf_fast - htf_slow) / close * 100

    # 8. Индикатор режима рынка (Market Regime)
    # ADX > 25 = ТРЕНД, ADX <= 25 = ФЛЭТ
    is_trend = data["adx"] > 25.0
    bull_regime = is_trend & (data["plus_di"] > data["minus_di"])
    bear_regime = is_trend & (data["minus_di"] > data["plus_di"])
    data["regime_code"] = np.select([bull_regime, bear_regime], [1.0, -1.0], default=0.0)

    return data
