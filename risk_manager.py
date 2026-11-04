"""
Модуль индивидуального риск-менеджмента (Risk Management) и микроструктурных фильтров рынка.
Реализует специализированную логику под каждый из 6 активов:
- XAU/USD (Золото)
- XAG/USD (Серебро)
- GBP/USD (Британский фунт)
- TECH100 (Nasdaq 100)
- BRENT (Сырая нефть)
- NATGAS (Природный газ)

Включает:
1. Индивидуальные торговые сессии (активные часы по GMT).
2. Пороги ADX для отсечения флэта / боковика.
3. Индивидуальные параметры TP/SL по ATR с учетом специфики инструмента.
4. Аварийный Kill Switch для защиты депозита от серийных стопов и дневной просадки.
"""

from datetime import datetime, timezone
import logging
from typing import Tuple, Dict, Any, Optional
import pandas as pd

from config import (
    FILTER_ASIAN_SESSION,
    ASIAN_SESSION_START_HOUR,
    ASIAN_SESSION_END_HOUR,
    KILL_SWITCH_ENABLED,
    KILL_SWITCH_MAX_CONSECUTIVE_LOSSES,
    KILL_SWITCH_MAX_DAILY_SIGNALS,
    KILL_SWITCH_MAX_DAILY_LOSS
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------
# ИНДИВИДУАЛЬНЫЕ ПРОФИЛИ ТОРГОВЫХ АКТИВОВ
# -------------------------------------------------------------
ASSET_PROFILES: Dict[str, Dict[str, Any]] = {
    "XAU/USD": {
        "name": "Gold",
        "category": "Precious Metals",
        "active_hours_gmt": (7, 21),       # Лондон + Нью-Йорк
        "min_adx": 20.0,                   # Требуется развитие направленного движения
        "min_probability": 0.64,
        "sl_atr_mult": 1.8,
        "tp_atr_mult": 2.4,                # R:R = 1:1.33
        "min_sl_dist": 5.0,
        "min_tp_dist": 6.5,
        "max_signals_per_day": 3,
        "cooldown_minutes": 45,
        "strict_ema200": True,
        "notes": "Высокая ликвидность на Лондоне/Нью-Йорке. Ночные азиатские шпильки блокируются."
    },
    "XAG/USD": {
        "name": "Silver",
        "category": "Precious Metals (High Beta)",
        "active_hours_gmt": (7, 21),
        "min_adx": 22.0,                   # Серебро требует более явного тренда
        "min_probability": 0.65,
        "sl_atr_mult": 2.0,                # Более широкий стоп из-за агрессивных теней свечей
        "tp_atr_mult": 2.6,                # R:R = 1:1.30
        "min_sl_dist": 0.28,
        "min_tp_dist": 0.40,
        "max_signals_per_day": 3,
        "cooldown_minutes": 50,
        "strict_ema200": True,
        "notes": "Высокая волатильность (бета выше золота в 2 раза). Защищен от ложного сбития стопа."
    },
    "GBP/USD": {
        "name": "GBP/USD",
        "category": "Forex Major",
        "active_hours_gmt": (7, 17),       # Строго европейская и начало американской сессии
        "min_adx": 18.0,                   # Форекс развивает устойчивые тренды при среднем ADX
        "min_probability": 0.62,
        "sl_atr_mult": 1.6,
        "tp_atr_mult": 2.2,                # R:R = 1:1.37
        "min_sl_dist": 0.0020,
        "min_tp_dist": 0.0028,
        "max_signals_per_day": 2,
        "cooldown_minutes": 60,
        "strict_ema200": True,
        "notes": "Торгуется строго во время работы Лондона. В Азии и поздним вечером — глухой боковик."
    },
    "TECH100": {
        "name": "USA Tech 100 (Nasdaq)",
        "category": "Equity Index",
        "active_hours_gmt": (13, 21),      # Премаркет США и регулярная сессия Wall Street
        "min_adx": 22.0,
        "min_probability": 0.65,
        "sl_atr_mult": 1.8,
        "tp_atr_mult": 2.5,                # R:R = 1:1.38
        "min_sl_dist": 45.0,
        "min_tp_dist": 65.0,
        "max_signals_per_day": 3,
        "cooldown_minutes": 45,
        "strict_ema200": True,
        "notes": "Американский индекс. Торговля ночью или утром бесполезна — ждем открытия США."
    },
    "BRENT": {
        "name": "Brent Crude Oil",
        "category": "Energy Commodity",
        "active_hours_gmt": (7, 20),
        "min_adx": 22.0,
        "min_probability": 0.65,
        "sl_atr_mult": 1.8,
        "tp_atr_mult": 2.4,                # R:R = 1:1.33
        "min_sl_dist": 0.80,
        "min_tp_dist": 1.20,
        "max_signals_per_day": 2,
        "cooldown_minutes": 60,
        "strict_ema200": True,
        "notes": "Сырая нефть. Трендовый актив с чувствительностью к геополитике и запасам EIA."
    },
    "NATGAS": {
        "name": "Natural Gas",
        "category": "Volatile Commodity",
        "active_hours_gmt": (13, 19),      # Пиковые ликвидные часы американского газового хаба
        "min_adx": 25.0,                   # СТРОГИЙ фильтр: никаких сделок без мощного импульса!
        "min_probability": 0.70,           # СТРОГИЙ фильтр: уверенность модели не менее 70%
        "sl_atr_mult": 2.2,                # Широкий запас под спред CFD и гэпы
        "tp_atr_mult": 3.0,                # R:R = 1:1.36
        "min_sl_dist": 0.08,
        "min_tp_dist": 0.12,
        "max_signals_per_day": 2,
        "cooldown_minutes": 75,
        "strict_ema200": True,
        "notes": "Экстремально волатильный актив (widow-maker). Вход только при безупречном сетапе."
    }
}


def get_asset_profile(symbol: str) -> Dict[str, Any]:
    """
    Возвращает профиль настроек для указанного актива с поддержкой алиасов (тикеров).
    Если инструмент неизвестен, возвращает сбалансированный профиль по умолчанию.
    """
    sym = symbol.strip().upper()

    # Алиасы
    if sym in ("XAU/USD", "GOLD", "XAU"):
        key = "XAU/USD"
    elif sym in ("XAG/USD", "SILVER", "SLV", "XAG"):
        key = "XAG/USD"
    elif sym in ("GBP/USD", "GBPUSD", "CABLE"):
        key = "GBP/USD"
    elif sym in ("TECH100", "QQQ", "NAS100", "US100", "NDX"):
        key = "TECH100"
    elif sym in ("BRENT", "XBR/USD", "BNO", "OIL", "CRUDE"):
        key = "BRENT"
    elif sym in ("NATGAS", "NG=F", "GAS", "UNG"):
        key = "NATGAS"
    else:
        key = "XAU/USD"

    return ASSET_PROFILES.get(key, ASSET_PROFILES["XAU/USD"])


def is_market_open(check_time: Optional[datetime] = None) -> Tuple[bool, str]:
    """
    Проверяет, открыт ли глобальный рынок металлов/сырья/форекс.
    Закрывается в пятницу в 21:00 GMT и открывается в воскресенье в 22:00 GMT.
    """
    if check_time is None:
        check_time = datetime.now(timezone.utc)

    weekday = check_time.weekday()
    hour = check_time.hour

    if weekday == 5:  # Суббота
        return False, "Рынок закрыт на выходные (Суббота)"
    if weekday == 4 and hour >= 21:  # Пятница вечер
        return False, "Рынок закрыт на выходные (Пятница вечер)"
    if weekday == 6 and hour < 22:  # Воскресенье до открытия
        return False, "Рынок закрыт на выходные (Воскресенье до открытия)"

    return True, "Рынок открыт"


def is_asian_session(check_time: Optional[datetime] = None) -> bool:
    """Проверяет интервал Азиатской сессии (23:00 - 08:00 GMT)."""
    if not FILTER_ASIAN_SESSION:
        return False

    if check_time is None:
        check_time = datetime.now(timezone.utc)

    hour = check_time.hour
    if ASIAN_SESSION_START_HOUR > ASIAN_SESSION_END_HOUR:
        if hour >= ASIAN_SESSION_START_HOUR or hour < ASIAN_SESSION_END_HOUR:
            return True
    else:
        if ASIAN_SESSION_START_HOUR <= hour < ASIAN_SESSION_END_HOUR:
            return True
    return False


def get_current_trading_session_name(check_time: Optional[datetime] = None) -> str:
    """Возвращает название текущей торговой сессии по GMT."""
    if check_time is None:
        check_time = datetime.now(timezone.utc)
    hour = check_time.hour

    if 7 <= hour < 12:
        return "Европейская (Лондон) 🇪🇺"
    elif 12 <= hour < 17:
        return "Лондон + Нью-Йорк (Максимальная активность) 🇪🇺🇺🇸"
    elif 17 <= hour < 21:
        return "Американская (Нью-Йорк) 🇺🇸"
    elif 21 <= hour < 23:
        return "Тихоокеанская / Закрытие 🌙"
    else:
        return "Азиатская (Низкая ликвидность) ⛩️"


def check_asset_trading_allowed(
    symbol: str,
    candle: Optional[pd.Series] = None,
    check_time: Optional[datetime] = None
) -> Tuple[bool, str]:
    """
    Индивидуальная проверка готовности актива к генерации сигналов:
    1. Глобальные выходные.
    2. Разрешенные часы торговой сессии по GMT именно для этого инструмента.
    3. Фильтр трендовой активности ADX (защита от распила во флэте).
    """
    if check_time is None:
        check_time = datetime.now(timezone.utc)

    profile = get_asset_profile(symbol)

    # 1. Выходные дни
    market_open, reason = is_market_open(check_time)
    if not market_open:
        return False, f"[{symbol}] {reason}"

    # 2. Индивидуальные часы сессии
    start_h, end_h = profile["active_hours_gmt"]
    hour = check_time.hour
    if hour < start_h or hour >= end_h:
        return False, f"[{symbol}] Вне торговой сессии актива ({start_h:02d}:00–{end_h:02d}:00 GMT). Защита от ночного шума."

    # 3. Фильтр флэта по ADX (если данные свечи переданы)
    if candle is not None and "adx" in candle:
        adx_val = float(candle.get("adx", 20.0))
        min_adx = profile["min_adx"]
        if adx_val < min_adx:
            return False, f"[{symbol}] ADX={adx_val:.1f} < {min_adx:.0f} — рынок в боковике (флэт). Вход заблокирован."

    return True, "OK"


def evaluate_kill_switch(daily_stats: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Проверяет дневные показатели на срабатывание аварийного Kill Switch:
    - 3 или более стоп-лосса подряд.
    - Достижение дневного потолка по количеству сигналов (защита от овертрейдинга).
    - Превышение порога чистого дневного убытка.
    """
    if not KILL_SWITCH_ENABLED:
        return False, ""

    consecutive_losses = daily_stats.get("consecutive_losses", 0)
    total_signals = daily_stats.get("total_signals", 0)
    net_pnl = daily_stats.get("net_pnl", 0.0)

    # 1. Серия стоп-лоссов подряд
    if consecutive_losses >= KILL_SWITCH_MAX_CONSECUTIVE_LOSSES:
        return True, f"Зафиксировано {consecutive_losses} стоп-лосса подряд за день. Рыночный режим агрессивен к стратегии."

    # 2. Дневной лимит выданных сигналов (ограничиваем спам и овертрейдинг)
    if total_signals >= KILL_SWITCH_MAX_DAILY_SIGNALS:
        return True, f"Достигнут дневной лимит качественных сигналов ({total_signals}/{KILL_SWITCH_MAX_DAILY_SIGNALS}). Торговля завершена до завтра."

    # 3. Превышение максимального дневного убытка
    if net_pnl <= -abs(KILL_SWITCH_MAX_DAILY_LOSS):
        return True, f"Дневной убыток ({net_pnl:.1f} п.) превысил допустимый порог ({KILL_SWITCH_MAX_DAILY_LOSS} п.)."

    return False, ""


def calculate_risk_levels(
    direction: str,
    entry_price: float,
    atr: float,
    symbol: str = "XAU/USD"
) -> Dict[str, Any]:
    """
    Рассчитывает динамические уровни Take Profit и Stop Loss с учетом индивидуального профиля актива.
    """
    profile = get_asset_profile(symbol)

    # Точность знаков после запятой
    sym_upper = symbol.upper()
    if "GBP" in sym_upper or "EUR" in sym_upper:
        decimals = 5
    elif "XAG" in sym_upper or "SILVER" in sym_upper:
        decimals = 4
    elif "NATGAS" in sym_upper or "GAS" in sym_upper or "NG" in sym_upper:
        decimals = 3
    else:
        decimals = 2

    # Защитный минимальный ATR (не менее 0.15% от цены актива)
    safe_atr = max(atr, entry_price * 0.0015)

    # Индивидуальные множители ATR из профиля актива
    sl_mult = profile["sl_atr_mult"]
    tp_mult = profile["tp_atr_mult"]
    min_sl = profile["min_sl_dist"]
    min_tp = profile["min_tp_dist"]

    raw_sl = safe_atr * sl_mult
    raw_tp = safe_atr * tp_mult

    sl_distance = max(raw_sl, min_sl)
    tp_distance = max(raw_tp, min_tp)

    # Гарантируем привлекательное соотношение Risk/Reward
    if tp_distance < sl_distance * 1.15:
        tp_distance = round(sl_distance * 1.25, decimals)

    if direction == "BUY":
        stop_loss = round(entry_price - sl_distance, decimals)
        take_profit = round(entry_price + tp_distance, decimals)
    elif direction == "SELL":
        stop_loss = round(entry_price + sl_distance, decimals)
        take_profit = round(entry_price - tp_distance, decimals)
    else:
        raise ValueError(f"Неизвестное направление сделки: {direction}")

    # Расчет рекомендуемого объема для риска $10 на Trading 212
    recommended_lot = 0.01
    if "XAU" in sym_upper or "GOLD" in sym_upper:
        recommended_lot = max(0.01, round(10.0 / max(sl_distance * 100.0, 1.0), 2))
    elif "XAG" in sym_upper or "SILVER" in sym_upper:
        recommended_lot = max(0.01, round(10.0 / max(sl_distance * 50.0, 1.0), 2))
    elif "GBP" in sym_upper:
        recommended_lot = max(0.01, round(10.0 / max(sl_distance * 100000.0, 1.0), 2))
    elif "TECH" in sym_upper or "NAS" in sym_upper or "100" in sym_upper:
        recommended_lot = max(0.01, round(10.0 / max(sl_distance, 1.0), 2))
    elif "BRENT" in sym_upper or "XBR" in sym_upper or "BNO" in sym_upper:
        recommended_lot = max(0.01, round(10.0 / max(sl_distance * 10.0, 1.0), 2))
    elif "NATGAS" in sym_upper or "GAS" in sym_upper or "NG" in sym_upper:
        recommended_lot = max(0.01, round(10.0 / max(sl_distance * 100.0, 1.0), 2))
    else:
        recommended_lot = max(0.01, round(10.0 / max(sl_distance, 1.0), 2))

    risk_reward_ratio = round(tp_distance / sl_distance, 2)

    return {
        "entry_price": round(entry_price, decimals),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "atr": round(safe_atr, decimals),
        "sl_distance": round(sl_distance, decimals),
        "tp_distance": round(tp_distance, decimals),
        "risk_reward_ratio": f"1:{risk_reward_ratio}",
        "recommended_lot": recommended_lot,
        "asset_profile": profile
    }
