"""
Модуль управления рисками (Risk Management) и временных фильтров рынка.
1. Рассчитывает динамические Take Profit и Stop Loss на основе индикатора волатильности ATR.
2. Фильтрует ложные сигналы во время низкой ликвидности: блокирует Азиатскую сессию (23:00 – 08:00 GMT).
3. Проверяет рыночные выходные дни (когда рынок золота закрыт).
"""

from datetime import datetime, timezone
import logging
from typing import Tuple, Dict, Any
from config import (
    FILTER_ASIAN_SESSION,
    ASIAN_SESSION_START_HOUR,
    ASIAN_SESSION_END_HOUR,
    SL_ATR_MULTIPLIER,
    TP_ATR_MULTIPLIER
)

logger = logging.getLogger(__name__)


def is_market_open(check_time: datetime = None) -> Tuple[bool, str]:
    """
    Проверяет, открыт ли глобальный рынок золота (XAU/USD).
    Рынок закрывается в пятницу в 21:00 GMT и открывается в воскресенье в 22:00 GMT.
    """
    if check_time is None:
        check_time = datetime.now(timezone.utc)

    # weekday: 0 = Понедельник, ..., 4 = Пятница, 5 = Суббота, 6 = Воскресенье
    weekday = check_time.weekday()
    hour = check_time.hour

    if weekday == 5:  # Суббота
        return False, "Рынок закрыт на выходные (Суббота)"
    if weekday == 4 and hour >= 21:  # Пятница после 21:00 GMT
        return False, "Рынок закрыт на выходные (Пятница вечер)"
    if weekday == 6 and hour < 22:  # Воскресенье до 22:00 GMT
        return False, "Рынок закрыт на выходные (Воскресенье до открытия)"

    return True, "Рынок открыт"


def is_asian_session(check_time: datetime = None) -> bool:
    """
    Проверяет, находится ли текущее время в периоде Азиатской сессии (23:00 - 08:00 GMT).
    В это время ликвидность на золоте низкая, и пробои уровней часто оказываются ложными.
    """
    if not FILTER_ASIAN_SESSION:
        return False

    if check_time is None:
        check_time = datetime.now(timezone.utc)

    hour = check_time.hour

    # Интервал 23:00 - 08:00 GMT
    if ASIAN_SESSION_START_HOUR > ASIAN_SESSION_END_HOUR:
        # Переход через полночь (23, 0, 1, 2, 3, 4, 5, 6, 7)
        if hour >= ASIAN_SESSION_START_HOUR or hour < ASIAN_SESSION_END_HOUR:
            return True
    else:
        if ASIAN_SESSION_START_HOUR <= hour < ASIAN_SESSION_END_HOUR:
            return True

    return False


def get_current_trading_session_name(check_time: datetime = None) -> str:
    """Возвращает название текущей торговой сессии по GMT."""
    if check_time is None:
        check_time = datetime.now(timezone.utc)
    hour = check_time.hour

    if 8 <= hour < 12:
        return "Европейская (Лондон) 🇪🇺"
    elif 12 <= hour < 17:
        return "Лондон + Нью-Йорк (Максимальная активность) 🇪🇺🇺🇸"
    elif 17 <= hour < 21:
        return "Американская (Нью-Йорк) 🇺🇸"
    elif 21 <= hour < 23:
        return "Тихоокеанская / Закрытие 🌙"
    else:
        return "Азиатская (Низкая ликвидность) ⛩️"


# Минимальные защитные дистанции (в пунктах цены) для исключения выбивания случайным рыночным шумом и спредом
MIN_SL_DISTANCES = {
    "XAU/USD": 5.0,     # Золото: минимум $5.00 защитного пространства (не ближе 50 пунктов)
    "GOLD": 5.0,
    "XAG/USD": 0.25,    # Серебро: минимум $0.25
    "SILVER": 0.25,
    "SLV": 0.25,
    "GBP/USD": 0.0020,  # GBP/USD: минимум 20 пунктов
    "EUR/USD": 0.0020,
    "TECH100": 45.0,    # Nasdaq: минимум 45 пунктов
    "QQQ": 45.0,
    "BRENT": 0.80,      # Нефть Brent: минимум $0.80
    "XBR/USD": 0.80,
    "BNO": 0.80,
    "NATGAS": 0.06,     # Природный газ: минимум $0.06
    "NG=F": 0.06
}

MIN_TP_DISTANCES = {
    "XAU/USD": 6.5,     # Золото: минимум $6.50
    "GOLD": 6.5,
    "XAG/USD": 0.35,
    "SILVER": 0.35,
    "SLV": 0.35,
    "GBP/USD": 0.0025,
    "EUR/USD": 0.0025,
    "TECH100": 60.0,
    "QQQ": 60.0,
    "BRENT": 1.10,
    "XBR/USD": 1.10,
    "BNO": 1.10,
    "NATGAS": 0.08,
    "NG=F": 0.08
}


def calculate_risk_levels(
    direction: str,
    entry_price: float,
    atr: float,
    symbol: str = "XAU/USD"
) -> Dict[str, Any]:
    """
    Рассчитывает динамические уровни Take Profit и Stop Loss на основе ATR с защитой от рыночного шума.
    Stop Loss вынесен за пределы естественного микро-шума свечей (1.8 * ATR + минимальный порог актива).
    Также рассчитывает безопасный объем лота для риска ровно $10.
    """
    # Точность знаков после запятой в зависимости от инструмента
    if "GBP" in symbol or "EUR" in symbol:
        decimals = 5
    elif "XAG" in symbol or "SILVER" in symbol.upper():
        decimals = 4
    elif "NATGAS" in symbol.upper() or "GAS" in symbol.upper() or "NG" in symbol.upper():
        decimals = 3
    else:
        decimals = 2

    # Защитный минимальный ATR (не менее 0.15% от цены актива)
    safe_atr = max(atr, entry_price * 0.0015)

    # Динамический расчет расстояний по ATR
    raw_sl = safe_atr * SL_ATR_MULTIPLIER
    raw_tp = safe_atr * TP_ATR_MULTIPLIER

    # Фильтр рыночного шума: гарантируем, что SL не окажется внутри обычного спреда или 1-минутной свечи
    sym_upper = symbol.upper()
    min_sl = entry_price * 0.0025
    for k, v in MIN_SL_DISTANCES.items():
        if k in sym_upper:
            min_sl = v
            break

    min_tp = entry_price * 0.0035
    for k, v in MIN_TP_DISTANCES.items():
        if k in sym_upper:
            min_tp = v
            break

    sl_distance = max(raw_sl, min_sl)
    tp_distance = max(raw_tp, min_tp)

    # Гарантируем адекватное соотношение Risk/Reward (цель должна стоить риска)
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

    # Расчет рекомендуемого объема для риска ровно $10 на сделку в Trading 212
    recommended_lot = 0.01
    if "XAU" in symbol or "GOLD" in symbol.upper():
        # Золото: 1 контракт = 100 унций (или 1 унция на Trading 212 CFD)
        recommended_lot = max(0.01, round(10.0 / max(sl_distance * 100.0, 1.0), 2))
    elif "XAG" in symbol or "SILVER" in symbol.upper():
        # Серебро spot
        recommended_lot = max(0.01, round(10.0 / max(sl_distance * 50.0, 1.0), 2))
    elif "GBP" in symbol:
        recommended_lot = max(0.01, round(10.0 / max(sl_distance * 100000.0, 1.0), 2))
    elif "TECH" in symbol.upper() or "NAS" in symbol.upper() or "100" in symbol:
        # USA Tech 100 CFD: расчет на риск $10
        recommended_lot = max(0.01, round(10.0 / max(sl_distance, 1.0), 2))
    elif "XBR" in symbol or "BRENT" in symbol.upper() or "BNO" in symbol:
        # Нефть Brent
        recommended_lot = max(0.01, round(10.0 / max(sl_distance * 10.0, 1.0), 2))
    elif "NATGAS" in symbol.upper() or "GAS" in symbol.upper() or "NG" in symbol.upper():
        # Природный газ (NATGAS)
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
        "recommended_lot": recommended_lot
    }
