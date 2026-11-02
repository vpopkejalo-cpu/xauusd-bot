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


def calculate_risk_levels(
    direction: str,
    entry_price: float,
    atr: float,
    symbol: str = "XAU/USD"
) -> Dict[str, Any]:
    """
    Рассчитывает динамические уровни Take Profit и Stop Loss на основе ATR.
    Сконфигурирован на умеренный и безопасный риск (SL = 0.8 * ATR, TP = 1.6 * ATR).
    Также рассчитывает безопасный объем лота для риска ровно $10.
    """
    # Защитный минимальный ATR
    safe_atr = max(atr, entry_price * 0.001)

    sl_distance = safe_atr * SL_ATR_MULTIPLIER
    tp_distance = safe_atr * TP_ATR_MULTIPLIER

    # Точность знаков после запятой в зависимости от инструмента
    if "GBP" in symbol or "EUR" in symbol:
        decimals = 5
    elif "XAG" in symbol or "SILVER" in symbol.upper():
        decimals = 4
    else:
        decimals = 2

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
    elif "XBR" in symbol or "BRENT" in symbol.upper() or "BNO" in symbol:
        # Нефть Brent
        recommended_lot = max(0.01, round(10.0 / max(sl_distance * 10.0, 1.0), 2))
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
