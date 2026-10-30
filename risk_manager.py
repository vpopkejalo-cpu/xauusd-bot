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
    atr: float
) -> Dict[str, Any]:
    """
    Рассчитывает динамические уровни Take Profit и Stop Loss на основе ATR.
    
    direction: "BUY" или "SELL"
    entry_price: цена входа
    atr: текущее значение Average True Range
    """
    # Защитный минимальный ATR (если вдруг рынок сверхспокоен или данные сжаты)
    safe_atr = max(atr, entry_price * 0.001)

    sl_distance = safe_atr * SL_ATR_MULTIPLIER
    tp_distance = safe_atr * TP_ATR_MULTIPLIER

    if direction == "BUY":
        stop_loss = round(entry_price - sl_distance, 2)
        take_profit = round(entry_price + tp_distance, 2)
    elif direction == "SELL":
        stop_loss = round(entry_price + sl_distance, 2)
        take_profit = round(entry_price - tp_distance, 2)
    else:
        raise ValueError(f"Неизвестное направление сделки: {direction}")

    # Соотношение риск / прибыль (Risk/Reward Ratio)
    risk_reward_ratio = round(tp_distance / sl_distance, 2)

    return {
        "entry_price": round(entry_price, 2),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "atr": round(safe_atr, 2),
        "sl_distance": round(sl_distance, 2),
        "tp_distance": round(tp_distance, 2),
        "risk_reward_ratio": f"1:{risk_reward_ratio}"
    }
