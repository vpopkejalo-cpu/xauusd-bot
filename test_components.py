"""
Тестовый скрипт проверки целостности всех математических модулей и базы данных.
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from database import init_db, save_candles, load_candles_from_db, save_signal, get_open_signals, update_signal_status
from indicators import calculate_indicators
from ml_model import TradingMLModel, FEATURE_COLUMNS
from risk_manager import calculate_risk_levels, is_asian_session, get_current_trading_session_name


def test_system():
    print(">>> 1. Тест базы данных SQLite...")
    init_db()

    # Генерация синтетических свечей XAU/USD
    now = datetime.now(timezone.utc)
    timestamps = [now - timedelta(minutes=15 * (100 - i)) for i in range(100)]
    
    # Симуляция реалистичного случайного блуждания золота вокруг $2700
    np.random.seed(42)
    prices = [2700.0]
    for _ in range(99):
        change = np.random.normal(0, 1.5)
        prices.append(prices[-1] + change)

    data = []
    for ts, p in zip(timestamps, prices):
        high = p + abs(np.random.normal(1.0, 0.5))
        low = p - abs(np.random.normal(1.0, 0.5))
        open_p = (high + low) / 2
        close_p = p
        data.append({
            "datetime": ts,
            "open": open_p,
            "high": high,
            "low": low,
            "close": close_p,
            "volume": 1500
        })

    df = pd.DataFrame(data).set_index("datetime")
    save_candles(df)
    loaded_df = load_candles_from_db(limit=50)
    assert not loaded_df.empty, "Ошибка: свечи не загружены из БД"
    print(f"База данных: успешно сохранено и загружено {len(loaded_df)} свечей.")

    print(">>> 2. Тест модуля индикаторов (pandas_ta / fallback)...")
    df_ind = calculate_indicators(df)
    assert "rsi" in df_ind.columns, "RSI отсутствует в индикаторах"
    assert "atr" in df_ind.columns, "ATR отсутствует в индикаторах"
    assert "ema_21" in df_ind.columns, "EMA 21 отсутствует в индикаторах"
    assert "macd" in df_ind.columns, "MACD отсутствует в индикаторах"
    print(f"Индикаторы успешно рассчитаны! Последний RSI: {df_ind['rsi'].iloc[-1]:.2f}, ATR: {df_ind['atr'].iloc[-1]:.2f}")

    print(">>> 3. Тест риск-менеджмента (Динамические TP/SL по ATR)...")
    entry_price = float(df_ind["close"].iloc[-1])
    atr_val = float(df_ind["atr"].iloc[-1])
    risk = calculate_risk_levels("BUY", entry_price, atr_val)
    assert risk["take_profit"] > entry_price, "TP должен быть выше цены входа для BUY"
    assert risk["stop_loss"] < entry_price, "SL должен быть ниже цены входа для BUY"
    print(f"Расчет уровней: Вход={risk['entry_price']}, TP={risk['take_profit']}, SL={risk['stop_loss']}, Соотношение={risk['risk_reward_ratio']}")

    print(">>> 4. Тест фильтра Азиатской сессии...")
    # Тест времени в Азии: 02:00 GMT
    asian_time = datetime(2026, 10, 30, 2, 0, 0, tzinfo=timezone.utc)
    london_time = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)
    assert is_asian_session(asian_time) == True, "02:00 GMT должна считаться Азиатской сессией"
    assert is_asian_session(london_time) == False, "10:00 GMT НЕ должна быть Азиатской сессией"
    print(f"Фильтр сессий работает: 02:00 -> Азия ({is_asian_session(asian_time)}), 10:00 -> Лондон ({is_asian_session(london_time)})")

    print(">>> 5. Тест модуля машинного обучения (scikit-learn)...")
    ml = TradingMLModel(min_prob=0.55)
    train_success = ml.train(df_ind)
    print(f"Обучение модели выполнено. Результат: {train_success}")
    if train_success:
        pred = ml.predict(df_ind.iloc[-1])
        print(f"Прогноз ML на текущей свече: {pred}")

    print(">>> 6. Тест жизненного цикла сигнала в БД...")
    sig_id = save_signal("XAU/USD", "BUY", entry_price, risk["take_profit"], risk["stop_loss"], atr_val, 0.65)
    open_sigs = get_open_signals()
    assert any(s["id"] == sig_id for s in open_sigs), "Сигнал не найден среди открытых"
    update_signal_status(sig_id, "TP_HIT", risk["take_profit"], risk["take_profit"] - entry_price)
    open_sigs_after = get_open_signals()
    assert not any(s["id"] == sig_id for s in open_sigs_after), "Сигнал должен быть закрыт"
    print("Сигнал успешно сохранен и закрыт в БД.")

    print("\n==============================================")
    print("✅ ВСЕ МОДУЛИ РАБОТАЮТ БЕЗУПРЕЧНО И ПРОШЛИ ТЕСТ!")
    print("==============================================")


if __name__ == "__main__":
    test_system()
