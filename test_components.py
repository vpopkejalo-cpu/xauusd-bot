"""
Тестовый скрипт проверки целостности всех математических модулей, базы данных и риск-менеджмента.
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from database import (
    init_db,
    save_candles,
    load_candles_from_db,
    save_signal,
    get_open_signals,
    update_signal_status,
    get_today_signal_stats,
    is_kill_switch_active,
    trigger_kill_switch,
    reset_kill_switch
)
from indicators import calculate_indicators
from ml_model import TradingMLModel, FEATURE_COLUMNS
from risk_manager import (
    calculate_risk_levels,
    check_asset_trading_allowed,
    evaluate_kill_switch,
    get_asset_profile,
    get_current_trading_session_name
)


def test_system():
    print(">>> 1. Тест базы данных SQLite...")
    init_db()

    # Генерация синтетических свечей XAU/USD
    now = datetime.now(timezone.utc)
    timestamps = [now - timedelta(minutes=5 * (120 - i)) for i in range(120)]
    
    # Симуляция реалистичного случайного блуждания золота вокруг $2700
    np.random.seed(42)
    prices = [2700.0]
    for _ in range(119):
        change = np.random.normal(0, 1.2)
        prices.append(prices[-1] + change)

    data = []
    for ts, p in zip(timestamps, prices):
        high = p + abs(np.random.normal(0.8, 0.3))
        low = p - abs(np.random.normal(0.8, 0.3))
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
    save_candles(df, symbol="XAU/USD")
    loaded_df = load_candles_from_db(symbol="XAU/USD", limit=50)
    assert not loaded_df.empty, "Ошибка: свечи не загружены из БД"
    print(f"База данных: успешно сохранено и загружено {len(loaded_df)} свечей.")

    print(">>> 2. Тест модуля индикаторов (RSI, ATR, ADX, EMA 200)...")
    df_ind = calculate_indicators(df)
    assert "rsi" in df_ind.columns, "RSI отсутствует в индикаторах"
    assert "atr" in df_ind.columns, "ATR отсутствует в индикаторах"
    assert "adx" in df_ind.columns, "ADX отсутствует в индикаторах"
    assert "ema_200" in df_ind.columns, "EMA 200 отсутствует в индикаторах"
    assert "dist_ema_200" in df_ind.columns, "dist_ema_200 отсутствует в индикаторах"
    print(f"Индикаторы рассчитаны: RSI={df_ind['rsi'].iloc[-1]:.2f}, ATR={df_ind['atr'].iloc[-1]:.2f}, ADX={df_ind['adx'].iloc[-1]:.1f}")

    print(">>> 3. Тест индивидуального риск-менеджмента по активам...")
    entry_price = float(df_ind["close"].iloc[-1])
    atr_val = float(df_ind["atr"].iloc[-1])

    # Золото
    risk_gold = calculate_risk_levels("BUY", entry_price, atr_val, symbol="XAU/USD")
    assert risk_gold["take_profit"] > entry_price, "TP должен быть выше входа для BUY"
    assert risk_gold["stop_loss"] < entry_price, "SL должен быть ниже входа для BUY"
    print(f"XAU/USD: Вход={risk_gold['entry_price']}, TP={risk_gold['take_profit']}, SL={risk_gold['stop_loss']}, R:R={risk_gold['risk_reward_ratio']}")

    # Природный газ (NATGAS)
    risk_gas = calculate_risk_levels("SELL", 2.65, 0.05, symbol="NATGAS")
    assert risk_gas["take_profit"] < 2.65, "TP должен быть ниже входа для SELL"
    assert risk_gas["stop_loss"] > 2.65, "SL должен быть выше входа для SELL"
    print(f"NATGAS: Вход={risk_gas['entry_price']}, TP={risk_gas['take_profit']}, SL={risk_gas['stop_loss']}, R:R={risk_gas['risk_reward_ratio']}")

    print(">>> 4. Тест индивидуальных сессий и фильтра флэта (ADX)...")
    profile_gas = get_asset_profile("NATGAS")
    assert profile_gas["min_probability"] == 0.70, "Для NATGAS должен быть повышенный порог 70%"
    assert profile_gas["min_adx"] == 25.0, "Для NATGAS требуется ADX >= 25.0"

    # Проверка ночной сессии для Nasdaq (02:00 GMT)
    night_time = datetime(2026, 11, 4, 2, 0, 0, tzinfo=timezone.utc)
    allowed, msg = check_asset_trading_allowed("TECH100", check_time=night_time)
    assert not allowed, "Nasdaq не должен торговаться в 02:00 GMT (ночь Азии)"
    print(f"Фильтр сессий активен: TECH100 в 02:00 GMT заблокирован ({msg})")

    # Проверка фильтра флэта ADX
    day_time = datetime(2026, 11, 4, 15, 0, 0, tzinfo=timezone.utc)
    flat_candle = pd.Series({"adx": 14.5})
    allowed_flat, msg_flat = check_asset_trading_allowed("XAU/USD", candle=flat_candle, check_time=day_time)
    assert not allowed_flat, "Низкий ADX должен блокировать вход во флэте"
    print(f"Фильтр флэта активен: XAU/USD при ADX=14.5 заблокирован ({msg_flat})")

    print(">>> 5. Тест модуля машинного обучения с Triple Barrier разметкой...")
    ml = TradingMLModel()
    train_success = ml.train(df_ind, symbol="XAU/USD")
    print(f"Обучение индивидуальной модели XAU/USD: {train_success}")
    assert train_success == True, "Обучение XAU/USD должно пройти успешно"
    pred = ml.predict("XAU/USD", df_ind.iloc[-1])
    print(f"ML Прогноз на свежей свече: {pred}")

    print(">>> 6. Тест Аварийного Kill Switch и демо-режима...")
    # Очищаем состояние Kill Switch
    reset_kill_switch()
    active, _ = is_kill_switch_active()
    assert not active

    # Симулируем 3 стоп-лосса подряд
    sim_stats = {"consecutive_losses": 3, "total_signals": 6, "net_pnl": -18.5}
    
    # В ДЕМО-режиме Kill Switch не должен блокировать непрерывную торговлю
    from config import DEMO_MODE, KILL_SWITCH_ENABLED
    if DEMO_MODE or not KILL_SWITCH_ENABLED:
        ks_triggered, _ = evaluate_kill_switch(sim_stats)
        assert not ks_triggered, "На демо-счете Kill Switch не должен блокировать сделки"
        print("Демо-режим: Kill Switch пассивен, бот не блокирует сбор опыта.")
    
    # Прямая проверка триггера и сброса
    trigger_kill_switch("Тестовая авария", force=True)
    active, current_reason = is_kill_switch_active()
    assert active, "Kill Switch должен быть активен в БД"
    print(f"Kill Switch успешно сработал: {current_reason}")

    # Сброс
    reset_kill_switch()
    active_after, _ = is_kill_switch_active()
    assert not active_after, "Kill Switch должен быть сброшен"
    print("Kill Switch успешно сброшен и разблокирован.")

    print(">>> 7. Тест жизненного цикла торгового сигнала и винрейта за Сегодня...")
    sig_id = save_signal("XAU/USD", "BUY", entry_price, risk_gold["take_profit"], risk_gold["stop_loss"], atr_val, 0.65)
    open_sigs = get_open_signals()
    assert any(s["id"] == sig_id for s in open_sigs), "Сигнал не найден среди открытых"
    update_signal_status(sig_id, "TP_HIT", risk_gold["take_profit"], risk_gold["take_profit"] - entry_price)
    open_sigs_after = get_open_signals()
    assert not any(s["id"] == sig_id for s in open_sigs_after), "Сигнал должен быть закрыт"
    
    # Проверка винрейта строго за СЕГОДНЯ
    from database import get_winrate_stats
    today_stats = get_winrate_stats(1)
    print(f"Статистика за Сегодня (календарный день): сигналов={today_stats['total_signals']}, TP={today_stats['wins']}, SL={today_stats['losses']}")
    assert today_stats["total_signals"] >= 1, "Должен учитывать свежий сигнал за сегодня"
    print("Сигнал успешно сохранен, отслежен и закрыт в БД.")

    print("\n==============================================================")
    print("✅ ВСЕ МОДУЛИ, ИНДИВИДУАЛЬНЫЕ ПРОФИЛИ И ЗАЩИТА УСПЕШНО ПРОШЛИ ТЕСТ!")
    print("==============================================================")


if __name__ == "__main__":
    test_system()
