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
    diag_data = {
        "ml_prediction": pred["diagnostics"],
        "risk": risk_gold,
        "market_context": {"rsi": 66.2, "adx": 26.5, "plus_di": 23.3, "minus_di": 21.0}
    }
    sig_id = save_signal("XAU/USD", "BUY", entry_price, risk_gold["take_profit"], risk_gold["stop_loss"], atr_val, 0.65, diagnostic_data=diag_data)
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

    print(">>> 8. Тест генератора глубоких диагностических логов (TXT/CSV) и весов...")
    from report_generator import generate_diagnostics_report
    rep = generate_diagnostics_report(days=1)
    assert rep["signals_count"] >= 1, "Отчет должен содержать сегодняшние сигналы"
    assert "ТАБЛИЦА ВЕСОВ" in rep["txt_content"] or "ДЕТАЛИЗАЦИЯ ВЕСОВ" in rep["txt_content"] or "ДЕЙСТВИЕ" in rep["txt_content"], "В отчете должен быть блок весов модели"
    assert "csv_content" in rep and len(rep["csv_content"]) > 0, "CSV отчет не должен быть пустым"
    print(f"Диагностический отчет успешно сгенерирован: сигналов={rep['signals_count']}, TXT={len(rep['txt_content'])} байт, CSV={len(rep['csv_content'])} байт")

    print(">>> 9. Тест Рыночных Режимов, Пирамидинга и AI-Аналитика...")
    from risk_manager import (
        detect_market_regime,
        check_pyramiding_allowed,
        check_symbol_consecutive_losses
    )
    from ai_analyst import generate_ai_session_analysis

    # Проверка новых индикаторов: rsi_diff, htf_trend, regime_code
    assert "rsi_diff" in df_ind.columns, "rsi_diff отсутствует в индикаторах"
    assert "htf_trend" in df_ind.columns, "htf_trend отсутствует в индикаторах"
    assert "regime_code" in df_ind.columns, "regime_code отсутствует в индикаторах"

    # Проверка детектора режимов
    sample_candle = df_ind.iloc[-1]
    regime_res = detect_market_regime(sample_candle)
    assert "regime" in regime_res and "allow_buy" in regime_res, "detect_market_regime вернул некорректный ответ"
    print(f"Детектор режимов: {regime_res['regime']} (BUY={regime_res['allow_buy']}, SELL={regime_res['allow_sell']})")

    # Проверка проверки пирамидинга и контролируемого добора на откате
    pyr_allowed, pyr_reason = check_pyramiding_allowed("XAU/USD", "BUY", entry_price, atr_val, max_positions=3)
    assert isinstance(pyr_allowed, bool), "check_pyramiding_allowed должен вернуть boolean статус"
    print(f"Проверка пирамидинга: allowed={pyr_allowed}, reason='{pyr_reason}'")

    # Юнит-тесты движка масштабирования и защиты от мартингейла
    # 1. Первая позиция по новому активу всегда разрешена
    ok_first, r_first = check_pyramiding_allowed("EUR/USD_TEST", "BUY", 1.0500, 0.0050)
    assert ok_first and "Первая позиция" in r_first, f"Ошибка первой позиции: {r_first}"

    # 2. Низкая уверенность при откате должна быть заблокирована (<90%)
    # Имитируем активную позицию BUY по 2000.0, SL=1980, цена откатила до 1995.0, prob=0.85
    from unittest.mock import patch
    mock_active = [{"id": 101, "symbol": "MOCK", "direction": "BUY", "entry_price": 2000.0, "stop_loss": 1980.0}]
    with patch("database.get_open_signals", return_value=mock_active):
        # 2.1 Низкая уверенность (<90%) -> Блокировка
        ok_low, r_low = check_pyramiding_allowed("MOCK", "BUY", 1995.0, 10.0, probability=0.85)
        assert not ok_low and "ниже строгого порога" in r_low, f"Низкая уверенность не заблокирована: {r_low}"

        # 2.2 Высокая уверенность (>=90%) и здоровый откат -> Санкционирован 1-й добор
        ok_hi, r_hi = check_pyramiding_allowed("MOCK", "BUY", 1994.0, 10.0, probability=0.92)
        assert ok_hi and "Добор на откате" in r_hi, f"Добор >=90% не разрешен: {r_hi}"

        # 2.3 Опасная зона перед Stop Loss (цена 1983 при SL=1980 -> потеряно 85% дистанции) -> Блокировка
        ok_sl, r_sl = check_pyramiding_allowed("MOCK", "BUY", 1983.0, 10.0, probability=0.95)
        assert not ok_sl and "Опасная зона" in r_sl, f"Вход перед стопом не заблокирован: {r_sl}"

    # 3. Лимит усреднений: если 1 добор уже совершен, второй категорически запрещен
    mock_two_trades = [
        {"id": 101, "symbol": "MOCK", "direction": "BUY", "entry_price": 2000.0, "stop_loss": 1980.0},
        {"id": 102, "symbol": "MOCK", "direction": "BUY", "entry_price": 1992.0, "stop_loss": 1980.0}
    ]
    with patch("database.get_open_signals", return_value=mock_two_trades):
        ok_limit, r_limit = check_pyramiding_allowed("MOCK", "BUY", 1988.0, 10.0, probability=0.95)
        assert not ok_limit and "Лимит усреднений исчерпан" in r_limit, f"Второе усреднение не заблокировано: {r_limit}"

    print("Движок пирамидинга и контролируемого добора: все 5 юнит-тестов успешно пройдены.")

    # Проверка заморозки после серии стопов
    is_cooling, cool_reason = check_symbol_consecutive_losses("XAU/USD", cooldown_minutes=90)
    assert isinstance(is_cooling, bool), "check_symbol_consecutive_losses должен вернуть boolean статус"
    print(f"Проверка серии стопов: cooling={is_cooling}, reason='{cool_reason}'")

    # Проверка запуска AI-Аналитика (локальный аудит или Gemini)
    ai_report = generate_ai_session_analysis(days=1)
    assert "AI-АНАЛИТИК" in ai_report or "РЕЦЕНЗИЯ" in ai_report, "AI-аналитик не сформировал отчет"
    print(f"AI-Аналитик успешно сформировал рецензию ({len(ai_report)} символов).")

    print(">>> 10. Тест модуля экспресс-проверки всех систем (System Health Check)...")
    from system_checker import run_full_system_check
    diag = run_full_system_check()
    assert "ДИАГНОСТИЧЕСКАЯ ПАНЕЛЬ" in diag["report_text"], "Диагностический отчет не сформирован"
    assert "api_results" in diag and "ml_results" in diag, "Словарь диагностики не полон"
    print(f"Диагностика систем успешно завершена за {diag['elapsed_seconds']} с:")
    print(f"API проверено: {len(diag['api_results'])}, Моделей проанализировано: {len(diag['ml_results'])}")

    print(">>> 11. Тест отчетов на след. день, атрибуции овернайт-сделок и подтверждения паузы...")
    from database import get_day_utc_bounds, get_winrate_stats, set_bot_paused, is_bot_paused, get_db, _adapt_sql
    from telegram_notifier import TelegramNotifier, _format_daily_report_message, get_daily_report_keyboard

    # 11.1 Проверка расчета временных границ дня пользователя
    st_today, end_today, d_today, lbl_today = get_day_utc_bounds("today")
    st_yest, end_yest, d_yest, lbl_yest = get_day_utc_bounds("yesterday")
    assert st_today == end_yest, f"Граница вчерашнего дня должна стыковаться с началом сегодняшнего: {end_yest} vs {st_today}"
    print(f"Границы дней (Europe/Chisinau): Сегодня={lbl_today} [{st_today}..{end_today}], Вчера={lbl_yest} [{st_yest}..{end_yest}]")

    # 11.2 Проверка атрибуции овернайт-сделки:
    # Имитируем сделку, открытую вчера (днем) и закрытую сегодня (ночью)
    yest_entry_time = "2026-11-16T12:00:00"
    today_exit_time = "2026-11-17T03:15:00"

    with get_db() as cursor:
        cursor.execute(_adapt_sql("""
            INSERT INTO signals (created_at, symbol, direction, entry_price, take_profit, stop_loss,
                                 atr, probability, status, closed_at, exit_price, pnl)
            VALUES (?, 'OVERNIGHT_TEST', 'BUY', 2000.0, 2010.0, 1990.0, 5.0, 0.85, 'TP_HIT', ?, 2010.0, 10.0)
        """), (yest_entry_time, today_exit_time))

    # Сделка ДОЛЖНА попасть в отчет за ВЧЕРА (по дате открытия)
    stats_yest = get_winrate_stats(target_date="yesterday")
    yest_signals = [s for s in stats_yest.get("signals", []) if s.get("symbol") == "OVERNIGHT_TEST"]
    assert len(yest_signals) == 1, "Овернайт-сделка не найдена в отчете за вчера (день открытия)"
    assert yest_signals[0]["status"] == "TP_HIT" and yest_signals[0]["pnl"] == 10.0, "Результат овернайта не учтен во вчерашнем дне"
    print("✅ Атрибуция овернайта подтверждена: сделка закрыта сегодня, но строго учтена в дне открытия (вчера)!")

    # Сделка НЕ ДОЛЖНА попасть в отчет за СЕГОДНЯ (так как открыта была вчера)
    stats_today = get_winrate_stats(target_date="today")
    today_signals = [s for s in stats_today.get("signals", []) if s.get("symbol") == "OVERNIGHT_TEST"]
    assert len(today_signals) == 0, "Овернайт-сделка ошибочно попала в отчет за сегодня (дублирование)"
    print("✅ Защита от задвоения: сделка вчерашнего дня не попала в сегодняшний отчет!")

    # 11.3 Проверка форматирования отчета за вчера
    formatted_rep = _format_daily_report_message(stats_yest, "Вчера")
    assert "ОТЧЕТ ЗА СЕССИЮ: ВЧЕРА" in formatted_rep, "Форматирование отчета за вчера некорректно"
    assert "OVERNIGHT_TEST" in formatted_rep, "Сделка отсутствует в тексте отчета"
    print("✅ Текстовый отчет за прошлый день успешно сформирован с овернайт-пометкой.")

    # 11.4 Проверка установки на паузу
    notifier = TelegramNotifier()
    notifier._execute_pause(has_active_deals=True)
    assert is_bot_paused(), "Бот должен быть поставлен на паузу"
    set_bot_paused(False)
    assert not is_bot_paused(), "Бот должен быть успешно снят с паузы"
    print("✅ Механика паузы с подтверждением и сохранением овернайт-позиций работает безупречно.")

    print("\n==============================================================")
    print("✅ ВСЕ МОДУЛИ, ИНДИВИДУАЛЬНЫЕ ПРОФИЛИ И ЗАЩИТА УСПЕШНО ПРОШЛИ ТЕСТ!")
    print("==============================================================")


if __name__ == "__main__":
    test_system()
