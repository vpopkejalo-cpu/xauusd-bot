"""
Главный модуль автономного Telegram-ассистента трейдера (Мульти-активы + Новости).
Работает в непрерывном цикле 24/7.
Анализирует 5 ключевых активов: XAU/USD, XAG/USD, GBP/USD, TECH100 (QQQ), BRENT (XBR/USD).
Оснащен экономическим календарем («Новостной щит») и анализом сантимента новостей.
"""

import sys
import time
import signal
import logging
import threading
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone
from typing import Optional

from config import (
    validate_config,
    SYMBOL,
    SYMBOLS,
    SYMBOL_NAMES,
    TIMEFRAME,
    POLL_INTERVAL_SECONDS,
    MIN_PROBABILITY,
    RETRAIN_INTERVAL_HOURS,
    ATR_PERIOD,
    NEWS_FILTER_ENABLED
)
from database import (
    init_db,
    save_candles,
    load_candles_from_db,
    save_signal,
    get_open_signals,
    update_signal_status,
    log_candle_analysis,
    is_bot_paused
)
from data_fetcher import DataFetcher, RateLimitError
from indicators import calculate_indicators
from ml_model import TradingMLModel
from news_filter import fetch_news_sentiment, check_economic_calendar_shield
from risk_manager import (
    is_market_open,
    is_asian_session,
    get_current_trading_session_name,
    calculate_risk_levels
)
from telegram_notifier import TelegramNotifier

# Настройка логирования в файл и консоль
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_bot.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("MainBot")


# Встроенный легковесный веб-сервер для облачных платформ (Render, Koyeb и др.)
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"OK: Multi-Asset XAU/USD/GBP/TECH/BRENT Trading Bot is running 24/7\n")

    def log_message(self, format, *args):
        pass


def start_health_server():
    """Запускает HTTP сервер в фоновом потоке для проверки работоспособности облаком."""
    port = int(os.getenv("PORT", "8080"))
    try:
        server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"Встроенный веб-сервер проверки здоровья запущен на порту {port}")
    except Exception as e:
        logger.warning(f"Не удалось запустить веб-сервер проверки здоровья: {e}")


# Флаг для корректной остановки процесса
running = True


def handle_shutdown_signal(signum, frame):
    """Обработчик сигналов завершения от ОС (SIGINT, SIGTERM)."""
    global running
    logger.info(f"Получен сигнал завершения ({signum}). Завершаем работу бота...")
    running = False


signal.signal(signal.SIGINT, handle_shutdown_signal)
signal.signal(signal.SIGTERM, handle_shutdown_signal)


def check_active_trades(current_candle, notifier: TelegramNotifier, symbol: Optional[str] = None):
    """
    Проверяет открытые сигналы на предмет достижения Take Profit или Stop Loss.
    Также закрывает устаревшие сделки (старше 4 часов), чтобы не блокировать бота.
    """
    open_signals = get_open_signals()
    if not open_signals:
        return

    high = float(current_candle["high"])
    low = float(current_candle["low"])
    close = float(current_candle["close"])
    now_utc = datetime.utcnow()

    for sig in open_signals:
        sig_symbol = sig.get("symbol", "XAU/USD")
        if symbol and sig_symbol != symbol:
            continue

        sig_id = sig["id"]
        direction = sig["direction"]
        entry = float(sig["entry_price"])
        tp = float(sig["take_profit"])
        sl = float(sig["stop_loss"])

        # Автоматическое закрытие сделок старше 1 часа (для динамичной внутридневной торговли)
        try:
            created_dt = datetime.fromisoformat(sig["created_at"])
            age_hours = (now_utc - created_dt).total_seconds() / 3600.0
            if age_hours >= 1.0:
                logger.info(f"Сделка #{sig_id} ({sig_symbol} {direction}) открыта более 1 часа. Автоматически закрываем (EXPIRED).")
                pnl = (close - entry) if direction == "BUY" else (entry - close)
                update_signal_status(sig_id, "EXPIRED", close, pnl)
                notifier.send_raw_message(f"⌛ <b>Сигнал #{sig_id} ({sig_symbol} {direction}) снят с отслеживания (прошло 60 мин).</b>\nБот готов к поиску новых сделок.")
                continue
        except Exception as e:
            logger.warning(f"Ошибка проверки возраста сделки #{sig_id}: {e}")

        status = None
        exit_price = None
        pnl = 0.0

        if direction == "BUY":
            if high >= tp:
                status = "TP_HIT"
                exit_price = tp
                pnl = tp - entry
            elif low <= sl:
                status = "SL_HIT"
                exit_price = sl
                pnl = sl - entry
        elif direction == "SELL":
            if low <= tp:
                status = "TP_HIT"
                exit_price = tp
                pnl = entry - tp
            elif high >= sl:
                status = "SL_HIT"
                exit_price = sl
                pnl = entry - sl

        if status:
            logger.info(f"Сделка #{sig_id} ({sig_symbol} {direction}) закрыта: {status}. PnL: {pnl:.2f}")
            update_signal_status(sig_id, status, exit_price, pnl)
            notifier.send_trade_outcome(sig_id, direction, status, pnl, exit_price)


def main():
    logger.info("==================================================")
    logger.info("Запуск мульти-активного торгового помощника с новостным щитом")
    logger.info(f"Отслеживаемые активы ({len(SYMBOLS)}): {', '.join(SYMBOLS)}")
    logger.info("==================================================")

    # 1. Проверка конфигурации .env
    errors = validate_config()
    if errors:
        for err in errors:
            logger.error(f"Ошибка настройки: {err}")
        logger.error("Пожалуйста, заполните параметры в файле .env и перезапустите бота.")
        sys.exit(1)

    # 2. Инициализация базы данных SQLite
    init_db()

    # 3. Запуск веб-сервера для удержания облака (Render, Koyeb и др.)
    start_health_server()

    # 4. Инициализация модулей
    fetcher = DataFetcher()
    ml_model = TradingMLModel()
    notifier = TelegramNotifier()

    # Запуск фонового слушателя кнопок и команд в Telegram
    notifier.start_polling()

    # 5. Отправка тестового стартового уведомления в Telegram со списком всех инструментов
    try:
        notifier.send_startup_notification(interval=TIMEFRAME, min_prob=MIN_PROBABILITY, symbols=SYMBOLS)
    except Exception as e:
        logger.error(f"Не удалось отправить приветственное сообщение в Telegram: {e}")

    # 6. Первоначальная загрузка исторических данных и обучение модели
    logger.info("Начальная загрузка исторических данных по активам для обучения...")
    all_training_dfs = []
    for sym in SYMBOLS:
        try:
            logger.info(f"Загрузка истории для {sym}...")
            candles_df = fetcher.fetch_time_series(symbol=sym, outputsize=150)
            if not candles_df.empty:
                save_candles(candles_df, symbol=sym)
                candles_with_ind = calculate_indicators(candles_df, atr_period=ATR_PERIOD)
                all_training_dfs.append(candles_with_ind)
        except Exception as e:
            logger.error(f"Ошибка начальной загрузки {sym}: {e}")

    # Если есть накопленные данные, обучаем единую масштабированную ML-модель
    if all_training_dfs and not ml_model.is_trained:
        try:
            import pandas as pd
            merged_df = pd.concat(all_training_dfs, axis=0)
            logger.info(f"Выполняем обучение ML модели на {len(merged_df)} свечах...")
            if ml_model.train(merged_df):
                notifier.send_retrain_notification(len(merged_df), ml_model.last_accuracy)
        except Exception as e:
            logger.error(f"Ошибка начального обучения модели: {e}")

    last_retrain_time = time.time()
    last_processed_timestamps = {s: None for s in SYMBOLS}

    logger.info(f"Основной цикл запущен. Сканирование каждые {POLL_INTERVAL_SECONDS} сек. 24/7.")

    # 7. Основной бесконечный цикл
    while running:
        try:
            # 0. Проверка: не поставил ли пользователь бота на паузу кнопкой в Telegram
            if is_bot_paused():
                logger.info("Бот находится на паузе (остановлен пользователем). Запросы к API не отправляются.")
                for _ in range(15):
                    if not running or not is_bot_paused():
                        break
                    time.sleep(1)
                continue

            now_utc = datetime.now(timezone.utc)

            # Проверка открытия мирового рынка золота / форекс (выходные дни)
            market_open, market_reason = is_market_open(now_utc)

            # Последовательный опрос всех активов
            for sym in SYMBOLS:
                if not running or is_bot_paused():
                    break

                # Если рынок закрыт на выходные, металлы и валюты пропускаем (кроме криптовалют, если есть)
                if not market_open and sym in ("XAU/USD", "XAG/USD", "GBP/USD", "QQQ"):
                    continue

                try:
                    # 1. Получаем свежие свечи с защитой лимитов Twelve Data
                    df = fetcher.fetch_time_series(symbol=sym, outputsize=60)
                    if df.empty:
                        continue

                    save_candles(df, symbol=sym)
                    df_with_ind = calculate_indicators(df, atr_period=ATR_PERIOD)
                    latest_candle = df_with_ind.iloc[-1]
                    latest_timestamp = df_with_ind.index[-1]
                    close_p = float(latest_candle["close"])
                    atr_val = float(latest_candle.get("atr", close_p * 0.002))

                    # Кэшируем актуальную цену для отображения в "Статус Бота"
                    notifier.update_price(sym, close_p)

                    # 2. Проверка открытых позиций по этому активу
                    check_active_trades(latest_candle, notifier, symbol=sym)

                    # 3. Проверяем закрытие новой свечи
                    if latest_timestamp != last_processed_timestamps.get(sym):
                        last_processed_timestamps[sym] = latest_timestamp
                        logger.info(f"[{sym}] Анализ свечи: {latest_timestamp} | Close: {close_p:.2f} | ATR: {atr_val:.2f}")

                        # 4. НОВОСТНОЙ ЩИТ (Economic Calendar Shield)
                        news_info = fetch_news_sentiment(sym)
                        if NEWS_FILTER_ENABLED and news_info["shield_active"]:
                            logger.info(f"[{sym}] НОВОСТНОЙ ЩИТ АКТИВЕН: {news_info['shield_reason']}. Сигналы заблокированы для защиты депозита.")
                            log_candle_analysis(sym, close_p, "HOLD", 0.0, "🛡 Новостной щит", news_info["shield_reason"])
                            continue

                        # 5. ФИЛЬТР АЗИАТСКОЙ СЕССИИ (для металлов и валют)
                        if sym in ("XAU/USD", "XAG/USD", "GBP/USD") and is_asian_session(now_utc):
                            logger.info(f"[{sym}] Фильтр Азиатской сессии АКТИВЕН (23:00-08:00 GMT). Пропуск.")
                            log_candle_analysis(sym, close_p, "HOLD", 0.0, "⛩ Азиатская сессия", "Низкая ликвидность")
                            continue

                        # 6. ML ПРОГНОЗ
                        prediction = ml_model.predict(latest_candle)
                        action = prediction["action"]
                        prob = prediction["probability"]
                        reason = prediction["reason"]

                        logger.info(f"[{sym}] ML Решение: {action} ({prob:.1%}) - {reason}")
                        outcome_label = f"ML {action}"

                        # 7. ТРЕНДОВЫЙ ФИЛЬТР (EMA 50)
                        if action in ("BUY", "SELL"):
                            ema_50 = float(latest_candle.get("ema_50", close_p))
                            if action == "BUY" and close_p < ema_50:
                                logger.info(f"[{sym}] Сигнал BUY отфильтрован: цена {close_p:.2f} ниже EMA50 {ema_50:.2f} (нисходящий тренд).")
                                action = "HOLD"
                                outcome_label = "❌ Ниже EMA50 (нисходящий тренд)"
                            elif action == "SELL" and close_p > ema_50:
                                logger.info(f"[{sym}] Сигнал SELL отфильтрован: цена {close_p:.2f} выше EMA50 {ema_50:.2f} (восходящий тренд).")
                                action = "HOLD"
                                outcome_label = "❌ Выше EMA50 (восходящий тренд)"

                        # 8. ПОДТВЕРЖДЕНИЕ НОВОСТНЫМ САНТИМЕНТОМ
                        if action == "BUY" and news_info["score"] <= -0.3:
                            logger.info(f"[{sym}] Сигнал BUY отфильтрован: новостной фон негативный ({news_info['score']}).")
                            action = "HOLD"
                            outcome_label = "❌ Негативные новости"
                        elif action == "SELL" and news_info["score"] >= 0.3:
                            logger.info(f"[{sym}] Сигнал SELL отфильтрован: новостной фон позитивный ({news_info['score']}).")
                            action = "HOLD"
                            outcome_label = "❌ Позитивные новости"

                        # 9. ГЕНЕРАЦИЯ И ОТПРАВКА СИГНАЛА
                        if action in ("BUY", "SELL"):
                            active_signals = get_open_signals()
                            same_dir_signals = [s for s in active_signals if s.get("symbol") == sym and s["direction"] == action]

                            allow_signal = True
                            if same_dir_signals:
                                last_sig = same_dir_signals[-1]
                                try:
                                    last_dt = datetime.fromisoformat(last_sig["created_at"])
                                    sig_age_min = (now_utc - last_dt).total_seconds() / 60.0
                                    if sig_age_min < 45.0:
                                        logger.info(f"[{sym}] Сигнал {action} пропущен: уже открыта свежая сделка #{last_sig['id']} ({sig_age_min:.0f} мин назад).")
                                        allow_signal = False
                                        outcome_label = f"⏳ Свежая сделка #{last_sig['id']}"
                                    else:
                                        logger.info(f"[{sym}] Предыдущая сделка #{last_sig['id']} открыта более 45 мин назад ({sig_age_min:.0f} мин). Разрешаем новый сигнал.")
                                except Exception:
                                    pass

                            if allow_signal:
                                levels = calculate_risk_levels(action, close_p, atr_val, symbol=sym)
                                session_name = get_current_trading_session_name(now_utc)

                                # Сохранение в БД
                                sig_id = save_signal(
                                    symbol=sym,
                                    direction=action,
                                    entry_price=levels["entry_price"],
                                    take_profit=levels["take_profit"],
                                    stop_loss=levels["stop_loss"],
                                    atr=levels["atr"],
                                    probability=prob
                                )

                                logger.info(f"[{sym}] Сформирован сигнал #{sig_id}: {action} по {levels['entry_price']}, TP: {levels['take_profit']}, SL: {levels['stop_loss']}")
                                outcome_label = f"✅ Сигнал #{sig_id} {action}"

                                # Отправка в Telegram с безопасным лотом и кнопками
                                notifier.send_trade_signal(
                                    signal_id=sig_id,
                                    symbol=sym,
                                    direction=action,
                                    entry_price=levels["entry_price"],
                                    stop_loss=levels["stop_loss"],
                                    take_profit=levels["take_profit"],
                                    probability=prob,
                                    atr=levels["atr"],
                                    risk_reward=levels["risk_reward_ratio"],
                                    session_name=session_name,
                                    recommended_lot=levels.get("recommended_lot", 0.01),
                                    news_sentiment=news_info["label"],
                                    headline=news_info["top_headline"]
                                )

                        # Записываем в журнал анализов для кнопки "📊 Статус Бота"
                        log_candle_analysis(sym, close_p, action, prob, outcome_label, reason)

                except RateLimitError as e:
                    logger.warning(f"[{sym}] Rate limit: {e}. Пауза...")
                    time.sleep(10)
                except Exception as e:
                    logger.error(f"Ошибка при обработке {sym}: {e}", exc_info=False)

            # Проверка необходимости периодического самообучения (раз в сутки)
            elapsed_hours = (time.time() - last_retrain_time) / 3600.0
            if elapsed_hours >= RETRAIN_INTERVAL_HOURS:
                logger.info(f"Прошло {elapsed_hours:.1f} ч. Плановое самообучение модели на свежих данных...")
                all_candles = load_candles_from_db(limit=2000)
                if len(all_candles) >= 100:
                    ind_all = calculate_indicators(all_candles, atr_period=ATR_PERIOD)
                    if ml_model.train(ind_all):
                        notifier.send_retrain_notification(len(ind_all), ml_model.last_accuracy)
                        last_retrain_time = time.time()

        except Exception as e:
            logger.error(f"Непредвиденная ошибка в основном цикле: {e}", exc_info=True)

        # Пауза перед следующим циклом проверки
        for _ in range(POLL_INTERVAL_SECONDS):
            if not running:
                break
            time.sleep(1)

    logger.info("Бот завершил работу.")


if __name__ == "__main__":
    main()
