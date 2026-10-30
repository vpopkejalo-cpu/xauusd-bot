"""
Главный модуль автономного Telegram-ассистента трейдера (XAU/USD).
Работает в непрерывном цикле 24/7 под управлением systemd.
Координирует получение данных, расчет индикаторов, ML-прогнозирование,
проверку открытых сделок и периодическое самообучение.
"""

import sys
import time
import signal
import logging
import threading
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone

from config import (
    validate_config,
    SYMBOL,
    TIMEFRAME,
    POLL_INTERVAL_SECONDS,
    MIN_PROBABILITY,
    RETRAIN_INTERVAL_HOURS,
    ATR_PERIOD
)
from database import (
    init_db,
    save_candles,
    load_candles_from_db,
    save_signal,
    get_open_signals,
    update_signal_status
)
from data_fetcher import DataFetcher, RateLimitError
from indicators import calculate_indicators
from ml_model import TradingMLModel
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
        self.wfile.write(b"OK: XAU/USD Trading Bot is running 24/7\n")

    def log_message(self, format, *args):
        # Отключаем спам в консоль при пингах от мониторинга
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


def check_active_trades(current_candle, notifier: TelegramNotifier):
    """
    Проверяет все открытые сигналы на предмет достижения Take Profit или Stop Loss.
    """
    open_signals = get_open_signals()
    if not open_signals:
        return

    high = float(current_candle["high"])
    low = float(current_candle["low"])
    close = float(current_candle["close"])

    for sig in open_signals:
        sig_id = sig["id"]
        direction = sig["direction"]
        entry = float(sig["entry_price"])
        tp = float(sig["take_profit"])
        sl = float(sig["stop_loss"])

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
            logger.info(f"Сделка #{sig_id} ({direction}) закрыта: {status}. PnL: {pnl:.2f}")
            update_signal_status(sig_id, status, exit_price, pnl)
            notifier.send_trade_outcome(sig_id, direction, status, pnl, exit_price)


def main():
    logger.info("==================================================")
    logger.info("Запуск автономного Telegram-помощника трейдера XAU/USD")
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

    # 5. Отправка тестового стартового уведомления в Telegram
    try:
        notifier.send_startup_notification(interval=TIMEFRAME, min_prob=MIN_PROBABILITY)
    except Exception as e:
        logger.error(f"Не удалось отправить приветственное сообщение в Telegram: {e}")

    # 5. Первоначальная загрузка исторических данных и обучение модели
    logger.info("Загрузка начальных рыночных данных для инициализации...")
    try:
        candles_df = fetcher.fetch_time_series(outputsize=150)
        if not candles_df.empty:
            save_candles(candles_df)
            candles_with_indicators = calculate_indicators(candles_df, atr_period=ATR_PERIOD)
            
            # Если модель еще не обучена, обучаем ее на полученных исторических данных
            if not ml_model.is_trained:
                logger.info("Модель не найдена на диске. Выполняем первичное обучение...")
                if ml_model.train(candles_with_indicators):
                    notifier.send_retrain_notification(len(candles_with_indicators), ml_model.last_accuracy)
    except Exception as e:
        logger.error(f"Ошибка при начальной загрузке данных: {e}", exc_info=True)

    last_retrain_time = time.time()
    last_processed_timestamp = None

    logger.info(f"Основной цикл запущен. Интервал опроса: {POLL_INTERVAL_SECONDS} сек. 24/7.")

    # 6. Основной бесконечный цикл
    while running:
        try:
            now_utc = datetime.now(timezone.utc)

            # Проверка открытия мирового рынка золота (выходные дни)
            market_open, market_reason = is_market_open(now_utc)
            if not market_open:
                logger.info(f"{market_reason}. Ожидание открытия рынка...")
                time.sleep(min(POLL_INTERVAL_SECONDS, 600))
                continue

            # Получаем свежие свечи через Twelve Data API
            df = fetcher.fetch_time_series(outputsize=80)
            if df.empty:
                logger.warning("Получен пустой датасет, пропускаем итерацию.")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            save_candles(df)
            df_with_ind = calculate_indicators(df, atr_period=ATR_PERIOD)
            latest_candle = df_with_ind.iloc[-1]
            latest_timestamp = df_with_ind.index[-1]

            # Проверка открытых позиций на срабатывание TP/SL
            check_active_trades(latest_candle, notifier)

            # Проверяем, появилась ли новая закрывшаяся свеча
            if latest_timestamp != last_processed_timestamp:
                last_processed_timestamp = latest_timestamp
                logger.info(f"Анализ закрытой свечи: {latest_timestamp} | Close: {latest_candle['close']:.2f} | ATR: {latest_candle.get('atr', 0):.2f}")

                # Проверка фильтра Азиатской сессии (микроструктура рынка)
                in_asian = is_asian_session(now_utc)
                if in_asian:
                    logger.info("Фильтр Азиатской сессии АКТИВЕН (23:00-08:00 GMT). Сигналы заблокированы для исключения ложных пробоев.")
                else:
                    # ML модуль анализирует свечу и рассчитывает вероятности
                    prediction = ml_model.predict(latest_candle)
                    action = prediction["action"]
                    prob = prediction["probability"]
                    reason = prediction["reason"]

                    logger.info(f"ML Решение: {action} (вероятность {prob:.1%}) - {reason}")

                    if action in ("BUY", "SELL"):
                        # Проверяем, нет ли уже активной сделки в том же направлении
                        active_signals = get_open_signals()
                        same_dir_signals = [s for s in active_signals if s["direction"] == action]

                        if same_dir_signals:
                            logger.info(f"Сигнал {action} пропущен: уже открыта активная сделка #{same_dir_signals[-1]['id']}.")
                        else:
                            # Расчет динамических TP и SL на основе ATR
                            entry_price = float(latest_candle["close"])
                            atr_val = float(latest_candle.get("atr", entry_price * 0.002))
                            session_name = get_current_trading_session_name(now_utc)

                            levels = calculate_risk_levels(action, entry_price, atr_val)

                            # Сохраняем в базу данных
                            sig_id = save_signal(
                                symbol=SYMBOL,
                                direction=action,
                                entry_price=levels["entry_price"],
                                take_profit=levels["take_profit"],
                                stop_loss=levels["stop_loss"],
                                atr=levels["atr"],
                                probability=prob
                            )

                            logger.info(f"Сформирован сигнал #{sig_id}: {action} по {entry_price:.2f}, TP: {levels['take_profit']}, SL: {levels['stop_loss']}")

                            # Отправляем форматированное сообщение в Telegram
                            notifier.send_trade_signal(
                                direction=action,
                                entry_price=levels["entry_price"],
                                stop_loss=levels["stop_loss"],
                                take_profit=levels["take_profit"],
                                probability=prob,
                                atr=levels["atr"],
                                risk_reward=levels["risk_reward_ratio"],
                                session_name=session_name
                            )

            # Проверка необходимости периодического самообучения (раз в RETRAIN_INTERVAL_HOURS)
            elapsed_hours = (time.time() - last_retrain_time) / 3600.0
            if elapsed_hours >= RETRAIN_INTERVAL_HOURS:
                logger.info(f"Прошло {elapsed_hours:.1f} ч. Запуск планового самообучения модели...")
                all_candles = load_candles_from_db(limit=1000)
                if len(all_candles) >= 80:
                    ind_all = calculate_indicators(all_candles, atr_period=ATR_PERIOD)
                    if ml_model.train(ind_all):
                        notifier.send_retrain_notification(len(ind_all), ml_model.last_accuracy)
                        last_retrain_time = time.time()

        except RateLimitError as e:
            logger.warning(f"Ограничение частоты запросов API (Rate Limit): {e}. Ожидание...")
        except Exception as e:
            logger.error(f"Непредвиденная ошибка в основном цикле: {e}", exc_info=True)

        # Пауза перед следующей проверкой
        for _ in range(POLL_INTERVAL_SECONDS):
            if not running:
                break
            time.sleep(1)

    logger.info("Бот завершил работу.")


if __name__ == "__main__":
    main()
