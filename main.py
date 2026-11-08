"""
Главный модуль автономного Telegram-ассистента трейдера (Индивидуальная логика под 6 активов).
Работает в непрерывном цикле 24/7.
Анализирует 6 ключевых инструментов:
- XAU/USD (Золото)
- XAG/USD (Серебро)
- GBP/USD (Британский фунт)
- TECH100 (Nasdaq 100)
- BRENT (Сырая нефть)
- NATGAS (Природный газ)

Ключевые механизмы защиты:
1. Индивидуальные модели ML под каждый инструмент (разметка Triple Barrier).
2. Аварийный Kill Switch (автоматическая заморозка торговли при серии из 3 стопов или дневной просадке).
3. Индивидуальные торговые сессии (защита от ночного азиатского распила по Nasdaq, Фунту и Газу).
4. Фильтр трендовой активности ADX (запрет входа во флэте).
5. Фильтр старшего тренда (EMA 200).
6. Новостной щит (Economic Calendar Shield).
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
    TRAINING_CANDLES_COUNT,
    RETRAIN_INTERVAL_HOURS,
    ATR_PERIOD,
    NEWS_FILTER_ENABLED,
    DEMO_MODE,
    KILL_SWITCH_ENABLED,
    KILL_SWITCH_MAX_DAILY_SIGNALS
)
from database import (
    init_db,
    save_candles,
    load_candles_from_db,
    save_signal,
    get_open_signals,
    log_candle_analysis,
    is_bot_paused,
    consume_retrain_request,
    set_signal_message_id,
    get_today_signal_stats,
    get_today_symbol_signal_count,
    is_kill_switch_active,
    trigger_kill_switch,
    reset_kill_switch
)
from data_fetcher import DataFetcher, RateLimitError
from indicators import calculate_indicators
from ml_model import TradingMLModel
from news_filter import fetch_news_sentiment, check_economic_calendar_shield
from risk_manager import (
    is_market_open,
    get_current_trading_session_name,
    calculate_risk_levels,
    check_asset_trading_allowed,
    evaluate_kill_switch,
    get_asset_profile,
    detect_market_regime,
    check_pyramiding_allowed,
    check_symbol_consecutive_losses
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


# Встроенный легковесный веб-сервер для облачных платформ (Render, Koyeb, GCP)
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"OK: Multi-Asset Trading Assistant (XAU, XAG, GBP, TECH, BRENT, NATGAS) is running 24/7\n")

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


def main():
    logger.info("==================================================")
    logger.info("Запуск мульти-активного торгового помощника (Специализированная логика)")
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

    # 3. Запуск веб-сервера для удержания облака
    start_health_server()

    # 4. Инициализация ключевых компонентов
    fetcher = DataFetcher()
    ml_model = TradingMLModel()
    notifier = TelegramNotifier()

    # Запуск фонового слушателя кнопок и команд в Telegram
    notifier.start_polling()

    # 5. Стартовое уведомление в Telegram
    try:
        notifier.send_startup_notification(interval=TIMEFRAME, min_prob=MIN_PROBABILITY, symbols=SYMBOLS)
    except Exception as e:
        logger.error(f"Не удалось отправить приветственное сообщение в Telegram: {e}")

    # 6. Первоначальная загрузка истории и индивидуальное обучение моделей по каждому активу
    logger.info(f"Начальная загрузка истории и обучение индивидуальных моделей (до {TRAINING_CANDLES_COUNT} свечей на актив)...")
    trained_assets = []
    for sym in SYMBOLS:
        try:
            logger.info(f"[{sym}] Загрузка исторических свечей...")
            candles_df = fetcher.fetch_time_series(symbol=sym, outputsize=TRAINING_CANDLES_COUNT)
            if not candles_df.empty:
                save_candles(candles_df, symbol=sym)
                candles_with_ind = calculate_indicators(candles_df, atr_period=ATR_PERIOD)
                # Обучаем индивидуальную модель под этот актив
                success = ml_model.train(candles_with_ind, symbol=sym)
                if success:
                    trained_assets.append(sym)
            time.sleep(1)  # Защитная пауза для уважения rate limits Twelve Data
        except Exception as e:
            logger.error(f"[{sym}] Ошибка начальной загрузки/обучения: {e}")

    if trained_assets:
        try:
            notifier.send_retrain_notification(len(trained_assets) * TRAINING_CANDLES_COUNT, ml_model.last_accuracy)
        except Exception:
            pass

    last_retrain_time = time.time()
    last_processed_timestamps = {s: None for s in SYMBOLS}
    last_kill_switch_alert_sent = False

    logger.info(f"Основной цикл запущен. Мониторинг каждые {POLL_INTERVAL_SECONDS} сек. 24/7.")

    # 7. Главный бесконечный цикл
    while running:
        try:
            # 0. Проверка ручной паузы пользователем
            if is_bot_paused():
                logger.info("Бот на паузе (остановлен пользователем). Запросы к API приостановлены.")
                for _ in range(15):
                    if not running or not is_bot_paused():
                        break
                    time.sleep(1)
                continue

            now_utc = datetime.now(timezone.utc)

            # 1. ПРОВЕРКА АВАРИЙНОГО KILL SWITCH
            daily_stats = get_today_signal_stats()
            ks_active, ks_reason = is_kill_switch_active()

            # Если Kill Switch еще не был зафиксирован в БД, оцениваем дневные показатели
            if not ks_active and KILL_SWITCH_ENABLED and not DEMO_MODE:
                should_trigger, trigger_reason = evaluate_kill_switch(daily_stats)
                if should_trigger:
                    trigger_kill_switch(trigger_reason)
                    ks_active = True
                    ks_reason = trigger_reason
                    if not last_kill_switch_alert_sent:
                        try:
                            notifier.send_kill_switch_alert(trigger_reason, daily_stats)
                            last_kill_switch_alert_sent = True
                        except Exception as ex:
                            logger.error(f"Не удалось отправить уведомление о Kill Switch: {ex}")

            if ks_active and not DEMO_MODE:
                logger.info(f"🛑 Kill Switch активен: {ks_reason}. Генерация сигналов приостановлена. Анализ свечей продолжается 24/7.")
            else:
                last_kill_switch_alert_sent = False

            # Проверка открытия мирового рынка (выходные)
            market_open, market_reason = is_market_open(now_utc)

            # 2. ПОСЛЕДОВАТЕЛЬНЫЙ АНАЛИЗ ВСЕХ АКТИВОВ
            for sym in SYMBOLS:
                if not running or is_bot_paused():
                    break

                # Пропускаем закрытые на выходные рынки
                if not market_open and sym in ("XAU/USD", "XAG/USD", "GBP/USD", "QQQ"):
                    continue

                try:
                    profile = get_asset_profile(sym)

                    # 1. Получаем свежие свечи с защитой от rate limits
                    df = fetcher.fetch_time_series(symbol=sym, outputsize=70)
                    if df.empty:
                        continue

                    save_candles(df, symbol=sym)
                    df_with_ind = calculate_indicators(df, atr_period=ATR_PERIOD)
                    latest_candle = df_with_ind.iloc[-1]
                    latest_timestamp = df_with_ind.index[-1]
                    close_p = float(latest_candle["close"])
                    atr_val = float(latest_candle.get("atr", close_p * 0.002))
                    adx_val = float(latest_candle.get("adx", 20.0))

                    # Кэшируем котировку для кнопки "📊 Статус Бота"
                    notifier.update_price(sym, close_p)

                    # 2. Проверяем закрытие новой свечи
                    if latest_timestamp != last_processed_timestamps.get(sym):
                        last_processed_timestamps[sym] = latest_timestamp
                        logger.info(f"[{sym}] Анализ свечи: {latest_timestamp} | Close: {close_p:.2f} | ATR: {atr_val:.2f} | ADX: {adx_val:.1f}")

                        # 3. ИНДИВИДУАЛЬНЫЙ ФИЛЬТР СЕССИИ И БОКОВИКА (ADX)
                        allowed, block_reason = check_asset_trading_allowed(sym, candle=latest_candle, check_time=now_utc)
                        if not allowed:
                            logger.info(f"[{sym}] Пропуск: {block_reason}")
                            log_candle_analysis(sym, close_p, "HOLD", 0.0, "⏸ Вне сессии / Флэт", block_reason)
                            continue

                        # 3.1. ЗАЩИТНЫЙ ТАЙМ-АУТ ПОСЛЕ СЕРИИ СТОПОВ ПО АКТИВУ
                        loss_ok, loss_reason = check_symbol_consecutive_losses(sym, cooldown_minutes=90)
                        if not loss_ok:
                            logger.info(loss_reason)
                            log_candle_analysis(sym, close_p, "HOLD", 0.0, "⏸ Тайм-аут актива", loss_reason)
                            continue

                        # 3.2. MARKET REGIME ENGINE: Определение режима рынка (Тренд vs Флэт)
                        regime_info = detect_market_regime(latest_candle)

                        # 4. ДНЕВНОЙ ЛИМИТ СИГНАЛОВ ПО ЭТОМУ АКТИВУ (в боевом режиме)
                        sym_signals_today = get_today_symbol_signal_count(sym)
                        max_sym_signals = profile.get("max_signals_per_day", 3)
                        if not DEMO_MODE and sym_signals_today >= max_sym_signals:
                            logger.info(f"[{sym}] Достигнут лимит сигналов на день ({sym_signals_today}/{max_sym_signals}). Отдых.")
                            log_candle_analysis(sym, close_p, "HOLD", 0.0, "⏸ Дневной лимит актива", f"{sym_signals_today}/{max_sym_signals} сигналов")
                            continue

                        # 5. ОБЩИЙ ДНЕВНОЙ ЛИМИТ СИГНАЛОВ ВСЕГО БОТА (в боевом режиме)
                        if not DEMO_MODE and daily_stats.get("total_signals", 0) >= KILL_SWITCH_MAX_DAILY_SIGNALS:
                            logger.info(f"[{sym}] Достигнут дневной потолок сигналов ({KILL_SWITCH_MAX_DAILY_SIGNALS}).")
                            log_candle_analysis(sym, close_p, "HOLD", 0.0, "⏸ Дневной лимит бота", "Защита от овертрейдинга")
                            continue

                        # 6. НОВОСТНОЙ ЩИТ (Economic Calendar Shield)
                        news_info = fetch_news_sentiment(sym)
                        if NEWS_FILTER_ENABLED and news_info["shield_active"]:
                            logger.info(f"[{sym}] НОВОСТНОЙ ЩИТ АКТИВЕН: {news_info['shield_reason']}. Блокировка.")
                            log_candle_analysis(sym, close_p, "HOLD", 0.0, "🛡 Новостной щит", news_info["shield_reason"])
                            continue

                        # 7. ИНДИВИДУАЛЬНЫЙ ML ПРОГНОЗ
                        prediction = ml_model.predict(sym, latest_candle)
                        action = prediction["action"]
                        prob = prediction["probability"]
                        reason = prediction["reason"]

                        logger.info(f"[{sym}] ML Вердикт: {action} ({prob:.1%}) — {reason} | Режим: {regime_info['regime']}")
                        outcome_label = f"ML {action}"

                        # 8. СТРУКТУРНЫЙ, РЕЖИМНЫЙ И МУЛЬТИ-ТАЙМФРЕЙМ ФИЛЬТР
                        if action in ("BUY", "SELL"):
                            ema_200 = float(latest_candle.get("ema_200", close_p))
                            ema_50 = float(latest_candle.get("ema_50", close_p))
                            ema_21 = float(latest_candle.get("ema_21", close_p))
                            rsi_val = float(latest_candle.get("rsi", 50.0))
                            rsi_diff = float(latest_candle.get("rsi_diff", 0.0))
                            htf_trend = float(latest_candle.get("htf_trend", 0.0))

                            # 8.1. СООТВЕТСТВИЕ РЕЖИМУ РЫНКА (Запрет контртренда в сильном тренде!)
                            if action == "BUY" and not regime_info["allow_buy"]:
                                logger.info(f"[{sym}] BUY заблокирован: {regime_info['description']}. Запрет покупки против нисходящего тренда.")
                                action = "HOLD"
                                outcome_label = "❌ Контртренд (Bear Trend)"
                            elif action == "SELL" and not regime_info["allow_sell"]:
                                logger.info(f"[{sym}] SELL заблокирован: {regime_info['description']}. Запрет продажи против восходящего тренда.")
                                action = "HOLD"
                                outcome_label = "❌ Контртренд (Bull Trend)"

                            # 8.2. СТАРШИЙ ЧАСОВОЙ ТРЕНД (HTF H1 Proxy)
                            elif action == "BUY" and htf_trend < 0.0 and profile.get("strict_ema200", True):
                                if prob < 0.72:
                                    logger.info(f"[{sym}] BUY отклонен: старший часовой тренд H1 нисходящий (уверенность {prob:.1%} < 72%).")
                                    action = "HOLD"
                                    outcome_label = "❌ Против H1 тренда"
                            elif action == "SELL" and htf_trend > 0.0 and profile.get("strict_ema200", True):
                                if prob < 0.72:
                                    logger.info(f"[{sym}] SELL отклонен: старший часовой тренд H1 восходящий (уверенность {prob:.1%} < 72%).")
                                    action = "HOLD"
                                    outcome_label = "❌ Против H1 тренда"

                            # 8.3. ЗАЩИТА ОТ ЛОВЛИ ПАДАЮЩИХ НОЖЕЙ (RSI Hook)
                            elif action == "BUY" and rsi_val < 32.0 and rsi_diff < 0.0:
                                logger.info(f"[{sym}] BUY отклонен: ловля падающего ножа (RSI={rsi_val:.1f} падает, разворотный крюк отсутствует).")
                                action = "HOLD"
                                outcome_label = "❌ Ловля ножа (RSI падает)"
                            elif action == "SELL" and rsi_val > 68.0 and rsi_diff > 0.0:
                                logger.info(f"[{sym}] SELL отклонен: продажа против ракеты (RSI={rsi_val:.1f} растет, разворотный крюк отсутствует).")
                                action = "HOLD"
                                outcome_label = "❌ Шорт против ракеты"

                            # 8.4. КЛАССИЧЕСКИЙ ФИЛЬТР СРЕДНИХ (EMA 200 / EMA 50)
                            elif action == "BUY":
                                if profile.get("strict_ema200", True) and close_p < ema_200:
                                    logger.info(f"[{sym}] BUY отклонен: цена ({close_p:.2f}) ниже базовой EMA 200 ({ema_200:.2f}). Тренд нисходящий.")
                                    action = "HOLD"
                                    outcome_label = "❌ Против тренда EMA200"
                                elif close_p < ema_50 or ema_21 < ema_50:
                                    logger.info(f"[{sym}] BUY отклонен: цена ниже EMA50 или нисходящий веер EMA.")
                                    action = "HOLD"
                                    outcome_label = "❌ Против тренда EMA50"
                                elif rsi_val > 68.0:
                                    logger.info(f"[{sym}] BUY отклонен: RSI={rsi_val:.1f} перекуплен (>68). Опасность покупки на пике.")
                                    action = "HOLD"
                                    outcome_label = f"❌ Перекуплен RSI={rsi_val:.1f}"

                            elif action == "SELL":
                                if profile.get("strict_ema200", True) and close_p > ema_200:
                                    logger.info(f"[{sym}] SELL отклонен: цена ({close_p:.2f}) выше базовой EMA 200 ({ema_200:.2f}). Тренд восходящий.")
                                    action = "HOLD"
                                    outcome_label = "❌ Против тренда EMA200"
                                elif close_p > ema_50 or ema_21 > ema_50:
                                    logger.info(f"[{sym}] SELL отклонен: цена выше EMA50 или восходящий веер EMA.")
                                    action = "HOLD"
                                    outcome_label = "❌ Против тренда EMA50"
                                elif rsi_val < 32.0:
                                    logger.info(f"[{sym}] SELL отклонен: RSI={rsi_val:.1f} перепродан (<32). Опасность шорта на дне.")
                                    action = "HOLD"
                                    outcome_label = f"❌ Перепродан RSI={rsi_val:.1f}"

                        # 9. ПОДТВЕРЖДЕНИЕ НОВОСТНЫМ САНТИМЕНТОМ
                        if action == "BUY" and news_info["score"] <= -0.5:
                            logger.info(f"[{sym}] BUY отклонен: сильный негативный новостной фон ({news_info['score']}).")
                            action = "HOLD"
                            outcome_label = "❌ Негативные новости"
                        elif action == "SELL" and news_info["score"] >= 0.5:
                            logger.info(f"[{sym}] SELL отклонен: сильный позитивный новостной фон ({news_info['score']}).")
                            action = "HOLD"
                            outcome_label = "❌ Позитивные новости"

                        # 10. ГЕНЕРАЦИЯ СИГНАЛА И ДВИЖОК ПИРАМИДИНГА (ДОЛИВКА В ПРИБЫЛЬ)
                        if action in ("BUY", "SELL"):
                            pyramid_ok, pyramid_msg = check_pyramiding_allowed(sym, action, close_p, atr_val)
                            if not pyramid_ok:
                                logger.info(f"[{sym}] Сигнал {action} отклонен: {pyramid_msg}")
                                outcome_label = "⏳ Усреднение запрещено"
                                action = "HOLD"
                            else:
                                logger.info(f"[{sym}] {pyramid_msg}")

                                # Проверка кулдауна с корректным учетом часовых поясов
                                active_signals = get_open_signals()
                                same_dir_signals = [s for s in active_signals if s.get("symbol") == sym and s["direction"] == action]
                                allow_signal = True

                                if same_dir_signals:
                                    last_sig = same_dir_signals[-1]
                                    try:
                                        last_dt = datetime.fromisoformat(last_sig["created_at"])
                                        if last_dt.tzinfo is None:
                                            last_dt = last_dt.replace(tzinfo=timezone.utc)
                                        sig_age_min = (now_utc - last_dt).total_seconds() / 60.0
                                        # При пирамидинге (доливке в прибыль) пауза между сделками 10 мин
                                        pyramid_cd = 10.0
                                        if sig_age_min < pyramid_cd:
                                            logger.info(f"[{sym}] Пауза доливки: прошло {sig_age_min:.0f} из {pyramid_cd:.0f} мин.")
                                            allow_signal = False
                                            outcome_label = f"⏳ Пауза доливки #{last_sig['id']}"
                                    except Exception as ex_dt:
                                        logger.warning(f"Ошибка проверки времени кулдауна: {ex_dt}")

                                if allow_signal:
                                    logger.info(f"[{sym}] {pyramid_msg}")
                                    levels = calculate_risk_levels(action, close_p, atr_val, symbol=sym)
                                    session_name = get_current_trading_session_name(now_utc)

                                    # Собираем исчерпывающие диагностические данные для детального анализа сделки
                                    try:
                                        from zoneinfo import ZoneInfo
                                        local_dt = now_utc.astimezone(ZoneInfo("Europe/Chisinau"))
                                        local_time_str = local_dt.strftime("%Y-%m-%d %H:%M:%S")
                                        is_after_17 = local_dt.hour >= 17
                                    except Exception:
                                        local_time_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
                                        is_after_17 = False

                                    diagnostic_payload = {
                                        "symbol": sym,
                                        "direction": action,
                                        "created_at_utc": now_utc.isoformat(),
                                        "created_at_local": local_time_str,
                                        "is_after_17_chisinau": is_after_17,
                                        "session_name": session_name,
                                        "category": profile.get("category", ""),
                                        "market_regime": regime_info.get("regime", "RANGE_FLAT"),
                                        "pyramiding_info": pyramid_msg,
                                        "prices": {
                                            "open": float(latest_candle.get("open", close_p)),
                                            "high": float(latest_candle.get("high", close_p)),
                                            "low": float(latest_candle.get("low", close_p)),
                                            "close": float(close_p),
                                            "volume": float(latest_candle.get("volume", 0.0))
                                        },
                                        "indicators": {
                                            "rsi": float(latest_candle.get("rsi", 50.0)),
                                            "atr": float(atr_val),
                                            "atr_pct": float(latest_candle.get("atr_pct", 0.0)),
                                            "adx": float(adx_val),
                                            "plus_di": float(latest_candle.get("plus_di", 0.0)),
                                            "minus_di": float(latest_candle.get("minus_di", 0.0)),
                                            "macd": float(latest_candle.get("macd", 0.0)),
                                            "macd_hist": float(latest_candle.get("macd_hist", 0.0)),
                                            "ema_21": float(latest_candle.get("ema_21", close_p)),
                                            "ema_50": float(latest_candle.get("ema_50", close_p)),
                                            "ema_200": float(latest_candle.get("ema_200", close_p)),
                                            "dist_ema_200": float(latest_candle.get("dist_ema_200", 0.0)),
                                            "ema_trend": float(latest_candle.get("ema_trend", 0.0)),
                                            "htf_trend": float(latest_candle.get("htf_trend", 0.0)),
                                            "return_1": float(latest_candle.get("return_1", 0.0)),
                                            "return_3": float(latest_candle.get("return_3", 0.0)),
                                            "return_5": float(latest_candle.get("return_5", 0.0)),
                                        },
                                        "ml_prediction": prediction.get("diagnostics", {}),
                                        "ml_action": action,
                                        "ml_probability": float(prob),
                                        "ml_reason": reason,
                                        "news": {
                                            "shield_active": bool(news_info.get("shield_active", False)),
                                            "shield_reason": str(news_info.get("shield_reason", "")),
                                            "score": float(news_info.get("score", 0.0)),
                                            "label": str(news_info.get("label", "Neutral")),
                                            "top_headline": str(news_info.get("top_headline", "")),
                                            "source": str(news_info.get("source", ""))
                                        },
                                        "risk": {
                                            "entry_price": float(levels["entry_price"]),
                                            "take_profit": float(levels["take_profit"]),
                                            "stop_loss": float(levels["stop_loss"]),
                                            "risk_reward_ratio": str(levels.get("risk_reward_ratio", "1:1.33")),
                                            "recommended_lot": float(levels.get("recommended_lot", 0.01)),
                                            "sl_distance": float(levels.get("sl_distance", 0.0)),
                                            "tp_distance": float(levels.get("tp_distance", 0.0))
                                        },
                                        "asset_profile": {
                                            "min_adx": profile.get("min_adx"),
                                            "min_probability": profile.get("min_probability")
                                        }
                                    }

                                    # Сохранение в базу данных с расширенным диагностическим пакетом
                                    sig_id = save_signal(
                                        symbol=sym,
                                        direction=action,
                                        entry_price=levels["entry_price"],
                                        take_profit=levels["take_profit"],
                                        stop_loss=levels["stop_loss"],
                                        atr=levels["atr"],
                                        probability=prob,
                                        diagnostic_data=diagnostic_payload
                                    )

                                    logger.info(f"[{sym}] Сформирован сигнал #{sig_id}: {action} @ {levels['entry_price']}, TP: {levels['take_profit']}, SL: {levels['stop_loss']}")
                                    outcome_label = f"✅ Сигнал #{sig_id} {action}"

                                    # Отправка в Telegram с полным контекстом
                                    sent_msg_id = notifier.send_trade_signal(
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
                                        headline=news_info["top_headline"],
                                        adx=adx_val,
                                        asset_category=profile.get("category", "")
                                    )
                                    if sent_msg_id and sig_id:
                                        try:
                                            set_signal_message_id(sig_id, sent_msg_id)
                                        except Exception as e:
                                            logger.warning(f"Не удалось связать message_id с сигналом #{sig_id}: {e}")

                        # Записываем в журнал анализов для кнопки "📊 Статус Бота"
                        log_candle_analysis(sym, close_p, action, prob, outcome_label, reason)

                except RateLimitError as e:
                    logger.warning(f"[{sym}] Rate limit Twelve Data: {e}. Пауза 10 сек...")
                    time.sleep(10)
                except Exception as e:
                    logger.error(f"Ошибка при обработке {sym}: {e}", exc_info=False)

            # 3. ПЕРИОДИЧЕСКОЕ САМООБУЧЕНИЕ МОДЕЛЕЙ (Каждые 4 часа или по запросу пользователя)
            elapsed_hours = (time.time() - last_retrain_time) / 3600.0
            manual_request = consume_retrain_request()
            if manual_request or elapsed_hours >= RETRAIN_INTERVAL_HOURS:
                trigger_reason = "по вашему запросу из Telegram" if manual_request else f"плановое (каждые {RETRAIN_INTERVAL_HOURS} ч)"
                logger.info(f"Запуск индивидуального самообучения моделей ({trigger_reason})...")

                retrained_count = 0
                for s in SYMBOLS:
                    try:
                        c_df = fetcher.fetch_time_series(symbol=s, outputsize=TRAINING_CANDLES_COUNT)
                        if not c_df.empty:
                            save_candles(c_df, symbol=s)
                            fresh_with_ind = calculate_indicators(c_df, atr_period=ATR_PERIOD)
                            if ml_model.train(fresh_with_ind, symbol=s):
                                retrained_count += 1
                        time.sleep(1)
                    except Exception as ex:
                        logger.warning(f"[{s}] Ошибка обновления модели: {ex}")

                if retrained_count > 0:
                    try:
                        notifier.send_retrain_notification(retrained_count * TRAINING_CANDLES_COUNT, ml_model.last_accuracy)
                    except Exception:
                        pass
                    last_retrain_time = time.time()

        except Exception as e:
            logger.error(f"Непредвиденная ошибка в основном цикле: {e}", exc_info=True)

        # Пауза перед следующим циклом опроса
        for _ in range(POLL_INTERVAL_SECONDS):
            if not running:
                break
            time.sleep(1)

    logger.info("Бот завершил работу.")


if __name__ == "__main__":
    main()
