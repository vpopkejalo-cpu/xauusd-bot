"""
Модуль интеграции с Telegram Bot API через прямой REST API POST-запрос (sendMessage).
Форматирует сигналы и служебные сообщения с использованием красивой HTML-разметки.
"""

import requests
import logging
from typing import Optional, Dict, Any
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SYMBOL

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, token: str = TELEGRAM_BOT_TOKEN, chat_id: str = TELEGRAM_CHAT_ID):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.token}"

    def send_raw_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Отправляет текстовое сообщение в Telegram через метод sendMessage.
        Использует requests.post с таймаутом и обработкой ошибок.
        """
        if not self.token or not self.chat_id:
            logger.warning("Telegram токен или Chat ID не заданы. Сообщение не отправлено.")
            return False

        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200 and response.json().get("ok"):
                logger.info("Сообщение успешно доставлено в Telegram.")
                return True
            else:
                logger.error(f"Ошибка отправки в Telegram: HTTP {response.status_code}, Ответ: {response.text}")
                return False
        except requests.RequestException as e:
            logger.error(f"Сетевой сбой при отправке сообщения в Telegram: {e}")
            return False

    def send_startup_notification(self, interval: str, min_prob: float):
        """Отправляет тестовое приветственное сообщение при старте бота."""
        message = (
            "🚀 <b>Торговый ассистент XAU/USD запущен 24/7!</b>\n\n"
            "✅ <b>Статус:</b> Активен и анализирует рынок\n"
            f"📈 <b>Инструмент:</b> {SYMBOL} (Золото)\n"
            f"⏱ <b>Таймфрейм:</b> {interval}\n"
            f"🧠 <b>ML Фильтр вероятности:</b> ≥ {min_prob:.0%}\n"
            "🛡 <b>Фильтр Азии:</b> Активен (23:00 - 08:00 GMT блокируются)\n"
            "⚙️ <b>Сервер:</b> Google Cloud (systemd автоперезапуск)\n\n"
            "<i>Ожидайте сигналов высокой точности при открытии Лондонской и Нью-Йоркской сессий.</i>"
        )
        return self.send_raw_message(message)

    def send_trade_signal(
        self,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        probability: float,
        atr: float,
        risk_reward: str,
        session_name: str
    ) -> bool:
        """
        Форматирует и отправляет готовый торговый сигнал заказчику в Telegram.
        """
        if direction.upper() == "BUY":
            dir_emoji = "🟢"
            dir_name = "ПОКУПКА (BUY)"
            color_badge = "📈 <b>ВХОД В ЛОНГ</b>"
        else:
            dir_emoji = "🔴"
            dir_name = "ПРОДАЖА (SELL)"
            color_badge = "📉 <b>ВХОД В ШОРТ</b>"

        message = (
            f"🔔 <b>НОВЫЙ ТОРГОВЫЙ СИГНАЛ: {SYMBOL}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"{color_badge} {dir_emoji}\n\n"
            f"🎯 <b>Направление:</b> {dir_name}\n"
            f"💵 <b>Цена входа:</b> <code>{entry_price:.2f}</code>\n"
            f"🛑 <b>Stop Loss:</b> <code>{stop_loss:.2f}</code>\n"
            f"🏆 <b>Take Profit:</b> <code>{take_profit:.2f}</code>\n"
            f"⚖️ <b>Соотношение R:R:</b> <code>{risk_reward}</code>\n\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"🤖 <b>Уверенность ML модели:</b> <b>{probability:.1%}</b>\n"
            f"🌊 <b>Волатильность (ATR 14):</b> <code>${atr:.2f}</code>\n"
            f"🌍 <b>Сессия:</b> {session_name}\n\n"
            f"💡 <i>Рекомендация: соблюдайте риск не более 1-2% от вашего депозита на сделку.</i>"
        )
        return self.send_raw_message(message)

    def send_trade_outcome(
        self,
        signal_id: int,
        direction: str,
        status: str,
        pnl: float,
        exit_price: float
    ) -> bool:
        """Отправляет уведомление о закрытии сделки по Take Profit или Stop Loss."""
        if status == "TP_HIT":
            outcome_emoji = "🎉✅"
            title = "TAKE PROFIT ДОСТИГНУТ!"
            pnl_sign = f"+${abs(pnl):.2f}"
        else:
            outcome_emoji = "⚠️🛑"
            title = "STOP LOSS СРАБОТАЛ"
            pnl_sign = f"-${abs(pnl):.2f}"

        message = (
            f"{outcome_emoji} <b>{title}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"📌 Сигнал #{signal_id} ({direction})\n"
            f"🏁 <b>Цена закрытия:</b> <code>{exit_price:.2f}</code>\n"
            f"💰 <b>Результат движения:</b> <code>{pnl_sign}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━"
        )
        return self.send_raw_message(message)

    def send_retrain_notification(self, samples_count: int, accuracy: float):
        """Отправляет уведомление об успешном сеансе самообучения модели."""
        message = (
            "🧠 <b>САМООБУЧЕНИЕ МОДЕЛИ ЗАВЕРШЕНО</b>\n\n"
            f"📊 <b>Обработано свечей:</b> {samples_count}\n"
            f"🎯 <b>Новая точность кросс-валидации:</b> {accuracy:.1%}\n"
            "💾 Веса модели успешно обновлены на сервере."
        )
        return self.send_raw_message(message)
