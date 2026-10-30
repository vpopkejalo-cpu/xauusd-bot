"""
Модуль интеграции с Telegram Bot API через прямой REST API POST-запрос.
Поддерживает интерактивные кнопки (Inline Keyboard), подтверждение сделок,
фиксацию ручных результатов (Take Profit / Stop Loss) и фоновую обработку ответов.
"""

import time
import threading
import requests
import logging
from typing import Optional, Dict, Any, Callable
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SYMBOL
from database import close_all_open_signals, update_signal_status, get_signal_by_id, get_open_signals

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, token: str = TELEGRAM_BOT_TOKEN, chat_id: str = TELEGRAM_CHAT_ID):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.last_update_id = 0
        self.is_polling = False

    def send_raw_message(self, text: str, reply_markup: Optional[Dict] = None, parse_mode: str = "HTML") -> Optional[int]:
        """
        Отправляет текстовое сообщение в Telegram через метод sendMessage.
        Возвращает message_id отправленного сообщения.
        """
        if not self.token or not self.chat_id:
            logger.warning("Telegram токен или Chat ID не заданы. Сообщение не отправлено.")
            return None

        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup

        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    msg_id = data["result"]["message_id"]
                    logger.info(f"Сообщение #{msg_id} успешно доставлено в Telegram.")
                    return msg_id
            logger.error(f"Ошибка отправки в Telegram: HTTP {response.status_code}, Ответ: {response.text}")
            return None
        except requests.RequestException as e:
            logger.error(f"Сетевой сбой при отправке сообщения в Telegram: {e}")
            return None

    def edit_message_text(self, chat_id: Any, message_id: int, text: str, reply_markup: Optional[Dict] = None) -> bool:
        """Редактирует текст и кнопки существующего сообщения."""
        url = f"{self.base_url}/editMessageText"
        payload = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup

        try:
            res = requests.post(url, json=payload, timeout=10)
            return res.status_code == 200 and res.json().get("ok", False)
        except Exception as e:
            logger.error(f"Ошибка редактирования сообщения #{message_id}: {e}")
            return False

    def answer_callback_query(self, callback_query_id: str, text: str, show_alert: bool = False):
        """Отправляет всплывающее уведомление пользователю при нажатии на кнопку."""
        url = f"{self.base_url}/answerCallbackQuery"
        payload = {
            "callback_query_id": callback_query_id,
            "text": text,
            "show_alert": show_alert
        }
        try:
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            logger.warning(f"Не удалось ответить на callback: {e}")

    def send_startup_notification(self, interval: str, min_prob: float):
        """Отправляет тестовое приветственное сообщение при старте бота."""
        message = (
            "🚀 <b>Торговый ассистент XAU/USD запущен 24/7!</b>\n\n"
            "✅ <b>Статус:</b> Активен и анализирует рынок\n"
            f"📈 <b>Инструмент:</b> {SYMBOL} (Золото)\n"
            f"⏱ <b>Таймфрейм:</b> {interval}\n"
            f"🧠 <b>ML Фильтр вероятности:</b> ≥ {min_prob:.0%}\n"
            "🛡 <b>Фильтр Азии:</b> Активен (23:00 - 08:00 GMT блокируются)\n\n"
            "🎮 <i>Под каждым сигналом теперь есть интерактивные кнопки: подтверждайте открытие сделок и фиксируйте результат нажатием одной кнопки.</i>\n\n"
            "📌 <b>Команды:</b>\n"
            "• <code>/status</code> — текущий статус бота и открытых позиций\n"
            "• <code>/reset</code> — принудительно сбросить все зависшие сделки"
        )
        return self.send_raw_message(message)

    def send_trade_signal(
        self,
        signal_id: int,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        probability: float,
        atr: float,
        risk_reward: str,
        session_name: str
    ) -> Optional[int]:
        """
        Форматирует и отправляет готовый торговый сигнал с интерактивными кнопками:
        [✅ Вошел в сделку] [⏭ Пропустил]
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
            f"🔔 <b>НОВЫЙ ТОРГОВЫЙ СИГНАЛ: #{signal_id}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"{color_badge} {dir_emoji}\n\n"
            f"🎯 <b>Направление:</b> {dir_name}\n"
            f"💵 <b>Цена входа:</b> <code>{entry_price:.2f}</code>\n"
            f"🛑 <b>Stop Loss:</b> <code>{stop_loss:.2f}</code>\n"
            f"🏆 <b>Take Profit:</b> <code>{take_profit:.2f}</code>\n"
            f"⚖️ <b>Соотношение R:R:</b> <code>{risk_reward}</code>\n\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"🤖 <b>Уверенность ML:</b> <b>{probability:.1%}</b>\n"
            f"🌊 <b>Волатильность (ATR 14):</b> <code>${atr:.2f}</code>\n"
            f"🌍 <b>Сессия:</b> {session_name}\n\n"
            f"👇 <b>Вы вошли в эту сделку? Нажмите кнопку:</b>"
        )

        reply_markup = {
            "inline_keyboard": [
                [
                    {"text": "✅ Вошел в сделку", "callback_data": f"enter_{signal_id}"},
                    {"text": "⏭ Пропустил", "callback_data": f"skip_{signal_id}"}
                ]
            ]
        }

        return self.send_raw_message(message, reply_markup=reply_markup)

    def send_trade_outcome(
        self,
        signal_id: int,
        direction: str,
        status: str,
        pnl: float,
        exit_price: float
    ) -> bool:
        """Отправляет уведомление об автоматическом закрытии по рынку."""
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
            f"🏁 <b>Цена выхода:</b> <code>{exit_price:.2f}</code>\n"
            f"💰 <b>Результат движения:</b> <code>{pnl_sign}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━"
        )
        return bool(self.send_raw_message(message))

    def send_retrain_notification(self, samples_count: int, accuracy: float):
        """Отправляет уведомление об успешном сеансе самообучения модели."""
        message = (
            "🧠 <b>САМООБУЧЕНИЕ МОДЕЛИ ЗАВЕРШЕНО</b>\n\n"
            f"📊 <b>Обработано свечей:</b> {samples_count}\n"
            f"🎯 <b>Новая точность кросс-валидации:</b> {accuracy:.1%}\n"
            "💾 Веса модели успешно обновлены на сервере."
        )
        return bool(self.send_raw_message(message))

    def _poll_updates(self):
        """Фоновый цикл обработки входящих нажатий кнопок и команд в Telegram."""
        logger.info("Фоновый слушатель кнопок Telegram запущен.")
        while self.is_polling:
            try:
                url = f"{self.base_url}/getUpdates"
                params = {
                    "offset": self.last_update_id + 1,
                    "timeout": 20
                }
                res = requests.get(url, params=params, timeout=25)
                if res.status_code != 200:
                    time.sleep(3)
                    continue

                data = res.json()
                if not data.get("ok"):
                    time.sleep(3)
                    continue

                for update in data.get("result", []):
                    self.last_update_id = update["update_id"]

                    # Обработка нажатий на inline-кнопки
                    if "callback_query" in update:
                        self._handle_callback(update["callback_query"])

                    # Обработка текстовых команд (/status, /reset)
                    elif "message" in update and "text" in update["message"]:
                        self._handle_command(update["message"])

            except Exception as e:
                logger.error(f"Ошибка в цикле Telegram polling: {e}")
                time.sleep(3)

    def _handle_callback(self, cb: Dict):
        cb_id = cb["id"]
        cb_data = cb.get("data", "")
        msg = cb.get("message", {})
        chat_id = msg.get("chat", {}).get("id")
        msg_id = msg.get("message_id")
        orig_text = msg.get("text", "")

        parts = cb_data.split("_")
        action = parts[0]
        sig_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

        logger.info(f"Нажата кнопка Telegram: action={action}, signal_id={sig_id}")

        if action == "enter":
            self.answer_callback_query(cb_id, "Сделка принята! Укажите результат после ее закрытия.", show_alert=False)
            new_text = orig_text + "\n\n<b>Статус:</b> ВЫ В СДЕЛКЕ 🟢\n<i>Когда закроете позицию брокера, нажмите результат:</i>"
            markup = {
                "inline_keyboard": [
                    [
                        {"text": "🏆 Закрыл в плюс (+TP)", "callback_data": f"win_{sig_id}"},
                        {"text": "🛑 Закрыл в минус (-SL)", "callback_data": f"loss_{sig_id}"}
                    ],
                    [
                        {"text": "✖️ Сбросить / Вышел в 0", "callback_data": f"manual_{sig_id}"}
                    ]
                ]
            }
            self.edit_message_text(chat_id, msg_id, new_text, reply_markup=markup)

        elif action == "skip":
            self.answer_callback_query(cb_id, "Сигнал пропущен вами.", show_alert=False)
            if sig_id:
                update_signal_status(sig_id, "SKIPPED", 0.0, 0.0)
            new_text = orig_text + "\n\n<b>Статус:</b> Сделка пропущена вами ⏭"
            self.edit_message_text(chat_id, msg_id, new_text, reply_markup=None)

        elif action == "win":
            self.answer_callback_query(cb_id, "Отлично! Прибыль зафиксирована.", show_alert=True)
            sig = get_signal_by_id(sig_id)
            tp_price = sig["take_profit"] if sig else 0.0
            pnl = abs(tp_price - sig["entry_price"]) if sig else 10.0
            update_signal_status(sig_id, "TP_HIT", tp_price, pnl)

            new_text = orig_text + "\n\n🎉 <b>РЕЗУЛЬТАТ: ЗАКРЫТО В ПРИБЫЛЬ (TP)!</b> ✅\n<i>Бот освобожден и сканирует новые точки входа.</i>"
            self.edit_message_text(chat_id, msg_id, new_text, reply_markup=None)

        elif action == "loss":
            self.answer_callback_query(cb_id, "Убыток зафиксирован. Данные пойдут в самообучение бота.", show_alert=True)
            sig = get_signal_by_id(sig_id)
            sl_price = sig["stop_loss"] if sig else 0.0
            pnl = -abs(sl_price - sig["entry_price"]) if sig else -10.0
            update_signal_status(sig_id, "SL_HIT", sl_price, pnl)

            new_text = orig_text + "\n\n🛑 <b>РЕЗУЛЬТАТ: ЗАКРЫТО В УБЫТОК (SL)</b> ⚠️\n<i>Бот разблокирован для новых сигналов. Ошибка учтена в статистике.</i>"
            self.edit_message_text(chat_id, msg_id, new_text, reply_markup=None)

        elif action == "manual":
            self.answer_callback_query(cb_id, "Сделка закрыта вручную.", show_alert=False)
            if sig_id:
                update_signal_status(sig_id, "MANUAL_CLOSE", 0.0, 0.0)
            new_text = orig_text + "\n\n✖️ <b>РЕЗУЛЬТАТ: Закрыто вручную / Сброшено</b>\n<i>Бот готов к новым сигналам.</i>"
            self.edit_message_text(chat_id, msg_id, new_text, reply_markup=None)

    def _handle_command(self, msg: Dict):
        text = msg.get("text", "").strip().lower()
        chat_id = msg.get("chat", {}).get("id")

        if text in ("/reset", "/сброс", "сброс"):
            count = close_all_open_signals("MANUAL_RESET")
            reply = f"🧹 <b>Сброс выполнен!</b>\nЗакрыто зависших позиций: <b>{count}</b>.\nБот полностью свободен и готов отправлять новые сигналы."
            self.send_raw_message(reply)

        elif text in ("/status", "/статус", "статус"):
            open_sigs = get_open_signals()
            count = len(open_sigs)
            if count == 0:
                sig_info = "Открытых позиций нет. Бот свободен и анализирует рынок."
            else:
                lines = [f"• #{s['id']} {s['direction']} по {s['entry_price']} (TP: {s['take_profit']}, SL: {s['stop_loss']})" for s in open_sigs]
                sig_info = f"<b>Открыто позиций: {count}</b>\n" + "\n".join(lines) + "\n\n<i>Отправьте <code>/reset</code>, если вы уже вышли из них у брокера.</i>"

            reply = (
                "📊 <b>ТЕКУЩИЙ СТАТУС БОТА:</b>\n\n"
                f"✅ <b>Служба:</b> Работает в облаке 24/7\n"
                f"📈 <b>Инструмент:</b> {SYMBOL}\n"
                f"{sig_info}"
            )
            self.send_raw_message(reply)

    def start_polling(self):
        """Запускает фоновый поток опроса Telegram API."""
        if not self.is_polling:
            self.is_polling = True
            thread = threading.Thread(target=self._poll_updates, daemon=True)
            thread.start()
