"""
Модуль интеграции с Telegram Bot API.
Поддерживает:
1. Нативную постоянную клавиатуру внизу экрана (Reply Keyboard):
   - [ 📊 Статус Бота ] [ 📈 Уровень качества (Винрейт) ]
   - [ ⏸ Поставить на паузу / ▶️ Запустить бота ] [ 🧹 Сброс сделок ]
2. Выбор периода для расчета уровня качества и точности (День, Неделя, Месяц, 3 мес, 6 мес, Год, Всего)
3. Отображение последних 5 анализов рынка для полной прозрачности
4. Интерактивные кнопки под сигналами для подтверждения сделок
5. Управление паузой/запуском бота для экономии ресурсов ночью
"""

import time
import threading
import requests
import logging
from typing import Optional, Dict, Any, List
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SYMBOL, SYMBOLS, SYMBOL_NAMES
from database import (
    close_all_open_signals,
    update_signal_status,
    get_signal_by_id,
    get_open_signals,
    get_recent_analyses,
    get_winrate_stats,
    set_bot_paused,
    is_bot_paused
)

logger = logging.getLogger(__name__)


def get_main_keyboard(is_paused: bool = False) -> Dict:
    """Возвращает постоянную нативную клавиатуру внизу экрана Telegram."""
    toggle_btn = "▶️ Запустить бота" if is_paused else "⏸ Поставить на паузу"
    return {
        "keyboard": [
            [{"text": "📊 Статус Бота"}, {"text": "📈 Винрейт"}],
            [{"text": toggle_btn}, {"text": "🧹 Сброс сделок"}]
        ],
        "resize_keyboard": True,
        "is_persistent": True
    }


def get_winrate_period_keyboard() -> Dict:
    """Возвращает inline-кнопки для выбора периода расчета статистики."""
    return {
        "inline_keyboard": [
            [
                {"text": "📅 1 День", "callback_data": "wr_1"},
                {"text": "📆 7 Дней", "callback_data": "wr_7"},
                {"text": "🗓 1 Месяц", "callback_data": "wr_30"}
            ],
            [
                {"text": "📊 3 Месяца", "callback_data": "wr_90"},
                {"text": "📈 6 Месяцев", "callback_data": "wr_180"},
                {"text": "🏛 1 Год", "callback_data": "wr_365"}
            ],
            [
                {"text": "🏆 За всё время", "callback_data": "wr_all"}
            ]
        ]
    }


class TelegramNotifier:
    def __init__(self, token: str = TELEGRAM_BOT_TOKEN, chat_id: str = TELEGRAM_CHAT_ID):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.last_update_id = 0
        self.is_polling = False
        self.latest_prices: Dict[str, float] = {}

    def update_price(self, symbol: str, price: float):
        """Кэширует последнюю цену инструмента для отображения в Статусе."""
        self.latest_prices[symbol] = price

    def send_raw_message(self, text: str, reply_markup: Optional[Dict] = None, parse_mode: str = "HTML") -> Optional[int]:
        """Отправляет текстовое сообщение в Telegram через метод sendMessage."""
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
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        else:
            # По умолчанию всегда поддерживаем главную нативную клавиатуру
            payload["reply_markup"] = get_main_keyboard(is_bot_paused())

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

    def send_startup_notification(self, interval: str, min_prob: float, symbols: Optional[list] = None):
        """Отправляет тестовое приветственное сообщение при старте бота с нативной клавиатурой."""
        sym_list = symbols or [SYMBOL]
        display_syms = "\n".join([f"• <b>{SYMBOL_NAMES.get(s, s)}</b>" for s in sym_list])
        paused = is_bot_paused()

        message = (
            "🚀 <b>Торговый помощник для Trading 212 запущен!</b>\n\n"
            f"✅ <b>Статус:</b> {'⏸ НА ПАУЗЕ' if paused else '🟢 АКТИВЕН И СКАНИРУЕТ РЫНОК'}\n"
            f"⏱ <b>Таймфрейм:</b> {interval}\n"
            "🛡 <b>Защита риска:</b> Короткий Stop Loss (0.8 ATR), умеренный Take Profit (1.6 ATR)\n"
            "🛡 <b>Новостной щит:</b> Активен (NFP, CPI, ФРС, запасы нефти)\n\n"
            "📊 <b>Отслеживаемые инструменты:</b>\n"
            f"{display_syms}\n\n"
            "📱 <b>Управление ботом доступно кнопками внизу экрана:</b>\n"
            "• <b>📊 Статус Бота</b> — текущие цены, последние 5 анализов и открытые сделки\n"
            "• <b>📈 Винрейт</b> — статистика винрейта за день, неделю, месяц, год\n"
            "• <b>⏸ Поставить на паузу</b> — выключайте бота на ночь для экономии ресурсов\n"
            "• <b>🧹 Сброс сделок</b> — мгновенная очистка зависших позиций"
        )
        return self.send_raw_message(message, reply_markup=get_main_keyboard(paused))

    def send_trade_signal(
        self,
        signal_id: int,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        probability: float,
        atr: float,
        risk_reward: str,
        session_name: str,
        recommended_lot: float = 0.01,
        news_sentiment: str = "",
        headline: str = ""
    ) -> Optional[int]:
        """
        Форматирует и отправляет готовый торговый сигнал с кнопками управления:
        [✅ Вошел в сделку] [⏭ Пропустил]
        """
        display_name = SYMBOL_NAMES.get(symbol, symbol)

        # Калибровка отображаемой уверенности: никогда не показывать нереалистичные 100%
        calibrated_prob = min(max(probability, 0.55), 0.88)

        if direction.upper() == "BUY":
            dir_emoji = "🟢"
            dir_name = "ПОКУПКА (BUY)"
            color_badge = "📈 <b>ВХОД В ЛОНГ</b>"
        else:
            dir_emoji = "🔴"
            dir_name = "ПРОДАЖА (SELL)"
            color_badge = "📉 <b>ВХОД В ШОРТ</b>"

        news_block = ""
        if news_sentiment:
            news_block = f"📰 <b>Новостной фон:</b> {news_sentiment}\n"
        if headline and headline != "Рыночный фон стабилен":
            clean_head = headline[:90] + ("..." if len(headline) > 90 else "")
            news_block += f"🗞 <b>Свежая новость:</b> <i>«{clean_head}»</i>\n"

        message = (
            f"🔔 <b>НОВЫЙ ТОРГОВЫЙ СИГНАЛ: #{signal_id}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"🏷 <b>Инструмент:</b> <b>{display_name}</b>\n"
            f"{color_badge} {dir_emoji}\n\n"
            f"🎯 <b>Направление:</b> {dir_name}\n"
            f"💵 <b>Цена входа:</b> <code>{entry_price}</code>\n"
            f"🛑 <b>Stop Loss:</b> <code>{stop_loss}</code> (короткий защитный)\n"
            f"🏆 <b>Take Profit:</b> <code>{take_profit}</code>\n"
            f"⚖️ <b>Соотношение R:R:</b> <code>{risk_reward}</code>\n\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"🤖 <b>Уверенность алгоритма:</b> <b>{calibrated_prob:.1%}</b>\n"
            f"{news_block}"
            f"🌊 <b>Волатильность (ATR 14):</b> <code>${atr}</code>\n"
            f"🌍 <b>Сессия:</b> {session_name}\n\n"
            f"💼 <b>Рекомендуемый объем лота (для риска $10):</b> <b>{recommended_lot}</b>\n"
            f"💡 <i>На Trading 212 открывайте микро-лот (0.01–0.02), чтобы контролировать риск!</i>\n\n"
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
            f"🏁 <b>Цена выхода:</b> <code>{exit_price}</code>\n"
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

                    # Обработка текстовых команд и нативных кнопок меню
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

        # 1. Выбор периода для расчета уровня качества / винрейта
        if cb_data.startswith("wr_"):
            period_str = cb_data.replace("wr_", "")
            days_map = {
                "1": (1, "1 День"),
                "7": (7, "7 Дней"),
                "30": (30, "1 Месяц"),
                "90": (90, "3 Месяца"),
                "180": (180, "6 Месяцев"),
                "365": (365, "1 Год"),
                "all": (None, "За всё время")
            }
            days_val, period_name = days_map.get(period_str, (None, "За всё время"))
            stats = get_winrate_stats(days_val)

            winrate = stats["winrate"]
            if winrate >= 70:
                rating = "⭐⭐⭐⭐⭐ (Отличный)"
            elif winrate >= 50:
                rating = "⭐⭐⭐⭐ (Хороший)"
            elif winrate > 0:
                rating = "⭐⭐⭐ (Удовлетворительный)"
            else:
                rating = "⚪ (Недостаточно закрытых сделок)"

            pnl_val = stats["total_pnl"]
            pnl_sign = f"+{pnl_val}" if pnl_val >= 0 else f"{pnl_val}"

            report = (
                f"🏆 <b>СТАТИСТИКА ВИНРЕЙТА</b>\n"
                f"Период: <b>{period_name}</b>\n"
                f"━━━━━━━━━━━━━━━━━━━\n"
                f"🎯 <b>Винрейт (Точность побед):</b> <b>{winrate}%</b>\n"
                f"🏅 <b>Рейтинг эффективности:</b> {rating}\n\n"
                f"📊 <b>Всего сигналов выдано:</b> {stats['total_signals']}\n"
                f"✅ <b>Успешных (Take Profit):</b> {stats['wins']}\n"
                f"🛑 <b>Убыточных (Stop Loss):</b> {stats['losses']}\n"
                f"⌛ <b>Истекло по времени:</b> {stats['expired']}\n"
                f"✖️ <b>Сброшено / Закрыто в 0:</b> {stats['manual']}\n"
                f"⏳ <b>Сейчас в рынке (Open):</b> {stats['open_count']}\n\n"
                f"💰 <b>Суммарный результат движения:</b> <code>{pnl_sign} пунктов</code>\n"
                f"━━━━━━━━━━━━━━━━━━━\n"
                f"<i>Статистика строится строго по фактическим результатам ваших сделок.</i>"
            )
            self.answer_callback_query(cb_id, f"Винрейт за {period_name}: {winrate}%", show_alert=False)
            self.send_raw_message(report)
            return

        # 2. Сброс зависших сделок из кнопки статуса
        if cb_data == "action_reset_trades":
            count = close_all_open_signals("MANUAL_RESET")
            self.answer_callback_query(cb_id, f"Сброшено сделок: {count}", show_alert=True)
            self.send_raw_message(f"🧹 <b>Сброс выполнен!</b> Закрыто зависших позиций: <b>{count}</b>. Бот свободен.")
            return

        # 3. Интерактивные действия по сигналам
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

            new_text = orig_text + "\n\n🎉 <b>РЕЗУЛЬТАТ: ЗАКРЫТО В ПРИБЫЛЬ (TP)!</b> ✅\n<i>Результат учтен в статистике винрейта. Бот готов к новым сигналам.</i>"
            self.edit_message_text(chat_id, msg_id, new_text, reply_markup=None)

        elif action == "loss":
            self.answer_callback_query(cb_id, "Убыток зафиксирован. Учтено в статистике качества.", show_alert=True)
            sig = get_signal_by_id(sig_id)
            sl_price = sig["stop_loss"] if sig else 0.0
            pnl = -abs(sl_price - sig["entry_price"]) if sig else -10.0
            update_signal_status(sig_id, "SL_HIT", sl_price, pnl)

            new_text = orig_text + "\n\n🛑 <b>РЕЗУЛЬТАТ: ЗАКРЫТО В УБЫТОК (SL)</b> ⚠️\n<i>Результат записан в статистику винрейта. Бот разблокирован.</i>"
            self.edit_message_text(chat_id, msg_id, new_text, reply_markup=None)

        elif action == "manual":
            self.answer_callback_query(cb_id, "Сделка закрыта вручную.", show_alert=False)
            if sig_id:
                update_signal_status(sig_id, "MANUAL_CLOSE", 0.0, 0.0)
            new_text = orig_text + "\n\n✖️ <b>РЕЗУЛЬТАТ: Закрыто вручную / Сброшено</b>\n<i>Бот готов к новым сигналам.</i>"
            self.edit_message_text(chat_id, msg_id, new_text, reply_markup=None)

    def _handle_command(self, msg: Dict):
        text = msg.get("text", "").strip()
        lower_text = text.lower()
        chat_id = msg.get("chat", {}).get("id")

        # 1. Кнопка "📊 Статус Бота"
        if "статус" in lower_text:
            paused = is_bot_paused()
            status_badge = "⏸ НА ПАУЗЕ (Остановлен вами)" if paused else "🟢 АКТИВЕН И СКАНИРУЕТ РЫНОК"

            # Текущие котировки активов
            price_lines = []
            for s in SYMBOLS:
                display_name = SYMBOL_NAMES.get(s, s)
                p = self.latest_prices.get(s)
                p_str = f"<code>{p}</code>" if p else "<i>ожидание свечи...</i>"
                price_lines.append(f"• {display_name}: {p_str}")
            prices_block = "\n".join(price_lines)

            # Открытые сделки
            open_sigs = get_open_signals()
            if not open_sigs:
                deals_block = "<i>Открытых сделок нет. Бот свободен.</i>"
            else:
                deals_lines = [f"• #{s['id']} <b>{s.get('symbol', 'XAU/USD')}</b> {s['direction']} по <code>{s['entry_price']}</code> (TP: {s['take_profit']}, SL: {s['stop_loss']})" for s in open_sigs]
                deals_block = "\n".join(deals_lines)

            # Последние 6 анализов свечей (по числу активов)
            recent = get_recent_analyses(6)
            if not recent:
                analyses_block = "<i>История анализов формируется при закрытии свечей...</i>"
            else:
                a_lines = []
                for a in recent:
                    dt = a["created_at"].split("T")[-1][:5] if "T" in a["created_at"] else ""
                    sym = a["symbol"]
                    act = a["action"]
                    prob = round(a["probability"] * 100, 1)
                    outcome = a["outcome"]
                    a_lines.append(f"• <b>{dt} [{sym}]</b>: ML {act} ({prob}%) | {outcome}")
                analyses_block = "\n".join(a_lines)

            reply = (
                "📊 <b>ТЕКУЩИЙ СТАТУС БОТА:</b>\n"
                f"Состояние: <b>{status_badge}</b>\n\n"
                "💰 <b>Котировки активов (Trading 212):</b>\n"
                f"{prices_block}\n\n"
                "📋 <b>Активные сделки в боте:</b>\n"
                f"{deals_block}\n\n"
                "🔍 <b>Последние 6 анализов рынка:</b>\n"
                f"{analyses_block}\n\n"
                "<i>Чтобы сбросить зависшие сделки, нажмите кнопку ниже:</i>"
            )

            inline_reset = {
                "inline_keyboard": [
                    [{"text": "🧹 Сбросить зависшие сделки", "callback_data": "action_reset_trades"}]
                ]
            }
            self.send_raw_message(reply, reply_markup=inline_reset)

        # 2. Кнопка "📈 Винрейт"
        elif "винрейт" in lower_text or "уровень качества" in lower_text or "качество" in lower_text:
            text_prompt = (
                "📈 <b>СТАТИСТИКА И ТОЧНОСТЬ СИГНАЛОВ (ВИНРЕЙТ)</b>\n\n"
                "Здесь вы можете посмотреть реальную статистику: сколько сигналов вы открыли, "
                "сколько из них закрылись в плюс по Take Profit, сколько по Stop Loss, и точный процент побед (Win Rate).\n\n"
                "👇 <b>Выберите период для расчета:</b>"
            )
            self.send_raw_message(text_prompt, reply_markup=get_winrate_period_keyboard())

        # 3. Кнопка "⏸ Поставить на паузу" / "Стоп"
        elif "пауз" in lower_text or lower_text in ("стоп", "stop"):
            set_bot_paused(True)
            msg_text = (
                "⏸ <b>БОТ ПОСТАВЛЕН НА ПАУЗУ</b>\n\n"
                "✅ Сканирование рынка приостановлено.\n"
                "✅ Запросы к API остановлены.\n"
                "🌙 Вы можете спокойно ложиться спать, ресурсы не тратятся!\n\n"
                "<i>Когда проснетесь, просто нажмите кнопку <b>▶️ Запустить бота</b> внизу экрана.</i>"
            )
            self.send_raw_message(msg_text, reply_markup=get_main_keyboard(is_paused=True))

        # 4. Кнопка "▶️ Запустить бота" / "Старт"
        elif "запустить" in lower_text or lower_text in ("старт", "start", "/start"):
            set_bot_paused(False)
            msg_text = (
                "▶️ <b>БОТ УСПЕШНО ЗАПУЩЕН!</b>\n\n"
                "✅ Сканирование 5 активов возобновлено.\n"
                "✅ Мониторинг новостей и экономический щит активны.\n"
                "🎯 Ожидайте новых качественных сигналов высокой точности!"
            )
            self.send_raw_message(msg_text, reply_markup=get_main_keyboard(is_paused=False))

        # 5. Кнопка "🧹 Сброс сделок" / /reset
        elif "сброс" in lower_text or lower_text in ("/reset", "reset"):
            count = close_all_open_signals("MANUAL_RESET")
            reply = f"🧹 <b>Сброс выполнен!</b>\nЗакрыто зависших позиций: <b>{count}</b>.\nБот полностью свободен и готов отправлять новые сигналы."
            self.send_raw_message(reply, reply_markup=get_main_keyboard(is_bot_paused()))

        else:
            # Неизвестная команда — просто возвращаем кнопки меню
            self.send_raw_message(
                "Используйте кнопки меню внизу экрана для управления ботом 👇",
                reply_markup=get_main_keyboard(is_bot_paused())
            )

    def start_polling(self):
        """Запускает фоновый поток опроса Telegram API."""
        if not self.is_polling:
            self.is_polling = True
            thread = threading.Thread(target=self._poll_updates, daemon=True)
            thread.start()
