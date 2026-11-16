"""Secure Telegram Mini App trade editor endpoints."""

import hashlib
import hmac
import html
import json
import logging
import math
import re
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qsl, urlparse

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_OWNER_USER_ID
from database import (
    close_all_open_signals,
    get_analysis_logs,
    get_model_snapshots_history,
    log_operational_event,
    get_open_signals,
    get_recent_analyses,
    get_recent_signals,
    get_signal_by_id,
    get_signal_events,
    get_signal_level_history,
    get_today_signal_stats,
    get_winrate_stats,
    is_bot_paused,
    request_retrain,
    reset_kill_switch,
    set_bot_paused,
    set_signal_entered_at,
    set_signal_position_quantity,
    set_signal_reported_pnl,
    update_signal_status,
    update_signal_trade_values,
)

logger = logging.getLogger(__name__)
_HTML_PATH = Path(__file__).with_name("miniapp.html")
_MAX_BODY_BYTES = 32_768


def _authorized_user(init_data: str):
    """Validate Telegram's signed initData and return the authenticated user."""
    if not init_data or not TELEGRAM_BOT_TOKEN:
        return None
    try:
        pairs = dict(parse_qsl(init_data, keep_blank_values=True, strict_parsing=True))
        supplied_hash = pairs.pop("hash", "")
        auth_date = int(pairs.get("auth_date", "0"))
        if not supplied_hash or auth_date > int(time.time()) + 60 or int(time.time()) - auth_date > 86_400:
            return None
        check_string = "\n".join(f"{key}={value}" for key, value in sorted(pairs.items()))
        secret_key = hmac.new(b"WebAppData", TELEGRAM_BOT_TOKEN.encode("utf-8"), hashlib.sha256).digest()
        expected_hash = hmac.new(secret_key, check_string.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected_hash, supplied_hash):
            return None
        user = json.loads(pairs.get("user", "{}"))
        user_id = int(user.get("id", 0))
        allowed_id = int(TELEGRAM_OWNER_USER_ID) if TELEGRAM_OWNER_USER_ID else int(TELEGRAM_CHAT_ID)
        if user_id <= 0 or allowed_id <= 0 or user_id != allowed_id:
            return None
        return user
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _trade_payload(signal_id: int):
    signal = get_signal_by_id(signal_id)
    if not signal:
        return None
    entry = signal.get("actual_entry_price") or signal.get("entry_price")
    return {
        "id": int(signal["id"]),
        "symbol": signal.get("symbol", ""),
        "direction": str(signal.get("direction", "BUY")).upper(),
        "status": signal.get("status", ""),
        "entry": float(entry),
        "reference_entry": float(signal.get("entry_price") or entry),
        "tp": float(signal["take_profit"]),
        "sl": float(signal["stop_loss"]),
        "initial_tp": float(signal.get("initial_take_profit") or signal["take_profit"]),
        "initial_sl": float(signal.get("initial_stop_loss") or signal["stop_loss"]),
        "exit_price": float(signal["exit_price"]) if signal.get("exit_price") is not None else None,
        "pnl": float(signal["pnl"]) if signal.get("pnl") is not None else None,
        "reported_pnl": float(signal["reported_pnl"]) if signal.get("reported_pnl") is not None else None,
        "reported_pnl_currency": signal.get("reported_pnl_currency") or "USD",
        "entered": bool(signal.get("entered_at")),
        "entered_at": signal.get("entered_at"),
        "created_at": signal.get("created_at"),
        "closed_at": signal.get("closed_at"),
        "probability": float(signal.get("probability") or 0),
        "atr": float(signal.get("atr") or 0),
        "quantity": signal.get("position_quantity"),
        "pnl_basis": signal.get("pnl_basis"),
        "exit_source": signal.get("exit_source"),
        "history": get_signal_level_history(signal_id),
        "events": get_signal_events(signal_id),
    }


def _signal_summary(signal):
    entry = signal.get("actual_entry_price") or signal.get("entry_price")
    return {
        "id": int(signal["id"]), "symbol": signal.get("symbol", ""),
        "direction": str(signal.get("direction", "BUY")).upper(),
        "status": signal.get("status", ""), "entry": entry,
        "reference_entry": signal.get("entry_price"),
        "tp": signal.get("take_profit"), "sl": signal.get("stop_loss"),
        "exit_price": signal.get("exit_price"), "pnl": signal.get("pnl"),
        "reported_pnl": signal.get("reported_pnl"),
        "reported_pnl_currency": signal.get("reported_pnl_currency") or "USD",
        "created_at": signal.get("created_at"), "entered_at": signal.get("entered_at"),
        "closed_at": signal.get("closed_at"),
        "entered": bool(signal.get("entered_at")), "quantity": signal.get("position_quantity"),
        "probability": signal.get("probability"),
    }


def _plain_text(markup: str) -> str:
    return html.unescape(re.sub(r"<[^>]*>", "", str(markup or "")))


def _parse_user_number(value) -> float:
    """Accept decimal comma/dot and common thousands separators from user input."""
    if value is None or isinstance(value, bool):
        raise ValueError("Число не задано")
    # Strip spaces, grouping marks and invisible characters copied by Telegram/markets.
    text = re.sub(r"[\s\u00a0\u202f\u200b\u200c\u200d\u2060\ufeff_'’]", "", str(value).strip())
    if not text:
        raise ValueError("Число не задано")
    if "," in text and "." in text:
        decimal_separator = "," if text.rfind(",") > text.rfind(".") else "."
        grouping_separator = "." if decimal_separator == "," else ","
        text = text.replace(grouping_separator, "")
        if decimal_separator == ",":
            text = text.replace(",", ".")
    elif "," in text:
        groups = text.split(",")
        text = "".join(groups[:-1]) + "." + groups[-1]
    elif text.count(".") > 1:
        groups = text.split(".")
        text = "".join(groups[:-1]) + "." + groups[-1]
    return float(text)


class MiniAppRequestHandler(BaseHTTPRequestHandler):
    notifier = None

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/miniapp":
            try:
                body = _HTML_PATH.read_bytes()
            except OSError:
                logger.exception("Mini App HTML could not be read")
                return self._send_json(503, {"error": "Mini App пока недоступен"})
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/health":
            return self._send_text(200, "OK\n")
        return self._send_text(200, "OK: Multi-Asset Trading Assistant is running\n")

    def do_POST(self):
        if urlparse(self.path).path not in ("/api/miniapp", "/api/miniapp/trade"):
            return self._send_json(404, {"error": "Маршрут не найден"})
        payload = {}
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > _MAX_BODY_BYTES:
                return self._send_json(413, {"error": "Некорректный размер запроса"})
            payload = json.loads(self.rfile.read(content_length))
            user = _authorized_user(self.headers.get("X-Telegram-Init-Data", ""))
            if not user:
                return self._send_json(401, {"error": "Не удалось подтвердить Telegram аккаунт"})
            action = str(payload.get("action", ""))
            signal_id = int(payload.get("trade_id", 0) or 0)
            log_operational_event("mini_app_action", "telegram_mini_app", {
                "action": action[:80], "trade_id": signal_id or None,
                "user_id": int(user.get("id", 0)),
            })

            if action == "overview":
                open_signals = get_open_signals()
                recent_analyses = get_recent_analyses(12)
                prices = dict(getattr(self.notifier, "latest_prices", {}) or {})
                for item in recent_analyses:
                    prices.setdefault(item.get("symbol"), item.get("close_price"))
                return self._send_json(200, {
                    "paused": is_bot_paused(),
                    "today": get_today_signal_stats(),
                    "week": get_winrate_stats(7),
                    "open_trades": [_signal_summary(item) for item in open_signals],
                    "recent_analyses": recent_analyses,
                    "prices": prices,
                })

            if action == "trades":
                status = payload.get("status")
                if status not in (None, "OPEN", "TP_HIT", "SL_HIT", "MANUAL_CLOSE", "SKIPPED", "EXPIRED", "MANUAL_RESET"):
                    return self._send_json(400, {"error": "Неизвестный фильтр сделок"})
                period = payload.get("period", "all")
                if period not in ("today", "week", "month", "all"):
                    return self._send_json(400, {"error": "Неизвестный период истории"})
                date_from = date_to_exclusive = None
                if period != "all":
                    local_today = datetime.now(ZoneInfo("Europe/Chisinau")).date()
                    if period == "today":
                        start_day = local_today
                        end_day = start_day + timedelta(days=1)
                    elif period == "week":
                        start_day = local_today - timedelta(days=local_today.weekday())
                        end_day = start_day + timedelta(days=7)
                    else:
                        start_day = local_today.replace(day=1)
                        end_day = start_day.replace(year=start_day.year + (start_day.month == 12),
                                                    month=1 if start_day.month == 12 else start_day.month + 1)
                    utc = timezone.utc
                    date_from = datetime.combine(start_day, datetime.min.time(), ZoneInfo("Europe/Chisinau"))
                    date_to_exclusive = datetime.combine(end_day, datetime.min.time(), ZoneInfo("Europe/Chisinau"))
                    date_from = date_from.astimezone(utc).replace(tzinfo=None).isoformat(timespec="seconds")
                    date_to_exclusive = date_to_exclusive.astimezone(utc).replace(tzinfo=None).isoformat(timespec="seconds")
                rows = get_recent_signals(
                    limit=1000, status=status,
                    date_from=date_from,
                    date_to_exclusive=date_to_exclusive,
                )
                return self._send_json(200, {"trades": [_signal_summary(row) for row in rows]})

            if action == "winrate":
                period = payload.get("period")
                period_days = None if period in (None, "all", "wr_all") else int(period)
                if period_days is not None and period_days not in (1, 7, 30, 90, 180, 365):
                    return self._send_json(400, {"error": "Неизвестный период"})
                return self._send_json(200, {"stats": get_winrate_stats(period_days), "period": period_days})

            if action == "model_history":
                symbol = payload.get("symbol") or None
                snapshots = get_model_snapshots_history(symbol=symbol, limit=200)
                return self._send_json(200, {"snapshots": snapshots})

            if action == "report":
                days = int(payload.get("days", 1))
                if days not in (1, 7, 30, 90, 365):
                    return self._send_json(400, {"error": "Неизвестный период отчёта"})
                from report_generator import generate_diagnostics_report
                report = generate_diagnostics_report(days=days)
                delivered = 0
                delivery_total = 3
                notifier = self.notifier
                if notifier:
                    files = (
                        ("txt_content", "txt_filename", "🧾 Диагностический отчёт"),
                        ("csv_content", "csv_filename", "📊 Сделки CSV"),
                        ("analysis_csv_content", "analysis_csv_filename", "🔬 Анализ свечей CSV"),
                    )
                    for content_key, filename_key, caption in files:
                        if notifier.send_document(
                            report.get(content_key, ""), report.get(filename_key, "report.txt"),
                            caption=f"{caption} · период: {days} дн.",
                        ):
                            delivered += 1
                return self._send_json(200, {
                    "report": report,
                    "chat_delivery": {"sent": delivered, "total": delivery_total},
                })

            if action == "ai_analysis":
                days = int(payload.get("days", 1))
                if days not in (1, 7, 30, 90, 365):
                    return self._send_json(400, {"error": "Неизвестный период анализа"})
                from ai_analyst import generate_ai_session_analysis
                return self._send_json(200, {"text": _plain_text(generate_ai_session_analysis(days=days))})

            if action == "system_check":
                from system_checker import run_full_system_check
                result = run_full_system_check()
                return self._send_json(200, {"text": _plain_text(result.get("report_text", "")), "errors": result.get("errors_count", 0)})

            if action == "pause":
                paused = bool(payload.get("paused"))
                set_bot_paused(paused)
                return self._send_json(200, {"paused": is_bot_paused(), "message": "Бот поставлен на паузу" if paused else "Бот запущен"})

            if action == "retrain":
                request_retrain()
                return self._send_json(200, {"message": "Запрос на переобучение передан боту"})

            if action == "reset_kill_switch":
                reset_kill_switch()
                return self._send_json(200, {"message": "Kill Switch сброшен"})

            if action == "reset_trades":
                if payload.get("confirm") is not True:
                    return self._send_json(400, {"error": "Для сброса нужна явная отметка подтверждения"})
                count = close_all_open_signals("MANUAL_RESET")
                return self._send_json(200, {"message": f"Сброшено открытых сделок: {count}", "count": count})

            signal = get_signal_by_id(signal_id) if signal_id else None
            if not signal:
                return self._send_json(404, {"error": "Сделка не найдена"})
            if action in ("get", "trade"):
                return self._send_json(200, {"trade": _trade_payload(signal_id)})
            if action == "enter":
                if signal.get("status") != "OPEN":
                    return self._send_json(409, {"error": "Сделка уже закрыта"})
                set_signal_entered_at(signal_id, source="telegram_mini_app")
                return self._send_json(200, {"trade": _trade_payload(signal_id), "message": "Вход в сделку отмечен"})
            if action == "skip":
                if signal.get("status") != "OPEN":
                    return self._send_json(409, {"error": "Сделка уже закрыта"})
                if not update_signal_status(signal_id, "SKIPPED", None, 0.0, source="telegram_mini_app_skip", pnl_basis="none"):
                    return self._send_json(409, {"error": "Сделка уже изменилась"})
                return self._send_json(200, {"trade": _trade_payload(signal_id), "message": "Сигнал отмечен как пропущенный"})
            if action == "quantity":
                if signal.get("status") != "OPEN":
                    return self._send_json(409, {"error": "Сделка уже закрыта"})
                quantity = _parse_user_number(payload.get("quantity", 0))
                if not math.isfinite(quantity) or quantity <= 0 or not set_signal_position_quantity(signal_id, quantity, source="telegram_mini_app"):
                    return self._send_json(400, {"error": "Введите положительный размер позиции"})
                return self._send_json(200, {"trade": _trade_payload(signal_id), "message": "Размер позиции сохранён"})
            if action == "actual_pnl":
                if signal.get("status") not in ("TP_HIT", "SL_HIT", "MANUAL_CLOSE"):
                    return self._send_json(409, {"error": "Фактический результат можно записать после закрытия сделки"})
                try:
                    amount = _parse_user_number(payload.get("amount"))
                except (TypeError, ValueError):
                    return self._send_json(400, {"error": "Введите сумму со знаком, например 100 или -100"})
                currency = str(payload.get("currency", "USD")).upper().strip()
                if not math.isfinite(amount) or not re.fullmatch(r"[A-Z]{3}", currency):
                    return self._send_json(400, {"error": "Проверьте сумму и трёхбуквенный код валюты"})
                if not set_signal_reported_pnl(signal_id, amount, currency, source="telegram_mini_app"):
                    return self._send_json(409, {"error": "Не удалось сохранить фактический P/L"})
                return self._send_json(200, {
                    "trade": _trade_payload(signal_id),
                    "message": "Фактический P/L сохранён; изменение записано в журнал сделки",
                })
            if action == "close":
                if signal.get("status") != "OPEN":
                    return self._send_json(409, {"error": "Сделка уже закрыта"})
                intent = payload.get("status")
                if intent not in ("TP_HIT", "SL_HIT", "MANUAL_CLOSE"):
                    return self._send_json(400, {"error": "Выберите результат закрытия"})
                try:
                    exit_price = _parse_user_number(payload.get("exit_price"))
                    entry = float(signal.get("actual_entry_price") or signal.get("entry_price") or 0)
                except (TypeError, ValueError):
                    return self._send_json(400, {"error": "Укажите фактическую цену выхода числом, например 94.66"})
                if not math.isfinite(exit_price) or exit_price <= 0 or entry <= 0:
                    return self._send_json(400, {"error": "Введите положительную цену выхода"})
                move = exit_price - entry
                pnl = move if str(signal.get("direction", "BUY")).upper() == "BUY" else -move
                final_status = intent
                if intent == "TP_HIT" and pnl < 0:
                    final_status = "SL_HIT"
                elif intent == "SL_HIT" and pnl > 0:
                    final_status = "TP_HIT"
                elif intent in ("TP_HIT", "SL_HIT") and pnl == 0:
                    final_status = "MANUAL_CLOSE"
                if not update_signal_status(
                    signal_id, final_status, exit_price, pnl,
                    source=f"telegram_mini_app_{intent.lower()}",
                    pnl_basis="signed_price_move_estimate",
                ):
                    return self._send_json(409, {"error": "Сделка уже закрылась; результат не перезаписан"})
                return self._send_json(200, {
                    "message": "Закрытие записано. Показано изменение цены, не денежный PnL брокера.",
                    "trade": _trade_payload(signal_id),
                })
            if action != "update":
                return self._send_json(400, {"error": "Неизвестное действие"})
            if signal.get("status") != "OPEN":
                return self._send_json(409, {"error": "Изменять уровни можно только у открытой сделки"})

            entry = _parse_user_number(payload["entry"])
            take_profit = _parse_user_number(payload["tp"])
            stop_loss = _parse_user_number(payload["sl"])
            if not all(math.isfinite(value) and value > 0 for value in (entry, take_profit, stop_loss)):
                return self._send_json(400, {"error": "Все цены должны быть положительными числами"})
            buy = str(signal.get("direction", "BUY")).upper() == "BUY"
            if not (stop_loss < entry < take_profit if buy else take_profit < entry < stop_loss):
                return self._send_json(400, {"error": "Проверь порядок Entry, TP и SL для направления сделки"})
            if not update_signal_trade_values(signal_id, entry, take_profit, stop_loss, source="telegram_mini_app"):
                return self._send_json(409, {"error": "Сделка уже закрыта; изменения не сохранены"})
            return self._send_json(200, {"message": "Уровни сохранены в истории сделки", "trade": _trade_payload(signal_id)})
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            logger.warning(
                "Mini App request validation failed: action=%s error_type=%s",
                payload.get("action", "unknown"), type(exc).__name__,
            )
            return self._send_json(400, {"error": "Проверьте введённые значения"})
        except Exception:
            logger.exception("Mini App trade request failed")
            return self._send_json(500, {"error": "Не удалось обработать запрос. Попробуйте ещё раз"})

    def _send_json(self, status: int, data):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, text: str):
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass
