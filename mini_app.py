"""Secure Telegram Mini App trade editor endpoints."""

import hashlib
import hmac
import json
import logging
import math
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qsl, urlparse

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_OWNER_USER_ID
from database import (
    get_signal_by_id,
    get_signal_level_history,
    update_signal_status,
    update_signal_trade_values,
)

logger = logging.getLogger(__name__)
_HTML_PATH = Path(__file__).with_name("miniapp.html")
_MAX_BODY_BYTES = 16_384


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
        "entered": bool(signal.get("entered_at")),
        "history": get_signal_level_history(signal_id),
    }


class MiniAppRequestHandler(BaseHTTPRequestHandler):
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
        if urlparse(self.path).path != "/api/miniapp/trade":
            return self._send_json(404, {"error": "Маршрут не найден"})
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > _MAX_BODY_BYTES:
                return self._send_json(413, {"error": "Некорректный размер запроса"})
            payload = json.loads(self.rfile.read(content_length))
            user = _authorized_user(self.headers.get("X-Telegram-Init-Data", ""))
            if not user:
                return self._send_json(401, {"error": "Не удалось подтвердить Telegram аккаунт"})
            signal_id = int(payload.get("trade_id", 0))
            signal = get_signal_by_id(signal_id)
            if not signal:
                return self._send_json(404, {"error": "Сделка не найдена"})
            if payload.get("action") == "get":
                return self._send_json(200, {"trade": _trade_payload(signal_id)})
            if payload.get("action") == "close":
                if signal.get("status") != "OPEN":
                    return self._send_json(409, {"error": "Сделка уже закрыта"})
                intent = payload.get("status")
                if intent not in ("TP_HIT", "SL_HIT", "MANUAL_CLOSE"):
                    return self._send_json(400, {"error": "Выберите результат закрытия"})
                exit_price = float(payload["exit_price"])
                entry = float(signal.get("actual_entry_price") or signal.get("entry_price") or 0)
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
            if payload.get("action") != "update":
                return self._send_json(400, {"error": "Неизвестное действие"})
            if signal.get("status") != "OPEN":
                return self._send_json(409, {"error": "Изменять уровни можно только у открытой сделки"})

            entry = float(payload["entry"])
            take_profit = float(payload["tp"])
            stop_loss = float(payload["sl"])
            if not all(math.isfinite(value) and value > 0 for value in (entry, take_profit, stop_loss)):
                return self._send_json(400, {"error": "Все цены должны быть положительными числами"})
            buy = str(signal.get("direction", "BUY")).upper() == "BUY"
            if not (stop_loss < entry < take_profit if buy else take_profit < entry < stop_loss):
                return self._send_json(400, {"error": "Проверь порядок Entry, TP и SL для направления сделки"})
            if not update_signal_trade_values(signal_id, entry, take_profit, stop_loss, source="telegram_mini_app"):
                return self._send_json(409, {"error": "Сделка уже закрыта; изменения не сохранены"})
            return self._send_json(200, {"message": "Уровни сохранены в истории сделки", "trade": _trade_payload(signal_id)})
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            return self._send_json(400, {"error": "Проверьте введённые значения"})
        except Exception:
            logger.exception("Mini App trade request failed")
            return self._send_json(500, {"error": "Не удалось обработать запрос. Попробуйте ещё раз"})

    def _send_json(self, status: int, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
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
