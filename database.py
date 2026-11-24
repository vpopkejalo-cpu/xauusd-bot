"""
Модуль базы данных для хранения исторических свечей,
торговых сигналов и результатов сделок.
Поддерживает два режима:
1. Облачный PostgreSQL (Neon.tech, Supabase, Render Postgres) при наличии DATABASE_URL.
2. Резервный локальный SQLite при отсутствии DATABASE_URL.
"""

import os
import json
import re
import sqlite3
import pandas as pd
from datetime import datetime, date, timedelta, timezone
from typing import Optional, List, Dict, Any, Tuple, Union
from zoneinfo import ZoneInfo
from contextlib import contextmanager
import logging
from config import DATABASE_PATH, DATABASE_URL

logger = logging.getLogger(__name__)

# Проверяем наличие конфигурации PostgreSQL
IS_POSTGRES = False
if DATABASE_URL and (DATABASE_URL.startswith("postgres://") or DATABASE_URL.startswith("postgresql://")):
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        import psycopg2.extensions
        try:
            import numpy as np
            psycopg2.extensions.register_adapter(np.float64, psycopg2.extensions.Float)
            psycopg2.extensions.register_adapter(np.float32, psycopg2.extensions.Float)
            psycopg2.extensions.register_adapter(np.int64, psycopg2.extensions.AsIs)
            psycopg2.extensions.register_adapter(np.int32, psycopg2.extensions.AsIs)
        except Exception:
            pass

        IS_POSTGRES = True
        logger.info("Подключение к базе данных: облачный PostgreSQL (вечная память активна).")
    except ImportError:
        logger.warning("Библиотека psycopg2 не найдена, используется локальный SQLite.")
        IS_POSTGRES = False
else:
    logger.info("Подключение к базе данных: локальный SQLite.")


def _adapt_sql(sql: str) -> str:
    """Заменяет плейсхолдеры ? на %s для совместимости с PostgreSQL."""
    if IS_POSTGRES:
        return sql.replace("?", "%s")
    return sql


@contextmanager
def get_db():
    """Контекстный менеджер соединения и транзакций."""
    if IS_POSTGRES:
        # Для облачного Postgres (Neon/Supabase)
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()
    else:
        # Для локального SQLite
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()


def init_db():
    """Инициализирует таблицы базы данных, если они еще не созданы."""
    with get_db() as cursor:
        if IS_POSTGRES:
            # Схема для PostgreSQL
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    symbol TEXT DEFAULT 'XAU/USD',
                    timestamp TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL DEFAULT 0,
                    CONSTRAINT uq_candles_sym_ts UNIQUE (symbol, timestamp)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_sym_ts ON candles (symbol, timestamp)")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    actual_entry_price REAL,
                    actual_entry_source TEXT,
                    position_quantity REAL,
                    take_profit REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    initial_take_profit REAL,
                    initial_stop_loss REAL,
                    atr REAL NOT NULL,
                    probability REAL NOT NULL,
                    status TEXT DEFAULT 'OPEN',
                    closed_at TEXT,
                    exit_price REAL,
                    pnl REAL,
                    pnl_basis TEXT,
                    exit_source TEXT,
                    reported_pnl REAL,
                    reported_pnl_currency TEXT,
                    message_id BIGINT,
                    model_snapshot_id INTEGER
                )
            """)
            cursor.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS message_id BIGINT")
            cursor.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS actual_entry_price REAL")
            cursor.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS actual_entry_source TEXT")
            cursor.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS position_quantity REAL")
            cursor.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS pnl_basis TEXT")
            cursor.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS exit_source TEXT")
            cursor.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS reported_pnl REAL")
            cursor.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS reported_pnl_currency TEXT")
            cursor.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS entered_at TEXT")
            cursor.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS diagnostic_data TEXT")
            cursor.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS initial_take_profit REAL")
            cursor.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS initial_stop_loss REAL")
            cursor.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS model_snapshot_id INTEGER")

            cursor.execute("UPDATE signals SET initial_take_profit = take_profit WHERE initial_take_profit IS NULL")
            cursor.execute("UPDATE signals SET initial_stop_loss = stop_loss WHERE initial_stop_loss IS NULL")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signal_level_history (
                    id SERIAL PRIMARY KEY,
                    signal_id INTEGER NOT NULL,
                    old_take_profit REAL NOT NULL,
                    old_stop_loss REAL NOT NULL,
                    new_take_profit REAL NOT NULL,
                    new_stop_loss REAL NOT NULL,
                    old_actual_entry_price REAL,
                    new_actual_entry_price REAL,
                    changed_at TEXT NOT NULL
                )
            """)
            cursor.execute("ALTER TABLE signal_level_history ADD COLUMN IF NOT EXISTS old_actual_entry_price REAL")
            cursor.execute("ALTER TABLE signal_level_history ADD COLUMN IF NOT EXISTS new_actual_entry_price REAL")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signal_level_history_signal ON signal_level_history (signal_id, id)")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signal_events (
                    id BIGSERIAL PRIMARY KEY,
                    signal_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    event_data TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signal_events_signal ON signal_events (signal_id, id)")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_logs (
                    id SERIAL PRIMARY KEY,
                    trained_at TEXT NOT NULL,
                    samples_count INTEGER NOT NULL,
                    accuracy REAL,
                    notes TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_logs (
                    id SERIAL PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    close_price REAL NOT NULL,
                    action TEXT NOT NULL,
                    probability REAL NOT NULL,
                    outcome TEXT NOT NULL,
                    details TEXT,
                    model_snapshot_id INTEGER,
                    diagnostics_json TEXT
                )
            """)
            cursor.execute("ALTER TABLE analysis_logs ADD COLUMN IF NOT EXISTS model_snapshot_id INTEGER")
            cursor.execute("ALTER TABLE analysis_logs ADD COLUMN IF NOT EXISTS diagnostics_json TEXT")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bot_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            cursor.execute("""
                INSERT INTO bot_settings (key, value) VALUES ('is_paused', 'false')
                ON CONFLICT (key) DO NOTHING
            """)

        else:
            # Схема для SQLite
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    timestamp TEXT,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL DEFAULT 0,
                    symbol TEXT DEFAULT 'XAU/USD'
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_sym_ts ON candles (symbol, timestamp)")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    actual_entry_price REAL,
                    actual_entry_source TEXT,
                    position_quantity REAL,
                    take_profit REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    initial_take_profit REAL,
                    initial_stop_loss REAL,
                    atr REAL NOT NULL,
                    probability REAL NOT NULL,
                    status TEXT DEFAULT 'OPEN',
                    closed_at TEXT,
                    exit_price REAL,
                    pnl REAL,
                    pnl_basis TEXT,
                    exit_source TEXT,
                    reported_pnl REAL,
                    reported_pnl_currency TEXT,
                    message_id INTEGER,
                    model_snapshot_id INTEGER
                )
            """)
            cursor.execute("PRAGMA table_info(signals)")
            cols = [row[1] if isinstance(row, (tuple, list)) else row["name"] for row in cursor.fetchall()]
            if "message_id" not in cols:
                cursor.execute("ALTER TABLE signals ADD COLUMN message_id INTEGER")
            if "actual_entry_price" not in cols:
                cursor.execute("ALTER TABLE signals ADD COLUMN actual_entry_price REAL")
            if "actual_entry_source" not in cols:
                cursor.execute("ALTER TABLE signals ADD COLUMN actual_entry_source TEXT")
            if "position_quantity" not in cols:
                cursor.execute("ALTER TABLE signals ADD COLUMN position_quantity REAL")
            if "pnl_basis" not in cols:
                cursor.execute("ALTER TABLE signals ADD COLUMN pnl_basis TEXT")
            if "exit_source" not in cols:
                cursor.execute("ALTER TABLE signals ADD COLUMN exit_source TEXT")
            if "reported_pnl" not in cols:
                cursor.execute("ALTER TABLE signals ADD COLUMN reported_pnl REAL")
            if "reported_pnl_currency" not in cols:
                cursor.execute("ALTER TABLE signals ADD COLUMN reported_pnl_currency TEXT")
            if "entered_at" not in cols:
                cursor.execute("ALTER TABLE signals ADD COLUMN entered_at TEXT")
            if "diagnostic_data" not in cols:
                cursor.execute("ALTER TABLE signals ADD COLUMN diagnostic_data TEXT")
            if "initial_take_profit" not in cols:
                cursor.execute("ALTER TABLE signals ADD COLUMN initial_take_profit REAL")
            if "initial_stop_loss" not in cols:
                cursor.execute("ALTER TABLE signals ADD COLUMN initial_stop_loss REAL")
            if "model_snapshot_id" not in cols:
                cursor.execute("ALTER TABLE signals ADD COLUMN model_snapshot_id INTEGER")

            cursor.execute("UPDATE signals SET initial_take_profit = take_profit WHERE initial_take_profit IS NULL")
            cursor.execute("UPDATE signals SET initial_stop_loss = stop_loss WHERE initial_stop_loss IS NULL")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signal_level_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER NOT NULL,
                    old_take_profit REAL NOT NULL,
                    old_stop_loss REAL NOT NULL,
                    new_take_profit REAL NOT NULL,
                    new_stop_loss REAL NOT NULL,
                    old_actual_entry_price REAL,
                    new_actual_entry_price REAL,
                    changed_at TEXT NOT NULL
                )
            """)
            cursor.execute("PRAGMA table_info(signal_level_history)")
            history_cols = [row[1] if isinstance(row, (tuple, list)) else row["name"] for row in cursor.fetchall()]
            if "old_actual_entry_price" not in history_cols:
                cursor.execute("ALTER TABLE signal_level_history ADD COLUMN old_actual_entry_price REAL")
            if "new_actual_entry_price" not in history_cols:
                cursor.execute("ALTER TABLE signal_level_history ADD COLUMN new_actual_entry_price REAL")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signal_level_history_signal ON signal_level_history (signal_id, id)")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signal_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    event_data TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signal_events_signal ON signal_events (signal_id, id)")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trained_at TEXT NOT NULL,
                    samples_count INTEGER NOT NULL,
                    accuracy REAL,
                    notes TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    close_price REAL NOT NULL,
                    action TEXT NOT NULL,
                    probability REAL NOT NULL,
                    outcome TEXT NOT NULL,
                    details TEXT,
                    model_snapshot_id INTEGER,
                    diagnostics_json TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bot_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            cursor.execute("INSERT OR IGNORE INTO bot_settings (key, value) VALUES ('is_paused', 'false')")

        artifact_type = "BYTEA" if IS_POSTGRES else "BLOB"
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS model_snapshots (
                id {"SERIAL PRIMARY KEY" if IS_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"},
                symbol TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                trained_at TEXT,
                accuracy REAL,
                samples_count INTEGER NOT NULL,
                features_json TEXT NOT NULL,
                weights_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                artifact_sha256 TEXT NOT NULL,
                artifact {artifact_type} NOT NULL
            )
        """)
        if IS_POSTGRES:
            cursor.execute("ALTER TABLE model_snapshots ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE")
        else:
            cursor.execute("PRAGMA table_info(model_snapshots)")
            model_snapshot_cols = [row[1] for row in cursor.fetchall()]
            if "is_active" not in model_snapshot_cols:
                cursor.execute("ALTER TABLE model_snapshots ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")
        # Older databases mark every historical snapshot active after migration;
        # retain only the newest active version per symbol, leaving candidates untouched.
        cursor.execute("""
            UPDATE model_snapshots SET is_active = FALSE
            WHERE is_active = TRUE
              AND id NOT IN (
                  SELECT MAX(id) FROM model_snapshots WHERE is_active = TRUE GROUP BY symbol
              )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_snapshots_symbol_id ON model_snapshots (symbol, id)")
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS runtime_logs (
                id {"SERIAL PRIMARY KEY" if IS_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"},
                created_at TEXT NOT NULL,
                level TEXT NOT NULL,
                logger_name TEXT NOT NULL,
                message TEXT NOT NULL,
                exception_text TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runtime_logs_time ON runtime_logs (created_at, level)")
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS operational_events (
                id {"BIGSERIAL PRIMARY KEY" if IS_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"},
                created_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                source TEXT NOT NULL,
                details_json TEXT NOT NULL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_operational_events_time ON operational_events (created_at, event_type)")
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS goal_plans (
                id {"SERIAL PRIMARY KEY" if IS_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"},
                created_at TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                starting_balance REAL NOT NULL,
                target_balance REAL NOT NULL,
                currency TEXT NOT NULL,
                method TEXT NOT NULL,
                is_active {"BOOLEAN" if IS_POSTGRES else "INTEGER"} NOT NULL DEFAULT TRUE
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_goal_plans_active ON goal_plans (is_active, id)")
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_goal_plans_single_active ON goal_plans (is_active) WHERE is_active = TRUE")
        if not IS_POSTGRES:
            cursor.execute("PRAGMA table_info(analysis_logs)")
            analysis_cols = [row[1] for row in cursor.fetchall()]
            if "model_snapshot_id" not in analysis_cols:
                cursor.execute("ALTER TABLE analysis_logs ADD COLUMN model_snapshot_id INTEGER")
            if "diagnostics_json" not in analysis_cols:
                cursor.execute("ALTER TABLE analysis_logs ADD COLUMN diagnostics_json TEXT")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_logs_model_snapshot ON analysis_logs (model_snapshot_id)")

    db_type = "облачной PostgreSQL" if IS_POSTGRES else "локальной SQLite"
    logger.info(f"База данных ({db_type}) успешно инициализирована.")


def log_candle_analysis(
    symbol: str,
    close_price: float,
    action: str,
    probability: float,
    outcome: str,
    details: str = "",
    model_snapshot_id: Optional[int] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
):
    """Записывает результат каждого проведенного анализа свечи."""
    sql = _adapt_sql("""
        INSERT INTO analysis_logs (
            created_at, symbol, close_price, action, probability, outcome, details,
            model_snapshot_id, diagnostics_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """)
    with get_db() as cursor:
        if is_active:
            cursor.execute(_adapt_sql("UPDATE model_snapshots SET is_active = FALSE WHERE symbol = ?"), (str(symbol),))
        cursor.execute(sql, (
            datetime.utcnow().isoformat(),
            str(symbol),
            float(close_price),
            str(action),
            float(probability),
            str(outcome),
            str(details),
            int(model_snapshot_id) if model_snapshot_id is not None else None,
            _json_dumps_safe(diagnostics or {}),
        ))


def _json_dumps_safe(value: Any) -> str:
    """Serialize diagnostics losslessly where possible, replacing NaN/Inf with null."""
    import math
    from datetime import date

    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple, set)):
            return [clean(val) for val in item]
        if isinstance(item, (datetime, date)):
            return item.isoformat()
        if hasattr(item, "item"):
            try:
                item = item.item()
            except Exception:
                return str(item)
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        return str(item)

    return json.dumps(clean(value), ensure_ascii=False, allow_nan=False)


def _record_operational_event(cursor, event_type: str, source: str, details: Dict[str, Any]):
    sql = _adapt_sql("""INSERT INTO operational_events (created_at, event_type, source, details_json)
        VALUES (?, ?, ?, ?)""")
    cursor.execute(sql, (datetime.utcnow().isoformat(), str(event_type)[:100],
                         str(source)[:100], _json_dumps_safe(details or {})))


def log_operational_event(event_type: str, source: str = "bot", details: Optional[Dict[str, Any]] = None):
    """Persist a timestamped lifecycle/action event for audit and later analysis."""
    with get_db() as cursor:
        _record_operational_event(cursor, event_type, source, details or {})


def get_operational_events(days: int = 1, limit: int = 1000) -> List[Dict[str, Any]]:
    from datetime import timedelta
    since = datetime.utcnow().strftime("%Y-%m-%d") + "T00:00:00" if days == 1 else (
        datetime.utcnow() - timedelta(days=max(1, int(days)))
    ).isoformat()
    sql = _adapt_sql("SELECT * FROM operational_events WHERE created_at >= ? ORDER BY id DESC LIMIT ?")
    with get_db() as cursor:
        cursor.execute(sql, (since, max(1, int(limit))))
        rows = [dict(row) for row in cursor.fetchall()]
    for row in rows:
        try:
            row["details"] = json.loads(row.pop("details_json", "{}") or "{}")
        except (TypeError, ValueError):
            row["details"] = {}
    return list(reversed(rows))


def get_active_goal_plan() -> Optional[Dict[str, Any]]:
    """Return the currently active version of the user's account-growth plan."""
    with get_db() as cursor:
        cursor.execute("SELECT * FROM goal_plans WHERE is_active = TRUE ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        return dict(row) if row else None


def get_goal_plan_history(limit: int = 12) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit), 100))
    with get_db() as cursor:
        cursor.execute(_adapt_sql("SELECT * FROM goal_plans ORDER BY id DESC LIMIT ?"), (limit,))
        return [dict(row) for row in cursor.fetchall()]


def save_goal_plan(
    start_date: str,
    end_date: str,
    starting_balance: float,
    target_balance: float,
    currency: str = "USD",
    method: str = "compound",
    source: str = "telegram_mini_app",
) -> int:
    """Save a new immutable plan version and deactivate the previous one."""
    from datetime import date

    start = date.fromisoformat(str(start_date))
    end = date.fromisoformat(str(end_date))
    start_balance = float(starting_balance)
    target = float(target_balance)
    currency = str(currency).strip().upper()
    method = str(method).strip().lower()
    if end < start or (end - start).days > 366:
        raise ValueError("Период должен быть не длиннее 367 календарных дней")
    if not (start_balance > 0 and target > start_balance):
        raise ValueError("Целевой баланс должен быть выше начального")
    if not re.fullmatch(r"[A-Z]{3}", currency):
        raise ValueError("Укажите трёхбуквенный код валюты")
    if method not in ("linear", "compound"):
        raise ValueError("Неизвестный способ расчёта цели")

    with get_db() as cursor:
        cursor.execute("UPDATE goal_plans SET is_active = FALSE WHERE is_active = TRUE")
        sql = _adapt_sql("""INSERT INTO goal_plans
            (created_at, start_date, end_date, starting_balance, target_balance, currency, method, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, TRUE)""")
        cursor.execute(sql, (datetime.utcnow().isoformat(), start.isoformat(), end.isoformat(),
                             start_balance, target, currency, method))
        if IS_POSTGRES:
            cursor.execute("SELECT LASTVAL() AS id")
            row = cursor.fetchone()
            plan_id = int(row["id"])
        else:
            plan_id = int(cursor.lastrowid)
        _record_operational_event(cursor, "goal_plan_saved", source, {
            "plan_id": plan_id, "start_date": start.isoformat(), "end_date": end.isoformat(),
            "starting_balance": start_balance, "target_balance": target,
            "currency": currency, "method": method,
        })
        return plan_id


def get_reported_pnl_for_period(start_utc: str, end_utc_exclusive: str) -> Dict[str, Any]:
    """Aggregate manually reported realized P/L by currency for deals closed in a UTC interval."""
    sql = _adapt_sql("""SELECT reported_pnl, reported_pnl_currency FROM signals
        WHERE status IN ('TP_HIT', 'SL_HIT', 'MANUAL_CLOSE') AND closed_at >= ? AND closed_at < ?""")
    with get_db() as cursor:
        cursor.execute(sql, (str(start_utc), str(end_utc_exclusive)))
        rows = [dict(row) for row in cursor.fetchall()]
    by_currency: Dict[str, float] = {}
    count_by_currency: Dict[str, int] = {}
    for row in rows:
        amount = row.get("reported_pnl")
        if amount is None:
            continue
        currency = str(row.get("reported_pnl_currency") or "USD").upper()
        by_currency[currency] = by_currency.get(currency, 0.0) + float(amount)
        count_by_currency[currency] = count_by_currency.get(currency, 0) + 1
    return {
        "closed_count": len(rows),
        "reported_count": sum(count_by_currency.values()),
        "by_currency": {key: round(value, 2) for key, value in by_currency.items()},
        "count_by_currency": count_by_currency,
    }


def get_recent_analyses(limit: int = 6) -> List[Dict[str, Any]]:
    """Возвращает последние N анализов рынка (по умолчанию 6 по числу активов)."""
    with get_db() as cursor:
        cursor.execute(f"SELECT * FROM analysis_logs ORDER BY id DESC LIMIT {limit}")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_analysis_logs(days: int = 1, limit: int = 2500) -> List[Dict[str, Any]]:
    """Return candle-level decisions with their exact model version and feature context."""
    from datetime import timedelta
    since = datetime.utcnow().strftime("%Y-%m-%d") + "T00:00:00" if days == 1 else (
        datetime.utcnow() - timedelta(days=max(1, int(days)))
    ).isoformat()
    sql = _adapt_sql("SELECT * FROM analysis_logs WHERE created_at >= ? ORDER BY id ASC LIMIT ?")
    with get_db() as cursor:
        cursor.execute(sql, (since, int(limit)))
        rows = [dict(row) for row in cursor.fetchall()]
    for row in rows:
        try:
            row["diagnostics"] = json.loads(row.pop("diagnostics_json", "{}") or "{}")
        except (TypeError, ValueError):
            row["diagnostics"] = {}
    if rows:
        snapshots = get_model_snapshots_by_ids([row.get("model_snapshot_id") for row in rows])
        for row in rows:
            snapshot_id = row.get("model_snapshot_id")
            row["model_snapshot"] = snapshots.get(int(snapshot_id)) if snapshot_id is not None else None
    return rows


def set_bot_paused(is_paused: bool):
    """Устанавливает статус паузы бота в базе данных."""
    val = "true" if is_paused else "false"
    sql = _adapt_sql("""
        INSERT INTO bot_settings (key, value) VALUES (?, ?)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
    """)
    with get_db() as cursor:
        cursor.execute(sql, ("is_paused", val))
        _record_operational_event(cursor, "bot_paused" if is_paused else "bot_resumed", "user", {})


def is_bot_paused() -> bool:
    """Проверяет, включена ли пользователем пауза бота."""
    with get_db() as cursor:
        cursor.execute("SELECT value FROM bot_settings WHERE key = 'is_paused'")
        row = cursor.fetchone()
        if row:
            return row["value"].lower() == "true"
        return False


def request_retrain():
    """Устанавливает флаг запроса на внеочередное переобучение модели."""
    sql = _adapt_sql("""
        INSERT INTO bot_settings (key, value) VALUES ('request_retrain', 'true')
        ON CONFLICT (key) DO UPDATE SET value = 'true'
    """)
    with get_db() as cursor:
        cursor.execute(sql)
        _record_operational_event(cursor, "retrain_requested", "user", {})


def consume_retrain_request() -> bool:
    """Проверяет и сбрасывает флаг запроса переобучения."""
    with get_db() as cursor:
        cursor.execute("SELECT value FROM bot_settings WHERE key = 'request_retrain'")
        row = cursor.fetchone()
        if row and row["value"].lower() == "true":
            cursor.execute("UPDATE bot_settings SET value = 'false' WHERE key = 'request_retrain'")
            _record_operational_event(cursor, "retrain_request_consumed", "bot", {})
            return True
        return False


USER_TIMEZONE = "Europe/Chisinau"


def get_day_utc_bounds(target_date: Optional[Union[str, date]] = None) -> Tuple[str, str, date, str]:
    """
    Возвращает (start_utc_iso, end_utc_iso, target_date_obj, date_label)
    для конкретного календарного дня в часовом поясе пользователя (Europe/Chisinau).
    Сделки всегда строго атрибутируются по времени открытия (created_at).
    """
    try:
        tz = ZoneInfo(USER_TIMEZONE)
    except Exception:
        tz = timezone.utc

    now_local = datetime.now(tz)

    if target_date is None or str(target_date).lower() in ("today", "сегодня", "1", "wr_1", "wr_today"):
        target_d = now_local.date()
        label = f"Сегодня ({target_d.strftime('%d.%m.%Y')})"
    elif str(target_date).lower() in ("yesterday", "вчера", "wr_yesterday"):
        target_d = now_local.date() - timedelta(days=1)
        label = f"Вчера ({target_d.strftime('%d.%m.%Y')})"
    elif isinstance(target_date, str):
        cleaned = target_date.replace("wr_date_", "").replace("wr_", "").strip()
        try:
            target_d = date.fromisoformat(cleaned[:10])
            label = target_d.strftime("%d.%m.%Y")
        except Exception:
            target_d = now_local.date()
            label = target_d.strftime("%d.%m.%Y")
    elif isinstance(target_date, date):
        target_d = target_date
        label = target_d.strftime("%d.%m.%Y")
    else:
        target_d = now_local.date()
        label = target_d.strftime("%d.%m.%Y")

    start_local = datetime.combine(target_d, datetime.min.time(), tz)
    end_local = datetime.combine(target_d + timedelta(days=1), datetime.min.time(), tz)

    start_utc_iso = start_local.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    end_utc_iso = end_local.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")

    return start_utc_iso, end_utc_iso, target_d, label


def get_winrate_stats(
    period_days: Optional[int] = None,
    target_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Рассчитывает статистику винрейта и точности рекомендаций.
    - target_date задан ("today", "yesterday", "YYYY-MM-DD") или period_days == 1:
      строго за указанный календарный день в местном времени (Europe/Chisinau).
      Сделки учитываются по дате ОТКРЫТИЯ (created_at), включая перешедшие на овернайт.
    - period_days > 1: за последние N календарных дней.
    - period_days is None и target_date is None: за всё время.
    """
    is_specific_day = bool(target_date is not None or period_days == 1)
    end_utc_str: Optional[str] = None

    with get_db() as cursor:
        if is_specific_day:
            day_arg = target_date if target_date is not None else "today"
            start_utc, end_utc, target_d, date_label = get_day_utc_bounds(day_arg)
            end_utc_str = end_utc
            sql = _adapt_sql("SELECT * FROM signals WHERE created_at >= ? AND created_at < ? ORDER BY id ASC")
            cursor.execute(sql, (start_utc, end_utc))
            period_name = date_label
        elif period_days is not None and period_days > 1:
            try:
                tz = ZoneInfo(USER_TIMEZONE)
            except Exception:
                tz = timezone.utc
            now_local = datetime.now(tz)
            since_local = datetime.combine(now_local.date() - timedelta(days=period_days - 1), datetime.min.time(), tz)
            since_utc = since_local.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
            sql = _adapt_sql("SELECT * FROM signals WHERE created_at >= ? ORDER BY id ASC")
            cursor.execute(sql, (since_utc,))
            period_name = f"{period_days} дней"
        else:
            cursor.execute("SELECT * FROM signals ORDER BY id ASC")
            period_name = "За всё время"

        rows = [dict(r) for r in cursor.fetchall()]

    total_signals = len(rows)
    wins = sum(1 for r in rows if r["status"] == "TP_HIT")
    losses = sum(1 for r in rows if r["status"] == "SL_HIT")
    expired = sum(1 for r in rows if r["status"] == "EXPIRED")
    manual = sum(1 for r in rows if r["status"] == "MANUAL_CLOSE")
    skipped = sum(1 for r in rows if r["status"] == "SKIPPED")
    open_count = sum(1 for r in rows if r["status"] == "OPEN")

    # Анализ овернайт-сделок (перешедших на следующий день или еще открытых)
    overnight_open_list = [r for r in rows if r["status"] == "OPEN"]
    overnight_closed_list = []
    if end_utc_str:
        for r in rows:
            cl = r.get("closed_at")
            if cl and cl >= end_utc_str:
                overnight_closed_list.append(r)

    finished = wins + losses
    winrate = round((wins / finished * 100), 1) if finished > 0 else 0.0
    total_pnl = sum((r["pnl"] or 0.0) for r in rows if r["status"] in ("TP_HIT", "SL_HIT"))
    actual_pnl = _summarize_actual_pnl(rows)

    return {
        "period_days": period_days,
        "target_date": target_date or ("today" if period_days == 1 else None),
        "period_name": period_name,
        "is_specific_day": is_specific_day,
        "total_signals": total_signals,
        "wins": wins,
        "losses": losses,
        "expired": expired,
        "manual": manual,
        "skipped": skipped,
        "open_count": open_count,
        "overnight_open": len(overnight_open_list),
        "overnight_closed": len(overnight_closed_list),
        "overnight_total": len(overnight_open_list) + len(overnight_closed_list),
        "finished": finished,
        "winrate": winrate,
        "total_pnl": round(total_pnl, 2),
        "signals": rows,
        **actual_pnl,
    }


def _summarize_actual_pnl(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize user-entered realized P/L without adding different currencies together."""
    totals: Dict[str, float] = {}
    wins = losses = breakeven = 0
    count = 0
    for row in rows:
        amount = row.get("reported_pnl")
        if amount is None:
            continue
        amount = float(amount)
        currency = str(row.get("reported_pnl_currency") or "USD").upper()
        totals[currency] = totals.get(currency, 0.0) + amount
        count += 1
        if amount > 0:
            wins += 1
        elif amount < 0:
            losses += 1
        else:
            breakeven += 1
    decisive = wins + losses
    return {
        "actual_pnl_by_currency": {currency: round(amount, 2) for currency, amount in sorted(totals.items())},
        "actual_pnl_count": count,
        "actual_profit_count": wins,
        "actual_loss_count": losses,
        "actual_breakeven_count": breakeven,
        "actual_winrate": round(wins / decisive * 100, 1) if decisive else None,
    }


def save_candles(df: pd.DataFrame, symbol: str = "XAU/USD"):
    """
    Сохраняет новые свечи в базу данных с привязкой к конкретному символу.
    Игнорирует дубликаты.
    """
    if df.empty:
        return

    data = df.copy()
    if "timestamp" not in data.columns:
        data = data.reset_index()
        if "datetime" in data.columns:
            data.rename(columns={"datetime": "timestamp"}, inplace=True)
        elif "index" in data.columns:
            data.rename(columns={"index": "timestamp"}, inplace=True)

    data["timestamp"] = data["timestamp"].astype(str)

    with get_db() as cursor:
        for _, row in data.iterrows():
            if IS_POSTGRES:
                cursor.execute("""
                    INSERT INTO candles (symbol, timestamp, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, timestamp) DO NOTHING
                """, (
                    symbol,
                    str(row["timestamp"]),
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row.get("volume", 0) or 0)
                ))
            else:
                cursor.execute("""
                    INSERT OR IGNORE INTO candles (symbol, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    str(row["timestamp"]),
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row.get("volume", 0) or 0)
                ))


def load_candles_from_db(symbol: Optional[str] = None, limit: int = 1000) -> pd.DataFrame:
    """Загружает последние сохраненные свечи для конкретного символа или всех символов."""
    with get_db() as cursor:
        if symbol:
            sql = _adapt_sql(f"SELECT * FROM candles WHERE symbol = ? ORDER BY timestamp DESC LIMIT {limit}")
            cursor.execute(sql, (symbol,))
        else:
            cursor.execute(f"SELECT * FROM candles ORDER BY timestamp DESC LIMIT {limit}")
        rows = [dict(r) for r in cursor.fetchall()]

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.set_index("timestamp", inplace=True)
    return df


def save_signal(
    symbol: str,
    direction: str,
    entry_price: float,
    take_profit: float,
    stop_loss: float,
    atr: float,
    probability: float,
    diagnostic_data: Optional[Dict[str, Any]] = None,
    model_snapshot_id: Optional[int] = None
) -> int:
    """Сохраняет сигнал в базу данных (включая глубокие диагностические данные весов и контекста) и возвращает его ID."""
    diag_json = _json_dumps_safe(diagnostic_data) if diagnostic_data else None

    with get_db() as cursor:
        if IS_POSTGRES:
            cursor.execute("""
                INSERT INTO signals (
                    created_at, symbol, direction, entry_price, take_profit, stop_loss,
                    initial_take_profit, initial_stop_loss, atr, probability, status, diagnostic_data,
                    model_snapshot_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'OPEN', %s, %s)
                RETURNING id
            """, (
                datetime.utcnow().isoformat(),
                str(symbol),
                str(direction),
                float(entry_price),
                float(take_profit),
                float(stop_loss),
                float(take_profit),
                float(stop_loss),
                float(atr),
                float(probability),
                diag_json,
                int(model_snapshot_id) if model_snapshot_id is not None else None
            ))
            row = cursor.fetchone()
            signal_id = row["id"] if row else 0
            if signal_id:
                _record_signal_event(cursor, signal_id, "signal_created", "bot", {
                    "entry_price": float(entry_price), "take_profit": float(take_profit),
                    "stop_loss": float(stop_loss), "atr": float(atr), "probability": float(probability)
                })
            return signal_id
        else:
            cursor.execute("""
                INSERT INTO signals (
                    created_at, symbol, direction, entry_price, take_profit, stop_loss,
                    initial_take_profit, initial_stop_loss, atr, probability, status, diagnostic_data,
                    model_snapshot_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                str(symbol),
                str(direction),
                float(entry_price),
                float(take_profit),
                float(stop_loss),
                float(take_profit),
                float(stop_loss),
                float(atr),
                float(probability),
                diag_json,
                int(model_snapshot_id) if model_snapshot_id is not None else None
            ))
            signal_id = cursor.lastrowid
            _record_signal_event(cursor, signal_id, "signal_created", "bot", {
                "entry_price": float(entry_price), "take_profit": float(take_profit),
                "stop_loss": float(stop_loss), "atr": float(atr), "probability": float(probability)
            })
            return signal_id


def _record_signal_event(cursor, signal_id: int, event_type: str, source: str, event_data: Dict[str, Any]):
    sql = _adapt_sql("""INSERT INTO signal_events
        (signal_id, event_type, source, event_data, created_at) VALUES (?, ?, ?, ?, ?)""")
    cursor.execute(sql, (int(signal_id), str(event_type), str(source),
                         _json_dumps_safe(event_data), datetime.utcnow().isoformat()))


def set_signal_message_id(signal_id: int, message_id: int):
    """Привязывает ID сообщения в Telegram к записи сигнала в базе данных."""
    sql = _adapt_sql("UPDATE signals SET message_id = ? WHERE id = ?")
    with get_db() as cursor:
        cursor.execute(sql, (int(message_id), int(signal_id)))
        if cursor.rowcount:
            _record_signal_event(cursor, signal_id, "telegram_message_linked", "telegram", {"message_id": int(message_id)})


def set_signal_entered_at(signal_id: int, source: str = "widget"):
    """Записывает момент нажатия кнопки 'Вошел в сделку' (UTC ISO-формат)."""
    from datetime import datetime, timezone
    now_str = datetime.now(timezone.utc).isoformat()
    sql = _adapt_sql("""UPDATE signals SET entered_at = COALESCE(entered_at, ?),
        actual_entry_price = COALESCE(actual_entry_price, entry_price),
        actual_entry_source = COALESCE(actual_entry_source, 'signal_reference')
        WHERE id = ? AND status = 'OPEN'""")
    with get_db() as cursor:
        cursor.execute(sql, (now_str, int(signal_id)))
        if cursor.rowcount:
            _record_signal_event(cursor, signal_id, "entered", source, {"entry_basis": "signal_reference"})


def update_signal_trade_values(signal_id: int, entry_price: float, take_profit: float, stop_loss: float,
                              source: str = "widget") -> bool:
    """Сохраняет фактический вход и уровни открытой сделки с историей TP/SL."""
    now = datetime.utcnow().isoformat()
    with get_db() as cursor:
        sql = _adapt_sql("SELECT actual_entry_price, take_profit, stop_loss FROM signals WHERE id = ? AND status = 'OPEN'")
        cursor.execute(sql, (int(signal_id),))
        row = cursor.fetchone()
        if not row:
            return False
        old_entry = row["actual_entry_price"]
        old_tp, old_sl = float(row["take_profit"]), float(row["stop_loss"])
        new_tp, new_sl = float(take_profit), float(stop_loss)
        update_sql = _adapt_sql("UPDATE signals SET actual_entry_price = ?, actual_entry_source = ?, take_profit = ?, stop_loss = ? WHERE id = ? AND status = 'OPEN'")
        cursor.execute(update_sql, (float(entry_price), str(source), new_tp, new_sl, int(signal_id)))
        if cursor.rowcount != 1:
            return False
        if old_tp != new_tp or old_sl != new_sl or old_entry is None or float(old_entry) != float(entry_price):
            history_sql = _adapt_sql("""
                INSERT INTO signal_level_history
                    (signal_id, old_take_profit, old_stop_loss, new_take_profit, new_stop_loss,
                     old_actual_entry_price, new_actual_entry_price, changed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """)
            cursor.execute(history_sql, (int(signal_id), old_tp, old_sl, new_tp, new_sl,
                                         old_entry, float(entry_price), now))
            _record_signal_event(cursor, signal_id, "levels_changed", source, {
                "old_entry": old_entry, "entry": float(entry_price), "old_tp": old_tp,
                "old_sl": old_sl, "tp": new_tp, "sl": new_sl
            })
        return True


def set_signal_position_quantity(signal_id: int, quantity: float, source: str = "widget") -> bool:
    """Сохраняет отображаемый брокером размер позиции; единицу измерения не предполагает."""
    quantity = float(quantity)
    if quantity <= 0:
        return False
    with get_db() as cursor:
        sql = _adapt_sql("UPDATE signals SET position_quantity = ? WHERE id = ? AND status = 'OPEN'")
        cursor.execute(sql, (quantity, int(signal_id)))
        if cursor.rowcount != 1:
            return False
        _record_signal_event(cursor, signal_id, "position_quantity_set", source, {"quantity": quantity})
        return True


def get_open_signals() -> List[Dict[str, Any]]:
    """Возвращает список всех активных (не закрытых) сигналов."""
    with get_db() as cursor:
        cursor.execute("SELECT * FROM signals WHERE status = 'OPEN' ORDER BY id ASC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_recent_signals(limit: int = 500, status: Optional[str] = None,
                       date_from: Optional[str] = None,
                       date_to_exclusive: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return recent signals for Mini App lists, optionally filtered by UTC creation date."""
    limit = max(1, min(int(limit), 2000))
    conditions = []
    params = []
    if status:
        conditions.append("status = ?")
        params.append(str(status))
    if date_from:
        conditions.append("created_at >= ?")
        params.append(str(date_from))
    if date_to_exclusive:
        conditions.append("created_at < ?")
        params.append(str(date_to_exclusive))
    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)
    sql = _adapt_sql(f"SELECT * FROM signals{where} ORDER BY id DESC LIMIT ?")
    with get_db() as cursor:
        cursor.execute(sql, tuple(params))
        return [dict(row) for row in cursor.fetchall()]


def update_signal_status(signal_id: int, status: str, exit_price: Optional[float], pnl: float,
                         source: str = "legacy", pnl_basis: str = "price_move_estimate") -> bool:
    """Фиксирует закрытие один раз, сохраняя источник и основу расчёта результата."""
    sql = _adapt_sql("""
        UPDATE signals
        SET status = ?, closed_at = ?, exit_price = ?, pnl = ?, exit_source = ?, pnl_basis = ?
        WHERE id = ? AND status = 'OPEN'
    """)
    with get_db() as cursor:
        cursor.execute(sql, (
            str(status),
            datetime.utcnow().isoformat(),
            float(exit_price) if exit_price is not None else None,
            float(pnl),
            str(source),
            str(pnl_basis),
            int(signal_id)
        ))
        if cursor.rowcount != 1:
            return False
        _record_signal_event(cursor, signal_id, "closed", source, {
            "status": str(status), "exit_price": float(exit_price) if exit_price is not None else None,
            "pnl": float(pnl),
            "pnl_basis": str(pnl_basis)
        })
        return True


def set_signal_reported_pnl(signal_id: int, amount: float, currency: str = "USD",
                            source: str = "mini_app") -> bool:
    """Store or correct the broker-reported result for an already closed trade."""
    currency = str(currency).upper()
    with get_db() as cursor:
        select_sql = _adapt_sql(
            "SELECT status, reported_pnl, reported_pnl_currency FROM signals WHERE id = ?"
        )
        cursor.execute(select_sql, (int(signal_id),))
        row = cursor.fetchone()
        if not row or row["status"] not in ("TP_HIT", "SL_HIT", "MANUAL_CLOSE"):
            return False
        old_amount = row["reported_pnl"]
        old_currency = row["reported_pnl_currency"]
        update_sql = _adapt_sql(
            "UPDATE signals SET reported_pnl = ?, reported_pnl_currency = ? WHERE id = ?"
        )
        cursor.execute(update_sql, (float(amount), currency, int(signal_id)))
        if cursor.rowcount != 1:
            return False
        _record_signal_event(cursor, signal_id, "actual_pnl_recorded", source, {
            "old_amount": old_amount, "old_currency": old_currency,
            "amount": float(amount), "currency": currency,
        })
        return True


def get_signal_events(signal_id: int) -> List[Dict[str, Any]]:
    sql = _adapt_sql("SELECT * FROM signal_events WHERE signal_id = ? ORDER BY id ASC")
    with get_db() as cursor:
        cursor.execute(sql, (int(signal_id),))
        rows = [dict(row) for row in cursor.fetchall()]
    for row in rows:
        try:
            row["event_data"] = json.loads(row.get("event_data") or "{}")
        except (TypeError, ValueError):
            row["event_data"] = {}
    return rows


def close_all_open_signals(reason: str = "MANUAL_RESET") -> int:
    """Закрывает все незакрытые сигналы, чтобы разблокировать поиск новых точек входа."""
    with get_db() as cursor:
        cursor.execute("SELECT id FROM signals WHERE status = 'OPEN'")
        ids = [int(row["id"]) for row in cursor.fetchall()]
        sql = _adapt_sql("""UPDATE signals SET status = ?, closed_at = ?, exit_source = ?, pnl_basis = 'none'
                           WHERE status = 'OPEN'""")
        cursor.execute(sql, (str(reason), datetime.utcnow().isoformat(), str(reason)))
        for signal_id in ids:
            _record_signal_event(cursor, signal_id, "closed", str(reason), {"status": str(reason)})
        return cursor.rowcount


def get_signal_by_id(signal_id: int) -> Optional[Dict[str, Any]]:
    """Возвращает информацию о сигнале по ID."""
    sql = _adapt_sql("SELECT * FROM signals WHERE id = ?")
    with get_db() as cursor:
        cursor.execute(sql, (int(signal_id),))
        row = cursor.fetchone()
        return dict(row) if row else None


def update_signal_levels(signal_id: int, take_profit: float, stop_loss: float) -> bool:
    """Сохраняет новую пару уровней открытой сделки и добавляет запись в историю."""
    now = datetime.utcnow().isoformat()
    with get_db() as cursor:
        sql = _adapt_sql("SELECT take_profit, stop_loss FROM signals WHERE id = ? AND status = 'OPEN'")
        cursor.execute(sql, (int(signal_id),))
        row = cursor.fetchone()
        if not row:
            return False

        old_tp = float(row["take_profit"])
        old_sl = float(row["stop_loss"])
        new_tp = float(take_profit)
        new_sl = float(stop_loss)
        if old_tp == new_tp and old_sl == new_sl:
            return True

        update_sql = _adapt_sql("""
            UPDATE signals SET take_profit = ?, stop_loss = ?
            WHERE id = ? AND status = 'OPEN'
        """)
        cursor.execute(update_sql, (new_tp, new_sl, int(signal_id)))
        if cursor.rowcount != 1:
            return False

        history_sql = _adapt_sql("""
            INSERT INTO signal_level_history
                (signal_id, old_take_profit, old_stop_loss, new_take_profit, new_stop_loss, changed_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """)
        cursor.execute(history_sql, (int(signal_id), old_tp, old_sl, new_tp, new_sl, now))
        _record_signal_event(cursor, signal_id, "levels_changed", "legacy_levels_form", {
            "old_tp": old_tp, "old_sl": old_sl, "tp": new_tp, "sl": new_sl
        })
        return True


def get_signal_level_history(signal_id: int) -> List[Dict[str, Any]]:
    """Возвращает изменения TP/SL сделки в хронологическом порядке."""
    sql = _adapt_sql("SELECT * FROM signal_level_history WHERE signal_id = ? ORDER BY id ASC")
    with get_db() as cursor:
        cursor.execute(sql, (int(signal_id),))
        return [dict(row) for row in cursor.fetchall()]


def log_training_run(samples_count: int, accuracy: float, notes: str = ""):
    """Записывает результат сессии переобучения модели."""
    sql = _adapt_sql("""
        INSERT INTO training_logs (trained_at, samples_count, accuracy, notes)
        VALUES (?, ?, ?, ?)
    """)
    with get_db() as cursor:
        cursor.execute(sql, (
            datetime.utcnow().isoformat(),
            int(samples_count),
            float(accuracy),
            str(notes)
        ))


def save_model_snapshot(
    symbol: str,
    trained_at: Optional[str],
    accuracy: float,
    samples_count: int,
    features: List[str],
    weights: Dict[str, Any],
    metadata: Dict[str, Any],
    artifact_sha256: str,
    artifact: bytes,
    is_active: bool = True,
) -> int:
    """Persist an active model or an unpromoted candidate with its exact artifact."""
    sql = _adapt_sql("""
        INSERT INTO model_snapshots (
            symbol, captured_at, trained_at, accuracy, samples_count,
            features_json, weights_json, metadata_json, artifact_sha256, artifact, is_active
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """)
    with get_db() as cursor:
        cursor.execute(sql, (
            str(symbol), datetime.utcnow().isoformat(), trained_at,
            float(accuracy), int(samples_count),
            json.dumps(features, ensure_ascii=False),
            json.dumps(weights, ensure_ascii=False, default=str, allow_nan=False),
            json.dumps(metadata, ensure_ascii=False, default=str, allow_nan=False),
            str(artifact_sha256), bytes(artifact), bool(is_active),
        ))
        if IS_POSTGRES:
            cursor.execute("SELECT LASTVAL() AS id")
            row = cursor.fetchone()
            return int(row["id"])
        return int(cursor.lastrowid)


def _decode_model_snapshot(row: Dict[str, Any], include_artifact: bool = False) -> Dict[str, Any]:
    snapshot = dict(row)
    for key in ("features_json", "weights_json", "metadata_json"):
        try:
            snapshot[key.removesuffix("_json")] = json.loads(snapshot.pop(key) or "{}")
        except (TypeError, ValueError):
            snapshot[key.removesuffix("_json")] = {}
    if include_artifact and snapshot.get("artifact") is not None:
        snapshot["artifact"] = bytes(snapshot["artifact"])
    else:
        snapshot.pop("artifact", None)
    return snapshot


def get_latest_model_snapshots(include_artifact: bool = True) -> List[Dict[str, Any]]:
    """Return the currently active durable model for each symbol."""
    artifact_column = ", s.artifact" if include_artifact else ""
    sql = f"""
        SELECT s.id, s.symbol, s.captured_at, s.trained_at, s.accuracy, s.samples_count,
               s.features_json, s.weights_json, s.metadata_json, s.artifact_sha256, s.is_active{artifact_column}
        FROM model_snapshots s
        JOIN (SELECT symbol, MAX(id) AS latest_id FROM model_snapshots WHERE is_active = TRUE GROUP BY symbol) latest
          ON latest.latest_id = s.id
        ORDER BY s.id ASC
    """
    with get_db() as cursor:
        cursor.execute(sql)
        return [_decode_model_snapshot(dict(row), include_artifact=include_artifact) for row in cursor.fetchall()]


def get_model_snapshots_history(symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Return inspectable model versions for Mini App diagnostics, excluding binary artifacts."""
    limit = max(1, min(int(limit), 500))
    selection = "id, symbol, captured_at, trained_at, accuracy, samples_count, features_json, weights_json, metadata_json, artifact_sha256, is_active"
    with get_db() as cursor:
        if symbol:
            sql = _adapt_sql(f"SELECT {selection} FROM model_snapshots WHERE symbol = ? ORDER BY id DESC LIMIT ?")
            cursor.execute(sql, (str(symbol), limit))
        else:
            sql = _adapt_sql(f"SELECT {selection} FROM model_snapshots ORDER BY id DESC LIMIT ?")
            cursor.execute(sql, (limit,))
        rows = [_decode_model_snapshot(dict(row)) for row in cursor.fetchall()]
    return rows


def activate_model_snapshot(snapshot_id: int) -> Optional[str]:
    """Make a reviewed model snapshot active for its symbol, preserving all history."""
    with get_db() as cursor:
        cursor.execute(_adapt_sql("SELECT symbol FROM model_snapshots WHERE id = ?"), (int(snapshot_id),))
        row = cursor.fetchone()
        if not row:
            return None
        symbol = str(row["symbol"])
        cursor.execute(_adapt_sql("UPDATE model_snapshots SET is_active = FALSE WHERE symbol = ?"), (symbol,))
        cursor.execute(_adapt_sql("UPDATE model_snapshots SET is_active = TRUE WHERE id = ?"), (int(snapshot_id),))
    log_operational_event("model_snapshot_activated", "user", {"symbol": symbol, "snapshot_id": int(snapshot_id)})
    return symbol


def get_model_snapshots_by_ids(snapshot_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """Load report-safe model metadata and weights for the referenced snapshots."""
    ids = sorted({int(snapshot_id) for snapshot_id in snapshot_ids if snapshot_id is not None})
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    sql = _adapt_sql(f"""
        SELECT id, symbol, captured_at, trained_at, accuracy, samples_count,
               features_json, weights_json, metadata_json, artifact_sha256, is_active
        FROM model_snapshots WHERE id IN ({placeholders})
    """)
    with get_db() as cursor:
        cursor.execute(sql, ids)
        rows = [_decode_model_snapshot(dict(row)) for row in cursor.fetchall()]
    return {int(row["id"]): row for row in rows}


def save_runtime_log(level: str, logger_name: str, message: str, exception_text: str = ""):
    """Persist warning/error diagnostics without interrupting the bot on logging failures."""
    sql = _adapt_sql("""
        INSERT INTO runtime_logs (created_at, level, logger_name, message, exception_text)
        VALUES (?, ?, ?, ?, ?)
    """)
    with get_db() as cursor:
        cursor.execute(sql, (
            datetime.utcnow().isoformat(), str(level)[:20], str(logger_name)[:255],
            str(message)[:10_000], str(exception_text)[:20_000] or None,
        ))


def get_runtime_logs(days: int = 1, limit: int = 200) -> List[Dict[str, Any]]:
    from datetime import timedelta
    since = datetime.utcnow().strftime("%Y-%m-%d") + "T00:00:00" if days == 1 else (
        datetime.utcnow() - timedelta(days=max(1, int(days)))
    ).isoformat()
    sql = _adapt_sql("SELECT * FROM runtime_logs WHERE created_at >= ? ORDER BY id DESC LIMIT ?")
    with get_db() as cursor:
        cursor.execute(sql, (since, int(limit)))
        rows = [dict(row) for row in cursor.fetchall()]
    return list(reversed(rows))


def get_signals_for_report(days: int = 1, limit: int = 100, target_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Возвращает список сигналов для формирования глубокого диагностического отчета.
    Если указан target_date ("today", "yesterday", "YYYY-MM-DD") или days == 1 —
    возвращает сделки, открытые в этот конкретный календарный день (по часовому поясу Europe/Chisinau).
    Автоматически декодирует поле diagnostic_data из JSON.
    """
    if target_date is not None:
        start_utc, end_utc, _, _ = get_day_utc_bounds(target_date)
        sql = _adapt_sql("SELECT * FROM signals WHERE created_at >= ? AND created_at < ? ORDER BY id ASC")
        with get_db() as cursor:
            cursor.execute(sql, (start_utc, end_utc))
            rows = [dict(r) for r in cursor.fetchall()]
    elif days == 1:
        start_utc, end_utc, _, _ = get_day_utc_bounds("today")
        sql = _adapt_sql("SELECT * FROM signals WHERE created_at >= ? AND created_at < ? ORDER BY id ASC")
        with get_db() as cursor:
            cursor.execute(sql, (start_utc, end_utc))
            rows = [dict(r) for r in cursor.fetchall()]
    else:
        start_date = (datetime.utcnow() - pd.Timedelta(days=days)).isoformat()
        sql = _adapt_sql("SELECT * FROM signals WHERE created_at >= ? ORDER BY id ASC")
        with get_db() as cursor:
            cursor.execute(sql, (start_date,))
            rows = [dict(r) for r in cursor.fetchall()]

    # Если за выбранный период сделок нет, и target_date не задан, подтягиваем последние доступные
    if not rows and target_date is None and days != 1:
        sql_fallback = _adapt_sql("SELECT * FROM signals ORDER BY id DESC LIMIT ?")
        with get_db() as cursor:
            cursor.execute(sql_fallback, (int(limit),))
            rows = [dict(r) for r in cursor.fetchall()]
            rows.reverse()

    # Декодируем JSON диагностических данных
    for r in rows:
        diag_raw = r.get("diagnostic_data")
        if isinstance(diag_raw, str):
            try:
                r["diagnostic_data"] = json.loads(diag_raw)
            except Exception:
                r["diagnostic_data"] = None
        elif not isinstance(diag_raw, dict):
            r["diagnostic_data"] = None

    if rows:
        ids = [int(r["id"]) for r in rows]
        placeholders = ",".join("?" for _ in ids)
        with get_db() as cursor:
            cursor.execute(_adapt_sql(
                f"SELECT * FROM signal_events WHERE signal_id IN ({placeholders}) ORDER BY id ASC"), ids)
            event_rows = [dict(row) for row in cursor.fetchall()]
            cursor.execute(_adapt_sql(
                f"SELECT * FROM signal_level_history WHERE signal_id IN ({placeholders}) ORDER BY id ASC"), ids)
            history_rows = [dict(row) for row in cursor.fetchall()]
        by_id = {signal_id: {"events": [], "level_history": []} for signal_id in ids}
        for event in event_rows:
            try:
                event["event_data"] = json.loads(event.get("event_data") or "{}")
            except (TypeError, ValueError):
                event["event_data"] = {}
            by_id[int(event["signal_id"])]["events"].append(event)
        for history in history_rows:
            by_id[int(history["signal_id"])]["level_history"].append(history)
        for row in rows:
            row.update(by_id[int(row["id"])])

        snapshots = get_model_snapshots_by_ids([row.get("model_snapshot_id") for row in rows])
        for row in rows:
            snapshot_id = row.get("model_snapshot_id")
            row["model_snapshot"] = snapshots.get(int(snapshot_id)) if snapshot_id is not None else None

    return rows


def get_today_signal_stats(target_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Рассчитывает статистику сигналов за календарный день (по умолчанию Сегодня в Europe/Chisinau).
    Возвращает словарь с количеством, винрейтом, серией убытков и распределением по активам.
    Сделки строго атрибутируются по времени открытия (created_at).
    """
    start_utc, end_utc, target_d, date_label = get_day_utc_bounds(target_date or "today")
    sql = _adapt_sql("SELECT * FROM signals WHERE created_at >= ? AND created_at < ? ORDER BY id ASC")
    with get_db() as cursor:
        cursor.execute(sql, (start_utc, end_utc))
        rows = [dict(r) for r in cursor.fetchall()]

    total_today = len(rows)
    wins = sum(1 for r in rows if r["status"] == "TP_HIT")
    losses = sum(1 for r in rows if r["status"] == "SL_HIT")
    manual = sum(1 for r in rows if r["status"] == "MANUAL_CLOSE")
    skipped = sum(1 for r in rows if r["status"] == "SKIPPED")
    open_count = sum(1 for r in rows if r["status"] == "OPEN")

    finished = wins + losses
    winrate = round((wins / finished * 100), 1) if finished > 0 else 0.0
    net_pnl = sum((r["pnl"] or 0.0) for r in rows if r["status"] in ("TP_HIT", "SL_HIT"))

    # Расчет текущей серии убытков подряд (consecutive losses)
    finished_rows = [r for r in rows if r["status"] in ("TP_HIT", "SL_HIT")]
    consecutive_losses = 0
    for r in reversed(finished_rows):
        if r["status"] == "SL_HIT":
            consecutive_losses += 1
        else:
            break

    # Распределение по активам
    by_symbol: Dict[str, int] = {}
    for r in rows:
        sym = r.get("symbol", "UNKNOWN")
        by_symbol[sym] = by_symbol.get(sym, 0) + 1
    actual_pnl = _summarize_actual_pnl(rows)

    return {
        "today_date": target_d.isoformat(),
        "date_label": date_label,
        "total_signals": total_today,
        "wins": wins,
        "losses": losses,
        "manual": manual,
        "skipped": skipped,
        "open_count": open_count,
        "finished": finished,
        "winrate": winrate,
        "net_pnl": round(net_pnl, 2),
        "consecutive_losses": consecutive_losses,
        "by_symbol": by_symbol,
        **actual_pnl,
    }


def get_today_symbol_signal_count(symbol: str) -> int:
    """Возвращает количество сигналов, выданных по конкретному активу за сегодня (UTC)."""
    today_start = datetime.utcnow().strftime("%Y-%m-%d") + "T00:00:00"
    sql = _adapt_sql("SELECT count(*) as cnt FROM signals WHERE symbol = ? AND created_at >= ?")
    with get_db() as cursor:
        cursor.execute(sql, (str(symbol), today_start))
        row = cursor.fetchone()
        return int(row["cnt"]) if row else 0


def is_kill_switch_active() -> Tuple[bool, str]:
    """
    Проверяет, активен ли аварийный Kill Switch на сегодня.
    Автоматически сбрасывается при наступлении новых суток UTC.
    """
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    with get_db() as cursor:
        cursor.execute("SELECT value FROM bot_settings WHERE key = 'kill_switch_active'")
        active_row = cursor.fetchone()
        cursor.execute("SELECT value FROM bot_settings WHERE key = 'kill_switch_date'")
        date_row = cursor.fetchone()
        cursor.execute("SELECT value FROM bot_settings WHERE key = 'kill_switch_reason'")
        reason_row = cursor.fetchone()

        if active_row and active_row["value"].lower() == "true":
            # Проверяем дату: если дата старая, Kill Switch автоматически деактивируется на новый день
            if date_row and date_row["value"] == today_str:
                reason = reason_row["value"] if reason_row else "Активирован аварийный лимит риска"
                return True, reason
            else:
                # Новые сутки — автоматический сброс
                reset_kill_switch(source="bot_daily_rollover")
                return False, ""
    return False, ""


def is_kill_switch_manually_overridden() -> bool:
    """Проверяет, был ли Kill Switch принудительно отключен пользователем сегодня."""
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    with get_db() as cursor:
        cursor.execute("SELECT value FROM bot_settings WHERE key = 'kill_switch_manual_override'")
        override_row = cursor.fetchone()
        cursor.execute("SELECT value FROM bot_settings WHERE key = 'kill_switch_override_date'")
        date_row = cursor.fetchone()
        if override_row and override_row["value"].lower() == "true":
            if date_row and date_row["value"] == today_str:
                return True
    return False


def trigger_kill_switch(reason: str, force: bool = False):
    """Активирует аварийный Kill Switch на текущие сутки UTC (если не было ручного снятия)."""
    if not force and is_kill_switch_manually_overridden():
        logger.info(f"Kill Switch пропущен: пользователь вручную разрешил торговлю на сегодня ({reason})")
        log_operational_event("kill_switch_activation_suppressed", "bot", {"reason": str(reason), "forced": False})
        return

    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    with get_db() as cursor:
        for k, v in [
            ("kill_switch_active", "true"),
            ("kill_switch_date", today_str),
            ("kill_switch_reason", str(reason)),
            ("kill_switch_manual_override", "false")
        ]:
            sql = _adapt_sql("""
                INSERT INTO bot_settings (key, value) VALUES (?, ?)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """)
            cursor.execute(sql, (k, v))
        _record_operational_event(cursor, "kill_switch_activated", "bot", {"reason": str(reason), "forced": bool(force)})
    logger.warning(f"🛑 KILL SWITCH АКТИВИРОВАН: {reason}")


def reset_kill_switch(source: str = "user"):
    """Сбрасывает флаг аварийного Kill Switch и запоминает ручную разблокировку на текущий день."""
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    with get_db() as cursor:
        for k, v in [
            ("kill_switch_active", "false"),
            ("kill_switch_manual_override", "true"),
            ("kill_switch_override_date", today_str)
        ]:
            sql = _adapt_sql("""
                INSERT INTO bot_settings (key, value) VALUES (?, ?)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """)
            cursor.execute(sql, (k, v))
        _record_operational_event(cursor, "kill_switch_reset", source, {})
    logger.info("🔓 Kill Switch сброшен. Ручная разблокировка зафиксирована на сегодня.")
