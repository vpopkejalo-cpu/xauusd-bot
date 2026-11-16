"""
Модуль базы данных для хранения исторических свечей,
торговых сигналов и результатов сделок.
Поддерживает два режима:
1. Облачный PostgreSQL (Neon.tech, Supabase, Render Postgres) при наличии DATABASE_URL.
2. Резервный локальный SQLite при отсутствии DATABASE_URL.
"""

import os
import json
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
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
            try:
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
            except Exception:
                pass

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
            try:
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
            except Exception:
                pass

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
        cursor.execute(sql, (
            datetime.utcnow().isoformat(),
            str(symbol),
            float(close_price),
            str(action),
            float(probability),
            str(outcome),
            str(details),
            int(model_snapshot_id) if model_snapshot_id is not None else None,
            json.dumps(diagnostics or {}, ensure_ascii=False, default=str),
        ))


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


def consume_retrain_request() -> bool:
    """Проверяет и сбрасывает флаг запроса переобучения."""
    with get_db() as cursor:
        cursor.execute("SELECT value FROM bot_settings WHERE key = 'request_retrain'")
        row = cursor.fetchone()
        if row and row["value"].lower() == "true":
            cursor.execute("UPDATE bot_settings SET value = 'false' WHERE key = 'request_retrain'")
            return True
        return False


def get_winrate_stats(period_days: Optional[int] = None) -> Dict[str, Any]:
    """
    Рассчитывает статистику винрейта и точности рекомендаций за период (в днях).
    period_days == 1: строго за СЕГОДНЯ (календарные сутки с 00:00:00 UTC).
    period_days > 1: за последние N календарных дней.
    period_days is None: за всё время.
    """
    from datetime import timedelta
    with get_db() as cursor:
        if period_days == 1:
            # Календарные сутки "Сегодня" (с 00:00:00 UTC)
            since = datetime.utcnow().strftime("%Y-%m-%d") + "T00:00:00"
            sql = _adapt_sql("SELECT * FROM signals WHERE created_at >= ? ORDER BY id ASC")
            cursor.execute(sql, (since,))
        elif period_days is not None and period_days > 1:
            # За последние N календарных дней
            since = (datetime.utcnow() - timedelta(days=period_days - 1)).strftime("%Y-%m-%d") + "T00:00:00"
            sql = _adapt_sql("SELECT * FROM signals WHERE created_at >= ? ORDER BY id ASC")
            cursor.execute(sql, (since,))
        else:
            cursor.execute("SELECT * FROM signals ORDER BY id ASC")
        rows = [dict(r) for r in cursor.fetchall()]

    total_signals = len(rows)
    wins = sum(1 for r in rows if r["status"] == "TP_HIT")
    losses = sum(1 for r in rows if r["status"] == "SL_HIT")
    expired = sum(1 for r in rows if r["status"] == "EXPIRED")
    manual = sum(1 for r in rows if r["status"] == "MANUAL_CLOSE")
    skipped = sum(1 for r in rows if r["status"] == "SKIPPED")
    open_count = sum(1 for r in rows if r["status"] == "OPEN")

    finished = wins + losses
    winrate = round((wins / finished * 100), 1) if finished > 0 else 0.0
    total_pnl = sum((r["pnl"] or 0.0) for r in rows if r["status"] in ("TP_HIT", "SL_HIT"))

    return {
        "period_days": period_days,
        "total_signals": total_signals,
        "wins": wins,
        "losses": losses,
        "expired": expired,
        "manual": manual,
        "skipped": skipped,
        "open_count": open_count,
        "finished": finished,
        "winrate": winrate,
        "total_pnl": round(total_pnl, 2)
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
    diag_json = json.dumps(diagnostic_data, ensure_ascii=False) if diagnostic_data else None

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
                         json.dumps(event_data, ensure_ascii=False, default=str), datetime.utcnow().isoformat()))


def set_signal_message_id(signal_id: int, message_id: int):
    """Привязывает ID сообщения в Telegram к записи сигнала в базе данных."""
    sql = _adapt_sql("UPDATE signals SET message_id = ? WHERE id = ?")
    with get_db() as cursor:
        cursor.execute(sql, (int(message_id), int(signal_id)))


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
) -> int:
    """Persist both the exact model artifact and a human-readable weight snapshot."""
    sql = _adapt_sql("""
        INSERT INTO model_snapshots (
            symbol, captured_at, trained_at, accuracy, samples_count,
            features_json, weights_json, metadata_json, artifact_sha256, artifact
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """)
    with get_db() as cursor:
        cursor.execute(sql, (
            str(symbol), datetime.utcnow().isoformat(), trained_at,
            float(accuracy), int(samples_count),
            json.dumps(features, ensure_ascii=False),
            json.dumps(weights, ensure_ascii=False, default=str, allow_nan=False),
            json.dumps(metadata, ensure_ascii=False, default=str, allow_nan=False),
            str(artifact_sha256), bytes(artifact),
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
    """Return the newest durable model version for each symbol."""
    artifact_column = ", s.artifact" if include_artifact else ""
    sql = f"""
        SELECT s.id, s.symbol, s.captured_at, s.trained_at, s.accuracy, s.samples_count,
               s.features_json, s.weights_json, s.metadata_json, s.artifact_sha256{artifact_column}
        FROM model_snapshots s
        JOIN (SELECT symbol, MAX(id) AS latest_id FROM model_snapshots GROUP BY symbol) latest
          ON latest.latest_id = s.id
        ORDER BY s.id ASC
    """
    with get_db() as cursor:
        cursor.execute(sql)
        return [_decode_model_snapshot(dict(row), include_artifact=include_artifact) for row in cursor.fetchall()]


def get_model_snapshots_by_ids(snapshot_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """Load report-safe model metadata and weights for the referenced snapshots."""
    ids = sorted({int(snapshot_id) for snapshot_id in snapshot_ids if snapshot_id is not None})
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    sql = _adapt_sql(f"""
        SELECT id, symbol, captured_at, trained_at, accuracy, samples_count,
               features_json, weights_json, metadata_json, artifact_sha256
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


def get_signals_for_report(days: int = 1, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Возвращает список сигналов для формирования глубокого диагностического отчета.
    Приоритетно возвращает сделки за сегодня (UTC), а если их нет — последние `limit` сигналов.
    Автоматически декодирует поле diagnostic_data из JSON.
    """
    if days == 1:
        start_date = datetime.utcnow().strftime("%Y-%m-%d") + "T00:00:00"
    else:
        start_date = (datetime.utcnow() - pd.Timedelta(days=days)).isoformat()

    sql = _adapt_sql("SELECT * FROM signals WHERE created_at >= ? ORDER BY id ASC")
    with get_db() as cursor:
        cursor.execute(sql, (start_date,))
        rows = [dict(r) for r in cursor.fetchall()]

    # Если за выбранный период сделок нет, подтягиваем последние доступные
    if not rows:
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


def get_today_signal_stats() -> Dict[str, Any]:
    """
    Рассчитывает статистику сигналов за текущие календарные сутки UTC.
    Возвращает словарь с количеством, винрейтом, серией убытков и распределением по активам.
    """
    today_start = datetime.utcnow().strftime("%Y-%m-%d") + "T00:00:00"
    sql = _adapt_sql("SELECT * FROM signals WHERE created_at >= ? ORDER BY id ASC")
    with get_db() as cursor:
        cursor.execute(sql, (today_start,))
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

    return {
        "today_date": datetime.utcnow().strftime("%Y-%m-%d"),
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
        "by_symbol": by_symbol
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
                reset_kill_switch()
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
    logger.warning(f"🛑 KILL SWITCH АКТИВИРОВАН: {reason}")


def reset_kill_switch():
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
    logger.info("🔓 Kill Switch сброшен. Ручная разблокировка зафиксирована на сегодня.")

