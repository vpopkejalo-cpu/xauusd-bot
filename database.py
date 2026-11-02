"""
Модуль базы данных для хранения исторических свечей,
торговых сигналов и результатов сделок.
Поддерживает два режима:
1. Облачный PostgreSQL (Neon.tech, Supabase, Render Postgres) при наличии DATABASE_URL.
2. Резервный локальный SQLite при отсутствии DATABASE_URL.
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any
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
                    take_profit REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    atr REAL NOT NULL,
                    probability REAL NOT NULL,
                    status TEXT DEFAULT 'OPEN',
                    closed_at TEXT,
                    exit_price REAL,
                    pnl REAL
                )
            """)

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
                    details TEXT
                )
            """)

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
                    take_profit REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    atr REAL NOT NULL,
                    probability REAL NOT NULL,
                    status TEXT DEFAULT 'OPEN',
                    closed_at TEXT,
                    exit_price REAL,
                    pnl REAL
                )
            """)

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
                    details TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bot_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            cursor.execute("INSERT OR IGNORE INTO bot_settings (key, value) VALUES ('is_paused', 'false')")

    db_type = "облачной PostgreSQL" if IS_POSTGRES else "локальной SQLite"
    logger.info(f"База данных ({db_type}) успешно инициализирована.")


def log_candle_analysis(
    symbol: str,
    close_price: float,
    action: str,
    probability: float,
    outcome: str,
    details: str = ""
):
    """Записывает результат каждого проведенного анализа свечи."""
    sql = _adapt_sql("""
        INSERT INTO analysis_logs (created_at, symbol, close_price, action, probability, outcome, details)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """)
    with get_db() as cursor:
        cursor.execute(sql, (
            datetime.utcnow().isoformat(),
            symbol,
            close_price,
            action,
            probability,
            outcome,
            details
        ))


def get_recent_analyses(limit: int = 6) -> List[Dict[str, Any]]:
    """Возвращает последние N анализов рынка (по умолчанию 6 по числу активов)."""
    with get_db() as cursor:
        cursor.execute(f"SELECT * FROM analysis_logs ORDER BY id DESC LIMIT {limit}")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def set_bot_paused(is_paused: bool):
    """Устанавливает статус паузы бота в базе данных."""
    val = "true" if is_paused else "false"
    sql = _adapt_sql("""
        INSERT INTO bot_settings (key, value) VALUES (?, ?)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
    """)
    with get_db() as cursor:
        cursor.execute(sql, (val, val))


def is_bot_paused() -> bool:
    """Проверяет, включена ли пользователем пауза бота."""
    with get_db() as cursor:
        cursor.execute("SELECT value FROM bot_settings WHERE key = 'is_paused'")
        row = cursor.fetchone()
        if row:
            return row["value"].lower() == "true"
        return False


def get_winrate_stats(period_days: Optional[int] = None) -> Dict[str, Any]:
    """Рассчитывает статистику винрейта и точности рекомендаций за период (в днях)."""
    from datetime import timedelta
    with get_db() as cursor:
        if period_days is not None:
            since = (datetime.utcnow() - timedelta(days=period_days)).isoformat()
            sql = _adapt_sql("SELECT * FROM signals WHERE created_at >= ?")
            cursor.execute(sql, (since,))
        else:
            cursor.execute("SELECT * FROM signals")
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
    probability: float
) -> int:
    """Сохраняет сигнал в базу данных и возвращает его ID."""
    with get_db() as cursor:
        if IS_POSTGRES:
            cursor.execute("""
                INSERT INTO signals (
                    created_at, symbol, direction, entry_price, take_profit, stop_loss, atr, probability, status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'OPEN')
                RETURNING id
            """, (
                datetime.utcnow().isoformat(),
                symbol,
                direction,
                entry_price,
                take_profit,
                stop_loss,
                atr,
                probability
            ))
            row = cursor.fetchone()
            return row["id"] if row else 0
        else:
            cursor.execute("""
                INSERT INTO signals (
                    created_at, symbol, direction, entry_price, take_profit, stop_loss, atr, probability, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
            """, (
                datetime.utcnow().isoformat(),
                symbol,
                direction,
                entry_price,
                take_profit,
                stop_loss,
                atr,
                probability
            ))
            return cursor.lastrowid


def get_open_signals() -> List[Dict[str, Any]]:
    """Возвращает список всех активных (не закрытых) сигналов."""
    with get_db() as cursor:
        cursor.execute("SELECT * FROM signals WHERE status = 'OPEN'")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def update_signal_status(signal_id: int, status: str, exit_price: float, pnl: float):
    """Обновляет статус сигнала при закрытии по TP или SL."""
    sql = _adapt_sql("""
        UPDATE signals
        SET status = ?, closed_at = ?, exit_price = ?, pnl = ?
        WHERE id = ?
    """)
    with get_db() as cursor:
        cursor.execute(sql, (
            status,
            datetime.utcnow().isoformat(),
            exit_price,
            pnl,
            signal_id
        ))


def close_all_open_signals(reason: str = "MANUAL_RESET") -> int:
    """Закрывает все незакрытые сигналы, чтобы разблокировать поиск новых точек входа."""
    sql = _adapt_sql("""
        UPDATE signals
        SET status = ?, closed_at = ?
        WHERE status = 'OPEN'
    """)
    with get_db() as cursor:
        cursor.execute(sql, (reason, datetime.utcnow().isoformat()))
        return cursor.rowcount


def get_signal_by_id(signal_id: int) -> Optional[Dict[str, Any]]:
    """Возвращает информацию о сигнале по ID."""
    sql = _adapt_sql("SELECT * FROM signals WHERE id = ?")
    with get_db() as cursor:
        cursor.execute(sql, (signal_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def log_training_run(samples_count: int, accuracy: float, notes: str = ""):
    """Записывает результат сессии переобучения модели."""
    sql = _adapt_sql("""
        INSERT INTO training_logs (trained_at, samples_count, accuracy, notes)
        VALUES (?, ?, ?, ?)
    """)
    with get_db() as cursor:
        cursor.execute(sql, (
            datetime.utcnow().isoformat(),
            samples_count,
            accuracy,
            notes
        ))
