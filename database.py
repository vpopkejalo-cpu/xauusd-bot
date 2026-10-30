"""
Модуль локальной базы данных SQLite для хранения исторических свечей,
торговых сигналов и результатов сделок для периодического самообучения.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
from config import DATABASE_PATH

logger = logging.getLogger(__name__)


def get_connection() -> sqlite3.Connection:
    """Возвращает соединение с базой данных SQLite."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Инициализирует таблицы базы данных, если они еще не созданы."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Таблица свечей
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                timestamp TEXT PRIMARY KEY,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL DEFAULT 0
            )
        """)
        
        # Таблица сгенерированных сигналов
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
                status TEXT DEFAULT 'OPEN',     -- OPEN, TP_HIT, SL_HIT, EXPIRED
                closed_at TEXT,
                exit_price REAL,
                pnl REAL
            )
        """)
        
        # Таблица истории переобучений модели
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trained_at TEXT NOT NULL,
                samples_count INTEGER NOT NULL,
                accuracy REAL,
                notes TEXT
            )
        """)
        
        conn.commit()
    logger.info("База данных SQLite успешно инициализирована.")


def save_candles(df: pd.DataFrame):
    """
    Сохраняет новые свечи в базу данных, игнорируя дубликаты по timestamp.
    Ожидает DataFrame с колонками: open, high, low, close, volume и индексом datetime или колонкой timestamp.
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

    with get_connection() as conn:
        cursor = conn.cursor()
        for _, row in data.iterrows():
            cursor.execute("""
                INSERT OR IGNORE INTO candles (timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                str(row["timestamp"]),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row.get("volume", 0) or 0)
            ))
        conn.commit()


def load_candles_from_db(limit: int = 1000) -> pd.DataFrame:
    """Загружает последние сохраненные свечи из базы данных в виде pandas DataFrame."""
    with get_connection() as conn:
        query = f"SELECT * FROM candles ORDER BY timestamp DESC LIMIT {limit}"
        df = pd.read_sql_query(query, conn)

    if not df.empty:
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
    with get_connection() as conn:
        cursor = conn.cursor()
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
        conn.commit()
        return cursor.lastrowid


def get_open_signals() -> List[Dict[str, Any]]:
    """Возвращает список всех активных (не закрытых) сигналов."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM signals WHERE status = 'OPEN'")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def update_signal_status(signal_id: int, status: str, exit_price: float, pnl: float):
    """Обновляет статус сигнала при закрытии по TP или SL."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE signals
            SET status = ?, closed_at = ?, exit_price = ?, pnl = ?
            WHERE id = ?
        """, (
            status,
            datetime.utcnow().isoformat(),
            exit_price,
            pnl,
            signal_id
        ))
        conn.commit()


def log_training_run(samples_count: int, accuracy: float, notes: str = ""):
    """Записывает результат сессии переобучения модели."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO training_logs (trained_at, samples_count, accuracy, notes)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            samples_count,
            accuracy,
            notes
        ))
        conn.commit()
