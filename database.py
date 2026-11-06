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
                    take_profit REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    atr REAL NOT NULL,
                    probability REAL NOT NULL,
                    status TEXT DEFAULT 'OPEN',
                    closed_at TEXT,
                    exit_price REAL,
                    pnl REAL,
                    message_id BIGINT
                )
            """)
            try:
                cursor.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS message_id BIGINT")
                cursor.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS entered_at TEXT")
                cursor.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS diagnostic_data TEXT")
            except Exception:
                pass

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
                    pnl REAL,
                    message_id INTEGER
                )
            """)
            try:
                cursor.execute("PRAGMA table_info(signals)")
                cols = [row[1] if isinstance(row, (tuple, list)) else row["name"] for row in cursor.fetchall()]
                if "message_id" not in cols:
                    cursor.execute("ALTER TABLE signals ADD COLUMN message_id INTEGER")
                if "entered_at" not in cols:
                    cursor.execute("ALTER TABLE signals ADD COLUMN entered_at TEXT")
                if "diagnostic_data" not in cols:
                    cursor.execute("ALTER TABLE signals ADD COLUMN diagnostic_data TEXT")
            except Exception:
                pass

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
            str(symbol),
            float(close_price),
            str(action),
            float(probability),
            str(outcome),
            str(details)
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
    diagnostic_data: Optional[Dict[str, Any]] = None
) -> int:
    """Сохраняет сигнал в базу данных (включая глубокие диагностические данные весов и контекста) и возвращает его ID."""
    diag_json = json.dumps(diagnostic_data, ensure_ascii=False) if diagnostic_data else None

    with get_db() as cursor:
        if IS_POSTGRES:
            cursor.execute("""
                INSERT INTO signals (
                    created_at, symbol, direction, entry_price, take_profit, stop_loss, atr, probability, status, diagnostic_data
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'OPEN', %s)
                RETURNING id
            """, (
                datetime.utcnow().isoformat(),
                str(symbol),
                str(direction),
                float(entry_price),
                float(take_profit),
                float(stop_loss),
                float(atr),
                float(probability),
                diag_json
            ))
            row = cursor.fetchone()
            return row["id"] if row else 0
        else:
            cursor.execute("""
                INSERT INTO signals (
                    created_at, symbol, direction, entry_price, take_profit, stop_loss, atr, probability, status, diagnostic_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
            """, (
                datetime.utcnow().isoformat(),
                str(symbol),
                str(direction),
                float(entry_price),
                float(take_profit),
                float(stop_loss),
                float(atr),
                float(probability),
                diag_json
            ))
            return cursor.lastrowid


def set_signal_message_id(signal_id: int, message_id: int):
    """Привязывает ID сообщения в Telegram к записи сигнала в базе данных."""
    sql = _adapt_sql("UPDATE signals SET message_id = ? WHERE id = ?")
    with get_db() as cursor:
        cursor.execute(sql, (int(message_id), int(signal_id)))


def set_signal_entered_at(signal_id: int):
    """Записывает момент нажатия кнопки 'Вошел в сделку' (UTC ISO-формат)."""
    from datetime import datetime, timezone
    now_str = datetime.now(timezone.utc).isoformat()
    sql = _adapt_sql("UPDATE signals SET entered_at = ? WHERE id = ?")
    with get_db() as cursor:
        cursor.execute(sql, (now_str, int(signal_id)))


def get_open_signals() -> List[Dict[str, Any]]:
    """Возвращает список всех активных (не закрытых) сигналов."""
    with get_db() as cursor:
        cursor.execute("SELECT * FROM signals WHERE status = 'OPEN' ORDER BY id ASC")
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
            str(status),
            datetime.utcnow().isoformat(),
            float(exit_price),
            float(pnl),
            int(signal_id)
        ))


def close_all_open_signals(reason: str = "MANUAL_RESET") -> int:
    """Закрывает все незакрытые сигналы, чтобы разблокировать поиск новых точек входа."""
    sql = _adapt_sql("""
        UPDATE signals
        SET status = ?, closed_at = ?
        WHERE status = 'OPEN'
    """)
    with get_db() as cursor:
        cursor.execute(sql, (str(reason), datetime.utcnow().isoformat()))
        return cursor.rowcount


def get_signal_by_id(signal_id: int) -> Optional[Dict[str, Any]]:
    """Возвращает информацию о сигнале по ID."""
    sql = _adapt_sql("SELECT * FROM signals WHERE id = ?")
    with get_db() as cursor:
        cursor.execute(sql, (int(signal_id),))
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
            int(samples_count),
            float(accuracy),
            str(notes)
        ))


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

