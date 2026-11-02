"""
Модуль конфигурации торгового помощника XAU/USD.
Загружает переменные окружения из файла .env и выполняет базовую валидацию.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Путь к корневой директории проекта
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

# Загрузка переменных из .env, если файл существует
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# Ключи API и доступы
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Параметры торговых инструментов (Мульти-активы)
DEFAULT_SYMBOLS = ["XAU/USD", "XAG/USD", "GBP/USD", "TECH100", "BRENT"]
raw_symbols = os.getenv("SYMBOLS", "").strip()
if raw_symbols:
    SYMBOLS = [s.strip() for s in raw_symbols.split(",") if s.strip()]
else:
    # Совместимость с одиночным SYMBOL, если задан в .env
    single_symbol = os.getenv("SYMBOL", "").strip()
    if single_symbol and single_symbol not in ("XAU/USD", ""):
        SYMBOLS = [single_symbol]
    else:
        SYMBOLS = DEFAULT_SYMBOLS

SYMBOL_NAMES = {
    "XAU/USD": "Gold (Trading 212: XAU/USD)",
    "XAG/USD": "Silver (Trading 212: Silver)",
    "SLV": "Silver (Trading 212: Silver)",
    "GBP/USD": "GBP/USD (Trading 212: GBP/USD)",
    "TECH100": "USA Tech 100 (Trading 212: USA Tech 100)",
    "QQQ": "USA Tech 100 (Trading 212: USA Tech 100)",
    "BRENT": "Brent Crude (Trading 212: Brent Oil)",
    "XBR/USD": "Brent Crude (Trading 212: Brent Oil)",
    "BNO": "Brent Crude (Trading 212: Brent Oil)"
}

SYMBOL = SYMBOLS[0]  # основной символ по умолчанию
TIMEFRAME = os.getenv("TIMEFRAME", "5min").strip()  # 5min для динамичного анализа
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "300"))  # каждые 5 минут

# Настройки Machine Learning
MIN_PROBABILITY = float(os.getenv("MIN_PROBABILITY", "0.60"))  # Минимальная вероятность 60%
RETRAIN_INTERVAL_HOURS = int(os.getenv("RETRAIN_INTERVAL_HOURS", "24"))  # Переобучение раз в сутки

# Настройки управления рисками (ATR) - Умеренный защитный режим
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
SL_ATR_MULTIPLIER = float(os.getenv("SL_ATR_MULTIPLIER", "0.8"))  # Короткий защитный SL = 0.8 * ATR
TP_ATR_MULTIPLIER = float(os.getenv("TP_ATR_MULTIPLIER", "1.6"))  # Реалистичный TP = 1.6 * ATR (R:R = 1:2)

# Фильтр времени: Азиатская сессия отключена по умолчанию,
# так как пользователь управляет ботом вручную кнопками Старт/Пауза
FILTER_ASIAN_SESSION = os.getenv("FILTER_ASIAN_SESSION", "false").lower() in ("true", "1", "yes")
ASIAN_SESSION_START_HOUR = int(os.getenv("ASIAN_SESSION_START_HOUR", "23"))
ASIAN_SESSION_END_HOUR = int(os.getenv("ASIAN_SESSION_END_HOUR", "8"))

# Фильтр новостей и экономический календарь
NEWS_FILTER_ENABLED = os.getenv("NEWS_FILTER_ENABLED", "true").lower() in ("true", "1", "yes")

# База данных и пути хранения
DATABASE_PATH = BASE_DIR / "trading_bot.db"
MODEL_PATH = BASE_DIR / "ml_model.joblib"


def validate_config() -> list[str]:
    """
    Проверяет наличие обязательных параметров.
    Возвращает список ошибок (если есть).
    """
    errors = []
    if not TWELVE_DATA_API_KEY or TWELVE_DATA_API_KEY == "your_twelve_data_api_key_here":
        errors.append("Не указан TWELVE_DATA_API_KEY в файле .env")
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN.startswith("1234567890:"):
        errors.append("Не указан TELEGRAM_BOT_TOKEN в файле .env")
    if not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == "123456789":
        errors.append("Не указан TELEGRAM_CHAT_ID в файле .env")
    return errors
