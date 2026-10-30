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

# Параметры торгового инструмента
SYMBOL = os.getenv("SYMBOL", "XAU/USD").strip()
TIMEFRAME = os.getenv("TIMEFRAME", "15min").strip()  # 15min, 1h, etc.
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "300"))  # каждые 5 минут

# Настройки Machine Learning
MIN_PROBABILITY = float(os.getenv("MIN_PROBABILITY", "0.60"))  # Минимальная вероятность 60%
RETRAIN_INTERVAL_HOURS = int(os.getenv("RETRAIN_INTERVAL_HOURS", "24"))  # Переобучение раз в сутки

# Настройки управления рисками (ATR)
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
SL_ATR_MULTIPLIER = float(os.getenv("SL_ATR_MULTIPLIER", "1.5"))  # SL = 1.5 * ATR
TP_ATR_MULTIPLIER = float(os.getenv("TP_ATR_MULTIPLIER", "2.5"))  # TP = 2.5 * ATR

# Фильтр времени: Азиатская сессия (23:00 – 08:00 GMT)
FILTER_ASIAN_SESSION = os.getenv("FILTER_ASIAN_SESSION", "true").lower() in ("true", "1", "yes")
ASIAN_SESSION_START_HOUR = int(os.getenv("ASIAN_SESSION_START_HOUR", "23"))
ASIAN_SESSION_END_HOUR = int(os.getenv("ASIAN_SESSION_END_HOUR", "8"))

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
