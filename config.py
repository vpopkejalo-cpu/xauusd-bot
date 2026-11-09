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

# Ключи для новостного анализа (опциональны — бот работает без них через Google RSS)
# Finnhub: https://finnhub.io — бесплатный план 60 запросов/мин
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
# NewsAPI: https://newsapi.org — бесплатный план 100 запросов/день
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "").strip()

# Параметры торговых инструментов (Мульти-активы: 6 ключевых инструментов)
DEFAULT_SYMBOLS = ["XAU/USD", "XAG/USD", "GBP/USD", "TECH100", "BRENT", "NATGAS"]
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
    "BNO": "Brent Crude (Trading 212: Brent Oil)",
    "NATGAS": "Natural Gas (Trading 212: Natural Gas)",
    "NG=F": "Natural Gas (Trading 212: Natural Gas)"
}

SYMBOL = SYMBOLS[0]  # основной символ по умолчанию
TIMEFRAME = os.getenv("TIMEFRAME", "5min").strip()  # 5min для динамичного анализа
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "300"))  # каждые 5 минут

# Настройки Machine Learning
# Защитный порог 65% — отсекает шум и генерирует сигналы только при реальном перевесе вероятности
MIN_PROBABILITY = float(os.getenv("MIN_PROBABILITY", "0.65"))
TRAINING_CANDLES_COUNT = int(os.getenv("TRAINING_CANDLES_COUNT", "800"))  # Объем выборки на актив для обучения
RETRAIN_INTERVAL_HOURS = int(os.getenv("RETRAIN_INTERVAL_HOURS", "4"))  # Переобучение каждые 4 часа

# Настройки управления рисками (ATR) - Защита от рыночного шума
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
SL_ATR_MULTIPLIER = float(os.getenv("SL_ATR_MULTIPLIER", "1.8"))  # Надежный SL = 1.8 * ATR (защита от случайных шпилек)
TP_ATR_MULTIPLIER = float(os.getenv("TP_ATR_MULTIPLIER", "2.2"))  # Сбалансированный Take Profit = 2.2 * ATR (R:R > 1:1.2)

# Фильтр времени: Азиатская сессия отключена по умолчанию,
# так как пользователь управляет ботом вручную кнопками Старт/Пауза
FILTER_ASIAN_SESSION = os.getenv("FILTER_ASIAN_SESSION", "false").lower() in ("true", "1", "yes")
ASIAN_SESSION_START_HOUR = int(os.getenv("ASIAN_SESSION_START_HOUR", "23"))
ASIAN_SESSION_END_HOUR = int(os.getenv("ASIAN_SESSION_END_HOUR", "8"))

# Фильтр новостей и экономический календарь
NEWS_FILTER_ENABLED = os.getenv("NEWS_FILTER_ENABLED", "true").lower() in ("true", "1", "yes")

# База данных и пути хранения
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DATABASE_PATH = BASE_DIR / "trading_bot.db"
MODEL_PATH = BASE_DIR / "ml_model.joblib"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Настройки AI-Аналитика (AI Mentor):
# 1. Groq (Рекомендуется: 100% бесплатно навсегда, без карт, модели Llama 3.3 70B / DeepSeek R1, https://console.groq.com)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

# 2. OpenRouter (Бесплатные модели с суффиксом :free, https://openrouter.ai)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free").strip()

# 3. Google Gemini API (Google AI Studio)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")).strip()
AI_ANALYST_ENABLED = os.getenv("AI_ANALYST_ENABLED", "true").lower() in ("true", "1", "yes")

# Параметры доливки в прибыль (Pyramiding):
# Разрешено держать до N прибыльных позиций по одному активу одновременно
MAX_PYRAMID_POSITIONS = int(os.getenv("MAX_PYRAMID_POSITIONS", "3"))

# Режим работы бота: DEMO_MODE = True гарантирует непрерывную работу, анализ свечей и сбор опыта
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() in ("true", "1", "yes")

# Защитный риск-менеджмент:
# На демо-счете KILL_SWITCH_ENABLED отключен по умолчанию, чтобы не прерывать сбор опыта и дообучение моделей
KILL_SWITCH_ENABLED = os.getenv("KILL_SWITCH_ENABLED", "false").lower() in ("true", "1", "yes")
KILL_SWITCH_MAX_CONSECUTIVE_LOSSES = int(os.getenv("KILL_SWITCH_MAX_CONSECUTIVE_LOSSES", "5"))
KILL_SWITCH_MAX_DAILY_SIGNALS = int(os.getenv("KILL_SWITCH_MAX_DAILY_SIGNALS", "30"))
KILL_SWITCH_MAX_DAILY_LOSS = float(os.getenv("KILL_SWITCH_MAX_DAILY_LOSS", "50.0"))



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
