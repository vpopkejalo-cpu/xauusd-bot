"""
Модуль получения рыночных данных через Twelve Data API.
Учитывает ограничения бесплатного тарифа (800 запросов/сутки, 8 запросов/мин).
Использует библиотеку tenacity для экспоненциальной задержки при HTTP 429.
"""

import time
import logging
import requests
import pandas as pd
from typing import Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from config import TWELVE_DATA_API_KEY, SYMBOL, TIMEFRAME

logger = logging.getLogger(__name__)

# Базовый URL API Twelve Data
BASE_URL = "https://api.twelvedata.com"


class RateLimitError(Exception):
    """Исключение, возникающее при превышении лимита запросов (HTTP 429)."""
    pass


class TwelveDataAPIError(Exception):
    """Исключение при ошибках API Twelve Data (неверный ключ, ошибка сервера и т.д.)."""
    pass


class DataFetcher:
    def __init__(self, api_key: str = TWELVE_DATA_API_KEY, symbol: str = SYMBOL, interval: str = TIMEFRAME):
        self.api_key = api_key
        self.symbol = symbol
        self.interval = interval
        self.last_call_time = 0.0
        # Минимальный интервал между вызовами (чтобы не превысить 8 запросов в минуту = минимум 7.5 секунд)
        self.min_interval_between_calls = 8.0

    def _respect_rate_limits(self):
        """Гарантирует, что между запросами проходит безопасное время."""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval_between_calls:
            sleep_needed = self.min_interval_between_calls - elapsed
            logger.debug(f"Соблюдение лимита запросов: пауза {sleep_needed:.2f} сек.")
            time.sleep(sleep_needed)

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=2, min=10, max=120),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def fetch_time_series(self, symbol: Optional[str] = None, outputsize: int = 120) -> pd.DataFrame:
        """
        Запрашивает исторические свечи через endpoint /time_series.
        В случае ошибки HTTP 429 или сообщения о лимитах автоматически повторяет с экспоненциальной задержкой.
        
        Возвращает DataFrame со свечами (индекс - datetime, колонки open, high, low, close, volume).
        """
        self._respect_rate_limits()
        raw_symbol = symbol if symbol else self.symbol

        # Автоматический маппинг символов на тикеры Twelve Data
        ticker_map = {
            "TECH100": "QQQ",
            "NAS100": "QQQ",
            "BRENT": "XBR/USD",
            "BNO": "XBR/USD",
            "XAGUSD": "XAG/USD"
        }
        target_symbol = ticker_map.get(raw_symbol, raw_symbol)

        url = f"{BASE_URL}/time_series"
        params = {
            "symbol": target_symbol,
            "interval": self.interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "format": "JSON"
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            self.last_call_time = time.time()
            
            if response.status_code == 429:
                logger.warning("Получен HTTP 429: Превышен лимит запросов Twelve Data. Активируем задержку tenacity.")
                raise RateLimitError("Twelve Data HTTP 429: Rate Limit Exceeded")

            response.raise_for_status()
            data = response.json()

            # Проверка статуса в теле ответа Twelve Data
            if data.get("status") == "error":
                code = data.get("code")
                message = data.get("message", "Неизвестная ошибка Twelve Data")
                if code == 429 or "rate limit" in message.lower():
                    logger.warning(f"Ошибка API (лимит): {message}")
                    raise RateLimitError(message)
                else:
                    logger.error(f"Ошибка API Twelve Data: {message}")
                    raise TwelveDataAPIError(message)

            values = data.get("values", [])
            if not values:
                logger.warning(f"Twelve Data вернула пустой список свечей для {target_symbol}.")
                return pd.DataFrame()

            # Преобразование в DataFrame
            df = pd.DataFrame(values)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.sort_values("datetime", inplace=True)
            df.set_index("datetime", inplace=True)

            # Приведение типов колонок к float
            numeric_cols = ["open", "high", "low", "close"]
            for col in numeric_cols:
                df[col] = df[col].astype(float)

            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
            else:
                df["volume"] = 0.0

            logger.info(f"Успешно получено {len(df)} свечей для {raw_symbol} ({self.interval}). Последняя: {df.index[-1]}")
            return df

        except requests.RequestException as e:
            logger.error(f"Сетевая ошибка при запросе к Twelve Data: {e}")
            raise
