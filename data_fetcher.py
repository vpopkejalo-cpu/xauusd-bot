"""
Модуль получения рыночных данных через Twelve Data API.
Учитывает ограничения бесплатного тарифа (800 запросов/сутки, 8 запросов/мин).
Использует библиотеку tenacity для экспоненциальной задержки при HTTP 429.
"""

import time
import logging
import requests
import urllib.request
import json
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
        # Минимальный интервал между вызовами Twelve Data (лимит 8 запросов в минуту)
        self.min_interval_between_calls = 8.0

    def _respect_rate_limits(self):
        """Гарантирует, что между запросами Twelve Data проходит безопасное время."""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval_between_calls:
            sleep_needed = self.min_interval_between_calls - elapsed
            logger.debug(f"Соблюдение лимита запросов: пауза {sleep_needed:.2f} сек.")
            time.sleep(sleep_needed)

    def _fetch_from_yahoo(self, symbol: str, outputsize: int = 120) -> pd.DataFrame:
        """
        Запасной и приоритетный поставщик рыночных данных через Yahoo Finance API.
        Обеспечивает точные данные 24/5 для индексов и сырья (USA Tech 100, Серебро, Brent, Золото).
        Не имеет жестких лимитов и возвращает живые котировки даже при закрытых сессиях акций.
        """
        yf_map = {
            "TECH100": "NQ=F",
            "USA TECH 100": "NQ=F",
            "US100": "NQ=F",
            "QQQ": "NQ=F",
            "XAG/USD": "SI=F",
            "XAGUSD": "SI=F",
            "SILVER": "SI=F",
            "BRENT": "BZ=F",
            "XBR/USD": "BZ=F",
            "UKOIL": "BZ=F",
            "BNO": "BZ=F",
            "NATGAS": "NG=F",
            "NATURAL GAS": "NG=F",
            "NG=F": "NG=F",
            "XAU/USD": "GC=F",
            "GBP/USD": "GBPUSD=X"
        }
        target = yf_map.get(symbol.upper(), symbol)
        
        # Определение таймфрейма для Yahoo Finance (5m, 15m, 1h, 1d)
        if "5" in self.interval:
            yf_interval = "5m"
        elif "15" in self.interval:
            yf_interval = "15m"
        elif "1h" in self.interval or "60" in self.interval:
            yf_interval = "1h"
        else:
            yf_interval = "5m"

        range_str = "5d" if outputsize > 60 else "2d"
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{target}?interval={yf_interval}&range={range_str}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode())

            chart_result = data.get("chart", {}).get("result", [])
            if not chart_result:
                logger.warning(f"Yahoo Finance вернул пустой результат для {target}.")
                return pd.DataFrame()

            result = chart_result[0]
            timestamps = result.get("timestamp", [])
            quote = result.get("indicators", {}).get("quote", [{}])[0]

            rows = []
            for i, t in enumerate(timestamps):
                c = quote.get("close", [])[i] if i < len(quote.get("close", [])) else None
                o = quote.get("open", [])[i] if i < len(quote.get("open", [])) else None
                h = quote.get("high", [])[i] if i < len(quote.get("high", [])) else None
                l = quote.get("low", [])[i] if i < len(quote.get("low", [])) else None
                v = quote.get("volume", [])[i] if i < len(quote.get("volume", [])) else 0
                if c is not None and o is not None and h is not None and l is not None:
                    rows.append({
                        "datetime": pd.to_datetime(t, unit="s"),
                        "open": float(o),
                        "high": float(h),
                        "low": float(l),
                        "close": float(c),
                        "volume": float(v) if v is not None else 0.0
                    })

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows)
            df.sort_values("datetime", inplace=True)
            df.set_index("datetime", inplace=True)

            if len(df) > outputsize:
                df = df.iloc[-outputsize:]

            logger.info(f"Успешно получено {len(df)} свечей для {symbol} ({target}) через Yahoo Finance. Последняя: {df.index[-1]} | Close: {df['close'].iloc[-1]}")
            return df

        except Exception as e:
            logger.error(f"Ошибка получения данных через Yahoo Finance для {symbol} ({target}): {e}")
            return pd.DataFrame()

    def fetch_time_series(self, symbol: Optional[str] = None, outputsize: int = 120) -> pd.DataFrame:
        """
        Запрашивает исторические свечи через Twelve Data API или Yahoo Finance.
        Для индексов и сырья (USA Tech 100, Серебро, Brent) сразу использует Yahoo Finance,
        так как они не поддерживаются на бесплатном тарифе Twelve Data или имеют ограничение по часам.
        Для валют (XAU/USD, GBP/USD) использует Twelve Data с автоматическим переходом на Yahoo Finance при ошибках.
        """
        raw_symbol = symbol if symbol else self.symbol

        # Символы, требующие прямого обращения к 24-часовым фьючерсам/CFD
        yahoo_preferred_symbols = (
            "TECH100", "USA TECH 100", "US100", "QQQ",
            "XAG/USD", "XAGUSD", "SILVER",
            "BRENT", "XBR/USD", "BNO", "UKOIL",
            "NATGAS", "NATURAL GAS", "NG=F"
        )

        if raw_symbol.upper() in yahoo_preferred_symbols:
            df = self._fetch_from_yahoo(raw_symbol, outputsize)
            if not df.empty:
                return df
            logger.warning(f"Yahoo Finance не вернул данные для {raw_symbol}, пробуем Twelve Data...")

        # Основной запрос к Twelve Data для валют и металлов (с защитой лимитов)
        self._respect_rate_limits()

        url = f"{BASE_URL}/time_series"
        params = {
            "symbol": raw_symbol,
            "interval": self.interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "format": "JSON"
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            self.last_call_time = time.time()

            if response.status_code == 429:
                logger.warning("Twelve Data 429: Лимит превышен. Переключаемся на резервный Yahoo Finance...")
                df = self._fetch_from_yahoo(raw_symbol, outputsize)
                if not df.empty:
                    return df
                raise RateLimitError("Twelve Data HTTP 429: Rate Limit Exceeded")

            if response.status_code == 404:
                logger.warning(f"Twelve Data 404 для {raw_symbol}. Переключаемся на Yahoo Finance...")
                df = self._fetch_from_yahoo(raw_symbol, outputsize)
                if not df.empty:
                    return df
                raise TwelveDataAPIError(f"Twelve Data HTTP 404 Not Found for {raw_symbol}")

            response.raise_for_status()
            data = response.json()

            if data.get("status") == "error":
                message = data.get("message", "Ошибка Twelve Data")
                logger.warning(f"Ошибка Twelve Data ({message}). Переключаемся на резервный источник...")
                df = self._fetch_from_yahoo(raw_symbol, outputsize)
                if not df.empty:
                    return df
                raise TwelveDataAPIError(message)

            values = data.get("values", [])
            if not values:
                logger.warning(f"Twelve Data вернула пустой список для {raw_symbol}. Пробуем Yahoo Finance...")
                df = self._fetch_from_yahoo(raw_symbol, outputsize)
                if not df.empty:
                    return df
                return pd.DataFrame()

            df = pd.DataFrame(values)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.sort_values("datetime", inplace=True)
            df.set_index("datetime", inplace=True)

            numeric_cols = ["open", "high", "low", "close"]
            for col in numeric_cols:
                df[col] = df[col].astype(float)

            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
            else:
                df["volume"] = 0.0

            logger.info(f"Успешно получено {len(df)} свечей для {raw_symbol} ({self.interval}). Последняя: {df.index[-1]}")
            return df

        except Exception as e:
            logger.warning(f"Исключение при запросе {raw_symbol} к Twelve Data: {e}. Резервный переход на Yahoo Finance...")
            df = self._fetch_from_yahoo(raw_symbol, outputsize)
            if not df.empty:
                return df
            logger.error(f"Не удалось получить данные для {raw_symbol} ни из одного источника.")
            return pd.DataFrame()

