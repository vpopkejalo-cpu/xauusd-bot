"""
Модуль мониторинга финансовых новостей и экономического календаря.
1. "Новостной щит" (Economic Calendar Shield): блокирует сделки во время выхода
   сверхволатильных данных (NFP, CPI, решения ФРС, запасы нефти EIA).
2. Анализ новостного фона (Sentiment Analysis): парсит свежие заголовки мировых агентств
   через RSS и оценивает бычий/медвежий сантимент.
"""

import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, Any, Tuple
import urllib.request
import logging

logger = logging.getLogger(__name__)

# Позитивные (бычьи) финансовые ключевые слова
BULLISH_KEYWORDS = [
    "surge", "soar", "rally", "jump", "gain", "high", "bullish", "demand",
    "rise", "cut", "stimulus", "buy", "breakout", "growth", "climb", "upbeat", "boost"
]

# Негативные (медвежьи) финансовые ключевые слова
BEARISH_KEYWORDS = [
    "drop", "plunge", "fall", "slump", "decline", "crash", "bearish", "selloff",
    "rate hike", "hawkish", "recession", "weak", "loss", "tumble", "fears", "pressure"
]

# Кэш новостей {symbol: (timestamp, result_dict)}
_news_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
CACHE_TTL_SECONDS = 900  # 15 минут


def check_economic_calendar_shield(symbol: str, check_time: datetime = None) -> Tuple[bool, str]:
    """
    Проверяет периоды выхода сверхволатильных новостей по расписанию:
    - Запасы нефти EIA (для BRENT): Среда 14:15 - 15:00 GMT.
    - Статистика США (NFP / CPI): 13:15 - 14:00 GMT (в дни публикаций).
    - Заседания FOMC / ФРС: 18:45 - 20:00 GMT.
    """
    if check_time is None:
        check_time = datetime.now(timezone.utc)

    weekday = check_time.weekday()  # 0=Пн, 2=Ср, 4=Пт
    hour = check_time.hour
    minute = check_time.minute

    # 1. Отчет по запасам сырой нефти EIA (Среда, 14:30 GMT) - критично для BRENT
    if symbol.upper() == "BRENT" and weekday == 2:
        if (hour == 14 and minute >= 15) or (hour == 15 and minute <= 0):
            return True, "Публикация отчета по запасам сырой нефти EIA в США (Среда 14:30 GMT)"

    # 2. Регулярные окна публикации важнейшей макростатистики США (13:30 GMT)
    # По пятницам (NFP) и во вторник/среду/четверг в 13:15-13:45 GMT
    if hour == 13 and 20 <= minute <= 45:
        return True, "Окно публикации макроэкономических отчетов США (Инфляция CPI / Занятость NFP)"

    # 3. Вечерние решения ФРС США по процентной ставке (FOMC) в 18:50 - 19:40 GMT
    if weekday == 2 and hour == 19:
        return True, "Окно решений Федеральной резервной системы (ФРС США)"

    return False, "Спокойный новостной фон"


def fetch_news_sentiment(symbol: str) -> Dict[str, Any]:
    """
    Загружает свежие новости по активу через RSS ленту Google News и определяет сантимент.
    Возвращает словарь с оценкой сантимента, заголовком и статусом щита.
    """
    now = time.time()
    if symbol in _news_cache:
        cached_time, cached_data = _news_cache[symbol]
        if now - cached_time < CACHE_TTL_SECONDS:
            return cached_data

    # Формируем поисковый запрос по активу
    search_queries = {
        "XAU/USD": "Gold price XAU",
        "XAG/USD": "Silver price XAG",
        "GBP/USD": "GBP USD Pound dollar",
        "TECH100": "Nasdaq tech stocks US100",
        "QQQ": "Nasdaq tech stocks QQQ",
        "BRENT": "Brent crude oil price",
        "BNO": "Brent crude oil price"
    }

    query = search_queries.get(symbol, f"{symbol} price")
    encoded_query = urllib.parse.quote(f"{query} when:2d")
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

    sentiment_score = 0.0
    top_headline = "Рыночный фон стабилен"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            content = resp.read()
            root = ET.fromstring(content)
            items = root.findall(".//item")

            bullish_count = 0
            bearish_count = 0
            headlines = []

            for item in items[:6]:  # смотрим первые 6 свежих заголовков
                title_elem = item.find("title")
                if title_elem is not None and title_elem.text:
                    title = title_elem.text.lower()
                    headlines.append(title_elem.text)

                    for word in BULLISH_KEYWORDS:
                        if word in title:
                            bullish_count += 1
                    for word in BEARISH_KEYWORDS:
                        if word in title:
                            bearish_count += 1

            if headlines:
                top_headline = headlines[0]

            total_hits = bullish_count + bearish_count
            if total_hits > 0:
                sentiment_score = round((bullish_count - bearish_count) / total_hits, 2)

    except Exception as e:
        logger.warning(f"Не удалось получить RSS новости для {symbol}: {e}")

    # Определение текстовой метки сантимента
    if sentiment_score >= 0.2:
        label = f"🟢 Бычий (+{sentiment_score})"
    elif sentiment_score <= -0.2:
        label = f"🔴 Медвежий ({sentiment_score})"
    else:
        label = "⚪ Нейтральный"

    # Проверка щита календаря
    shield_active, shield_reason = check_economic_calendar_shield(symbol)

    result = {
        "score": sentiment_score,
        "label": label,
        "top_headline": top_headline,
        "shield_active": shield_active,
        "shield_reason": shield_reason
    }

    _news_cache[symbol] = (now, result)
    return result
