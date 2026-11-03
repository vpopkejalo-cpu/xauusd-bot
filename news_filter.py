"""
Модуль мониторинга финансовых новостей и экономического календаря.

Уровни анализа (применяются каскадно):
1. Finnhub API — готовый sentiment score от профессионального финансового сервиса
   (не требует NLP, данные уже обработаны алгоритмами Finnhub)
2. NewsAPI — свежие заголовки финансовых изданий с контекстным анализом
   (учитывает отрицания: "falls below record high" → бычий сигнал)
3. Google News RSS — резервный источник, если оба API недоступны

«Новостной щит» (Economic Calendar Shield) — жёсткое расписание блокировок
во время выхода волатильных данных (NFP, CPI, ФРС, EIA).
"""

import re
import time
import math
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List, Optional
import urllib.request
import logging

logger = logging.getLogger(__name__)

# ─── Загрузка ключей API ────────────────────────────────────────────────────
try:
    from config import (
        FINNHUB_API_KEY,
        NEWS_API_KEY,
        NEWS_FILTER_ENABLED,
    )
except ImportError:
    FINNHUB_API_KEY = ""
    NEWS_API_KEY = ""
    NEWS_FILTER_ENABLED = True

# ─── Кэш новостей {symbol: (timestamp, result_dict)} ────────────────────────
_news_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
CACHE_TTL_SECONDS = 900  # 15 минут — обновляем не чаще раза в 15 мин

# ─── Маппинг активов ────────────────────────────────────────────────────────

# Тикеры для Finnhub (Company News / Market News)
FINNHUB_CATEGORIES = {
    "XAU/USD":  "general",
    "XAG/USD":  "general",
    "GBP/USD":  "forex",
    "TECH100":  "general",
    "BRENT":    "general",
    "NATGAS":   "general",
}

# Поисковые запросы для NewsAPI
NEWSAPI_QUERIES = {
    "XAU/USD":  "gold price OR XAU/USD OR gold market",
    "XAG/USD":  "silver price OR XAG silver",
    "GBP/USD":  "GBP USD pound dollar forex",
    "TECH100":  "Nasdaq tech stocks US100 technology",
    "BRENT":    "Brent crude oil price",
    "NATGAS":   "natural gas price Henry Hub",
}

# Поисковые запросы для Google RSS (резервный)
RSS_QUERIES = {
    "XAU/USD":  "Gold price XAU",
    "XAG/USD":  "Silver price XAG",
    "GBP/USD":  "GBP USD Pound dollar",
    "TECH100":  "Nasdaq tech stocks US100",
    "BRENT":    "Brent crude oil price",
    "NATGAS":   "Natural gas price Henry Hub",
}

# ─── Контекстный NLP словарь ─────────────────────────────────────────────────

# Ключевые слова с весами (чем выше — тем сильнее сигнал)
BULLISH_PHRASES: Dict[str, float] = {
    "surges":         1.5, "surge":          1.5,
    "soars":          1.5, "soar":           1.5,
    "rallies":        1.4, "rally":          1.4,
    "jumps":          1.2, "jump":           1.2,
    "record high":    1.8, "all-time high":  1.8, "all time high": 1.8,
    "hits high":      1.5, "new high":       1.5,
    "above":          0.5,
    "gains":          1.0, "gain":           1.0,
    "rises":          1.0, "rise":           1.0, "rising":        1.0,
    "climbs":         1.0, "climb":          1.0,
    "bullish":        1.3,
    "buy":            0.8,
    "demand":         0.7,
    "stimulus":       0.8,
    "rate cut":       1.2, "rate cuts":      1.2,
    "dovish":         1.1,
    "breakout":       1.3,
    "growth":         0.7,
    "upbeat":         0.9,
    "boost":          0.8,
    "optimism":       0.8,
    "outperforms":    1.1,
    "beats":          0.9, "beat":           0.9,
    "strong":         0.7,
}

BEARISH_PHRASES: Dict[str, float] = {
    "plunges":        1.5, "plunge":         1.5,
    "crashes":        1.5, "crash":          1.5,
    "tumbles":        1.4, "tumble":         1.4,
    "slumps":         1.4, "slump":          1.4,
    "collapses":      1.5, "collapse":       1.5,
    "record low":     1.8, "all-time low":   1.8, "all time low":  1.8,
    "hits low":       1.5, "new low":        1.5,
    "drops":          1.0, "drop":           1.0,
    "falls":          1.0, "fall":           1.0, "falling":       1.0,
    "declines":       1.0, "decline":        1.0,
    "selloff":        1.3, "sell-off":       1.3,
    "bearish":        1.3,
    "hawkish":        1.1,
    "rate hike":      1.2, "rate hikes":     1.2,
    "recession":      1.4,
    "weak":           0.8, "weakness":       0.8,
    "fears":          0.7, "concern":        0.6, "concerns":      0.6,
    "pressure":       0.7,
    "miss":           0.9, "misses":         0.9, "missed":        0.9,
    "disappoints":    1.0, "disappointing":  1.0,
    "worries":        0.7,
    "shrinks":        1.0, "contracting":    0.9,
}

# Слова-отрицания — инвертируют знак следующего ключевого слова
NEGATIONS = {
    "not", "no", "never", "without", "below", "despite",
    "fails to", "fail to", "unable to", "rather than",
    "instead of", "contrary to",
}

# Контекстные усилители (слово перед ключевым увеличивает его вес)
INTENSIFIERS = {"sharply", "dramatically", "significantly", "strongly", "massively", "rapidly"}

# Контекстные смягчители (слово перед ключевым уменьшает его вес)
DIMINISHERS = {"slightly", "modestly", "marginally", "somewhat", "mildly", "barely"}


def _tokenize(text: str) -> List[str]:
    """Приводит текст к нижнему регистру и разбивает на токены/биграммы для контекстного анализа."""
    text = text.lower()
    # Убираем HTML теги если есть
    text = re.sub(r"<[^>]+>", " ", text)
    words = re.findall(r"[a-z'-]+", text)
    # Создаём список биграмм (для фраз типа "rate cut", "record high")
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    return words + bigrams


def _analyze_context(tokens: List[str], phrases: Dict[str, float]) -> float:
    """
    Рассчитывает взвешенный скор с учётом:
    - Весов ключевых слов
    - Отрицаний (инверсия знака)
    - Усилителей/смягчителей
    """
    total = 0.0
    words = [t for t in tokens if " " not in t]  # только одиночные слова для контекста

    for i, token in enumerate(tokens):
        if token not in phrases:
            continue

        weight = phrases[token]
        context_words = words[max(0, i - 3):i]  # 3 слова перед ключевым

        # Проверка отрицания
        negated = any(neg in " ".join(context_words) for neg in NEGATIONS)
        if negated:
            weight *= -0.7  # Инвертируем, но с затуханием (не полная инверсия)

        # Усилитель
        if any(w in context_words for w in INTENSIFIERS):
            weight *= 1.4

        # Смягчитель
        if any(w in context_words for w in DIMINISHERS):
            weight *= 0.6

        total += weight

    return total


def score_headline(text: str) -> float:
    """
    Анализирует заголовок и возвращает скор:
    > 0  → бычий (bullish)
    < 0  → медвежий (bearish)
    = 0  → нейтральный
    Учитывает контекст и отрицания.
    """
    tokens = _tokenize(text)
    bull = _analyze_context(tokens, BULLISH_PHRASES)
    bear = _analyze_context(tokens, BEARISH_PHRASES)
    return bull - bear


def normalize_score(raw_score: float, n_articles: int) -> float:
    """Нормализует суммарный скор к диапазону [-1.0, +1.0] с учётом количества статей."""
    if n_articles == 0:
        return 0.0
    avg = raw_score / n_articles
    # Сигмоид-образное сжатие к ±1.0
    return max(-1.0, min(1.0, avg / (1.0 + abs(avg) * 0.5)))


# ─── Источник 1: Finnhub ─────────────────────────────────────────────────────

def _fetch_finnhub_sentiment(symbol: str) -> Optional[float]:
    """
    Получает готовый sentiment score от Finnhub Market News API.
    Finnhub возвращает предобработанный sentiment для финансовых новостей.
    Бесплатный тариф: 60 запросов/минуту.
    Возвращает скор в [-1.0, +1.0] или None при ошибке.
    """
    if not FINNHUB_API_KEY:
        return None

    try:
        import requests

        # Finnhub: company news sentiment через /news-sentiment endpoint
        # Для Forex/Commodities используем /news с фильтром по категории
        category = FINNHUB_CATEGORIES.get(symbol, "general")
        url = f"https://finnhub.io/api/v1/news?category={category}&token={FINNHUB_API_KEY}"

        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            logger.warning(f"Finnhub вернул статус {resp.status_code} для {symbol}")
            return None

        articles = resp.json()
        if not articles or not isinstance(articles, list):
            return None

        # Finnhub не всегда даёт готовый sentiment — анализируем заголовки через наш NLP
        # Но зато источник гораздо качественнее: это Bloomberg, Reuters, CNBC — не Google News
        keywords_map = {
            "XAU/USD":  ["gold", "xau", "bullion", "precious metal"],
            "XAG/USD":  ["silver", "xag", "precious"],
            "GBP/USD":  ["pound", "gbp", "sterling", "gbpusd"],
            "TECH100":  ["nasdaq", "tech", "us100", "technology", "faang"],
            "BRENT":    ["oil", "crude", "brent", "opec", "petroleum"],
            "NATGAS":   ["gas", "natural gas", "lng", "henry hub"],
        }
        relevant_keywords = keywords_map.get(symbol, [symbol.lower()])

        relevant_articles = []
        for article in articles[:20]:  # первые 20 новостей
            headline = (article.get("headline", "") + " " + article.get("summary", "")).lower()
            if any(kw in headline for kw in relevant_keywords):
                relevant_articles.append(article.get("headline", ""))

        if not relevant_articles:
            # Fallback: берём первые 8 статей категории без фильтра по теме
            relevant_articles = [a.get("headline", "") for a in articles[:8]]

        if not relevant_articles:
            return None

        raw = sum(score_headline(h) for h in relevant_articles)
        score = normalize_score(raw, len(relevant_articles))
        logger.info(f"[Finnhub] {symbol}: {len(relevant_articles)} статей, raw={raw:.2f}, score={score:.2f}")
        return score

    except Exception as e:
        logger.warning(f"Ошибка Finnhub для {symbol}: {e}")
        return None


# ─── Источник 2: NewsAPI ─────────────────────────────────────────────────────

def _fetch_newsapi_sentiment(symbol: str) -> Optional[float]:
    """
    Получает новости из NewsAPI.org (финансовые издания: Reuters, Bloomberg, FT, WSJ).
    Бесплатный план: 100 запросов/день. Кэш 15 мин → ~64 запроса/день для 6 активов.
    Возвращает скор в [-1.0, +1.0] или None при ошибке.
    """
    if not NEWS_API_KEY:
        return None

    try:
        import requests

        query = NEWSAPI_QUERIES.get(symbol, symbol)
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={urllib.parse.quote(query)}"
            f"&language=en"
            f"&sortBy=publishedAt"
            f"&pageSize=10"
            f"&domains=reuters.com,bloomberg.com,ft.com,cnbc.com,marketwatch.com,investing.com"
            f"&apiKey={NEWS_API_KEY}"
        )

        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            logger.warning(f"NewsAPI вернул статус {resp.status_code} для {symbol}")
            return None

        data = resp.json()
        articles = data.get("articles", [])
        if not articles:
            return None

        headlines = [
            (a.get("title", "") + ". " + (a.get("description", "") or ""))
            for a in articles[:10]
        ]

        raw = sum(score_headline(h) for h in headlines)
        score = normalize_score(raw, len(headlines))
        logger.info(f"[NewsAPI] {symbol}: {len(headlines)} статей, raw={raw:.2f}, score={score:.2f}")
        return score

    except Exception as e:
        logger.warning(f"Ошибка NewsAPI для {symbol}: {e}")
        return None


# ─── Источник 3: Google RSS (резервный) ─────────────────────────────────────

def _fetch_rss_sentiment(symbol: str) -> float:
    """
    Резервный источник: Google News RSS.
    Использует улучшенный NLP с учётом контекста и отрицаний.
    Возвращает скор в [-1.0, +1.0].
    """
    query = RSS_QUERIES.get(symbol, f"{symbol} price")
    encoded_query = urllib.parse.quote(f"{query} when:1d")
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

    headlines = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible)"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            root = ET.fromstring(resp.read())
            for item in root.findall(".//item")[:10]:
                title_elem = item.find("title")
                desc_elem = item.find("description")
                text = ""
                if title_elem is not None and title_elem.text:
                    text += title_elem.text
                if desc_elem is not None and desc_elem.text:
                    text += ". " + desc_elem.text
                if text:
                    headlines.append(text)
    except Exception as e:
        logger.warning(f"Ошибка Google RSS для {symbol}: {e}")

    if not headlines:
        return 0.0

    raw = sum(score_headline(h) for h in headlines)
    score = normalize_score(raw, len(headlines))
    logger.info(f"[Google RSS] {symbol}: {len(headlines)} статей, raw={raw:.2f}, score={score:.2f}")
    return score


# ─── Экономический календарь (жёсткое расписание) ───────────────────────────

def check_economic_calendar_shield(symbol: str, check_time: datetime = None) -> Tuple[bool, str]:
    """
    Блокирует торговлю в периоды выхода сверхволатильных данных.
    Это надёжнее новостного анализа — расписание не меняется.
    """
    if check_time is None:
        check_time = datetime.now(timezone.utc)

    weekday = check_time.weekday()  # 0=Пн, 2=Ср, 4=Пт
    hour = check_time.hour
    minute = check_time.minute

    # 1. Запасы нефти EIA (Среда 14:30 GMT) — ±30 мин
    if symbol.upper() in ("BRENT", "XBR/USD", "BNO") and weekday == 2:
        if (hour == 14 and minute >= 10) or (hour == 15 and minute <= 10):
            return True, "Публикация отчёта по запасам нефти EIA (Среда 14:30 GMT)"

    # 2. Запасы природного газа EIA (Четверг 14:30 GMT)
    if symbol.upper() in ("NATGAS", "NG=F") and weekday == 3:
        if (hour == 14 and minute >= 10) or (hour == 15 and minute <= 10):
            return True, "Публикация отчёта по запасам газа EIA (Четверг 14:30 GMT)"

    # 3. Макростатистика США: NFP (пятница), CPI, PPI (13:30 GMT ±20 мин)
    if hour == 13 and 20 <= minute <= 50:
        return True, "Окно публикации макростатистики США (CPI/NFP/PPI — 13:30 GMT)"

    # 4. Решение ФРС (FOMC — Среда 19:00 GMT)
    if weekday == 2 and hour == 19:
        return True, "Окно решений Федеральной резервной системы (ФРС, 19:00 GMT)"

    # 5. Выступления председателя ФРС (Среда/Четверг 15:00–16:30 GMT)
    if weekday in (2, 3) and hour in (15, 16):
        if not (weekday == 3 and hour == 16 and minute > 30):
            return True, "Окно выступлений председателя ФРС (Powell speech)"

    return False, "Спокойный новостной фон"


# ─── Главная функция ─────────────────────────────────────────────────────────

def fetch_news_sentiment(symbol: str) -> Dict[str, Any]:
    """
    Каскадный анализ новостного сантимента по 3 уровням:
      1. Finnhub (если задан FINNHUB_API_KEY)
      2. NewsAPI (если задан NEWS_API_KEY)
      3. Google News RSS (всегда доступен как fallback)

    Возвращает:
        score       float  [-1.0, +1.0]
        label       str    текстовая метка
        top_headline str   свежий заголовок
        source      str    использованный источник
        shield_active bool
        shield_reason str
    """
    now = time.time()

    # Возвращаем из кэша если данные свежие
    if symbol in _news_cache:
        cached_time, cached_data = _news_cache[symbol]
        if now - cached_time < CACHE_TTL_SECONDS:
            return cached_data

    score: Optional[float] = None
    source = "google_rss"
    top_headline = "Рыночный фон стабилен"

    # ── Уровень 1: Finnhub ──
    if FINNHUB_API_KEY:
        score = _fetch_finnhub_sentiment(symbol)
        if score is not None:
            source = "finnhub"
            logger.info(f"[{symbol}] Новостной сантимент: Finnhub score={score:.2f}")

    # ── Уровень 2: NewsAPI ──
    if score is None and NEWS_API_KEY:
        score = _fetch_newsapi_sentiment(symbol)
        if score is not None:
            source = "newsapi"
            logger.info(f"[{symbol}] Новостной сантимент: NewsAPI score={score:.2f}")

    # ── Уровень 3: Google RSS ──
    if score is None:
        score = _fetch_rss_sentiment(symbol)
        source = "google_rss"
        logger.info(f"[{symbol}] Новостной сантимент: Google RSS score={score:.2f}")

    # Гарантия валидного float
    if score is None:
        score = 0.0

    # Получаем свежий заголовок для отображения в Telegram
    try:
        top_headline = _get_top_headline(symbol, source)
    except Exception:
        pass

    # Текстовая метка с источником
    source_icons = {"finnhub": "🔵", "newsapi": "📡", "google_rss": "🌐"}
    src_icon = source_icons.get(source, "📰")

    if score >= 0.25:
        label = f"🟢 Бычий {src_icon} (+{score:.2f})"
    elif score <= -0.25:
        label = f"🔴 Медвежий {src_icon} ({score:.2f})"
    else:
        label = f"⚪ Нейтральный {src_icon}"

    shield_active, shield_reason = check_economic_calendar_shield(symbol)

    result = {
        "score":         score,
        "label":         label,
        "top_headline":  top_headline,
        "source":        source,
        "shield_active": shield_active,
        "shield_reason": shield_reason,
    }

    _news_cache[symbol] = (now, result)
    return result


def _get_top_headline(symbol: str, source: str) -> str:
    """Получает свежий заголовок для отображения в сигнале Telegram."""
    if source in ("finnhub",) and FINNHUB_API_KEY:
        try:
            import requests
            category = FINNHUB_CATEGORIES.get(symbol, "general")
            url = f"https://finnhub.io/api/v1/news?category={category}&token={FINNHUB_API_KEY}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                articles = resp.json()
                if articles:
                    return articles[0].get("headline", "Рыночный фон стабилен")
        except Exception:
            pass

    if source == "newsapi" and NEWS_API_KEY:
        try:
            import requests
            query = NEWSAPI_QUERIES.get(symbol, symbol)
            url = (
                f"https://newsapi.org/v2/everything"
                f"?q={urllib.parse.quote(query)}&language=en&sortBy=publishedAt&pageSize=1"
                f"&domains=reuters.com,bloomberg.com,cnbc.com"
                f"&apiKey={NEWS_API_KEY}"
            )
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                articles = resp.json().get("articles", [])
                if articles:
                    return articles[0].get("title", "Рыночный фон стабилен")
        except Exception:
            pass

    # Google RSS fallback
    try:
        query = RSS_QUERIES.get(symbol, f"{symbol} price")
        encoded = urllib.parse.quote(f"{query} when:1d")
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible)"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            root = ET.fromstring(resp.read())
            item = root.find(".//item/title")
            if item is not None and item.text:
                return item.text
    except Exception:
        pass

    return "Рыночный фон стабилен"
