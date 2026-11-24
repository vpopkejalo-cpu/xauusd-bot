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

def _scored_articles(articles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the exact scored headlines and source metadata used for one sentiment result."""
    scored = []
    for article in articles:
        headline = str(article.get("headline") or "").strip()
        text = str(article.get("score_text") or headline).strip()
        if not text:
            continue
        scored.append({
            "headline": headline[:300],
            "score_text_excerpt": text[:400],
            "published_at": article.get("published_at"),
            "url": str(article.get("url") or "")[:1000],
            "score": float(score_headline(text)),
        })
    if not scored:
        return None
    raw = sum(item["score"] for item in scored)
    return {"score": normalize_score(raw, len(scored)), "articles": scored, "article_count": len(scored)}


def _fetch_finnhub_news(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch and return the same Finnhub articles that are scored and logged."""
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
        for article in articles[:20]:
            headline = (article.get("headline", "") + " " + article.get("summary", "")).lower()
            if any(kw in headline for kw in relevant_keywords):
                relevant_articles.append(article)

        if not relevant_articles:
            relevant_articles = articles[:8]

        result = _scored_articles([{
            "headline": article.get("headline", ""),
            "score_text": article.get("headline", ""),  # Keep the original headline-only scoring behavior.
            "published_at": article.get("datetime"),
            "url": article.get("url", ""),
        } for article in relevant_articles])
        if result:
            logger.info("[Finnhub] %s: %s статей, score=%.2f", symbol, result["article_count"], result["score"])
        return result

    except Exception as e:
        logger.warning(f"Ошибка Finnhub для {symbol}: {e}")
        return None


# ─── Источник 2: NewsAPI ─────────────────────────────────────────────────────

def _fetch_newsapi_news(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch and return the exact NewsAPI articles used for the sentiment score."""
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

        result = _scored_articles([{
            "headline": article.get("title", ""),
            "score_text": (article.get("title", "") + ". " + (article.get("description", "") or "")),
            "published_at": article.get("publishedAt"),
            "url": article.get("url", ""),
        } for article in articles[:10]])
        if result:
            logger.info("[NewsAPI] %s: %s статей, score=%.2f", symbol, result["article_count"], result["score"])
        return result

    except Exception as e:
        logger.warning(f"Ошибка NewsAPI для {symbol}: {e}")
        return None


# ─── Источник 3: Google RSS (резервный) ─────────────────────────────────────

def _fetch_rss_news(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch and return the exact Google News RSS items used for the score."""
    query = RSS_QUERIES.get(symbol, f"{symbol} price")
    encoded_query = urllib.parse.quote(f"{query} when:1d")
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

    articles = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible)"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            root = ET.fromstring(resp.read())
            for item in root.findall(".//item")[:10]:
                title_elem = item.find("title")
                desc_elem = item.find("description")
                title = title_elem.text if title_elem is not None and title_elem.text else ""
                description = desc_elem.text if desc_elem is not None and desc_elem.text else ""
                link_elem = item.find("link")
                date_elem = item.find("pubDate")
                text = (title + ". " + description).strip()
                if text:
                    articles.append({
                        "headline": title,
                        "score_text": text,
                        "published_at": date_elem.text if date_elem is not None else None,
                        "url": link_elem.text if link_elem is not None else "",
                    })
    except Exception as e:
        logger.warning(f"Ошибка Google RSS для {symbol}: {e}")

    result = _scored_articles(articles[:10])
    if result:
        logger.info("[Google RSS] %s: %s статей, score=%.2f", symbol, result["article_count"], result["score"])
    return result


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

    Помимо итогового score сохраняет точные заголовки, даты и ссылки, вошедшие в оценку.
    Заголовок для карточки берётся из этой же выборки, без отдельного повторного запроса.
    """
    now = time.time()

    # Возвращаем из кэша если данные свежие
    if symbol in _news_cache:
        cached_time, cached_data = _news_cache[symbol]
        if now - cached_time < CACHE_TTL_SECONDS:
            result = dict(cached_data)
            result["cache_hit"] = True
            result["cache_age_seconds"] = round(now - cached_time, 1)
            # Calendar windows are time-sensitive and must not be served stale from the news cache.
            result["shield_active"], result["shield_reason"] = check_economic_calendar_shield(symbol)
            return result

    source_fetchers = []
    if FINNHUB_API_KEY:
        source_fetchers.append(("finnhub", _fetch_finnhub_news))
    if NEWS_API_KEY:
        source_fetchers.append(("newsapi", _fetch_newsapi_news))
    source_fetchers.append(("google_rss", _fetch_rss_news))

    source = "unavailable"
    scored_news: Dict[str, Any] = {}
    for source_name, fetcher in source_fetchers:
        try:
            scored_news = fetcher(symbol) or {}
        except Exception as exc:
            logger.warning("Не удалось получить новости из %s для %s: %s", source_name, symbol, exc)
            scored_news = {}
        if scored_news.get("articles"):
            source = source_name
            break

    score = float(scored_news.get("score", 0.0))
    articles = list(scored_news.get("articles") or [])
    # The displayed headline now comes from the same article set used by sentiment scoring.
    top_headline = str(articles[0].get("headline") or "Рыночный фон стабилен") if articles else "Нет доступных новостных заголовков"

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
    fetched_at_utc = datetime.now(timezone.utc).isoformat()
    now = time.time()

    result = {
        "score":         score,
        "label":         label,
        "top_headline":  top_headline,
        "source":        source,
        "fetched_at_utc": fetched_at_utc,
        "cache_hit":     False,
        "cache_age_seconds": 0.0,
        "article_count": int(scored_news.get("article_count", len(articles))),
        "articles":      articles,
        "news_available": bool(articles),
        "shield_active": shield_active,
        "shield_reason": shield_reason,
    }

    _news_cache[symbol] = (now, result)
    return result
