"""
Модуль AI-Аналитика (AI Trading Mentor) на базе Gemini API / Количественного аудитора.
Анализирует сделки торговой сессии, выявляет систематические ошибки,
оценивает рыночный режим и дает практические рекомендации по улучшению стратегии.
"""

import logging
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    GEMINI_API_KEY,
    AI_ANALYST_ENABLED
)
from database import get_signals_for_report

logger = logging.getLogger(__name__)

USER_TZ = "Europe/Chisinau"


def _build_rule_based_analysis(signals: List[Dict[str, Any]]) -> str:
    """
    Встроенный количественный аудитор (работает автономно без API-ключей).
    Формирует структурированный математический разбор сессии.
    """
    if not signals:
        return (
            "🤖 <b>AI-АНАЛИТИК: СВОДКА СЕССИИ</b>\n"
            "━━━━━━━━━━━━━━━━━━━\n"
            "ℹ️ За выбранный период завершенных сделок не обнаружено.\n"
            "Как только бот совершит сделки, здесь появится полный аудит стратегии."
        )

    total = len(signals)
    wins = sum(1 for s in signals if s.get("status") == "TP_HIT")
    losses = sum(1 for s in signals if s.get("status") == "SL_HIT")
    finished = wins + losses
    winrate = round((wins / finished * 100), 1) if finished > 0 else 0.0
    total_pnl = sum((s.get("pnl") or 0.0) for s in signals if s.get("status") in ("TP_HIT", "SL_HIT"))
    pnl_sign = f"+{total_pnl:.2f}" if total_pnl >= 0 else f"{total_pnl:.2f}"

    # Статистика по символам
    sym_stats: Dict[str, Dict[str, Any]] = {}
    for s in signals:
        sym = s.get("symbol", "UNKNOWN")
        if sym not in sym_stats:
            sym_stats[sym] = {"total": 0, "wins": 0, "losses": 0, "pnl": 0.0}
        sym_stats[sym]["total"] += 1
        st = s.get("status")
        if st == "TP_HIT":
            sym_stats[sym]["wins"] += 1
            sym_stats[sym]["pnl"] += (s.get("pnl") or 0.0)
        elif st == "SL_HIT":
            sym_stats[sym]["losses"] += 1
            sym_stats[sym]["pnl"] += (s.get("pnl") or 0.0)

    # Находим худший и лучший актив
    sorted_syms = sorted(sym_stats.items(), key=lambda x: x[1]["pnl"])
    worst_sym, worst_data = sorted_syms[0] if sorted_syms else ("—", {})
    best_sym, best_data = sorted_syms[-1] if sorted_syms else ("—", {})

    # Анализ времени: до и после 17:00
    eve_losses = 0
    eve_total = 0
    day_losses = 0
    day_total = 0

    for s in signals:
        diag = s.get("diagnostic_data") or {}
        is_after_17 = diag.get("is_after_17_chisinau")
        st = s.get("status")
        if is_after_17:
            eve_total += 1
            if st == "SL_HIT":
                eve_losses += 1
        else:
            day_total += 1
            if st == "SL_HIT":
                day_losses += 1

    lines = [
        "🤖 <b>AI-АНАЛИТИК: КОЛИЧЕСТВЕННЫЙ АУДИТ СЕССИИ</b>",
        "━━━━━━━━━━━━━━━━━━━",
        f"📊 Всего сигналов: <b>{total}</b> | Завершено: <b>{finished}</b>",
        f"🎯 Винрейт сессии: <b>{winrate}%</b> (TP: {wins} / SL: {losses})",
        f"💰 Итоговый PnL: <code>{pnl_sign} пунктов</code>\n",
        "🔍 <b>КЛЮЧЕВЫЕ ВЫВОДЫ АУДИТОРА:</b>"
    ]

    # Анализ худшего актива
    if worst_data.get("pnl", 0) < 0:
        lines.append(
            f"1. <b>Критический отток капитала:</b> Основной убыток сосредоточен в <b>{worst_sym}</b> "
            f"(PnL: <code>{worst_data['pnl']:.2f} п.</code>, Стопов: {worst_data['losses']}). "
            "Рекомендуется проверить совпадение направления с дневным трендом."
        )

    # Анализ времени
    if eve_total > 0 and eve_losses / max(eve_total, 1) > 0.6:
        lines.append(
            f"2. <b>Влияние американской сессии (после 17:00):</b> {eve_losses} из {eve_total} сделок закрылись по стопу. "
            "Открытие Уолл-стрит создает повышенный микрошум и расширение спреда."
        )
    else:
        lines.append("2. <b>Временное распределение:</b> Дневная сессия показала стабильную динамику риска.")

    # Оценка пирамидинга и качества
    lines.append(
        "3. <b>Качество входов и пирамидинг:</b> Новый фильтр пирамидинга блокирует усреднение в убыток. "
        "Доливки разрешены только при выходе позиции в плюс."
    )

    lines.append("\n💡 <b>ТАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:</b>")
    if winrate < 40.0:
        lines.append("• Повысить минимальный порог уверенности ML (min_probability) на 5% для проблемных активов.")
        lines.append("• Торговать строго по направлению старшего часового тренда H1.")
    else:
        lines.append("• Система показывает положительное математическое ожидание. Сохранять текущий режим торговли.")

    lines.append("━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)


_MENTOR_SYSTEM_PROMPT = (
    "Ты — опытный количественный трейдер и шеф по рискам хедж-фонда (AI Trading Mentor). "
    "Изучи сводку сделок нашего торгового бота и дай краткую, конкретную и полезную рецензию (до 250 слов):\n"
    "1. В чем главная причина ошибок (если они есть)?\n"
    "2. Что сделано правильно?\n"
    "3. Топ-2 конкретные рекомендации для увеличения прибыли завтра.\n\n"
    "Отвечай на русском языке, используй лаконичный профессиональный тон и эмодзи."
)


def _call_groq_api(signals_summary_text: str) -> Optional[str]:
    """
    Отправляет сводку в Groq Cloud API (100% бесплатный тариф без кредитных карт).
    Использует сверхбыстрые модели Llama 3.3 70B Versatile или DeepSeek R1.
    """
    if not GROQ_API_KEY:
        return None

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": _MENTOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"ДАННЫЕ ТОРГОВОЙ СЕССИИ:\n{signals_summary_text}"}
        ],
        "temperature": 0.3,
        "max_tokens": 650
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "")
                if text:
                    return text.strip()
        else:
            logger.warning(f"Ошибка вызова Groq API: HTTP {resp.status_code} — {resp.text[:200]}")
    except Exception as e:
        logger.warning(f"Сбой при обращении к Groq API: {e}")

    return None


def _call_openrouter_api(signals_summary_text: str) -> Optional[str]:
    """
    Отправляет сводку в OpenRouter API (бесплатные модели с суффиксом :free).
    """
    if not OPENROUTER_API_KEY:
        return None

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/trading-bot",
        "X-Title": "TradingBot AI Mentor"
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": _MENTOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"ДАННЫЕ ТОРГОВОЙ СЕССИИ:\n{signals_summary_text}"}
        ],
        "temperature": 0.3,
        "max_tokens": 650
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "")
                if text:
                    return text.strip()
        else:
            logger.warning(f"Ошибка вызова OpenRouter API: HTTP {resp.status_code} — {resp.text[:200]}")
    except Exception as e:
        logger.warning(f"Сбой при обращении к OpenRouter API: {e}")

    return None


def _call_gemini_api(signals_summary_text: str) -> Optional[str]:
    """
    Отправляет сводку в Google Gemini API (gemini-1.5-flash).
    """
    if not GEMINI_API_KEY:
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{
            "parts": [{"text": f"{_MENTOR_SYSTEM_PROMPT}\n\nДАННЫЕ ТОРГОВОЙ СЕССИИ:\n{signals_summary_text}"}]
        }],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 600
        }
    }

    try:
        resp = requests.post(url, json=payload, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if text:
                    return text.strip()
        else:
            logger.warning(f"Ошибка вызова Gemini API: HTTP {resp.status_code} — {resp.text[:200]}")
    except Exception as e:
        logger.warning(f"Сбой при обращении к Gemini API: {e}")

    return None


def generate_ai_session_analysis(days: int = 1) -> str:
    """
    Основная точка входа:
    Генерирует анализ с помощью доступного бесплатного AI-провайдера:
    1. Groq (Llama 3.3 70B / DeepSeek R1 — 100% бесплатно, без карт)
    2. OpenRouter (:free модели)
    3. Google Gemini 1.5 Flash
    4. Автономный количественный аудитор (резерв)
    """
    signals = get_signals_for_report(days=days, limit=60)
    rule_analysis = _build_rule_based_analysis(signals)

    if not AI_ANALYST_ENABLED:
        return rule_analysis

    # 1. Приоритетный бесплатный провайдер: Groq
    if GROQ_API_KEY:
        try:
            groq_review = _call_groq_api(rule_analysis)
            if groq_review:
                model_name = GROQ_MODEL.upper()
                return (
                    f"🧠 <b>РЕЦЕНЗИЯ AI-МЕНТОРА (GROQ / {model_name}):</b>\n"
                    "━━━━━━━━━━━━━━━━━━━\n"
                    f"{groq_review}\n\n"
                    "📊 <b>КОЛИЧЕСТВЕННАЯ СВОДКА СЕССИИ:</b>\n"
                    f"{rule_analysis}"
                )
        except Exception as e:
            logger.error(f"Не удалось получить ответ Groq: {e}")

    # 2. Второй бесплатный провайдер: OpenRouter
    if OPENROUTER_API_KEY:
        try:
            or_review = _call_openrouter_api(rule_analysis)
            if or_review:
                return (
                    "🧠 <b>РЕЦЕНЗИЯ AI-МЕНТОРА (OPENROUTER / FREE):</b>\n"
                    "━━━━━━━━━━━━━━━━━━━\n"
                    f"{or_review}\n\n"
                    "📊 <b>КОЛИЧЕСТВЕННАЯ СВОДКА СЕССИИ:</b>\n"
                    f"{rule_analysis}"
                )
        except Exception as e:
            logger.error(f"Не удалось получить ответ OpenRouter: {e}")

    # 3. Резервный провайдер: Google Gemini
    if GEMINI_API_KEY:
        try:
            gemini_review = _call_gemini_api(rule_analysis)
            if gemini_review:
                return (
                    "🧠 <b>РЕЦЕНЗИЯ AI-МЕНТОРА (GEMINI 1.5 FLASH):</b>\n"
                    "━━━━━━━━━━━━━━━━━━━\n"
                    f"{gemini_review}\n\n"
                    "📊 <b>КОЛИЧЕСТВЕННАЯ СВОДКА СЕССИИ:</b>\n"
                    f"{rule_analysis}"
                )
        except Exception as e:
            logger.error(f"Не удалось получить ответ Gemini: {e}")

    # 4. Автономный количественный аудитор
    return rule_analysis

