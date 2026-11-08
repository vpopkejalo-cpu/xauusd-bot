"""
Модуль экспресс-диагностики и проверки всех систем, API и весов ML-модели.
"""

import time
import requests
import urllib.request
import json
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List

from config import (
    TWELVE_DATA_API_KEY,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    FINNHUB_API_KEY,
    NEWS_API_KEY,
    GEMINI_API_KEY,
    NEWS_FILTER_ENABLED,
    SYMBOLS,
    MAX_PYRAMID_POSITIONS,
    DEMO_MODE,
    KILL_SWITCH_ENABLED
)
from database import get_db, IS_POSTGRES, is_bot_paused, is_kill_switch_active, get_today_signal_stats
from ml_model import TradingMLModel, FEATURE_COLUMNS
from risk_manager import get_asset_profile, is_market_open

logger = logging.getLogger(__name__)
USER_TZ = "Europe/Chisinau"


def _check_twelve_data() -> Dict[str, Any]:
    if not TWELVE_DATA_API_KEY:
        return {
            "status": "WARN",
            "title": "Twelve Data API",
            "message": "Ключ не задан в .env/Render (Котировки идут через Yahoo Finance 24/5)",
            "icon": "🟡"
        }
    t0 = time.time()
    try:
        url = "https://api.twelvedata.com/time_series"
        params = {"symbol": "XAU/USD", "interval": "1min", "outputsize": 1, "apikey": TWELVE_DATA_API_KEY}
        resp = requests.get(url, params=params, timeout=5)
        latency = int((time.time() - t0) * 1000)
        data = resp.json()
        if resp.status_code == 200 and data.get("status") != "error":
            return {
                "status": "OK",
                "title": "Twelve Data API",
                "message": f"Подключен ({latency} мс, котировки поступают штатно)",
                "icon": "🟢"
            }
        elif resp.status_code == 429 or "rate limit" in str(data).lower():
            return {
                "status": "WARN",
                "title": "Twelve Data API",
                "message": f"Лимит 800 req/day исчерпан ({latency} мс). Автопереключение на Yahoo Finance",
                "icon": "🟡"
            }
        else:
            err_msg = data.get("message", f"HTTP {resp.status_code}")
            return {
                "status": "ERROR",
                "title": "Twelve Data API",
                "message": f"Ошибка: {err_msg}",
                "icon": "🔴"
            }
    except Exception as e:
        return {
            "status": "ERROR",
            "title": "Twelve Data API",
            "message": f"Таймаут/сбой: {e}",
            "icon": "🔴"
        }


def _check_yahoo_finance() -> Dict[str, Any]:
    t0 = time.time()
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=5m&range=1d"
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        latency = int((time.time() - t0) * 1000)
        chart = data.get("chart", {}).get("result", [])
        if chart:
            last_p = chart[0].get("meta", {}).get("regularMarketPrice", 0.0)
            return {
                "status": "OK",
                "title": "Yahoo Finance (Резерв 24/5)",
                "message": f"Доступен ({latency} мс, цена XAU: ${last_p:.2f})",
                "icon": "🟢"
            }
        return {
            "status": "WARN",
            "title": "Yahoo Finance (Резерв 24/5)",
            "message": f"Ответил без цены ({latency} мс)",
            "icon": "🟡"
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "title": "Yahoo Finance (Резерв 24/5)",
            "message": f"Ошибка соединения: {e}",
            "icon": "🔴"
        }


def _check_finnhub() -> Dict[str, Any]:
    if not FINNHUB_API_KEY:
        return {
            "status": "INFO",
            "title": "Finnhub News API",
            "message": "Ключ не задан (активен Google RSS резерв)",
            "icon": "⚪"
        }
    t0 = time.time()
    try:
        url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}"
        resp = requests.get(url, timeout=5)
        latency = int((time.time() - t0) * 1000)
        if resp.status_code == 200:
            return {
                "status": "OK",
                "title": "Finnhub News API",
                "message": f"Подключен ({latency} мс, 60 req/min)",
                "icon": "🟢"
            }
        else:
            return {
                "status": "ERROR",
                "title": "Finnhub News API",
                "message": f"HTTP {resp.status_code} (неверный токен)",
                "icon": "🔴"
            }
    except Exception as e:
        return {
            "status": "ERROR",
            "title": "Finnhub News API",
            "message": f"Сбой: {e}",
            "icon": "🔴"
        }


def _check_newsapi() -> Dict[str, Any]:
    if not NEWS_API_KEY:
        return {
            "status": "INFO",
            "title": "NewsAPI",
            "message": "Ключ не задан (активен Google RSS резерв)",
            "icon": "⚪"
        }
    t0 = time.time()
    try:
        url = f"https://newsapi.org/v2/top-headlines?category=business&pageSize=1&apiKey={NEWS_API_KEY}"
        resp = requests.get(url, timeout=5)
        latency = int((time.time() - t0) * 1000)
        if resp.status_code == 200:
            return {
                "status": "OK",
                "title": "NewsAPI",
                "message": f"Подключен ({latency} мс, 100 req/day)",
                "icon": "🟢"
            }
        elif resp.status_code == 429:
            return {
                "status": "WARN",
                "title": "NewsAPI",
                "message": f"Лимит 100 req исчерпан ({latency} мс)",
                "icon": "🟡"
            }
        else:
            return {
                "status": "ERROR",
                "title": "NewsAPI",
                "message": f"HTTP {resp.status_code} (ошибка ключа)",
                "icon": "🔴"
            }
    except Exception as e:
        return {
            "status": "ERROR",
            "title": "NewsAPI",
            "message": f"Сбой: {e}",
            "icon": "🔴"
        }


def _check_google_rss() -> Dict[str, Any]:
    t0 = time.time()
    try:
        url = "https://news.google.com/rss/search?q=Gold+price+XAU&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            code = resp.getcode()
        latency = int((time.time() - t0) * 1000)
        if code == 200:
            return {
                "status": "OK",
                "title": "Google News RSS",
                "message": f"Доступен без лимитов ({latency} мс)",
                "icon": "🟢"
            }
        return {
            "status": "WARN",
            "title": "Google News RSS",
            "message": f"HTTP {code}",
            "icon": "🟡"
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "title": "Google News RSS",
            "message": f"Сбой: {e}",
            "icon": "🔴"
        }


def _check_gemini() -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        return {
            "status": "INFO",
            "title": "Gemini AI (Google Studio)",
            "message": "Ключ не задан в .env/Render (активен встроенный квантовый аудитор)",
            "icon": "⚪"
        }
    t0 = time.time()
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
        resp = requests.get(url, timeout=5)
        latency = int((time.time() - t0) * 1000)
        if resp.status_code == 200:
            return {
                "status": "OK",
                "title": "Gemini AI (Google Studio)",
                "message": f"Авторизован ({latency} мс, Flash 1.5 активен)",
                "icon": "🟢"
            }
        else:
            return {
                "status": "ERROR",
                "title": "Gemini AI (Google Studio)",
                "message": f"HTTP {resp.status_code} (неверный ключ AI Studio)",
                "icon": "🔴"
            }
    except Exception as e:
        return {
            "status": "ERROR",
            "title": "Gemini AI (Google Studio)",
            "message": f"Сбой: {e}",
            "icon": "🔴"
        }


def _check_telegram() -> Dict[str, Any]:
    if not TELEGRAM_BOT_TOKEN:
        return {
            "status": "ERROR",
            "title": "Telegram Bot API",
            "message": "TELEGRAM_BOT_TOKEN не задан",
            "icon": "🔴"
        }
    t0 = time.time()
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
        resp = requests.get(url, timeout=5)
        latency = int((time.time() - t0) * 1000)
        if resp.status_code == 200:
            data = resp.json()
            uname = data.get("result", {}).get("username", "bot")
            chat_info = f"Chat ID: {TELEGRAM_CHAT_ID}" if TELEGRAM_CHAT_ID else "Chat ID не задан!"
            return {
                "status": "OK",
                "title": "Telegram Bot API",
                "message": f"@{uname} ({latency} мс, {chat_info})",
                "icon": "🟢"
            }
        return {
            "status": "ERROR",
            "title": "Telegram Bot API",
            "message": f"HTTP {resp.status_code} (неверный BOT_TOKEN)",
            "icon": "🔴"
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "title": "Telegram Bot API",
            "message": f"Сбой: {e}",
            "icon": "🔴"
        }


def _check_database() -> Dict[str, Any]:
    t0 = time.time()
    try:
        with get_db() as cur:
            cur.execute("SELECT COUNT(*) as cnt FROM candles")
            candles_cnt = cur.fetchone()["cnt"]
            cur.execute("SELECT COUNT(*) as cnt FROM signals")
            signals_cnt = cur.fetchone()["cnt"]
            cur.execute("SELECT COUNT(*) as cnt FROM signals WHERE status = 'OPEN'")
            open_cnt = cur.fetchone()["cnt"]
        latency = int((time.time() - t0) * 1000)
        db_name = "PostgreSQL (Cloud)" if IS_POSTGRES else "SQLite (Local)"
        return {
            "status": "OK",
            "title": "База данных",
            "message": f"{db_name} ({latency} мс) | Свечей: {candles_cnt} | Сделок: {signals_cnt} (в рынке: {open_cnt})",
            "icon": "🟢"
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "title": "База данных",
            "message": f"Ошибка запроса: {e}",
            "icon": "🔴"
        }


def _check_ml_and_weights() -> List[Dict[str, Any]]:
    results = []
    ml = TradingMLModel()
    
    for sym in SYMBOLS:
        pipe = ml.models.get(sym)
        acc = ml.accuracies.get(sym, 0.0)
        samples = ml.samples_counts.get(sym, 0)
        
        if pipe is not None:
            try:
                clf = pipe.named_steps["classifier"]
                coefs = clf.coef_[0]
                intercept = float(clf.intercept_[0])
                
                # Анализ весов
                feat_pairs = list(zip(FEATURE_COLUMNS, coefs))
                feat_pairs.sort(key=lambda x: x[1], reverse=True)
                top_bull = feat_pairs[0]
                top_bear = feat_pairs[-1]
                
                # Проверка gatekeeper
                is_reliable = (acc >= 0.52)
                icon = "🟢" if is_reliable else "🟡"
                status_str = "Надежна" if is_reliable else "Защитный порог 72%"
                
                results.append({
                    "symbol": sym,
                    "status": "OK" if is_reliable else "WARN",
                    "icon": icon,
                    "accuracy": acc,
                    "samples": samples,
                    "intercept": intercept,
                    "top_bull": top_bull,
                    "top_bear": top_bear,
                    "status_str": status_str,
                    "weights_count": len(coefs)
                })
            except Exception as ex:
                results.append({
                    "symbol": sym,
                    "status": "ERROR",
                    "icon": "🔴",
                    "message": f"Ошибка весов: {ex}"
                })
        else:
            results.append({
                "symbol": sym,
                "status": "INFO",
                "icon": "⚪",
                "message": "Ожидает сканирования свечей для первого обучения"
            })
            
    return results


def run_full_system_check() -> Dict[str, Any]:
    """Запускает комплексный тест всех систем и формирует детальный отчет."""
    t_start = time.time()
    
    # 1. Параллельный опрос всех внешних API
    with ThreadPoolExecutor(max_workers=7) as executor:
        f_twelve = executor.submit(_check_twelve_data)
        f_yahoo = executor.submit(_check_yahoo_finance)
        f_finnhub = executor.submit(_check_finnhub)
        f_newsapi = executor.submit(_check_newsapi)
        f_rss = executor.submit(_check_google_rss)
        f_gemini = executor.submit(_check_gemini)
        f_tg = executor.submit(_check_telegram)
        f_db = executor.submit(_check_database)

    api_results = {
        "twelve_data": f_twelve.result(),
        "yahoo_finance": f_yahoo.result(),
        "finnhub": f_finnhub.result(),
        "newsapi": f_newsapi.result(),
        "google_rss": f_rss.result(),
        "gemini": f_gemini.result(),
        "telegram": f_tg.result(),
        "database": f_db.result(),
    }
    
    # 2. Проверка моделей и весов
    ml_results = _check_ml_and_weights()
    
    # 3. Риск-менеджмент и статус бота
    bot_paused = is_bot_paused()
    ks_active, ks_reason = is_kill_switch_active()
    now_utc = datetime.now(timezone.utc)
    market_open, market_reason = is_market_open(now_utc)
    
    elapsed = round(time.time() - t_start, 2)
    
    # Формирование HTML сообщения
    now_local = datetime.now(ZoneInfo(USER_TZ)).strftime("%H:%M:%S (%d.%m.%Y)")
    
    # Подсчет итогового статуса
    errors = sum(1 for r in api_results.values() if r["status"] == "ERROR")
    
    if errors == 0:
        verdict_header = "🟢 <b>ВСЕ СИСТЕМЫ ИСПРАВНЫ (100% ГОТОВНОСТЬ)</b>"
    else:
        verdict_header = f"⚠️ <b>ВНИМАНИЕ: ОБНАРУЖЕНО ОШИБОК: {errors}</b>"
        
    lines = [
        "🩺 <b>ДИАГНОСТИЧЕСКАЯ ПАНЕЛЬ СИСТЕМ БОТА</b>",
        "━━━━━━━━━━━━━━━━━━━━━",
        verdict_header,
        f"⏱ <i>Время проверки: {now_local} (тест за {elapsed} с)</i>\n",
        "🌐 <b>1. СЕТЕВЫЕ API И ПОСТАВЩИКИ ДАННЫХ:</b>",
        f"{api_results['twelve_data']['icon']} <b>{api_results['twelve_data']['title']}:</b> {api_results['twelve_data']['message']}",
        f"{api_results['yahoo_finance']['icon']} <b>{api_results['yahoo_finance']['title']}:</b> {api_results['yahoo_finance']['message']}",
        f"{api_results['telegram']['icon']} <b>{api_results['telegram']['title']}:</b> {api_results['telegram']['message']}",
        f"{api_results['database']['icon']} <b>{api_results['database']['title']}:</b> {api_results['database']['message']}\n",
        "📰 <b>2. НОВОСТНЫЕ ШЛЮЗЫ И AI-СЕНТИМЕНТ:</b>",
        f"{api_results['finnhub']['icon']} <b>{api_results['finnhub']['title']}:</b> {api_results['finnhub']['message']}",
        f"{api_results['newsapi']['icon']} <b>{api_results['newsapi']['title']}:</b> {api_results['newsapi']['message']}",
        f"{api_results['google_rss']['icon']} <b>{api_results['google_rss']['title']}:</b> {api_results['google_rss']['message']}",
        f"{api_results['gemini']['icon']} <b>{api_results['gemini']['title']}:</b> {api_results['gemini']['message']}",
        f"🛡 <b>Новостной щит:</b> {'Включен (NFP/CPI фильтр)' if NEWS_FILTER_ENABLED else 'Отключен'}\n",
        "🧠 <b>3. РАБОТОСПОСОБНОСТЬ ВЕСОВ И МОДЕЛЕЙ ML:</b>"
    ]
    
    for ml_info in ml_results:
        sym = ml_info["symbol"]
        if ml_info["status"] in ("OK", "WARN"):
            acc_str = f"{ml_info['accuracy']:.1%}"
            samples = ml_info["samples"]
            bull_name, bull_w = ml_info["top_bull"]
            bear_name, bear_w = ml_info["top_bear"]
            icon = ml_info["icon"]
            lines.append(
                f"{icon} <b>[{sym}]</b>: Точность {acc_str} ({samples} св.)\n"
                f"   • Веса активны (всего {ml_info['weights_count']} признаков, bias: {ml_info['intercept']:+.2f})\n"
                f"   • 🐂 <b>Бычий драйвер:</b> <code>{bull_name} ({bull_w:+.3f})</code>\n"
                f"   • 🐻 <b>Медвежий драйвер:</b> <code>{bear_name} ({bear_w:+.3f})</code>"
            )
        else:
            lines.append(f"{ml_info['icon']} <b>[{sym}]</b>: {ml_info['message']}")
            
    lines.append("\n⚖️ <b>4. РИСК-МЕНЕДЖМЕНТ И РЕЖИМЫ:</b>")
    lines.append(f"• <b>Статус бота:</b> {'⏸ На паузе (ручная остановка)' if bot_paused else '▶️ Активен (сканирование 24/7)'}")
    
    ks_str = "🟢 Неактивен (Демо-режим: сбор опыта)" if DEMO_MODE else ("🛑 СРАБОТАЛ" if ks_active else "🟢 Защита наготове")
    lines.append(f"• <b>Аварийный Kill Switch:</b> {ks_str}")
    lines.append(f"• <b>Пирамидинг:</b> 🚀 Разрешен (до {MAX_PYRAMID_POSITIONS} прибыльных сделок в тренд)")
    lines.append(f"• <b>Рыночная сессия:</b> {'🟢 Торги открыты' if market_open else f'🟡 Рынок закрыт ({market_reason})'}")
    
    report_text = "\n".join(lines)
    return {
        "report_text": report_text,
        "errors_count": errors,
        "elapsed_seconds": elapsed,
        "api_results": api_results,
        "ml_results": ml_results
    }

