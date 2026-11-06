"""
Модуль формирования глубоких диагностических отчетов по сделкам торгового бота.
Генерирует подробнейший текстовый анализ (.txt) и табличные данные (.csv):
- Влияние весов ML модели на каждое решение (Feature Attribution)
- Разбор сделок по сессиям (до 17:00 vs после 17:00 Кишинёв / сессия США)
- Новостной фон, экономический календарь и макро-сантимент
- Как исходы сделок меняют датасет Triple Barrier и веса при переобучении
"""

import csv
import io
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from zoneinfo import ZoneInfo

from database import get_signals_for_report

logger = logging.getLogger(__name__)

USER_TZ = "Europe/Chisinau"


def _format_local_time(iso_str: Optional[str]) -> Tuple[str, Optional[int]]:
    """
    Преобразует ISO строку UTC в строку местного времени Кишинёва и возвращает (время_строка, час).
    """
    if not iso_str:
        return "—", None
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local_dt = dt.astimezone(ZoneInfo(USER_TZ))
        return local_dt.strftime("%Y-%m-%d %H:%M:%S"), local_dt.hour
    except Exception:
        return str(iso_str)[:19], None


def generate_diagnostics_report(days: int = 1, limit: int = 100) -> Dict[str, Any]:
    """
    Генерирует исчерпывающий диагностический пакет:
    1. txt_content: Подробный текстовый аналитический отчет с таблицами весов и выводами.
    2. csv_content: Табличный файл с количественными метриками по всем сделкам.
    3. txt_filename, csv_filename: Имена файлов с текущей датой.
    4. summary_stats: Краткая сводка для сообщения в Telegram.
    """
    now_utc = datetime.now(timezone.utc)
    local_now = now_utc.astimezone(ZoneInfo(USER_TZ))
    date_str = local_now.strftime("%Y-%m-%d")
    gen_time_str = local_now.strftime("%Y-%m-%d %H:%M:%S")

    signals = get_signals_for_report(days=days, limit=limit)

    # Статистика сессии
    total_count = len(signals)
    wins = sum(1 for s in signals if s.get("status") == "TP_HIT")
    losses = sum(1 for s in signals if s.get("status") == "SL_HIT")
    open_count = sum(1 for s in signals if s.get("status") == "OPEN")
    manual_count = sum(1 for s in signals if s.get("status") == "MANUAL_CLOSE")
    skipped_count = sum(1 for s in signals if s.get("status") == "SKIPPED")
    finished_count = wins + losses
    winrate = round((wins / finished_count * 100), 1) if finished_count > 0 else 0.0
    total_pnl = sum((s.get("pnl") or 0.0) for s in signals if s.get("status") in ("TP_HIT", "SL_HIT"))

    # Сравнение: До 17:00 (Европа/утро) vs После 17:00 (Сессия США / вечер)
    day_sigs = []
    evening_sigs = []

    for s in signals:
        created_at = s.get("created_at")
        _, hour = _format_local_time(created_at)
        # Проверяем флаг в диагностике или час
        diag = s.get("diagnostic_data") or {}
        is_after_17 = diag.get("is_after_17_chisinau")
        if is_after_17 is None:
            is_after_17 = (hour is not None and hour >= 17)

        if is_after_17:
            evening_sigs.append(s)
        else:
            day_sigs.append(s)

    def calc_sub_stats(sub_list: List[Dict]):
        w = sum(1 for s in sub_list if s.get("status") == "TP_HIT")
        l = sum(1 for s in sub_list if s.get("status") == "SL_HIT")
        fin = w + l
        wr = round((w / fin * 100), 1) if fin > 0 else 0.0
        pnl = sum((s.get("pnl") or 0.0) for s in sub_list if s.get("status") in ("TP_HIT", "SL_HIT"))
        return len(sub_list), w, l, wr, pnl

    day_total, day_w, day_l, day_wr, day_pnl = calc_sub_stats(day_sigs)
    eve_total, eve_w, eve_l, eve_wr, eve_pnl = calc_sub_stats(evening_sigs)

    # -------------------------------------------------------------
    # 1. ФОРМИРОВАНИЕ ТЕКСТОВОГО ОТЧЕТА (.TXT)
    # -------------------------------------------------------------
    lines = []
    lines.append("=" * 80)
    lines.append("📊 ГЛУБОКИЙ ДИАГНОСТИЧЕСКИЙ ОТЧЕТ ПО ТОРГОВЫМ СИГНАЛАМ И ВЕСАМ ML МОДЕЛИ")
    lines.append(f"Время формирования: {gen_time_str} (Кишинёв / {USER_TZ}) | UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Всего сигналов в выборке: {total_count}")
    lines.append("=" * 80)
    lines.append("")

    # Раздел 1: Сводная статистика
    lines.append("1. ОБЩИЕ ИТОГИ СЕССИИ:")
    lines.append("-" * 80)
    pnl_sign = f"+{total_pnl:.2f}" if total_pnl >= 0 else f"{total_pnl:.2f}"
    lines.append(f"• Всего сигналов:        {total_count}")
    lines.append(f"• Закрыто по Take Profit: {wins} (TP)")
    lines.append(f"• Закрыто по Stop Loss:   {losses} (SL)")
    lines.append(f"• Активно в рынке (OPEN): {open_count}")
    lines.append(f"• Сброшено вручную:       {manual_count}")
    lines.append(f"• Пропущено (SKIP):       {skipped_count}")
    lines.append(f"• Общий Винрейт:          {winrate}% (по {finished_count} завершенным сделкам)")
    lines.append(f"• Суммарный PnL:          {pnl_sign} пунктов")
    lines.append("")

    # Раздел 2: Анализ гипотезы рынка США после 17:00
    lines.append("-" * 80)
    lines.append("🚨 АНАЛИЗ ГИПОТЕЗЫ ВЛИЯНИЯ РЫНКА США (СРАВНЕНИЕ ДО И ПОСЛЕ 17:00 КИШИНЁВА):")
    lines.append("-" * 80)
    day_pnl_str = f"+{day_pnl:.2f}" if day_pnl >= 0 else f"{day_pnl:.2f}"
    eve_pnl_str = f"+{eve_pnl:.2f}" if eve_pnl >= 0 else f"{eve_pnl:.2f}"

    lines.append(f"☀️ ДНЕВНОЙ ПЕРИОД (до 17:00 Кишинёв — Лондонская сессия и утро):")
    lines.append(f"   Сигналов: {day_total} | TP: {day_w} | SL: {day_l} | Винрейт: {day_wr}% | PnL: {day_pnl_str} п.")
    lines.append("")
    lines.append(f"🌙 ВЕЧЕРНИЙ ПЕРИОД (после 17:00 Кишинёв — Сессия США & Уолл-стрит):")
    lines.append(f"   Сигналов: {eve_total} | TP: {eve_w} | SL: {eve_l} | Винрейт: {eve_wr}% | PnL: {eve_pnl_str} п.")
    lines.append("")

    if eve_total > 0 and day_total > 0:
        wr_diff = round(eve_wr - day_wr, 1)
        if wr_diff < -15.0:
            lines.append(f"⚠️ КРИТИЧЕСКОЕ НАБЛЮДЕНИЕ: После 17:00 винрейт падает на {abs(wr_diff)}%!")
            lines.append("   Причина: На американской сессии происходят резкие импульсы крупных фондов,")
            lines.append("   сбор стоп-приказов (stop-hunting) и расширение спредов, ломающее классические")
            lines.append("   трендовые веера EMA и индикаторы моментума.")
        elif wr_diff > 15.0:
            lines.append(f"💡 НАБЛЮДЕНИЕ: Вечерняя сессия США показала винрейт выше дневного на {wr_diff}%.")
        else:
            lines.append("ℹ️ Разница между дневной и вечерней сессией незначительна на текущей выборке.")
    lines.append("")

    # Раздел 3: Детальный разбор каждой сделки
    lines.append("=" * 80)
    lines.append("2. ДЕТАЛЬНЫЙ РАЗБОР КАЖДОЙ СДЕЛКИ (DEEP-DIVE):")
    lines.append("=" * 80)

    if not signals:
        lines.append("\nЗа выбранный период сигналы в базе данных отсутствуют.")
        lines.append("Как только бот сформирует первую сделку, здесь появится полный лог весов.")
    else:
        for idx, s in enumerate(signals, 1):
            s_id = s.get("id", idx)
            sym = s.get("symbol", "UNKNOWN")
            direction = s.get("direction", "HOLD")
            status = s.get("status", "OPEN")
            pnl_val = s.get("pnl") or 0.0
            pnl_str = f"+{pnl_val:.2f}" if pnl_val >= 0 else f"{pnl_val:.2f}"

            created_local, c_hour = _format_local_time(s.get("created_at"))
            entered_local, _ = _format_local_time(s.get("entered_at"))
            closed_local, _ = _format_local_time(s.get("closed_at"))

            diag = s.get("diagnostic_data") or {}
            is_after_17 = diag.get("is_after_17_chisinau")
            if is_after_17 is None:
                is_after_17 = (c_hour is not None and c_hour >= 17)

            session_tag = "🇺🇸 ВЕЧЕРНЯЯ СЕССИЯ США (ПОСЛЕ 17:00)" if is_after_17 else "☀️ ДНЕВНАЯ СЕССИЯ (ДО 17:00)"

            lines.append("")
            lines.append("-" * 80)
            lines.append(f"🎯 СДЕЛКА #{s_id}: [{sym}] Направление: {direction} | Статус: {status} | PnL: {pnl_str}")
            lines.append(f"   {session_tag}")
            lines.append("-" * 80)

            # Временные метки
            lines.append(f"⏱ ВРЕМЕННАЯ ШКАЛА (Кишинёв):")
            lines.append(f"   • Сигнал сформирован:   {created_local} (UTC: {s.get('created_at', '')[:19]})")
            lines.append(f"   • Нажатие 'Вошел':      {entered_local}")
            lines.append(f"   • Закрытие позиции:     {closed_local}")

            # Цены и риск
            entry_p = s.get("entry_price", 0.0)
            tp_p = s.get("take_profit", 0.0)
            sl_p = s.get("stop_loss", 0.0)
            exit_p = s.get("exit_price")
            atr_v = s.get("atr", 0.0)
            prob_v = s.get("probability", 0.0)

            lines.append(f"💰 ПАРАМЕТРЫ СДЕЛКИ И РИСКА:")
            lines.append(f"   • Вход: {entry_p} | TP: {tp_p} | SL: {sl_p} | Выход: {exit_p or '—'}")
            lines.append(f"   • Волатильность (ATR): {atr_v:.4f} | Вероятность ML: {prob_v:.1%}")

            risk_info = diag.get("risk", {})
            if risk_info:
                rr_val = risk_info.get("risk_reward_ratio", "1:1.33")
                rr_str = str(rr_val)
                if not rr_str.startswith("1:"):
                    rr_str = f"1:{rr_str}"
                lines.append(f"   • Соотношение R:R: {rr_str} | Реком. лот: {risk_info.get('recommended_lot', 0.01)}")

            # Рыночный контекст и индикаторы
            m_ctx = diag.get("market_context") or diag.get("indicators") or {}
            if m_ctx:
                lines.append(f"📈 РЫНОЧНЫЙ КОНТЕКСТ В МОМЕНТ ВХОДА:")
                lines.append(f"   • RSI(14): {m_ctx.get('rsi', '—')} | ADX(14): {m_ctx.get('adx', '—')} | +DI: {m_ctx.get('plus_di', '—')} | -DI: {m_ctx.get('minus_di', '—')}")
                lines.append(f"   • EMA 21: {m_ctx.get('ema_21', '—')} | EMA 50: {m_ctx.get('ema_50', '—')} | EMA 200: {m_ctx.get('ema_200', '—')}")
                lines.append(f"   • Дистанция до EMA 200: {m_ctx.get('dist_ema_200', '—')} | MACD Hist: {m_ctx.get('macd_hist', '—')}")

            # Новостной контекст
            news_info = diag.get("news") or {}
            if news_info:
                shield = "🛡 АКТИВЕН" if news_info.get("shield_active") else "✅ Отключен (спокойный фон)"
                lines.append(f"📰 НОВОСТНОЙ ФОН:")
                lines.append(f"   • Новостной щит: {shield}")
                lines.append(f"   • Сантимент: {news_info.get('label', 'Neutral')} (оценка: {news_info.get('score', 0.0):+.2f})")
                if news_info.get("top_headline"):
                    lines.append(f"   • Заголовок: \"{news_info.get('top_headline')}\" [источник: {news_info.get('source', 'news')}]")

            # Веса ML модели (Feature Attribution)
            ml_pred = diag.get("ml_prediction") or {}
            feat_list = ml_pred.get("features", [])

            if feat_list:
                lines.append(f"🧠 ДЕТАЛИЗАЦИЯ ВЕСОВ И ВКЛАДА ПРИЗНАКОВ В РЕШЕНИЕ:")
                lines.append(f"   • CV Точность модели: {ml_pred.get('model_accuracy', 0.0):.1%} | Обучено на {ml_pred.get('samples_count', 0)} свечах")
                lines.append(f"   • Свободный член (Intercept): {ml_pred.get('intercept', 0.0):+.4f}")
                lines.append(f"   • P(BUY): {ml_pred.get('buy_probability', 0.0):.1%} | P(SELL): {ml_pred.get('sell_probability', 0.0):.1%} | Порог: {ml_pred.get('min_probability', 0.0):.1%}")
                lines.append("")
                lines.append("   ┌──────────────────┬──────────┬──────────┬──────────┬──────────┬────────────────────────┐")
                lines.append("   │ Признак          │ Значение │ Scaled   │ Вес (W)  │ Вклад(W*S│ Влияние на решение     │")
                lines.append("   ├──────────────────┼──────────┼──────────┼──────────┼──────────┼────────────────────────┤")

                for f in feat_list:
                    name_s = f"{f.get('name', ''):<16}"[:16]
                    raw_s = f"{f.get('raw_value', 0.0):>8.2f}"
                    scale_s = f"{f.get('scaled_value', 0.0):>+8.2f}"
                    w_s = f"{f.get('weight', 0.0):>+8.3f}"
                    c_s = f"{f.get('contribution', 0.0):>+8.3f}"
                    bias = f.get("bias", "—")
                    bias_text = f"🟢 Push к BUY (+{c_s.strip()})" if bias == "BUY" else f"🔴 Push к SELL ({c_s.strip()})"
                    lines.append(f"   │ {name_s} │ {raw_s} │ {scale_s} │ {w_s} │ {c_s} │ {bias_text:<22} │")

                lines.append("   └──────────────────┴──────────┴──────────┴──────────┴──────────┴────────────────────────┘")

                top_b = ", ".join(ml_pred.get("top_bullish", [])) or "нет"
                top_s = ", ".join(ml_pred.get("top_bearish", [])) or "нет"
                driver = ml_pred.get("primary_driver", "—")
                lines.append(f"   🎯 Главный драйвер: {driver}")
                lines.append(f"   🟢 Топ-3 за BUY: {top_b}")
                lines.append(f"   🔴 Топ-3 за SELL: {top_s}")
            else:
                lines.append("   ℹ️ Детальные коэффициенты весов будут доступны для новых сделок, сформированных ботом.")

            # Динамика весов при переобучении (Learning dynamics)
            lines.append("")
            lines.append("🔄 ВЛИЯНИЕ ИСХОДА СДЕЛКИ НА ПЕРЕОБУЧЕНИЕ МОДЕЛИ (WEIGHT DYNAMICS):")
            if status == "SL_HIT":
                lines.append("   • Сделка завершилась убытком (SL).")
                lines.append("   • В следующем цикле переобучения метод Triple Barrier присвоит этой свече штрафную метку.")
                lines.append("   • Градиентный спуск снизит веса тех индикаторов, которые ошибочно подтолкнули к входу,")
                if is_after_17:
                    lines.append("     а также повысит требования к входу во время американской сессии после 17:00.")
                else:
                    lines.append("     скорректировав порог уверенности для аналогичной рыночной структуры.")
            elif status == "TP_HIT":
                lines.append("   • Сделка закрылась с прибылью (TP).")
                lines.append("   • Разметка Triple Barrier подтверждает паттерн. Веса признаков-драйверов укрепляются.")
            elif status == "OPEN":
                lines.append("   • Сделка еще в рынке. Влияние на веса определится в момент фиксации TP или SL.")
            else:
                lines.append(f"   • Статус сделки: {status}. Учтено в статистике.")

    # Раздел 4: Рекомендации
    lines.append("")
    lines.append("=" * 80)
    lines.append("3. РЕКОМЕНДАЦИИ И НАБЛЮДЕНИЯ ПО РЕЗУЛЬТАТАМ ДИАГНОСТИКИ:")
    lines.append("=" * 80)
    lines.append("1. Рынок США после 17:00 Кишинёва (15:00 GMT / 10:00 NY):")
    lines.append("   В первые часы после открытия Нью-Йоркской биржи наблюдается высокая активность институционалов,")
    lines.append("   ложные пробои трендовых линий и расширение торгового спреда брокеров.")
    lines.append("2. Мониторинг весов:")
    lines.append("   Если после сбора статистики за 2-3 торговых дня подтвердится перекос убытков на вечерние часы,")
    lines.append("   рекомендуется:")
    lines.append("   а) Повысить порог min_probability для вечерних часов (например, с 64% до 72%).")
    lines.append("   б) Или включить паузу на первые 30-45 минут американской сессии (16:30–17:15).")
    lines.append("=" * 80)

    txt_content = "\n".join(lines)

    # -------------------------------------------------------------
    # 2. ФОРМИРОВАНИЕ ТАБЛИЧНОГО CSV ФАЙЛА
    # -------------------------------------------------------------
    csv_buffer = io.StringIO()
    csv_writer = csv.writer(csv_buffer)

    headers = [
        "id", "symbol", "direction", "status", "pnl",
        "created_at_chisinau", "entered_at_chisinau", "closed_at_chisinau",
        "is_after_17_chisinau", "entry_price", "exit_price", "take_profit", "stop_loss",
        "atr", "adx", "rsi", "macd_hist", "ml_probability", "min_probability",
        "primary_driver", "news_score", "news_label"
    ]
    csv_writer.writerow(headers)

    for s in signals:
        created_local, c_hour = _format_local_time(s.get("created_at"))
        entered_local, _ = _format_local_time(s.get("entered_at"))
        closed_local, _ = _format_local_time(s.get("closed_at"))

        diag = s.get("diagnostic_data") or {}
        is_after_17 = diag.get("is_after_17_chisinau")
        if is_after_17 is None:
            is_after_17 = (c_hour is not None and c_hour >= 17)

        m_ctx = diag.get("market_context") or diag.get("indicators") or {}
        ml_pred = diag.get("ml_prediction") or {}
        news_info = diag.get("news") or {}

        csv_writer.writerow([
            s.get("id"),
            s.get("symbol"),
            s.get("direction"),
            s.get("status"),
            s.get("pnl", 0.0),
            created_local,
            entered_local,
            closed_local,
            "YES" if is_after_17 else "NO",
            s.get("entry_price"),
            s.get("exit_price", ""),
            s.get("take_profit"),
            s.get("stop_loss"),
            s.get("atr"),
            m_ctx.get("adx", ""),
            m_ctx.get("rsi", ""),
            m_ctx.get("macd_hist", ""),
            s.get("probability"),
            ml_pred.get("min_probability", ""),
            ml_pred.get("primary_driver", ""),
            news_info.get("score", ""),
            news_info.get("label", "")
        ])

    csv_content = csv_buffer.getvalue()

    pnl_str = f"+{total_pnl:.2f}" if total_pnl >= 0 else f"{total_pnl:.2f}"

    return {
        "txt_content": txt_content,
        "csv_content": csv_content,
        "txt_filename": f"diagnostics_{date_str}.txt",
        "csv_filename": f"trades_data_{date_str}.csv",
        "signals_count": total_count,
        "winrate": winrate,
        "pnl_str": f"{pnl_str} п.",
        "day_signals": day_total,
        "evening_signals": eve_total,
        "day_winrate": day_wr,
        "evening_winrate": eve_wr
    }
