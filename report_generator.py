"""
Модуль формирования глубоких диагностических отчетов по сделкам торгового бота.
Генерирует подробнейший текстовый анализ (.txt) и табличные данные (.csv):
- Влияние весов ML модели на каждое решение (Feature Attribution)
- Разбор сделок по сессиям (до 17:00 vs после 17:00 Кишинёв / сессия США)
- Новостной фон, экономический календарь и макро-сантимент
- История сигналов, ручных исходов и изменений уровней
"""

import csv
import io
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from zoneinfo import ZoneInfo

from database import (
    get_signals_for_report,
    get_latest_model_snapshots,
    get_runtime_logs,
    get_analysis_logs,
)

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
    1. txt_content: Подробный диагностический отчет с таблицами весов и журналом событий.
    2. csv_content: Табличный файл с количественными метриками по всем сделкам.
    3. txt_filename, csv_filename: Имена файлов с текущей датой.
    4. summary_stats: Краткая сводка для сообщения в Telegram.
    """
    now_utc = datetime.now(timezone.utc)
    local_now = now_utc.astimezone(ZoneInfo(USER_TZ))
    date_str = local_now.strftime("%Y-%m-%d")
    gen_time_str = local_now.strftime("%Y-%m-%d %H:%M:%S")

    signals = get_signals_for_report(days=days, limit=limit)
    runtime_logs = get_runtime_logs(days=days, limit=250)
    analysis_rows = get_analysis_logs(days=days, limit=10000)

    # Статистика сессии
    total_count = len(signals)
    wins = sum(1 for s in signals if s.get("status") == "TP_HIT")
    losses = sum(1 for s in signals if s.get("status") == "SL_HIT")
    open_count = sum(1 for s in signals if s.get("status") == "OPEN")
    manual_count = sum(1 for s in signals if s.get("status") == "MANUAL_CLOSE")
    skipped_count = sum(1 for s in signals if s.get("status") == "SKIPPED")
    finished_count = wins + losses
    winrate = round((wins / finished_count * 100), 1) if finished_count > 0 else 0.0
    actual_items = [s for s in signals if s.get("reported_pnl") is not None]
    actual_by_currency = {}
    actual_profit_count = actual_loss_count = 0
    for item in actual_items:
        amount = float(item.get("reported_pnl") or 0.0)
        currency = str(item.get("reported_pnl_currency") or "USD").upper()
        actual_by_currency[currency] = actual_by_currency.get(currency, 0.0) + amount
        actual_profit_count += amount > 0
        actual_loss_count += amount < 0
    actual_winrate = round(actual_profit_count / (actual_profit_count + actual_loss_count) * 100, 1) if actual_profit_count + actual_loss_count else None
    actual_pnl_text = "; ".join(f"{currency} {amount:+.2f}" for currency, amount in sorted(actual_by_currency.items())) or "нет внесённых значений"
    def price_moves_by_symbol(items):
        moves = {}
        for item in items:
            if item.get("status") in ("TP_HIT", "SL_HIT") and item.get("pnl") is not None:
                symbol = item.get("symbol", "UNKNOWN")
                moves[symbol] = moves.get(symbol, 0.0) + float(item.get("pnl") or 0.0)
        return moves

    def format_price_moves(items):
        moves = price_moves_by_symbol(items)
        return "; ".join(f"{symbol} {value:+.3f}" for symbol, value in sorted(moves.items())) or "нет закрытых сделок"

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
    lines.append(f"• Всего сигналов:        {total_count}")
    lines.append(f"• Записей свечного анализа с признаками и версией модели: {len(analysis_rows)}")
    lines.append(f"• Предупреждений / ошибок процесса: {sum(1 for item in runtime_logs if item.get('level') == 'WARNING')} / {sum(1 for item in runtime_logs if item.get('level') in ('ERROR', 'CRITICAL'))}")
    lines.append(f"• Закрыто по Take Profit: {wins} (TP)")
    lines.append(f"• Закрыто по Stop Loss:   {losses} (SL)")
    lines.append(f"• Активно в рынке (OPEN): {open_count}")
    lines.append(f"• Сброшено вручную:       {manual_count}")
    lines.append(f"• Пропущено (SKIP):       {skipped_count}")
    lines.append(f"• Точность TP/SL:         {winrate}% (TP против SL, {finished_count} сделок)")
    lines.append(f"• Фактический P/L:        {actual_pnl_text} ({len(actual_items)} записей)")
    lines.append(f"• Прибыльность по P/L:    {actual_winrate}%" if actual_winrate is not None else "• Прибыльность по P/L:    пока недостаточно фактических записей")
    lines.append(f"• Изменение цены по инструментам: {format_price_moves(signals)}")
    lines.append("  Это оценка движения цены, не прибыль в валюте счёта.")
    lines.append("")

    # Preserve the exact model version used for each signal and print inspectable parameters.
    snapshots_by_id = {}
    for signal in signals:
        snapshot = signal.get("model_snapshot")
        if snapshot and snapshot.get("id") is not None:
            snapshots_by_id[int(snapshot["id"])] = snapshot
    # Include each current model too, even on days when that asset issued no signal.
    try:
        for snapshot in get_latest_model_snapshots(include_artifact=False):
            snapshots_by_id[int(snapshot["id"])] = snapshot
    except Exception as exc:
        logger.warning("Не удалось добавить текущие веса ML-моделей в отчёт: %s", exc)
    lines.append("=" * 80)
    lines.append("2. СОХРАНЁННЫЕ ВЕРСИИ ML-МОДЕЛЕЙ И ИХ ВЕСА:")
    lines.append("=" * 80)
    if not snapshots_by_id:
        lines.append("Для сигналов в выборке снимок весов не найден. Снимки начнут появляться после обновления бота.")
    for snapshot_id, snapshot in sorted(snapshots_by_id.items()):
        weights = snapshot.get("weights") or {}
        metadata = snapshot.get("metadata") or {}
        features = weights.get("features") or snapshot.get("features") or []
        coefficients = weights.get("standardized_coefficients") or []
        coefficient_row = coefficients[0] if coefficients else []
        lines.append("")
        lines.append(
            f"Модель #{snapshot_id} | {snapshot.get('symbol')} | обучение: {snapshot.get('trained_at') or 'время неизвестно'} | "
            f"CV accuracy: {float(snapshot.get('accuracy') or 0):.1%} | samples: {snapshot.get('samples_count')}"
        )
        lines.append(f"   SHA-256 артефакта: {snapshot.get('artifact_sha256') or '—'}")
        cv = metadata.get("cross_validation") or {}
        if cv:
            lines.append(
                f"   TimeSeriesSplit: folds={cv.get('folds')} | fold accuracy={cv.get('fold_accuracies', [])} | "
                f"обучающая выборка={metadata.get('training_start')} .. {metadata.get('training_end')}"
            )
        if features and coefficient_row:
            ranked = sorted(zip(features, coefficient_row), key=lambda pair: abs(float(pair[1])), reverse=True)
            lines.append("   Коэффициенты LogisticRegression (стандартизированные; знак + смещает к BUY, − к SELL):")
            lines.extend(f"      {feature}: {float(value):+.7f}" for feature, value in ranked)
            scaler = weights.get("standard_scaler") or {}
            scales = scaler.get("scale") or []
            means = scaler.get("mean") or []
            if len(scales) == len(features) and len(means) == len(features):
                raw_coef = weights.get("raw_feature_coefficients", [[]])[0]
                lines.append("   StandardScaler: mean / scale по FEATURE_COLUMNS:")
                lines.extend(
                    f"      {feature}: mean={float(mean):+.7g}, scale={float(scale):.7g}, raw_coef={float(raw_coef[i]):+.7g}"
                    for i, (feature, mean, scale) in enumerate(zip(features, means, scales))
                )
        lines.append("   Коэффициенты объясняют вклад признаков модели, но не доказывают причинность результата сделки.")
    lines.append("")

    lines.append("=" * 80)
    lines.append("3. СОХРАНЁННЫЕ ПРЕДУПРЕЖДЕНИЯ И ОШИБКИ ПРОЦЕССА:")
    lines.append("=" * 80)
    if not runtime_logs:
        lines.append("За выбранный период предупреждений и ошибок не записано.")
    for item in runtime_logs:
        lines.append(f"[{item.get('created_at', '')} UTC] {item.get('level')} · {item.get('logger_name')}: {item.get('message')}")
        if item.get("exception_text"):
            lines.append(str(item["exception_text"])[:3000])
    lines.append("")

    # Раздел 2: Анализ гипотезы рынка США после 17:00
    lines.append("-" * 80)
    lines.append("🚨 АНАЛИЗ ГИПОТЕЗЫ ВЛИЯНИЯ РЫНКА США (СРАВНЕНИЕ ДО И ПОСЛЕ 17:00 КИШИНЁВА):")
    lines.append("-" * 80)
    lines.append(f"☀️ ДНЕВНОЙ ПЕРИОД (до 17:00 Кишинёв — Лондонская сессия и утро):")
    lines.append(f"   Сигналов: {day_total} | TP: {day_w} | SL: {day_l} | Точность TP/SL: {day_wr}%")
    lines.append(f"   Изменение цены по инструментам: {format_price_moves(day_sigs)}")
    lines.append("")
    lines.append(f"🌙 ВЕЧЕРНИЙ ПЕРИОД (после 17:00 Кишинёв — Сессия США & Уолл-стрит):")
    lines.append(f"   Сигналов: {eve_total} | TP: {eve_w} | SL: {eve_l} | Точность TP/SL: {eve_wr}%")
    lines.append(f"   Изменение цены по инструментам: {format_price_moves(evening_sigs)}")
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
    lines.append("4. ДЕТАЛЬНЫЙ РАЗБОР КАЖДОЙ СДЕЛКИ (DEEP-DIVE):")
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
            lines.append(f"🎯 СДЕЛКА #{s_id}: [{sym}] Направление: {direction} | Статус: {status} | Изм. цены: {pnl_str}")
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
            actual_entry = s.get("actual_entry_price")
            shown_entry = actual_entry if actual_entry is not None else entry_p
            lines.append(f"   • Вход по сигналу: {entry_p} | Фактический вход: {shown_entry} ({s.get('actual_entry_source') or 'нет отметки'})")
            lines.append(f"   • TP: {tp_p} | SL: {sl_p} | Фактический выход: {exit_p if exit_p is not None else '—'} ({s.get('exit_source') or '—'})")
            lines.append(f"   • Изменение цены с учётом направления: {pnl_str} | Основа: {s.get('pnl_basis') or 'историческая запись без указания'}")
            if s.get("position_quantity") is not None:
                lines.append(f"   • Размер позиции (со слов пользователя): {s['position_quantity']}")
            if s.get("reported_pnl") is not None:
                lines.append(f"   • PnL брокера: {s['reported_pnl']} {s.get('reported_pnl_currency') or ''}")
            for event in s.get("events", []):
                event_data = event.get("event_data") or {}
                lines.append(f"   • История [{event.get('created_at', '')} UTC] {event.get('event_type')} / {event.get('source')}: {json.dumps(event_data, ensure_ascii=False, default=str)}")
            for change in s.get("level_history", []):
                lines.append(f"   • Уровни [{change.get('changed_at', '')} UTC]: Entry {change.get('old_actual_entry_price')} → {change.get('new_actual_entry_price')}; TP {change.get('old_take_profit')} → {change.get('new_take_profit')}; SL {change.get('old_stop_loss')} → {change.get('new_stop_loss')}")
            lines.append(f"   • Волатильность (ATR): {atr_v:.4f} | Вероятность ML: {prob_v:.1%}")
            snapshot = s.get("model_snapshot") or {}
            snapshot_ref = diag.get("model_snapshot") or {}
            snapshot_id = s.get("model_snapshot_id") or snapshot_ref.get("snapshot_id")
            if snapshot_id:
                lines.append(
                    f"   • Версия модели: #{snapshot_id} | SHA-256: "
                    f"{snapshot.get('artifact_sha256') or snapshot_ref.get('sha256') or '—'}"
                )

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

            # Влияние записанного результата на обучение модели
            lines.append("")
            lines.append("🔄 КАК ИСХОДЫ СДЕЛОК СВЯЗАНЫ С ОБУЧЕНИЕМ МОДЕЛИ:")
            if status == "SL_HIT":
                lines.append("   • В журнале сделка отмечена как SL.")
                lines.append("   • Событие записано в журнал, но текущий код обучения не использует его как метку сделки.")
            elif status == "TP_HIT":
                lines.append("   • В журнале сделка отмечена как TP.")
                lines.append("   • Событие записано в журнал, но текущий код обучения не использует его как метку сделки.")
            elif status == "OPEN":
                lines.append("   • Открытая сделка пока не имеет результата; журнал сделки не является входом текущего обучения.")
            else:
                lines.append(f"   • Статус сделки: {status}. Учтено в статистике.")

    # Раздел 4: Рекомендации
    lines.append("")
    lines.append("=" * 80)
    lines.append("5. РЕКОМЕНДАЦИИ И НАБЛЮДЕНИЯ ПО РЕЗУЛЬТАТАМ ДИАГНОСТИКИ:")
    lines.append("=" * 80)
    lines.append("1. Сравнивайте результаты по инструменту, направлению, времени и фактическим ценам входа/выхода.")
    lines.append("2. Не трактуйте сумму ценовых изменений по разным инструментам как прибыль в валюте счёта.")
    lines.append("3. Ручные результаты и изменения уровней сохраняются для анализа; текущая ML-модель обучается на рыночных свечах, а не на этих отметках.")
    lines.append("4. Наблюдаемые связи в малой выборке не устанавливают причину результата и сами по себе не обосновывают изменение стратегии.")
    lines.append("=" * 80)

    txt_content = "\n".join(lines)

    # -------------------------------------------------------------
    # 2. ФОРМИРОВАНИЕ ТАБЛИЧНОГО CSV ФАЙЛА
    # -------------------------------------------------------------
    csv_buffer = io.StringIO()
    csv_writer = csv.writer(csv_buffer)

    headers = [
        "id", "symbol", "direction", "status", "pnl_price_move_estimate", "pnl_basis", "exit_source",
        "created_at_chisinau", "entered_at_chisinau", "closed_at_chisinau",
        "is_after_17_chisinau", "signal_entry_price", "actual_entry_price", "actual_entry_source",
        "exit_price", "take_profit", "stop_loss", "initial_take_profit", "initial_stop_loss",
        "position_quantity", "reported_pnl", "reported_pnl_currency", "level_history_json", "events_json",
        "atr", "adx", "rsi", "macd_hist", "ml_probability", "min_probability",
        "model_snapshot_id", "model_snapshot_sha256", "model_weights_json",
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
        model_snapshot = s.get("model_snapshot") or {}

        csv_writer.writerow([
            s.get("id"),
            s.get("symbol"),
            s.get("direction"),
            s.get("status"),
            s.get("pnl", 0.0),
            s.get("pnl_basis", ""),
            s.get("exit_source", ""),
            created_local,
            entered_local,
            closed_local,
            "YES" if is_after_17 else "NO",
            s.get("entry_price"),
            s.get("actual_entry_price", ""),
            s.get("actual_entry_source", ""),
            s.get("exit_price", ""),
            s.get("take_profit"),
            s.get("stop_loss"),
            s.get("initial_take_profit", ""),
            s.get("initial_stop_loss", ""),
            s.get("position_quantity", ""),
            s.get("reported_pnl", ""),
            s.get("reported_pnl_currency", ""),
            json.dumps(s.get("level_history", []), ensure_ascii=False, default=str),
            json.dumps(s.get("events", []), ensure_ascii=False, default=str),
            s.get("atr"),
            m_ctx.get("adx", ""),
            m_ctx.get("rsi", ""),
            m_ctx.get("macd_hist", ""),
            s.get("probability"),
            ml_pred.get("min_probability", ""),
            s.get("model_snapshot_id") or (diag.get("model_snapshot") or {}).get("snapshot_id", ""),
            model_snapshot.get("artifact_sha256", ""),
            json.dumps(model_snapshot.get("weights", {}), ensure_ascii=False, default=str),
            ml_pred.get("primary_driver", ""),
            news_info.get("score", ""),
            news_info.get("label", "")
        ])

    csv_content = csv_buffer.getvalue()

    analysis_buffer = io.StringIO()
    analysis_writer = csv.writer(analysis_buffer)
    analysis_writer.writerow([
        "analysis_id", "created_at_utc", "symbol", "close_price", "ml_action",
        "ml_probability", "final_outcome", "reason", "model_snapshot_id",
        "model_snapshot_sha256", "candle_timestamp", "ohlcv_json", "features_json",
        "market_regime_json", "news_json", "model_diagnostics_json", "full_diagnostics_json",
    ])
    for item in analysis_rows:
        diagnostics = item.get("diagnostics") or {}
        snapshot = item.get("model_snapshot") or {}
        analysis_writer.writerow([
            item.get("id"), item.get("created_at"), item.get("symbol"), item.get("close_price"),
            diagnostics.get("model_action", item.get("action")), item.get("probability"),
            item.get("outcome"), item.get("details"), item.get("model_snapshot_id") or "",
            snapshot.get("artifact_sha256", ""), diagnostics.get("candle_timestamp", ""),
            json.dumps(diagnostics.get("ohlcv", {}), ensure_ascii=False, default=str),
            json.dumps(diagnostics.get("features", {}), ensure_ascii=False, default=str),
            json.dumps(diagnostics.get("market_regime", {}), ensure_ascii=False, default=str),
            json.dumps(diagnostics.get("news", {}), ensure_ascii=False, default=str),
            json.dumps(diagnostics.get("model_diagnostics", {}), ensure_ascii=False, default=str),
            json.dumps(diagnostics, ensure_ascii=False, default=str),
        ])
    analysis_csv_content = analysis_buffer.getvalue()

    return {
        "txt_content": txt_content,
        "csv_content": csv_content,
        "analysis_csv_content": analysis_csv_content,
        "txt_filename": f"diagnostics_{date_str}.txt",
        "csv_filename": f"trades_data_{date_str}.csv",
        "analysis_csv_filename": f"analysis_data_{date_str}.csv",
        "signals_count": total_count,
        "winrate": winrate,
        "pnl_str": actual_pnl_text,
        "day_signals": day_total,
        "evening_signals": eve_total,
        "day_winrate": day_wr,
        "evening_winrate": eve_wr
    }
