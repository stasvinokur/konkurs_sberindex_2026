"""HTML-блоки лендинга: результаты, разборы и справка.

Таблицы и диаграммы строятся из данных отчёта; выводы — формулировки `sbx.core.report` и
`sbx.core.storyline`, те же, что в методологическом отчёте. Блок без данных возвращает пустую
строку, и страница его пропускает.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from sbx.core.report import (
    DETECTOR_NAMES,
    REFERENCE_KIND,
    counted,
    detector_rows,
    fact_above_statement,
    format_delta,
    format_number,
    horizon_label,
    model_family,
    model_title,
    reference_wins,
)
from sbx.shell.render import charts
from sbx.shell.render.blocks import (
    REFERENCE,
    _e,
    cap,
    effect,
    embedded_json,
    inline_html,
    legend,
    linked,
    month_name,
    ticks,
)

# Семейства моделей в таблице горизонтов: порядок и заголовки групп.
FAMILIES = (
    (("foundation",), "Foundation-модели"),
    (("stats",), "Статистические модели"),
    (("ml",), "Машинное обучение и нейросети"),
    (("ensemble",), "Ансамбли"),
    (("reference", "naive"), "Эталон и наивные прогнозы"),
)
KIND_NAMES = {"online": "онлайн", "offline": "офлайн", "baseline": "эталон"}
INPUT_NAMES = {"residual": "остатки прогноза", "raw": "сырой ряд"}
ABLATION_TITLES = {
    "A": "LightGBM только на данных Сбера",
    "B": "Плюс календарь и национальные ряды",
    "C": "Плюс новости и календарь событий",
    "D": "Набор моделей с foundation-моделями",
    "E": "Набор моделей с foundation-моделями",
    "F": "Набор E и детектор по остаткам прогноза",
}
NOT_BUILT = (
    "Не строится: глобальному LightGBM при h=12 не из чего собрать обучающую выборку, нейросетям "
    "не остаётся входного окна."
)
COPY_ICON = (
    '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">'
    '<rect x="5.5" y="5.5" width="8" height="8" rx="1.5"/>'
    '<path d="M3.5 10.5h-1v-8h8v1"/></svg>'
)
PIN_ICON = (
    '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">'
    '<path d="M8 14.5s4.5-4.2 4.5-8a4.5 4.5 0 0 0-9 0c0 3.8 4.5 8 4.5 8z"/>'
    '<circle cx="8" cy="6.5" r="1.6"/></svg>'
)


def _known(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(value)


def _reference_label(model: str) -> str:
    """Имя модели в строке таблицы; у эталона — метка."""
    label = _e(model_title(model))
    return f'{label} <span class="tag-ref">эталон</span>' if model == REFERENCE else label


# ── 3.1. MAE по горизонтам ───────────────────────────────────────────────────────────────────


def horizon_table(
    rows: Sequence[Mapping[str, Any]],
    winners: Sequence[Mapping[str, Any]] = (),
    coverage: Mapping[Any, Mapping[str, Any]] | None = None,
    significance: Mapping[str, Any] | None = None,
) -> tuple[str, bool]:
    """Таблица MAE «модель × горизонт»: модели сгруппированы по типу, лидер горизонта выделен.

    Сверху — строка лучших моделей со счётом окон против эталона. Пропуск в ячейке означает,
    что модель на этом горизонте построить нельзя, а не что она отработала плохо; есть ли такие
    ячейки, возвращается вторым значением — для сноски.
    """
    if not rows:
        return "", False
    columns = [key for key in rows[0] if key != "Модель"]
    horizons = [int(key.split("=")[-1]) for key in columns]
    best = {
        column: min((float(row[column]) for row in rows if _known(row[column])), default=None)
        for column in columns
    }
    wins = reference_wins(significance or {}, REFERENCE)
    head = "".join(f'<th scope="col">{horizon} мес.</th>' for horizon in horizons)
    windows = ""
    if wins:
        cells = "".join(f"<td>{wins[h][1] if h in wins else '—'}</td>" for h in horizons)
        windows = f"<tr><td>окон проверки, общих для всех моделей</td>{cells}</tr>"

    body = []
    if winners:
        by_horizon = {int(w["horizon"]): w for w in winners}
        cells = []
        for horizon in horizons:
            winner = by_horizon.get(horizon)
            if not winner:
                cells.append("<td></td>")
                continue
            count = ""
            if horizon in wins:
                better, folds = wins[horizon]
                count = f'<div class="wins">{ticks(better, folds, " big soft")}{better} из {folds}</div>'
            cells.append(
                f"<td><span>{_e(model_title(winner['model']))}</span>"
                f"<b>{_e(format_number(winner['mae'], 1))}</b>{count}</td>"
            )
        note = f"<small>и в скольких окнах она точнее {REFERENCE}</small>" if wins else ""
        body.append(
            f'<tr class="leaders"><th scope="row">Лучшая на горизонте{note}</th>{"".join(cells)}</tr>'
        )
    missing = False
    for families, title in FAMILIES:
        members = [row for row in rows if model_family(row["Модель"], REFERENCE) in families]
        if not members:
            continue
        # Эталон открывает свою группу: с ним сравнивают все остальные строки.
        members.sort(key=lambda row: model_family(row["Модель"], REFERENCE) != "reference")
        body.append(
            f'<tr class="fam"><th colspan="{len(columns) + 1}" scope="colgroup">{title}</th></tr>'
        )
        for row in members:
            cells = []
            for column in columns:
                value = row[column]
                if not _known(value):
                    missing = True
                    cells.append('<td class="na">не строится¹</td>')
                    continue
                cls = ' class="win"' if float(value) == best[column] else ""
                cells.append(f"<td{cls}>{_e(format_number(float(value), 1))}</td>")
            cls = ' class="reference"' if row["Модель"] == REFERENCE else ""
            body.append(
                f'<tr{cls}><th scope="row">{_reference_label(str(row["Модель"]))}</th>{"".join(cells)}</tr>'
            )
    foot = ""
    if coverage:
        counts = []
        for horizon in horizons:
            entry = coverage.get(str(horizon)) or coverage.get(horizon) or {}
            counts.append(
                f"<td>{entry.get('models', '—')} / {_e(format_number(entry.get('observations')))}</td>"
            )
        foot = f'<tfoot><tr><th scope="row">моделей / наблюдений</th>{"".join(counts)}</tr></tfoot>'
    table = (
        '<div class="scroll-x"><table class="board">'
        "<caption>MAE, руб. — ниже лучше; модели сравниваются внутри каждого горизонта и "
        "сгруппированы по типу.</caption>"
        f'<thead><tr><th scope="col">Модель</th>{head}</tr>{windows}</thead>'
        f"<tbody>{''.join(body)}</tbody>{foot}</table></div>"
    )
    return table, missing


def horizons_html(
    rows: Sequence[Mapping[str, Any]],
    winners: Sequence[Mapping[str, Any]],
    coverage: Mapping[Any, Mapping[str, Any]],
    significance: Mapping[str, Any],
    windows_note: str,
) -> str:
    """Таблица горизонтов, график MAE и примечания к ним."""
    table, missing = horizon_table(rows, winners, coverage, significance)
    notes = [
        "<li><b>Шкала логарифмическая:</b> равные шаги по вертикали — равные отношения ошибок. На "
        f"обычной шкале ошибка {REFERENCE} на длинном горизонте сжала бы остальные линии в одну.</li>"
    ]
    if windows_note:
        notes.append(f"<li><b>Счёт окон.</b> {_e(windows_note)}</li>")
    if missing:
        notes.append(f"<li><b>¹</b> {_e(NOT_BUILT)}</li>")
    notes.append(
        "<li><b>Окна.</b> Горизонт 12 месяцев — три перекрывающихся фолда, обучение 10–12 "
        "месяцев.</li>"
    )
    chart = (
        '<div class="chart-col"><h3>MAE по горизонтам, ₽</h3>'
        f"<p>Лидеры, {REFERENCE} и наивный прогноз. Кольцо — лучшая модель горизонта.</p>"
        f'<figure><div class="scroll-x">{charts.horizon_chart(rows, winners, REFERENCE)}</div></figure>'
        f'<ul class="notes">{"".join(notes)}</ul></div>'
    )
    return f'<div class="two">{table}{chart}</div>'


# ── 3.2. Foundation-модели ───────────────────────────────────────────────────────────────────


def foundation_statement(view: Mapping[str, Any]) -> str:
    """Вывод блока foundation-моделей — из того, на каких горизонтах они выиграли."""
    horizons = view["horizons"]

    def listed(values: Sequence[int]) -> str:
        names = [str(v) for v in values]
        text = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} и {names[-1]}"
        return f"{text} мес."

    wins, beats = view["wins"], view["beats_reference"]
    if not wins:
        first = "Foundation-модели не обошли лучшую из остальных одиночных моделей ни на одном горизонте"
    elif len(wins) == len(horizons):
        first = "Foundation-модели лучше остальных одиночных моделей на всех горизонтах"
    else:
        first = (
            f"Foundation-модели лучше остальных одиночных моделей на горизонтах {listed(wins)}, "
            f"на {listed([h for h in horizons if h not in wins])} — уступают"
        )
    if len(beats) == len(horizons):
        second = f"{REFERENCE} они превосходят на всех горизонтах"
    elif beats:
        second = f"{REFERENCE} они превосходят на горизонтах {listed(beats)}"
    else:
        second = f"{REFERENCE} они не превосходят"
    return f"{first}; {second}."


def foundation_html(view: Mapping[str, Any]) -> str:
    """Лучшая foundation-модель против лучшей из остальных на каждом горизонте и полная таблица."""
    horizons = view["horizons"]
    foundation = [row for row in view["rows"] if row["role"] == "foundation"]
    other = next((row for row in view["rows"] if row["role"] == "best_other"), None)
    reference = next((row for row in view["rows"] if row["role"] == "reference"), None)
    cards = []
    for i, horizon in enumerate(horizons):
        known = [row for row in foundation if _known(row["values"][i])]
        if not known or other is None or not _known(other["values"][i]):
            continue
        ours = min(known, key=lambda row: row["values"][i])
        value, rival = float(ours["values"][i]), float(other["values"][i])
        top = max(value, rival)
        won = value < rival
        tail = (
            f'<div class="hcard-ref">{REFERENCE} — {_e(format_number(reference["values"][i], 1))}</div>'
            if reference and _known(reference["values"][i])
            else ""
        )
        cards.append(
            '<div class="hcard"><div class="hcard-head">'
            f"<b>{_e(horizon_label(horizon))}</b>"
            f"<span{' class="won"' if won else ''}>{'foundation точнее' if won else 'остальные точнее'}</span></div>"
            '<div class="hcard-rows">'
            f'<div><div class="lbl{" win" if won else ""}"><span>{_e(model_title(ours["model"]))}</span>'
            f"<span>{_e(format_number(value, 1))}</span></div>"
            f'<div class="hbar"><i class="found" style="width:{100 * value / top:.1f}%"></i></div></div>'
            f'<div><div class="lbl{"" if won else " win"}"><span>{_e(model_title(other["names"][i]))}</span>'
            f"<span>{_e(format_number(rival, 1))}</span></div>"
            f'<div class="hbar"><i style="width:{100 * rival / top:.1f}%"></i></div></div>'
            f"</div>{tail}</div>"
        )
    single = [row for row in view["rows"] if row["role"] != "ensemble"]
    best = [
        min((row["values"][i] for row in single if _known(row["values"][i])), default=None)
        for i in range(len(horizons))
    ]
    body = []
    for row in view["rows"]:
        cells = []
        for i, value in enumerate(row["values"]):
            win = row["role"] != "ensemble" and _known(value) and value == best[i]
            name = f"<small>{_e(model_title(row['names'][i]))}</small>" if row.get("names") else ""
            text = format_number(value, 1) if _known(value) else "—"
            cells.append(f"<td{' class="win"' if win else ''}>{_e(text)}{name}</td>")
        if row["role"] == "foundation":
            cls, label = "", f'<i class="sq found"></i>{_e(model_title(row["model"]))}'
        elif row["role"] == "best_other":
            cls, label = ' class="other"', f'<i class="sq"></i>{_e(row["model"])}'
        elif row["role"] == "reference":
            cls, label = ' class="reference"', _reference_label(str(row["model"]))
        else:
            cls = ' class="aside"'
            label = f"{_e(model_title(row['model']))} <small>· для сравнения</small>"
        body.append(f'<tr{cls}><th scope="row">{label}</th>{"".join(cells)}</tr>')
    head = "".join(f'<th scope="col" class="th">{horizon} мес.</th>' for horizon in horizons)
    table = (
        '<div class="scroll-x"><table class="board flat">'
        "<caption>Все foundation-модели, MAE, руб. Жирным — лучшая одиночная модель горизонта; "
        "ансамбль приведён для сравнения.</caption>"
        f'<thead><tr><th scope="col" class="th">Модель</th>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )
    weights = "; ".join(
        f"<code>{_e(w['repo'])}</code> (ревизия {_e(w['revision'])}, лицензия {_e(w['license'])})"
        for w in view["weights"]
    )
    note = (
        f'<p class="note small">Веса используются без дообучения, ревизии закреплены: {weights}.</p>'
        if weights
        else ""
    )
    keys = legend(
        [
            '<i class="sw found"></i>лучшая foundation-модель',
            '<i class="sw other"></i>лучшая из остальных одиночных моделей',
            "короче полоса — меньше ошибка",
        ]
    )
    grid = f'{keys}<div class="cards4">{"".join(cards)}</div>' if cards else ""
    return f"{grid}{table}{note}"


# ── 3.3–3.5. Примеры прогноза, выбор ряда, все метрики ───────────────────────────────────────

SERIES_KEYS = (
    '<i class="k-line dots"></i>факт',
    '<i class="k-line acc"></i>прогноз ансамбля',
    '<i class="sw band"></i>интервал 0,1–0,9',
)


def short_region(name: Any) -> str:
    """Название региона для подписи в одну строку: «Чукотский АО»."""
    return str(name or "").replace("автономный округ", "АО")


def showcase_html(showcase: Sequence[Mapping[str, Any]]) -> str:
    """По отдельному графику на ряд: на общей оси ряды разного уровня не читаются."""
    cards = []
    for item in showcase:
        place = cap(str(item.get("mo") or item.get("region") or ""))
        about = " · ".join(
            part
            for part in (
                short_region(item.get("region")),
                str(item.get("category", "")).lower(),
                charts.series_unit(item),
            )
            if part
        )
        above = fact_above_statement(item["y"], item["y_hat"])
        cards.append(
            f'<figure class="mini"><figcaption><b>{_e(place)}</b><span>{_e(about)}</span></figcaption>'
            f"{charts.series_chart(item)}{f'<p>{_e(above)}</p>' if above else ''}</figure>"
        )
    return f'{legend(SERIES_KEYS)}<div class="cards3">{"".join(cards)}</div>'


def explorer_html(explorer: Mapping[str, Any]) -> str:
    """Выбор региона и категории; график рисует сценарий страницы.

    Данные встроены в страницу: лендинг статический, серверной части нет. Объём ограничен
    выборкой рядов в `explorer_data`.
    """
    if not explorer.get("series"):
        return ""
    keys = "".join(
        f'<span class="key">{item}</span>'
        for item in (
            *SERIES_KEYS[:1],
            '<i class="k-line acc"></i>прогноз',
            SERIES_KEYS[2],
            '<i class="k-tri"></i>тревога детектора',
        )
    )
    return (
        f'<div id="explorer">{embedded_json(explorer)}'
        '<div class="controls">'
        '<div class="field"><label for="ex-region">Регион</label>'
        '<div class="select"><select id="ex-region"></select></div></div>'
        '<div class="field"><label for="ex-category">Категория расходов</label>'
        '<div class="select"><select id="ex-category"></select></div></div>'
        f'<div class="status" aria-live="polite">{PIN_ICON}<span>Показан МО из оценочной выборки: '
        '<b data-f="place"></b></span></div></div>'
        '<div class="split"><figure>'
        f'<div class="fig-cap"><b data-f="title"></b><span class="legend">{keys}</span></div>'
        '<div id="ex-plot" class="scroll-x"></div></figure>'
        '<aside class="aside-card" aria-live="polite"><div class="cap">Выбранный месяц</div>'
        '<div class="month" data-f="month"></div><dl class="kv">'
        '<div><dt>Факт</dt><dd class="b" data-f="y"></dd></div>'
        '<div><dt>Прогноз</dt><dd class="acc" data-f="forecast"></dd></div>'
        '<div><dt>Интервал 0,1–0,9</dt><dd data-f="interval"></dd></div></dl>'
        '<div class="alarm-note" data-f="alarm" hidden><i class="k-tri"></i>Тревога детектора в этом месяце</div>'
        '<p class="p13">Тревог детектора у ряда: <b data-f="alarms"></b></p>'
        '<p class="p13 dim">Наведите на график или выберите месяц стрелками ← →.</p></aside></div>'
        '<noscript><p class="note">Выбор ряда работает со сценарием страницы; три примера прогноза '
        "показаны выше.</p></noscript></div>"
    )


def metrics_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Сводная таблица метрик: число знаков задаётся по столбцу, по умолчанию 3."""
    if not rows:
        return ""
    digits = {"MAE, руб.": 1, "MAE macro, руб.": 1}
    columns = [key for key in rows[0] if key != "Модель"]
    head = "".join(f'<th scope="col" class="th">{_e(column)}</th>' for column in columns)
    body = []
    for row in rows:
        cells = "".join(
            f"<td>{_e(format_number(row[c], digits.get(c, 3)) if _known(row[c]) else '—')}</td>"
            for c in columns
        )
        cls = ' class="reference"' if row["Модель"] == REFERENCE else ""
        body.append(
            f'<tr{cls}><th scope="row">{_reference_label(str(row["Модель"]))}</th>{cells}</tr>'
        )
    return (
        '<div class="scroll-x"><table class="board flat">'
        f'<thead><tr><th scope="col" class="th">Модель</th>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


# ── 4.1. Сравнение детекторов ────────────────────────────────────────────────────────────────


def match_html(match: Mapping[str, Any]) -> str:
    """Детектор и расписание при равной частоте ложных тревог: доля найденных сломов, задержка
    и интервал разности."""
    needed = {
        "detector", "period", "detector_recall", "schedule_recall", "detector_delay",
        "schedule_delay", "detector_false_alarms", "schedule_false_alarms", "recall_diff",
        "recall_diff_low", "recall_diff_high",
    }  # fmt: skip
    if not needed <= set(match):
        return ""
    name = DETECTOR_NAMES.get(str(match["detector"]), str(match["detector"]))
    period = int(match["period"])
    schedule = "раз в месяц" if period == 1 else f"раз в {horizon_label(period)}"

    def row(title: str, cls: str, recall: float, alarms: float, delay: float) -> str:
        share = round(100 * float(recall))
        return (
            f'<div><div class="rate-head"><b{cls}>{_e(title)}</b>'
            f"<span>{_e(format_number(alarms, 2))} ложной тревоги на ряд в год</span></div>"
            f'<div class="rate-bar"><div><i{cls} style="width:{share}%"></i></div>'
            f"<span><b>{share}%</b> <small>сломов</small></span></div>"
            f"<small>срабатывает в среднем через <b>{_e(format_number(delay, 1))} мес.</b> после слома</small></div>"
        )

    low, high = 100 * float(match["recall_diff_low"]), 100 * float(match["recall_diff_high"])
    point = 100 * float(match["recall_diff"])
    span = max(high, 0.0) - min(low, 0.0)
    left, right = min(low, 0.0) - 0.12 * span, max(high, 0.0) + 0.12 * span

    def at(value: float) -> str:
        return f"{100 * (value - left) / (right - left):.1f}%"

    level = round(100 * float(match.get("level", 0.95)))
    verdict_text = (
        f"Интервал от {format_number(low, 1)} до {format_number(high, 1)} п.п. включает ноль: "
        "перевес не доказан."
        if low <= 0 <= high
        else f"Интервал от {format_number(low, 1)} до {format_number(high, 1)} п.п. ноль не включает."
    )
    return (
        '<div class="match"><div><h3>При равной частоте ложных тревог</h3>'
        "<p>Период расписания подобран на калибровочной половине бенчмарка, сравнение — на тестовой</p>"
        '<div class="rate">'
        + row(
            f"{name} по остаткам прогноза",
            "",
            match["detector_recall"],
            match["detector_false_alarms"],
            match["detector_delay"],
        )
        + row(
            f"Расписание: тревога {schedule}",
            ' class="ref"',
            match["schedule_recall"],
            match["schedule_false_alarms"],
            match["schedule_delay"],
        )
        + "</div></div>"
        "<div><h3>Разница в доле найденных сломов</h3>"
        f"<p>{level}%-й интервал бутстрепа по рядам</p>"
        f'<div class="gap"><b>{_e(format_delta(point))}</b><span>процентного пункта</span></div>'
        f'<div class="ci" role="img" aria-label="Разность {_e(format_number(point, 1))} п.п., интервал от '
        f'{_e(format_number(low, 1))} до {_e(format_number(high, 1))} п.п.">'
        f'<i class="range" style="left:{at(low)};width:calc({at(high)} - {at(low)})"></i>'
        f'<i class="zero" style="left:{at(0.0)}"></i><i class="point" style="left:{at(point)}"></i>'
        f'<span class="z" style="left:{at(0.0)}">0</span>'
        f'<span style="left:{at(low)}">{_e(format_number(low, 1))}</span>'
        f'<span style="left:{at(high)}">{_e(format_number(high, 1))}</span></div>'
        f'<p class="p14">{_e(verdict_text)}</p></div></div>'
    )


def findings_html(left: Sequence[tuple[str, str]], right: Sequence[tuple[str, str]]) -> str:
    """Пронумерованные выводы в две колонки: что показал бенчмарк и оговорки к нему."""
    number = 0

    def column(title: str, items: Sequence[tuple[str, str]]) -> str:
        nonlocal number
        lines = []
        for name, text in items:
            if not text:
                continue
            number += 1
            lines.append(f"<li><i>{number:02d}</i><div><b>{_e(name)}</b> {_e(text)}</div></li>")
        return f"<div><h3>{_e(title)}</h3><ol>{''.join(lines)}</ol></div>" if lines else ""

    columns = column("Что показал бенчмарк", left) + column("Оговорки", right)
    return f'<div class="findings">{columns}</div>' if columns else ""


def _variant_marks(contrast: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    """Буквы у трёх вариантов, о которых говорит вывод: (детектор, вход) → буква."""
    marks: dict[tuple[str, str], str] = {}
    if not contrast:
        return marks
    marks[(str(contrast["detector"]), "residual")] = "A"
    best_raw = contrast.get("best_raw")
    if best_raw:
        marks[(str(best_raw["detector"]), "raw")] = "B"
        if (
            str(best_raw["detector"]) != str(contrast["detector"])
            and contrast.get("f1_raw") is not None
        ):
            marks[(str(contrast["detector"]), "raw")] = "C"
    return marks


def variants_html(detectors: Sequence[Mapping[str, Any]], contrast: Mapping[str, Any]) -> str:
    """Таблица вариантов «метод × вход» с эталоном: F1 рядом с линией расписания."""
    frame = detector_rows(pd.DataFrame(list(detectors)))
    rows = frame.to_dict("records")
    if not rows:
        return ""
    marks = _variant_marks(contrast)
    name = DETECTOR_NAMES.get(str(contrast.get("detector")), "") if contrast else ""
    scores = [float(row["f1"]) for row in rows if _known(row.get("f1"))]
    low, high = min(scores), max(scores)
    pad = 0.08 * ((high - low) or 1.0)
    low, high = low - pad, high + pad
    reference = next((row for row in rows if row.get("kind") == REFERENCE_KIND), None)

    def at(value: float) -> str:
        return f"{100 * (value - low) / (high - low):.1f}%"

    line = (
        f'<span class="at" style="left:{at(float(reference["f1"]))}"></span>' if reference else ""
    )
    body, plain = [], []
    for row in rows:
        detector, source = str(row["detector"]), str(row.get("input"))
        is_reference = row.get("kind") == REFERENCE_KIND
        title = cap(DETECTOR_NAMES.get(detector, detector))
        letter = marks.get((detector, source), "")
        if is_reference:
            period = int(row["param"]) if _known(row.get("param")) else None
            source_text = (
                "раз в месяц"
                if period == 1
                else f"раз в {horizon_label(period)}"
                if period
                else "—"
            )
            cls, dot = ' class="reference"', "ref"
        else:
            source_text = INPUT_NAMES.get(source, source)
            cls = f' class="marked{" a" if letter == "A" else ""}"' if letter else ""
            dot = "acc" if source == "residual" else ""
        f1 = float(row["f1"]) if _known(row.get("f1")) else None
        badge = f'<span class="letter">{letter}</span>' if letter else ""
        plot = (
            f'<span class="dotline">{line}<i class="{dot}" style="left:{at(f1)}"></i></span>'
            if f1 is not None
            else ""
        )
        resid = ' class="resid"' if source == "residual" else ' class="muted"'
        body.append(
            f'<tr{cls}><td>{badge}</td><th scope="row">{_e(title)}</th>'
            + (
                '<td><span class="tag-ref">эталон</span></td>'
                if is_reference
                else f'<td class="muted">{_e(KIND_NAMES.get(str(row.get("kind")), str(row.get("kind"))))}</td>'
            )
            + f"<td{resid}>{_e(source_text)}</td>"
            f'<td class="r">{_e(format_number(f1, 3))}</td><td>{plot}</td>'
            f'<td class="r">{_e(format_number(_number(row.get("mean_delay")), 3))}</td>'
            f'<td class="r">{_e(format_number(_number(row.get("false_alarms_per_series_year")), 3))}</td></tr>'
        )
        label = " · ".join(
            part for part in (letter, title, "" if is_reference else source_text) if part
        )
        plain.append(
            f'<tr{' class="reference"' if is_reference else ""}><th scope="row">{_e(label)}</th>'
            f'<td class="r">{_e(format_number(_number(row.get("precision")), 3))}</td>'
            f'<td class="r">{_e(format_number(_number(row.get("recall")), 3))}</td></tr>'
        )
    scale = (
        f"<span><span>{_e(format_number(low + pad, 3))}</span>"
        f"<span>{_e(format_number(high - pad, 3))}</span></span>"
    )
    keys = []
    for letter, text in (
        ("A", "лучший по остаткам прогноза"),
        ("B", "лучший онлайн по сырому ряду"),
        ("C", f"тот же {name} по сырому ряду"),
    ):
        if letter in marks.values():
            keys.append(f'<span class="letter">{letter}</span>{text}')
    keys += [
        '<i class="dot acc"></i>вход — остатки прогноза',
        '<i class="dot"></i>вход — сырой ряд',
    ]
    if reference:
        keys.append('<i class="k-line ref"></i>F1 эталона')
    has_scores = any("precision" in row or "recall" in row for row in rows)
    details = (
        '<details class="more"><summary>Точность и полнота по всем вариантам</summary>'
        '<div class="scroll-x"><table class="plain"><thead><tr><th scope="col">Вариант</th>'
        '<th scope="col" class="r">Точность</th><th scope="col" class="r">Полнота</th></tr></thead>'
        f"<tbody>{''.join(plain)}</tbody></table></div></details>"
        if has_scores
        else ""
    )
    return (
        legend(keys)
        + '<div class="scroll-x"><table class="variants"><thead><tr><th></th><th scope="col">Метод</th>'
        '<th scope="col">Режим</th><th scope="col">Вход</th><th scope="col" class="r f1">F1</th>'
        f'<th class="scale">{scale}</th><th scope="col" class="r">Задержка, мес.</th>'
        '<th scope="col" class="r">Ложных тревог на ряд в год</th></tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>{details}"
    )


def _number(value: Any) -> float | None:
    return float(value) if _known(value) else None


def curve_html(
    curve: Sequence[Mapping[str, Any]], contrast: Mapping[str, Any], match: Mapping[str, Any]
) -> str:
    """Кривая «задержка — ложные тревоги» с пояснениями к её отметкам."""
    if not curve:
        return ""
    focus = str(contrast.get("detector", ""))
    pair = dict(match) if match.get("period") is not None else None
    if pair is not None and "threshold" not in pair and contrast.get("threshold") is not None:
        pair["threshold"] = contrast["threshold"]
    chart = charts.delay_chart(curve, focus=focus, match=pair)
    notes = []
    if 'class="ref-area"' in chart:
        notes.append(
            '<div><i class="k-area"></i><div><b>Ниже пунктира — раньше расписания</b> при той же '
            "частоте ложных тревог.</div></div>"
        )
    if 'class="match"' in chart and pair is not None:
        name = DETECTOR_NAMES.get(str(pair.get("detector")), str(pair.get("detector")))
        notes.append(
            '<div><i class="k-match"></i><div><b>Отрезок</b> соединяет точки, взятые для сравнения '
            f"при равной частоте ложных тревог: {_e(name)} при пороге "
            f"{_e(format_number(float(pair['threshold']), 1))} и расписание раз в "
            f"{_e(horizon_label(int(pair['period'])))}.</div></div>"
        )
    notes.append(
        '<div><i class="k-grey"></i><div>Серые кривые — другие онлайн-детекторы по остаткам '
        "прогноза. Кривая построена по всему бенчмарку, включая калибровочную половину: это "
        "описание, а не проверка.</div></div>"
    )
    return (
        '<div class="curve"><figure><figcaption><b class="fig-title">Задержка против частоты '
        "ложных тревог</b><span>Каждая точка — один порог детектора. Левее и ниже — "
        f'лучше.</span></figcaption><div class="scroll-x">{chart}</div></figure>'
        f'<div class="curve-notes">{"".join(notes)}</div></div>'
    )


# ── 5.1. Абляции и вклад источников ──────────────────────────────────────────────────────────


def ablation_table(ablations: Sequence[Mapping[str, Any]]) -> str:
    """Конфигурации абляции: состав, MAE, разница с первой и F1 обнаружения."""
    deltas = [float(row["delta_mae"]) for row in ablations if _known(row.get("delta_mae"))]
    top = max((abs(value) for value in deltas), default=0.0) or 1.0
    below = any(value < 0 for value in deltas)
    above = any(value > 0 for value in deltas)
    zero = 100.0 if not above else 0.0 if not below else 50.0
    reach = 100.0 if not (below and above) else 50.0
    has_f1 = any(_known(row.get("f1")) for row in ablations)
    has_models = any(row.get("models") for row in ablations)
    seen: dict[str, str] = {}
    body = []
    for row in ablations:
        name = str(row["name"])
        title = ABLATION_TITLES.get(name, "")
        delta = float(row["delta_mae"]) if _known(row.get("delta_mae")) else None
        bar = ""
        if delta is not None:
            width = reach * abs(delta) / top
            start = zero - width if delta < 0 else zero
            bar = f'<span style="--zero:{zero:.0f}%"><i style="left:{start:.1f}%;width:{width:.1f}%"></i></span>'
        cells = [
            f'<th scope="row"><span class="cfg"><i>{_e(name)}</i><span>{_e(title)}</span></span></th>'
        ]
        if has_models:
            members = ", ".join(
                model_title(part) for part in str(row.get("models") or "").split(", ")
            )
            shown = f"Как в {seen[members]}" if members and members in seen else members
            seen.setdefault(members, name)
            cells.append(f'<td class="muted">{_e(shown)}</td>')
        cells.append(f'<td class="r">{_e(format_number(_number(row.get("mae")), 1))}</td>')
        cells.append(
            f'<td class="r"><div class="delta"><b>{_e(format_delta(delta))}</b>{bar}</div></td>'
        )
        if has_f1:
            cells.append(f'<td class="r">{_e(format_number(_number(row.get("f1")), 3))}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")
    head = '<th scope="col">Конфигурация</th>'
    head += '<th scope="col">Модели в наборе</th>' if has_models else ""
    head += '<th scope="col" class="r">MAE, ₽</th><th scope="col" class="r dm">∆MAE к A, ₽</th>'
    head += '<th scope="col" class="r">F1 обнаружения</th>' if has_f1 else ""
    return (
        '<div class="scroll-x"><table class="grid abl">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def _track(delta: float, top: float, direction: str) -> str:
    width = 50 * min(abs(delta) / top, 1.0)
    left = 50 - width if delta < 0 else 50
    return f'<span class="track"><i class="{direction}" style="left:{left:.1f}%;width:{width:.1f}%"></i></span>'


DIVERGING_HEAD = (
    '<div class="div-head"><span></span><span><span>← ошибка меньше</span><b>0</b>'
    "<span>ошибка больше →</span></span><span>∆MAE · окна · вывод</span></div>"
)


def scale_top(*groups: Sequence[Mapping[str, Any]]) -> float:
    """Общий предел шкалы для диаграмм вклада: наибольшая разность, округлённая вверх до 5."""
    top = max((abs(float(row["delta"])) for rows in groups for row in rows), default=0.0)
    return float(5 * math.ceil(top / 5)) or 5.0


def layers_html(rows: Sequence[Mapping[str, Any]], top: float) -> str:
    """Вклад слоёв решения: полоса влево — ошибка меньше, вправо — больше.

    Цветная полоса — знак разности держится по окнам проверки; штриховка — меняется от окна к
    окну.
    """
    lines = "".join(
        f'<div class="div-row"><div>{_e(row["subject"])}</div>'
        f"{_track(float(row['delta']), top, row['direction'])}{effect(row, wide=True)}</div>"
        for row in rows
    )
    label = "Изменение MAE от добавления каждого слоя решения"
    return f'{DIVERGING_HEAD}<div class="div-rows" role="img" aria-label="{label}">{lines}</div>'


def contribution_legend() -> str:
    return legend(
        [
            '<i class="sw better"></i>помог — ошибка меньше, и знак держится по окнам проверки',
            '<i class="sw worse"></i>навредил — ошибка больше, и знак держится',
            '<i class="sw none"></i>неясно — знак меняется от окна к окну',
            f"{ticks(6, 10)}окна, где знак тот же, что в среднем",
        ],
        boxed=True,
    )


def source_tiles(rows: Sequence[Mapping[str, Any]]) -> str:
    """Сколько сравнений «источник против модели без него» дали устойчивый результат."""
    total = len(rows)
    better = sum(1 for row in rows if row["direction"] == "better")
    worse = sum(1 for row in rows if row["direction"] == "worse")
    unclear = total - better - worse
    return (
        '<div class="tiles">'
        f'<div class="tile better"><b>{better}</b><span>сравнений: источник устойчиво помог</span></div>'
        f'<div class="tile worse"><b>{worse}</b><span>сравнений: устойчиво навредил</span></div>'
        f'<div class="tile none"><b>{unclear} из {total}</b><span>неясно: знак меняется от окна к окну</span></div>'
        "</div>"
    )


def sources_diagram(rows: Sequence[Mapping[str, Any]], top: float) -> str:
    """Вклад каждого источника по отдельности: строки сгруппированы по источнику."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("title") or row["subject"]), []).append(row)
    parts = []
    for title, members in groups.items():
        lines = "".join(
            f'<div class="div-row"><div>{_e(row.get("base") or "")}</div>'
            f"{_track(float(row['delta']), top, row['direction'])}{effect(row, wide=True)}</div>"
            for row in members
        )
        parts.append(f'<div class="div-group"><b>{_e(title)}</b>{lines}</div>')
    label = "Изменение MAE от добавления каждого источника по отдельности"
    return f'{DIVERGING_HEAD}<div class="div-rows" role="img" aria-label="{label}">{"".join(parts)}</div>'


# ── 6. Разборы ───────────────────────────────────────────────────────────────────────────────


def cases_legend(margin: int = 2) -> str:
    return legend(
        [
            '<i class="k-line"></i>ряд',
            '<i class="k-dia"></i>размеченный шок найден',
            '<i class="k-dia miss"></i>не найден',
            '<i class="k-tri"></i>тревога метода',
            '<i class="sw window"></i>окно допуска '
            f"±{counted(margin, 'месяц', 'месяца', 'месяцев')} вокруг тревоги",
        ],
        boxed=True,
    )


def tiles_html(items: Sequence[tuple[str, str, str]]) -> str:
    cells = "".join(
        f'<div class="tile{f" {cls}" if cls else ""}"><b>{_e(value)}</b><span>{_e(label)}</span></div>'
        for value, label, cls in items
    )
    return f'<div class="tiles wide">{cells}</div>'


def national_case_html(case: Mapping[str, Any]) -> str:
    """Национальный ряд: тревоги офлайн-метода, окна допуска и размеченные события."""
    events = list(case.get("events", []))
    found = sum(1 for event in events if event.get("detected"))
    margin = int(case.get("margin", 2))
    tiles = [
        (
            f"{found} из {len(events)}",
            f"размеченных шоков найдено в пределах ±{margin} месяцев",
            "",
        ),
        (format_number(int(case.get("alarms") or 0)), "тревог за весь ряд", ""),
    ]
    if case.get("coverage") is not None:
        tiles.append(
            (
                f"{round(100 * float(case['coverage']))}%",
                "ряда накрыто окнами допуска вокруг тревог",
                "warn",
            )
        )
    listed = "".join(
        f"<li><b>{number}</b><span>{_e(month_name(str(event['date'])))}</span>"
        f"<span>{_e(event['event'])}</span>"
        f"<em{' class="hit"' if event.get('detected') else ''}>{'найден' if event.get('detected') else 'не найден'}</em></li>"
        for number, event in enumerate(events, start=1)
    )
    return (
        tiles_html(tiles)
        + '<figure><div class="fig-cap"><span>Расходы, млрд руб. в месяц · цифры над ромбами — номера '
        f'событий из списка ниже</span></div><div class="scroll-x">{charts.case_chart(case)}</div></figure>'
        + f'<div class="events"><ol>{listed}</ol>'
        "<p>Метод работает задним числом, по всему ряду, и для сигнализации не подходит. Шок "
        f"считается найденным, если тревога стоит не дальше {counted(margin, 'месяца', 'месяцев', 'месяцев')} "
        "от его даты.</p></div>"
    )


def municipal_case_html(case: Mapping[str, Any], history: int | None = None) -> str:
    """Ряд МО с крупнейшим остатком: факт, прогноз ансамбля, интервал и тревоги детектора.

    `history` — длина панели в месяцах: столько точек истории у каждого ряда.
    """
    months = list(case["ds"])
    alarms = [m for m in dict.fromkeys(str(a)[:7] for a in case.get("alarms", [])) if m in months]
    tiles = [
        (f"{len(alarms)} из {len(months)}", "месяцев с тревогой детектора", "warn"),
        (
            f"|z| = {format_number(case.get('max_abs_z'), 1)}",
            "крупнейший остаток прогноза в потоке живого детектора",
            "",
        ),
    ]
    if history:
        noun = counted(history, "месяц", "месяца", "месяцев").split(" ", 1)[1]
        tiles.append((str(history), f"{noun} истории у ряда", ""))
    unit = charts.series_unit(case)
    keys = legend([*SERIES_KEYS, '<i class="k-tri"></i>тревога детектора'])
    return (
        tiles_html(tiles)
        + f'<figure><div class="fig-cap"><span>{_e(cap(str(case.get("category", ""))))}, {unit} в месяц</span>'
        f'{keys}</div><div class="scroll-x">'
        f"{charts.series_chart(case, size=charts.WIDE_SIZE, alarms=alarms, every_month=True)}</div></figure>"
    )


# ── 7. Справка ───────────────────────────────────────────────────────────────────────────────


def limits_html(items: Sequence[Any]) -> str:
    """Ограничения карточками: номер, заголовок, область и текст."""
    cards = []
    for number, item in enumerate(items, start=1):
        if isinstance(item, Mapping):
            title, area, text = str(item["title"]), str(item.get("area", "")), str(item["text"])
            # Заголовок не повторяется в тексте: формулировка вклада начинается с названия слоя.
            if text.startswith(f"{title}: "):
                text = cap(text[len(title) + 2 :])
            head = f"<h3>{_e(title)}</h3>" + (f"<em>{_e(area)}</em>" if area else "")
        else:
            head, text = "", str(item)
        cards.append(
            f'<article class="limit"><div><i>{number:02d}</i>{head}</div><p>{inline_html(text)}</p></article>'
        )
    return f'<div class="limits">{"".join(cards)}</div>'


def criteria_html(rows: Sequence[Mapping[str, Any]]) -> str:
    """Критерий конкурса → разделы страницы → раздел отчёта → артефакт."""
    top = max((int(row["weight"]) for row in rows), default=1)
    body = []
    for row in rows:
        links = ", ".join(
            f'<a href="{_e(link["href"])}">{_e(link["text"])}</a>' for link in row["links"]
        )
        code = f"<i>{_e(row['code'])}</i> " if row.get("code") else ""
        weight = int(row["weight"])
        body.append(
            f'<tr><th scope="row">{code}{_e(row["criterion"])}</th>'
            f'<td><span class="weight">{weight}%<span><i style="width:{100 * weight / top:.0f}%"></i></span></span></td>'
            f"<td>{links or '—'}</td><td>{_e(row['report'])}</td>"
            f"<td><code>{_e(row['artifact'])}</code></td></tr>"
        )
    head = "".join(
        f'<th scope="col">{name}</th>'
        for name in ("Критерий", "Вес", "На этой странице", "В отчёте", "Артефакт")
    )
    return (
        '<div class="scroll-x"><table class="grid criteria">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def commands_html() -> str:
    """Команды воспроизведения с кнопками копирования."""
    commands = (
        ("Всё решение одной командой", "make all"),
        ("То же в контейнере", "docker compose run --rm pipeline"),
        ("Собрать эту страницу", "make report"),
    )
    boxes = "".join(
        f'<div><small>{_e(title)}</small><div class="cmd"><code>{_e(command)}</code>'
        f'<button type="button" data-copy aria-label="Скопировать команду {_e(command)}">{COPY_ICON}</button></div></div>'
        for title, command in commands
    )
    return (
        f'<div class="cmds">{boxes}</div>'
        '<p class="p15" style="margin-top:14px">Шаги по отдельности перечислены в '
        "<code>README.md</code>. Условия использования всех данных — в файле "
        "<code>DATA_LICENSES.md</code>.</p>"
    )


def citations_html(lines: Sequence[str]) -> str:
    """Обязательные подписи источников нумерованным списком; адрес в подписи — ссылка."""
    items = "".join(
        f"<li><i>[{number}]</i><span>{linked(line)}</span></li>"
        for number, line in enumerate(lines, start=1)
    )
    return f'<ol class="refs">{items}</ol>'
