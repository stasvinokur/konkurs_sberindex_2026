"""HTML-блоки лендинга: первый экран, данные и «как считали».

Каждая функция получает готовую структуру из `sbx.core.storyline` или данные отчёта и возвращает
разметку раздела. Оформление — классы из `assets/landing.css`; цвета в разметке не задаются.
Диаграммы из полос и столбцов собраны из HTML и CSS-сетки, линии — встроенным SVG
(`sbx.shell.render.charts`): текст остаётся текстом страницы, графики меняются вместе с темой,
внешних библиотек не нужно. Число рядом с меткой всегда продублировано подписью.
"""

from __future__ import annotations

import html
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from sbx.core.report import (
    DETECTOR_NAMES,
    counted,
    detector_headline,
    format_delta,
    format_number,
    horizon_label,
    identification_facts,
    minority_note,
    model_title,
    reference_wins,
    schedule_brief,
)
from sbx.core.storyline import MONTH_NAMES_GENITIVE, month_label
from sbx.shell.render import charts

REFERENCE = "Prophet"
KIND_TITLES = {
    "key_rate_hike": "повышение ключевой ставки",
    "key_rate_cut": "снижение ключевой ставки",
    "flood": "паводок",
    "emergency_evacuation": "эвакуация",
    "local_disruption": "локальное нарушение",
    "macro_shock": "макрошок",
    "pandemic_restrictions": "ограничения пандемии",
}
VERDICTS = {"better": "помог", "worse": "навредил", "none": "неясно"}
LOGO = (
    '<svg width="26" height="26" viewBox="0 0 32 32" aria-hidden="true">'
    '<rect class="logo-bg" width="32" height="32" rx="7"/>'
    '<polyline class="logo-line" points="6,22 11,17 15,19.5 20,12.5"/>'
    '<circle class="logo-dot" cx="25" cy="9" r="3.25"/></svg>'
)


def _e(value: Any) -> str:
    return html.escape(str(value))


def cap(text: str) -> str:
    """Первая буква — прописная: формулировки ядра начинаются со строчной."""
    return text[:1].upper() + text[1:]


def month_name(month: str) -> str:
    """«сентябрь 2024» из «2024-09»."""
    return month_label(pd.Timestamp(f"{str(month)[:7]}-01"))


def month_genitive(month: str) -> str:
    """«сентября 2024 года» из «2024-09» — для оборота «в конце …»."""
    return f"{MONTH_NAMES_GENITIVE[int(str(month)[5:7]) - 1]} {str(month)[:4]} года"


def inline_html(text: str) -> str:
    """Текст абзаца: экранирование, а фрагмент в обратных кавычках — как код."""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", _e(text))


def linked(text: str) -> str:
    """Подпись источника: адрес становится ссылкой, остальное экранируется."""
    return re.sub(
        r"https?://[^\s()<>&]+",
        lambda match: f'<a href="{match.group(0)}">{match.group(0)}</a>',
        _e(text),
    )


def embedded_json(data: Any) -> str:
    """Данные для сценария страницы: JSON без NaN, безопасный внутри `<script>`."""

    def clean(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, Mapping):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [clean(item) for item in value]
        return value

    text = json.dumps(clean(data), ensure_ascii=False, separators=(",", ":"))
    return f'<script type="application/json">{text.replace("</", "<\\/")}</script>'


def ticks(agree: int, total: int, cls: str = "") -> str:
    """Отметки окон проверки: закрашено столько, в скольких окнах знак тот же, что в среднем."""
    return (
        f'<i class="ticks{cls}" style="--n:{int(agree)};--of:{int(total)}" role="img" '
        f'aria-label="{int(agree)} из {int(total)}"></i>'
    )


def verdict(direction: str) -> str:
    return f'<span class="verdict {direction}">{VERDICTS.get(direction, direction)}</span>'


def effect(row: Mapping[str, Any], wide: bool = False) -> str:
    """Изменение MAE, счёт согласных окон и вывод — одной строкой."""
    value = f"<b{' class="wide"' if wide else ''}>{_e(format_delta(row['delta']))} ₽</b>"
    if row.get("agree") is None or not row.get("folds"):
        return f'<div class="effect">{value}<span class="cnt">по окнам не сверено</span></div>'
    return (
        f'<div class="effect">{value}{ticks(row["agree"], row["folds"])}'
        f'<span class="cnt">{row["agree"]} из {row["folds"]}</span>'
        f"{verdict(row['direction'])}</div>"
    )


def conclusion(text: str) -> str:
    return f'<div class="concl"><span>Вывод</span><p>{inline_html(text)}</p></div>' if text else ""


def provenance(data: str, method: str, report: str, config: str) -> str:
    """Происхождение результата: на каких данных, каким методом и где подробности."""
    return (
        '<dl class="prov">'
        f"<div><dt>Данные</dt><dd>{_e(cap(data))}</dd></div>"
        f"<div><dt>Метод</dt><dd>{_e(cap(method))}</dd></div>"
        f"<div><dt>Подробнее</dt><dd>Отчёт, {_e(report)} · <code>{_e(config)}</code></dd></div>"
        "</dl>"
    )


def legend(items: Sequence[str], boxed: bool = False) -> str:
    cls = "legend boxed" if boxed else "legend"
    return (
        f'<div class="{cls}">{"".join(f'<span class="key">{item}</span>' for item in items)}</div>'
    )


# ── Первый экран: цепочка «данные → расчёт → результат» ──────────────────────────────────────


def _chain_head(title: str, arrow: bool) -> str:
    number, _, name = title.partition(". ")
    tail = '<span class="chain-arrow" aria-hidden="true"></span>' if arrow else ""
    return f'<div class="chain-head"><i>{_e(number)}</i><h2>{_e(name)}</h2>{tail}</div>'


def _result_table(winners: Sequence[Mapping[str, Any]], significance: Mapping[str, Any]) -> str:
    wins = reference_wins(significance, REFERENCE)
    rows = []
    for winner in winners:
        horizon = int(winner["horizon"])
        windows = ""
        if horizon in wins:
            better, folds = wins[horizon]
            windows = f'<span class="wins">{ticks(better, folds, " big")}{better} из {folds}</span>'
        reference = winner.get("reference_mae")
        rows.append(
            f'<tr><td>{horizon} мес.</td><td class="model">{_e(model_title(winner["model"]))}</td>'
            f'<td class="best">{_e(format_number(winner["mae"], 1))}</td>'
            f'<td class="ref">{_e(format_number(reference, 1))}</td>'
            + (f"<td>{windows}</td>" if wins else "")
            + "</tr>"
        )
    head = (
        '<th>Горизонт</th><th>Лучшая модель</th><th class="r">MAE, ₽</th>'
        f'<th class="r">{REFERENCE}, ₽</th>' + (f"<th>Точнее {REFERENCE}</th>" if wins else "")
    )
    return (
        '<div class="scroll-x"><table class="hero-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _detector_tiles(contrast: Mapping[str, Any]) -> str:
    def values(f1: Any, delay: Any) -> str:
        out = f"<span><b>{_e(format_number(f1, 3))}</b> <small>F1</small></span>"
        if delay is not None:
            out += f"<span><b>{_e(format_number(delay, 1))}</b> <small>мес. задержка</small></span>"
        return f'<div class="duo-vals">{out}</div>'

    tiles = [
        '<div><div class="duo-name">Детектор по остаткам прогноза</div>'
        f"{values(contrast['f1_residual'], contrast.get('delay_residual'))}</div>"
    ]
    reference = contrast.get("reference")
    if reference:
        tiles.append(
            '<div class="ref"><div class="duo-name">Тревога по расписанию '
            '<span class="tag-ref">эталон</span></div>'
            f"{values(reference['f1'], reference.get('delay'))}</div>"
        )
    return f'<div class="duo">{"".join(tiles)}</div>'


def chain_html(
    chain: Mapping[str, Mapping[str, Any]], meta: Mapping[str, Any], note: str = ""
) -> str:
    """Три колонки «Данные → Расчёт → Результат»; каждая строка — ссылка на раздел.

    Колонка результата — не пересказ, а сами числа: лучшая модель каждого горизонта рядом с
    эталоном и счётом окон, детектор рядом с тревогой по расписанию.
    """
    columns = []
    data = chain.get("data")
    if data:
        rows = "".join(
            f'<a href="{_e(line["href"])}"><span>{_e(line["label"])}</span>'
            f"<span>{_e(line['value'])}</span></a>"
            for line in data["lines"]
            if line.get("value")
        )
        more = "".join(
            f'<a class="chain-more" href="{_e(line["href"])}">{_e(line["label"])} →</a>'
            for line in data["lines"]
            if not line.get("value")
        )
        columns.append(
            f'<div class="chain-col data">{_chain_head(data["title"], True)}'
            '<p class="chain-sub">Какие источники взяты и как согласованы по времени</p>'
            f'<div class="chain-rows">{rows}</div>'
            + (f'<p class="chain-note">{_e(note)}</p>' if note else "")
            + f"{more}</div>"
        )
    method = chain.get("method")
    if method:
        rows = "".join(
            f'<a href="{_e(line["href"])}"><b>{_e(line["label"])}</b>'
            f"<span>{_e(cap(line['value']))}</span></a>"
            for line in method["lines"]
        )
        columns.append(
            f'<div class="chain-col method">{_chain_head(method["title"], True)}'
            '<p class="chain-sub">Чем проверено</p>'
            f'<div class="chain-rows kv">{rows}</div></div>'
        )
    result = chain.get("result")
    if result:
        columns.append(_result_column(result, meta))
    return (
        '<section id="chain" class="chain" aria-label="Данные, расчёт, результат">'
        + "".join(columns)
        + "</section>"
    )


def _result_column(result: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
    lines = list(result["lines"])
    winners = meta.get("winners") or []
    contrast = meta.get("detector_contrast") or {}
    by_target: dict[str, list[Mapping[str, Any]]] = {}
    for line in lines:
        by_target.setdefault(line["href"], []).append(line)
    parts = [_chain_head(result["title"], False)]
    forecast = by_target.pop("#horizons", [])
    if forecast and winners:
        varies = len({str(w["model"]) for w in winners}) > 1
        title = "лучшая модель зависит от горизонта" if varies else "лучшая модель по горизонтам"
        note = f"{forecast[0]['text']}."
        extra = minority_note(meta.get("significance") or {}, REFERENCE)
        parts.append(
            f'<a class="cap" href="#horizons">Прогноз · {title}</a>'
            + _result_table(winners, meta.get("significance") or {})
            + f'<p class="p14">{_e(note)}{f" {_e(extra)}" if extra else ""}</p>'
        )
    elif forecast:
        parts += [
            f'<p class="p14"><a href="#horizons">{_e(line["text"])}</a></p>' for line in forecast
        ]
    shifts = by_target.pop("#changepoints", [])
    if shifts and contrast:
        brief = schedule_brief(contrast, meta.get("schedule_match") or {})
        parts.append(
            '<div class="hero-part"><a class="cap" href="#changepoints">Сдвиги · ищем в ошибках '
            "прогноза</a>"
            + _detector_tiles(contrast)
            + f'<p class="p14">{_e(brief or detector_headline(contrast))}</p></div>'
        )
    elif shifts:
        parts += [
            f'<p class="p14"><a href="#changepoints">{_e(line["text"])}</a></p>' for line in shifts
        ]
    links = "".join(
        f'<a href="{_e(line["href"])}">{_e(line["text"])}</a>'
        for group in by_target.values()
        for line in group
    )
    if links:
        parts.append(f'<div class="hero-links">{links}</div>')
    return f'<div class="chain-col result">{"".join(parts)}</div>'


# ── 1. Данные ────────────────────────────────────────────────────────────────────────────────


def sources_legend() -> str:
    return legend(
        [
            "<span><b>Что дал</b> — изменение MAE, ₽; минус — ошибка стала меньше</span>",
            f"{ticks(6, 10)}<span>окна проверки, где знак тот же, что в среднем</span>",
            f"{verdict('better')}{verdict('worse')}{verdict('none')}"
            "<span>устойчиво лучше · устойчиво хуже · знак меняется от окна к окну</span>",
        ],
        boxed=True,
    )


def _gave_cell(row: Mapping[str, Any]) -> str:
    """Столбец «Что дал»: измеренный вклад источника или его роль, если вклад не измеряется."""
    effects = row.get("effects") or []
    if effects:
        return (
            '<div class="effects">'
            + "".join(
                f"<div><small>{_e(item['base'])}</small>{effect(item)}</div>" for item in effects
            )
            + "</div>"
        )
    if row.get("key") == "spending":
        return f'<span class="dark-pill">{_e(row["gave"])}</span>'
    return f'<div class="src-plain">{_e(cap(row["gave"]))}</div>'


def sources_html(rows: Sequence[Mapping[str, Any]]) -> str:
    """Таблица источников: одна строка — один источник, строки сгруппированы по уровню."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("group") or row.get("level") or ""), []).append(row)
    bodies = []
    for name, members in groups.items():
        lines = [f'<tr class="lvl"><th colspan="5" scope="colgroup">{_e(name)}</th></tr>']
        for row in members:
            provider = (
                f'<div class="src-by">{_e(row["provider"])}</div>' if row.get("provider") else ""
            )
            note = f'<div class="src-note">{_e(row["note"])}</div>' if row.get("note") else ""
            lines.append(
                f'<tr><td><div class="src-name">{_e(row.get("title") or row["source"])}</div>{provider}'
                f'<div class="src-lic">{_e(row["license"])}</div></td>'
                f"<td>{_e(cap(row['what']))}</td><td>{_e(row['volume'])}</td>"
                f'<td><span class="chip">{_e(row.get("tag") or row["timing"])}</span>{note}</td>'
                f"<td>{_gave_cell(row)}</td></tr>"
            )
        bodies.append(f"<tbody>{''.join(lines)}</tbody>")
    head = (
        "<th>Источник</th><th>Что берём</th><th>Объём и период</th><th>Согласование по времени</th>"
        "<th>Что дал · изменение MAE, окна, вывод</th>"
    )
    return (
        '<div class="scroll-x"><table class="grid sources">'
        f"<thead><tr>{head}</tr></thead>{''.join(bodies)}</table></div>"
    )


def territories_html(entity: Mapping[str, Any], summary: Sequence[str] = ()) -> str:
    """Сверка идентификации: сколько рядов попало бы в чужой регион при сопоставлении по названиям."""
    facts = identification_facts(entity)
    if not facts:
        return "".join(f'<p class="note">{inline_html(line)}</p>' for line in summary)
    wrong = facts.get("wrong")
    if wrong:
        left = (
            '<div><div class="cap">Если бы сопоставляли по названиям</div>'
            f'<div class="big"><b>{_e(format_number(wrong["series"]))}</b>'
            f"<span>рядов из {_e(format_number(wrong['of']))} · "
            f"{_e(format_number(wrong['share'], 1))}%</span></div>"
            f'<p class="p17">отнесены к чужому региону: {_e(wrong["spread"])}</p></div>'
        )
    else:
        left = (
            '<div><div class="cap">Сверка с идентификацией по названиям</div>'
            f'<p class="p17">{_e(facts.get("matched", facts["panel"]))}</p></div>'
        )
    pairs = [("Панель", cap(facts["panel"])), ("Почему не по названиям", facts["same_names"])]
    if facts.get("check"):
        pairs.append(("Как проверили", facts["check"]))
    right = "".join(
        f"<div><dt>{_e(name)}</dt><dd>{inline_html(text)}</dd></div>" for name, text in pairs
    )
    return f'<div class="panel soft">{left}<dl class="facts">{right}</dl></div>'


def timing_steps(max_lag: int | None) -> str:
    """Четыре шага согласования: событие, публикация, момент прогноза, целевые месяцы."""
    publication = "Конец периода плюс лаг источника"
    if max_lag:
        publication += f": от 0 до {max_lag} дней"
    steps = [
        ("", "Событие", "Произошло в отчётном периоде источника"),
        ("", "Публикация", publication),
        (
            "now",
            "Момент прогноза",
            "Конец последнего месяца: берём только опубликованное к этой дате",
        ),
        (
            "after",
            "Целевые месяцы",
            "На всех шагах — одни и те же значения; на целевой месяц — только производственный "
            "календарь",
        ),
    ]
    items = "".join(
        f"<li{f' class="{cls}"' if cls else ''}><div><i>{number:02d}</i><b>{_e(name)}</b>"
        f"<span>{_e(text)}</span></div></li>"
        for number, (cls, name, text) in enumerate(steps, start=1)
    )
    return f'<ol class="steps">{items}</ol>'


def sees_html(timing: Mapping[str, Any], before: int = 4, after: int = 3) -> str:
    """Что видит прогноз от заданного месяца: по полосе на источник.

    Закрашенная полоса — месяцы, известные на момент прогноза; штриховка — период прошёл, но
    данные ещё не опубликованы. Чем больше лаг публикации, тем раньше обрывается полоса.
    """
    origin = pd.Period(timing["origin"], "M")
    months = [origin + shift for shift in range(-before, after + 1)]
    now = before + 2  # первый целевой месяц: столбцы сетки считаются с единицы, первый — источник
    last = len(months) + 2
    day = origin.to_timestamp(how="end")
    flag = f"Момент прогноза · {day.day} {MONTH_NAMES_GENITIVE[day.month - 1]} {day.year}"
    head = "".join(
        f'<span class="sees-m{" target" if i > before else ""}" style="grid-column:{i + 2}">'
        f"{charts.MONTHS_SHORT[month.month - 1]}</span>"
        for i, month in enumerate(months)
    )
    lines = [
        f'<div class="sees-row head flag"><span class="sees-now">{_e(flag)}</span></div>',
        '<div class="sees-row head"><span class="sees-now"></span>'
        f'<div class="sees-src">Источник · лаг публикации</div>{head}'
        '<div class="sees-known">Последний известный период</div></div>',
    ]
    for row in timing["rows"]:
        kind = row.get("kind", "published")
        if kind == "ahead":
            bars = f'<i class="bar-known" style="grid-column:2 / {last}"></i>'
        elif kind == "static":
            bars = f'<i class="bar-static" style="grid-column:2 / {last}"></i>'
        else:
            known = pd.Period(row["known_month"], "M") if row.get("known_month") else origin
            through = min((known - months[0]).n, before)  # индекс последнего известного месяца
            bars = ""
            if through >= 0:
                bars += f'<i class="bar-known" style="grid-column:2 / {through + 3}"></i>'
            if through < before:
                bars += f'<i class="bar-wait" style="grid-column:{max(through, -1) + 3} / {now + 1}"></i>'
        known_cls = " ahead" if kind == "ahead" else ""
        lines.append(
            '<div class="sees-row"><span class="sees-now"></span>'
            f'<div class="sees-src">{_e(row["source"])} '
            f'<span class="chip small">{_e(row.get("tag") or row["lag"])}</span></div>{bars}'
            f'<div class="sees-known{known_cls}">{_e(row["known"])}</div></div>'
        )
    label = (
        f"Что известно прогнозу, сделанному в конце месяца {month_name(str(origin))}, по источникам"
    )
    keys = legend(
        [
            '<i class="bar-known"></i>известно на момент прогноза',
            '<i class="bar-wait"></i>период прошёл, данные ещё не опубликованы',
            '<i class="bar-static"></i>постоянная характеристика',
            '<i class="now-mark"></i>целевые месяцы прогноза',
        ]
    )
    return (
        f'<div class="scroll-x"><div class="sees" role="img" aria-label="{_e(label)}" '
        f'style="--cols:{len(months)};--now:{now}">{"".join(lines)}</div></div>{keys}'
    )


def _day(date: str) -> str:
    day = pd.Timestamp(date)
    return f"{day.day:02d}.{day.month:02d}.{day.year}"


def news_html(timeline: Mapping[str, Any]) -> str:
    """Число новостей по месяцам столбцами; под столбцом — ромб, если в месяце были события
    календаря, и их число, если событий несколько."""
    months, counts = timeline["months"], timeline["counts"]
    top = max(counts) or 1
    by_month: dict[str, list[Mapping[str, Any]]] = {}
    for event in timeline["events"]:
        by_month.setdefault(event["month"], []).append(event)
    bars, marks = [], []
    for month, count in zip(months, counts, strict=True):
        tip = f"{month_name(month)}: {format_number(count)} событий"
        peak = ' class="top"' if count == top else ""
        bars.append(f'<i style="height:{100 * count / top:.1f}%"{peak} title="{_e(tip)}"></i>')
        happened = by_month.get(month, [])
        if happened:
            names = "; ".join(str(e["description"]) for e in happened)
            several = f"<small>×{len(happened)}</small>" if len(happened) > 1 else ""
            marks.append(
                f'<span title="{_e(month_name(month) + ": " + names)}"><b></b>{several}</span>'
            )
        else:
            marks.append("<span></span>")
    axis = []
    for i, month in enumerate(months):
        if i == 0:
            axis.append(f'<span style="left:0">{charts.month_tick(month, True)}</span>')
        elif i == len(months) - 1:
            axis.append(f'<span style="right:0">{charts.month_tick(month)}</span>')
        elif month[5:7] in ("01", "07"):
            at = 100 * (i + 0.5) / len(months)
            text = charts.month_tick(month, month.endswith("-01"))
            axis.append(f'<span style="left:{at:.1f}%;transform:translateX(-50%)">{text}</span>')
    events = "".join(
        f"<li><b>{_e(_day(event['date']))}</b> — {_e(event['scope'])}: "
        f"{_e(KIND_TITLES.get(event['kind'], event['kind']))}. {_e(event['description'])}</li>"
        for event in timeline["events"]
    )
    label = (
        f"Число новостей о России по месяцам, {month_name(months[0])} — {month_name(months[-1])}; "
        f"максимум {format_number(top)} событий в месяц"
    )
    return (
        '<div class="news">'
        f'<div class="news-bars" role="img" aria-label="{_e(label)}">'
        f'<span class="news-max">{_e(format_number(top))}</span>{"".join(bars)}</div>'
        f'<div class="news-marks">{"".join(marks)}</div>'
        f'<div class="news-axis">{"".join(axis)}</div></div>'
        f'<details class="more"><summary>События календаря за этот период: {len(timeline["events"])}'
        f"</summary><ul>{events}</ul></details>"
    )


# ── 2. Как считали ───────────────────────────────────────────────────────────────────────────


def _node(
    title: str,
    text: str,
    column: int,
    cls: str = "",
    right: bool = False,
    down: bool = False,
    note: str = "",
) -> str:
    arrows = ('<i class="arr-r"></i>' if right else "") + ('<i class="arr-d"></i>' if down else "")
    hint = f'<span class="to-note">{_e(note)}</span>' if note else ""
    return (
        f'<div class="node{f" {cls}" if cls else ""}" style="grid-column:{column}">'
        f"<b>{_e(title)}</b>{_e(text)}{hint}{arrows}</div>"
    )


def scheme_html(steps: Mapping[str, str]) -> str:
    """Схема решения: основная цепочка прогноза, боковой вход внешних данных и ветка детектора.

    На узком экране узлы идут сверху вниз; куда уходит боковая ветка, там сказано словами.
    """
    side = (
        _node("Внешние источники", steps["sources"], 1, "side-in", right=True)
        + _node("Согласование по времени", steps["timing"], 2, "side-in", right=True)
        + _node(
            "Внешние признаки", steps["features"], 3, "side-in", down=True, note="↓ входят в модели"
        )
    )
    main = (
        _node("Расходы МО", steps["spending"], 1, right=True)
        + _node("Панель", steps["panel"], 2, right=True)
        + _node("Модели", steps["models"], 3, right=True)
        + _node("Ансамбль", steps["ensemble"], 4, right=True, down=True)
        + _node("Прогноз", steps["forecast"], 5, "out")
    )
    branch = (
        '<p class="scheme-foot"><b>Сдвиги ищем в ошибках прогноза,</b> а не в самом ряде: '
        "сезонный подъём прогноз объясняет сам, и тревогу он не вызывает.</p>"
        + _node("Остатки прогноза", steps["residuals"], 4, right=True)
        + _node("Детектор сдвигов", steps["detector"], 5, "out")
    )
    return (
        '<div class="scheme">'
        '<p class="cap">Боковой вход · внешние данные</p>'
        f'<div class="scheme-row">{side}</div>'
        '<p class="cap">Основная цепочка · прогноз</p>'
        f'<div class="scheme-row">{main}</div>'
        '<p class="cap branch">Ветка от ансамбля · сдвиги</p>'
        f'<div class="scheme-row">{branch}</div></div>'
    )


def architecture_html() -> str:
    """Ядро и оболочка: что где лежит в коде."""
    return (
        '<p class="note">Код разделён на ядро и оболочку — Functional Core / Imperative Shell:</p>'
        '<div class="pair">'
        "<div><b>Ядро</b><span>чистые функции, без чтения файлов и сети</span>"
        "<p>Окна проверки, признаки, метрики, детекторы сдвигов и веса ансамбля</p></div>"
        "<div><b>Оболочка</b><span>всё, что касается внешнего мира</span>"
        "<p>Загрузка данных, адаптеры моделей, командная строка и сборка этой страницы</p></div>"
        "</div>"
    )


def metric_html(where: str) -> str:
    """Определение MAE и то, что нужно знать, чтобы читать таблицы результатов."""
    formula = (
        '<div class="formula" role="img" aria-label="MAE равна сумме модулей разностей факта и '
        'прогноза, делённой на число наблюдений">'
        '<span>MAE</span><span class="eq">=</span>'
        '<span class="frac"><span>1</span><span>N</span></span><span class="sum">∑</span>'
        "<span>| факт<sub>i</sub> − прогноз<sub>i</sub> |</span></div>"
    )
    pairs = [
        (
            "Окно проверки",
            "Пара «последний месяц обучения, горизонт»: модель видит ряды по этот месяц и "
            "прогнозирует следующие месяцы.",
        ),
    ]
    if where:
        pairs.append(("Где сравниваем", where))
    facts = "".join(f"<div><dt>{_e(name)}</dt><dd>{_e(text)}</dd></div>" for name, text in pairs)
    facts += (
        "<div><dt>Счёт окон</dt><dd>В скольких окнах проверки одна модель точнее другой: "
        "показывает, держится ли знак разности от окна к окну. "
        f'<span class="inline-ticks">{ticks(3, 4, " big")} 3 из 4</span></dd></div>'
    )
    return (
        '<div class="panel framed"><div><div class="cap">Метрика конкурса</div>'
        f"{formula}"
        '<p class="p17">MAE — средний модуль разности факта и прогноза по всем парам «ряд × '
        "месяц» проверки, в рублях; ниже — лучше.</p></div>"
        f'<dl class="facts">{facts}</dl></div>'
    )


def folds_html(timeline: Mapping[str, Any], common: Mapping[int, int] | None = None) -> str:
    """Окна проверки на оси месяцев: полоса обучения и полоса проверки у каждого окна.

    `common` — горизонт → число окон, на которых сравниваются все модели: если их меньше, чем
    окон горизонта, это сказано в заголовке группы.
    """
    months = timeline["months"]
    years: list[tuple[str, int, int]] = []
    for position, month in enumerate(months):
        if years and years[-1][0] == month[:4]:
            years[-1] = (month[:4], years[-1][1], years[-1][2] + 1)
        else:
            years.append((month[:4], position, 1))
    head = "".join(
        f'<b style="grid-column:{start + 2} / span {length}">{_e(year)}</b>'
        for year, start, length in years
    )
    marks = "".join(
        f'<span style="grid-column:{i + 2} / span 3">{charts.month_tick(month)}</span>'
        for i, month in enumerate(months)
        if month[5:7] in ("01", "04", "07", "10") and i + 3 <= len(months)
    )
    groups = {int(group["horizon"]): group for group in timeline["horizons"]}
    lines = [
        f'<div class="folds-row years"><span>Последний месяц обучения</span>{head}</div>',
        f'<div class="folds-row months">{marks}</div>',
    ]
    current = None
    for row in timeline["rows"]:
        horizon = int(row["horizon"])
        if horizon != current:
            current = horizon
            group = groups[horizon]
            low, high = group["train_months"]
            train = f"{low}–{high}" if low != high else f"{low}"
            summary = (
                f"{counted(group['folds'], 'окно', 'окна', 'окон')} · обучение {train} мес. · "
                f"проверка {group['test_months']} мес."
            )
            shared = (common or {}).get(horizon)
            if shared is not None and shared < int(group["folds"]):
                summary += f" · общих для всех моделей — {shared}"
            lines.append(
                f'<div class="folds-group"><b>Горизонт {_e(horizon_label(horizon))}</b>'
                f"<span>{_e(summary)}</span></div>"
            )
        train_from, train_to = row["train"]
        test_from, test_to = row["test"]
        learned = f"обучение: {month_name(months[train_from])} — {month_name(months[train_to])}"
        checked = (
            f"проверка: {month_name(months[test_from])}"
            if test_from == test_to
            else f"проверка: {month_name(months[test_from])} — {month_name(months[test_to])}"
        )
        lines.append(
            f'<div class="folds-row window"><span class="cut">{_e(charts.month_tick(row["cutoff"], True))}</span>'
            f'<i class="train" style="grid-column:{train_from + 2} / {train_to + 3}" title="{_e(learned)}"></i>'
            f'<i class="test" style="grid-column:{test_from + 2} / {test_to + 3}" title="{_e(checked)}"></i></div>'
        )
    label = f"Окна проверки: {len(timeline['rows'])} окон на оси {len(months)} месяцев панели"
    keys = legend(
        [
            '<i class="sw train"></i>обучение',
            '<i class="sw test"></i>проверка: с этими месяцами сравнивают прогноз',
        ]
    )
    return (
        f'<div class="scroll-x"><div class="folds" role="img" aria-label="{_e(label)}" '
        f'style="--m:{len(months)}">{"".join(lines)}</div></div>{keys}'
    )


def detector_demo_html(demo: Mapping[str, Any]) -> str:
    """Пример работы детектора: три панели на одном ряду и значения выбранного месяца."""
    name = DETECTOR_NAMES.get(str(demo["detector"]), str(demo["detector"]))
    alarm = int(demo["alarms"][0])
    months = list(demo["months"])
    delay = int(demo["delay"])
    when = (
        "в том же месяце"
        if delay == 0
        else f"«{month_name(months[alarm])}», через {counted(delay, 'месяц', 'месяца', 'месяцев')} "
        "после слома"
    )

    def value(key: str, digits: int, unit: str = "") -> str:
        number = demo[key][alarm]
        return "—" if number is None else f"{format_number(number, digits)}{unit}"

    payload = {key: demo[key] for key in ("months", "y", "forecast", "z", "statistic")}
    payload["alarms"] = [int(a) for a in demo["alarms"]]
    keys = legend(
        [
            '<i class="k-line"></i>факт и ошибка прогноза',
            '<i class="k-line acc"></i>прогноз на шаг вперёд и статистика детектора',
            '<i class="k-dia"></i>внесённый слом',
            '<i class="k-tri"></i>тревога детектора',
            '<i class="k-line thr"></i>порог',
        ]
    )
    panel = (
        '<aside class="aside-card" aria-live="polite">'
        '<div class="cap">Выбранный месяц</div>'
        f'<div class="month" data-f="month">{_e(cap(month_name(months[alarm])))}</div>'
        '<dl class="kv">'
        f'<div><dt>Факт</dt><dd class="b" data-f="y">{_e(value("y", 0, " ₽"))}</dd></div>'
        f'<div><dt>Прогноз на шаг вперёд</dt><dd class="acc" data-f="forecast">{_e(value("forecast", 0, " ₽"))}</dd></div>'
        f'<div><dt>Ошибка, в обычных ошибках</dt><dd data-f="z">{_e(value("z", 1))}</dd></div>'
        f'<div><dt>Статистика {_e(name)}</dt><dd data-f="statistic">{_e(value("statistic", 1))}</dd></div>'
        "</dl>"
        '<div class="alarm-note" data-f="alarm"><i class="k-tri"></i>Тревога детектора в этом месяце</div>'
        f'<p class="p13 marked"><i class="k-dia"></i>Слом внесён в месяце «{_e(month_name(demo["break_month"]))}»: '
        f"уровень ряда изменён на {_e(format_number(100 * demo['delta'], 0))}%.</p>"
        f'<p class="p13 marked"><i class="k-tri"></i>Тревога — {_e(when)}: статистика '
        f"{_e(format_number(demo['statistic'][alarm], 1))} при пороге "
        f"{_e(format_number(demo['threshold'], 1))}.</p>"
        '<p class="p13">До слома статистика остаётся ниже порога: обычные ошибки прогноза тревоги '
        "не вызывают.</p>"
        '<p class="p13 dim">Наведите на график или выберите месяц стрелками ← →, чтобы посмотреть '
        "другие значения.</p></aside>"
    )
    return (
        f'{keys}<div class="split" data-demo>{embedded_json(payload)}'
        f'<figure><div class="scroll-x">{charts.demo_chart(demo)}</div></figure>{panel}</div>'
    )
