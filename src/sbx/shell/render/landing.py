"""Статический лендинг с итогами решения: одна страница без внешних ресурсов.

Страница собирается из данных отчёта (`report_data.json`): тексты выводов — формулировки ядра,
числа — из артефактов. Оформление, шрифты и сценарий встроены в файл (`sbx.shell.render.assets`),
графики — встроенный SVG (`sbx.shell.render.charts`), поэтому страница открывается с диска,
с GitHub Pages и из контейнера без сети.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from sbx.core.report import (
    DETECTOR_NAMES,
    ablation_basis,
    contribution_statements,
    control_alarm_statement,
    counted,
    detector_statement,
    forecast_bias_statement,
    format_number,
    hazard_brief,
    leaders_statement,
    live_alarm_statement,
    reference_windows,
    reference_wins,
    schedule_brief,
    schedule_match_statement,
    schedule_reference_statement,
)
from sbx.shell.render import assets, blocks, results
from sbx.shell.render.blocks import LOGO, REFERENCE, _e, cap, conclusion, inline_html, provenance

DEFAULT_TITLE = "Прогнозирование расходов МО и раннее обнаружение шоков"
BRAND = "Прогноз расходов МО"
# Значок вкладки — тот же знак, что в шапке; без него браузер запрашивает /favicon.ico.
ICON = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
    "%3Crect width='32' height='32' rx='7' fill='%232350C8'/%3E"
    "%3Cpolyline points='6,22 11,17 15,19.5 20,12.5' fill='none' stroke='%23fff' "
    "stroke-width='2.75' stroke-linecap='round' stroke-linejoin='round'/%3E"
    "%3Ccircle cx='25' cy='9' r='3.25' fill='%23FF8A5C'/%3E%3C/svg%3E"
)

TIMING_LEAD = (
    "В прогноз входит только то, что опубликовано к моменту прогноза: дата публикации — конец "
    "отчётного периода плюс лаг источника."
)
TIMING_RULE = (
    "Момент прогноза — конец последнего наблюдаемого месяца. На всех шагах многошагового "
    "прогноза стоят одни и те же значения, известные на момент прогноза: прогноз от июня на "
    "июль, август и сентябрь не получает ни июльских новостей, ни августовского решения по "
    "ставке. На целевой месяц берётся только производственный календарь — он известен заранее."
)
SOURCES_NOTE = (
    "Одна строка — один источник. Столбец «Что дал» — измеренное изменение ошибки прогноза от "
    "добавления источника (абляция), а не ожидание."
)
SCHEME_LEAD = (
    "Прогноз и поиск сдвигов — одна цепочка. Внешние данные входят сбоку и только после "
    "согласования по времени; детектор смотрит на ошибки готового прогноза."
)
DEMO_LEAD = (
    "Детектор смотрит не на сам ряд, а на ошибку прогноза на шаг вперёд, делённую на её обычный "
    "размах до этого месяца: сезонный подъём прогноз объясняет сам, и тревогу он не вызывает."
)
DEMO_NOTE = (
    "Ряд бенчмарка — настоящий ряд расходов, в который внесён слом известной величины в "
    "известный месяц."
)
HORIZONS_NOTE = (
    "Критерий конкурса требует четырёх горизонтов: 1, 3, 6 и 12 месяцев. Модели сравниваются "
    "внутри каждого горизонта; метрика — MAE, руб., ниже — лучше. "
    f"{REFERENCE} — эталон из условий конкурса — выделен строкой."
)
METRICS_NOTE = (
    "Справочная сводка. Среднее берётся по тем горизонтам, на которых модель определена, поэтому "
    "модели без горизонта 12 месяцев (LightGBM, нейросети) выглядят в ней лучше, чем есть: самый "
    "трудный горизонт в их среднее не входит. Выбирать модель по этой таблице нельзя — только по "
    "таблице горизонтов выше."
)
SHOWCASE_NOTE = (
    "Три крупнейших ряда оценочной подвыборки из разных регионов. На каждый месяц 2024 года — "
    "прогноз итогового ансамбля с самым коротким доступным горизонтом (1 или 3 месяца) и "
    "интервал 0,1–0,9."
)
ABLATION_NOTE = (
    "Абляция — прогон решения с отключённой частью, чтобы измерить её вклад. Конфигурации A–F "
    "посчитаны на одном наборе фолдов; все внешние признаки взяты по состоянию на момент "
    "прогноза."
)
F1_COLUMN_NOTE = (
    "В столбце «F1 обнаружения» у A–E стоит лучший офлайн-метод по сырому ряду, у F — "
    "онлайн-детектор по остаткам прогноза."
)
SOURCES_ALONE_NOTE = (
    "LightGBM с одним добавленным блоком признаков против той же модели без него: к базовой "
    "модели и поверх календаря и национальных рядов. Источник, который помогает базовой модели, "
    "но ничего не добавляет поверх лучшего набора, повторяет уже известное модели."
)


@dataclass
class Section:
    """Раздел страницы: якорь, подпись в оглавлении, шапка и содержимое."""

    anchor: str
    toc: str
    eyebrow: str
    title: str
    body: str
    kind: str = "plain"  # plain — обычная шапка; fold — свёрнутый блок; case — разбор примера
    extra: dict[str, str] = field(default_factory=dict)


@dataclass
class Group:
    """Группа разделов с общей шапкой: «Данные», «Как считали» и так далее."""

    toc: str
    name: str
    caption: str
    sections: list[Section]
    intro: str = ""
    pill_anchor: str = ""


def _lead(text: str, cls: str = "lead") -> str:
    return f'<p class="{cls}">{inline_html(text)}</p>' if text else ""


def _sub(title: str, note: str = "") -> str:
    tail = f'<p class="sub-note">{_e(note)}</p>' if note else ""
    return f'<h3 class="sub">{_e(title)}</h3>{tail}'


def _sample(story: Mapping[str, Any]) -> str:
    series = (story.get("numbers") or {}).get("evaluation_series")
    if series:
        return f"оценочная выборка: {counted(series, 'ряд', 'ряда', 'рядов')}"
    return "оценочная выборка рядов"


def _windows(story: Mapping[str, Any]) -> str:
    folds = (story.get("numbers") or {}).get("folds")
    if folds:
        return f"{counted(folds, 'окно', 'окна', 'окон')} проверки"
    return "окна проверки на четырёх горизонтах"


# ── Группы разделов ──────────────────────────────────────────────────────────────────────────


def _data_group(story: Mapping[str, Any], meta: Mapping[str, Any]) -> Group:
    sections = []
    sources = story.get("sources") or []
    if sources:
        keys = {row.get("key") for row in sources}
        count = f"{counted(len(sources), 'источник', 'источника', 'источников')}"
        if {"spending", "reference"} <= keys and len(sources) > 2:
            count += f": цель прогноза, справочник территорий и {len(sources) - 2} внешних"
        measured = any(row.get("effects") for row in sources)
        sections.append(
            Section(
                "data",
                "Источники",
                "Источники",
                "Источники: что берём, как согласуем по времени и что это дало",
                _lead(str(story.get("source_verdict") or ""))
                + f'<p class="note">{_e(count)}. {_e(SOURCES_NOTE)}</p>'
                + (blocks.sources_legend() if measured else "")
                + blocks.sources_html(sources),
            )
        )
        territories = blocks.territories_html(
            meta.get("entity") or {}, meta.get("entity_summary") or []
        )
        if territories:
            sections.append(
                Section(
                    "territories",
                    "Сверка территорий",
                    "Сверка территорий",
                    "Территории — по официальным кодам, а не по названиям",
                    territories,
                )
            )
    timing = story.get("timing")
    if timing:
        lags = [int(item["lag_days"]) for item in (story.get("facts") or {}).get("national", [])]
        body = (
            _lead(TIMING_LEAD)
            + f'<p class="note">{_e(TIMING_RULE)}</p>'
            + blocks.timing_steps(max(lags) if lags else None)
            + _sub(
                f"Что видит прогноз, сделанный в конце {blocks.month_genitive(timing['origin'])}",
                "Пример для одного момента прогноза. Чем больше лаг публикации, тем раньше "
                "обрывается полоса.",
            )
            + blocks.sees_html(timing)
        )
        news = story.get("news_timeline")
        if news:
            body += _sub(
                "Новости GDELT о России по месяцам",
                f"Событий в месяц, максимум — {format_number(max(news['counts']))}. Ромб под "
                "столбцом — событие календаря событий в этом месяце; число рядом — сколько их.",
            ) + blocks.news_html(news)
        sections.append(
            Section(
                "timing",
                "Согласование по времени",
                "Согласование по времени",
                "Согласование внешних данных по времени",
                body,
            )
        )
    return Group("Данные", "Данные", "что взяли и как согласовали по времени", sections)


def _scheme_steps(story: Mapping[str, Any]) -> dict[str, str]:
    """Подписи узлов схемы решения; числа — из собранных фактов."""
    facts, numbers = story.get("facts") or {}, story.get("numbers") or {}
    spending = facts.get("spending") or {}
    external = ["национальные ряды", "календари"]
    if facts.get("news"):
        external.insert(1, "новости")
    if facts.get("weather"):
        external.append("погода")
    if (facts.get("population") or {}).get("used"):
        external.append("население")
    kinds = "бейзлайны, Prophet, LightGBM, нейросети, foundation-модели"
    models = kinds
    if numbers.get("models") and numbers.get("folds"):
        models = (
            f"{counted(numbers['models'], 'модель', 'модели', 'моделей')} на "
            f"{counted(numbers['folds'], 'окне', 'окнах', 'окнах')} проверки: {kinds}"
        )
    horizons = [str(group["horizon"]) for group in (story.get("folds") or {}).get("horizons", [])]
    ahead = (
        f"расходов на {', '.join(horizons[:-1])} и {horizons[-1]} месяцев вперёд"
        if len(horizons) > 1
        else "расходов на заданные горизонты"
    )
    return {
        "sources": ", ".join(external),
        "timing": "только опубликованное к моменту прогноза",
        "features": "одни и те же на всех шагах прогноза",
        "spending": (
            f"{format_number(spending['territories'])} МО по {spending.get('categories', 6)} "
            "категориям — цель прогноза"
            if spending
            else "расходы МО по категориям — цель прогноза"
        ),
        "panel": (
            f"{counted(spending['series'], 'ряд', 'ряда', 'рядов')} «МО × категория», территории — "
            "по официальным кодам"
            if spending
            else "ряды «МО × категория», территории — по официальным кодам"
        ),
        "models": models,
        "ensemble": "веса подобраны по ошибкам прошлых месяцев",
        "forecast": ahead,
        "residuals": "факт минус прогноз в единицах обычной ошибки",
        "detector": "тревога, когда ошибки перестают быть обычными",
    }


def _method_group(story: Mapping[str, Any], meta: Mapping[str, Any]) -> Group:
    caption = "схема решения, проверка, метрика и детектор"
    if not story:
        section = Section(
            "pipeline",
            "Архитектура решения",
            "Архитектура",
            "Архитектура решения",
            blocks.architecture_html(),
        )
        return Group("Как считали", "Как считали", caption, [section])
    facts, numbers = story.get("facts") or {}, story.get("numbers") or {}
    sections = [
        Section(
            "pipeline",
            "Схема решения",
            "Схема решения",
            "Схема решения",
            _lead(SCHEME_LEAD)
            + blocks.scheme_html(_scheme_steps(story))
            + blocks.architecture_html(),
        )
    ]
    series = (facts.get("spending") or {}).get("series")
    where = ""
    if numbers.get("evaluation_series") and series:
        where = (
            f"Модели сравниваются на {format_number(numbers['evaluation_series'])} рядах оценочной "
            f"выборки; глобальные модели учатся на всех {format_number(series)} рядах панели."
        )
    folds = story.get("folds")
    lead = (
        f"Модели проверяем на {counted(numbers['folds'], 'окне', 'окнах', 'окнах')}: учим на "
        "данных по некоторый месяц и сравниваем прогноз с тем, что было дальше."
        if numbers.get("folds")
        else ""
    )
    body = _lead(lead) + blocks.metric_html(where)
    if folds:
        common = {
            h: total
            for h, (_, total) in reference_wins(meta.get("significance") or {}, REFERENCE).items()
        }
        body += _sub(
            f"{cap(counted(len(folds['rows']), 'окно', 'окна', 'окон'))} проверки на оси месяцев панели"
        ) + blocks.folds_html(folds, common)
    sections.append(
        Section(
            "protocol",
            "Окна проверки и MAE",
            "Проверка и метрика",
            "Окна проверки и метрика MAE",
            body,
        )
    )
    demo = story.get("detector_demo")
    if demo:
        sections.append(
            Section(
                "detector-demo",
                "Как детектор находит сдвиг",
                "Детектор на примере",
                "Как детектор находит сдвиг",
                _lead(DEMO_LEAD)
                + f'<p class="note">{_e(DEMO_NOTE)}</p>'
                + blocks.detector_demo_html(demo),
            )
        )
    return Group("Как считали", "Как считали", caption, sections)


def _forecast_group(
    data: Mapping[str, Any], meta: Mapping[str, Any], story: Mapping[str, Any]
) -> Group:
    sections = []
    winners = meta.get("winners") or []
    rows = data.get("horizon_leaderboard") or []
    significance = meta.get("significance") or {}
    if rows:
        prov = (
            provenance(
                _sample(story),
                f"{_windows(story)}, MAE; с {REFERENCE} сверено по окнам проверки",
                "раздел 4",
                "configs/backtest.yaml",
            )
            if story
            else ""
        )
        sections.append(
            Section(
                "horizons",
                "Лучшая модель по горизонтам",
                "Главный результат",
                "Лучшая модель зависит от горизонта прогноза",
                conclusion(leaders_statement(winners, REFERENCE))
                + prov
                + f'<p class="note">{_e(HORIZONS_NOTE)}</p>'
                + results.horizons_html(
                    rows,
                    winners,
                    meta.get("horizon_coverage") or {},
                    significance,
                    reference_windows(significance, REFERENCE),
                ),
            )
        )
    foundation = story.get("foundation")
    if foundation:
        sections.append(
            Section(
                "foundation",
                "Foundation-модели",
                "Foundation-модели",
                "Foundation-модели против остальных",
                conclusion(results.foundation_statement(foundation))
                + provenance(
                    _sample(story),
                    "предобученные веса без дообучения, те же окна проверки",
                    "раздел 6",
                    "configs/models.yaml",
                )
                + results.foundation_html(foundation),
            )
        )
    live = meta.get("live") or {}
    if data.get("showcase"):
        sections.append(
            Section(
                "forecast-examples",
                "Как выглядит прогноз",
                "Прогноз на рядах",
                "Как выглядит прогноз",
                _lead(forecast_bias_statement(live))
                + f'<p class="note">{_e(SHOWCASE_NOTE)}</p>'
                + results.showcase_html(data["showcase"]),
            )
        )
    if (data.get("explorer") or {}).get("series"):
        detector = DETECTOR_NAMES.get(str(live.get("detector")), "")
        sections.append(
            Section(
                "explorer",
                "Выбор ряда",
                "Выбор ряда",
                "Выберите территорию и категорию",
                '<p class="note">Факт, прогноз итогового ансамбля (на каждый месяц — самый короткий '
                "доступный горизонт), интервал 0,1–0,9 и тревоги детектора"
                f"{f' {_e(detector)}' if detector else ''} по остаткам прогноза. Для каждой пары «регион × "
                "категория» показан один муниципалитет из оценочной подвыборки.</p>"
                + results.explorer_html(data["explorer"]),
            )
        )
    view = data.get("leaderboard_view") or []
    if view:
        metrics = len([key for key in view[0] if key != "Модель"])
        sections.append(
            Section(
                "metrics",
                "Все метрики · справка",
                "Справка",
                "Все метрики: среднее по горизонтам",
                f'<p class="note small">{_e(METRICS_NOTE)}</p>' + results.metrics_table(view),
                kind="fold",
                extra={
                    "summary": f"{counted(len(view), 'модель', 'модели', 'моделей')} × "
                    f"{counted(metrics, 'метрика', 'метрики', 'метрик')}. Выбирать модель по этой "
                    "таблице нельзя — только по таблице горизонтов."
                },
            )
        )
    return Group(
        "Прогноз", "Результаты · прогноз", "какая модель точнее и как выглядит прогноз", sections
    )


def _shifts_group(
    data: Mapping[str, Any],
    meta: Mapping[str, Any],
    story: Mapping[str, Any],
    curve: Sequence[Mapping[str, Any]],
) -> Group:
    detectors = data.get("detectors") or []
    if not detectors:
        return Group("Сдвиги", "Результаты · сдвиги", "", [])
    contrast = meta.get("detector_contrast") or {}
    match = meta.get("schedule_match") or {}
    live = meta.get("live") or {}
    frame = pd.DataFrame(list(detectors))
    benchmark = (story.get("facts") or {}).get("benchmark") or {}
    source = "полусинтетический бенчмарк: настоящие ряды с внесёнными сломами"
    if benchmark:
        source = (
            f"бенчмарк: {format_number(benchmark['series'])} настоящих рядов, в "
            f"{format_number(benchmark['breaks'])} внесён слом известной величины"
        )
    prov = (
        provenance(
            source,
            "пороги подобраны на одной половине бенчмарка, метрики — на другой",
            "раздел 5",
            "configs/cpd.yaml",
        )
        if story and contrast
        else ""
    )
    findings = results.findings_html(
        [
            ("Остатки прогноза и сырой ряд.", detector_statement(contrast)),
            ("Эталон — тревога по расписанию.", schedule_reference_statement(frame)),
            ("Сравнение при равной частоте ложных тревог.", schedule_match_statement(match)),
        ],
        [
            ("Тревоги на рядах без слома.", control_alarm_statement(contrast)),
            ("Тревоги на настоящих рядах.", live_alarm_statement(live)),
            ("Модель вероятности шока.", hazard_brief(meta.get("hazard") or {})),
        ],
    )
    body = (
        conclusion(schedule_brief(contrast, match) or detector_statement(contrast))
        + prov
        + results.match_html(match)
        + findings
        + _sub(
            "Варианты «метод × вход»",
            "Пороги калиброваны на отдельной части бенчмарка, метрики — на тестовой.",
        )
        + results.variants_html(detectors, contrast)
        + results.curve_html(curve, contrast, match)
    )
    section = Section(
        "changepoints",
        "Сравнение детекторов",
        "Сравнение детекторов",
        "Обнаружение структурных изменений",
        body,
    )
    return Group("Сдвиги", "Результаты · сдвиги", "какой метод раньше замечает слом", [section])


def _contribution_group(
    data: Mapping[str, Any], meta: Mapping[str, Any], story: Mapping[str, Any]
) -> Group:
    ablations = data.get("ablations") or []
    if not ablations:
        return Group("Вклад частей", "Вклад частей решения", "", [])
    layers = story.get("contributions") or []
    alone = story.get("source_contributions") or []
    dm = (meta.get("contributions") or {}).get("diebold_mariano") or {}
    basis = ablation_basis(ablations)
    prov = (
        provenance(
            "те же окна проверки; LightGBM с разными наборами признаков и ансамбли",
            "абляция: слой добавляется, остальное не меняется; вывод — по окнам проверки",
            "раздел 9",
            "configs/ablations.yaml",
        )
        if layers
        else ""
    )
    footnote = F1_COLUMN_NOTE if any(row.get("f1") is not None for row in ablations) else ""
    # Вывод раздела — об источниках по отдельности; не измерены — о первых двух слоях решения.
    verdict = str(story.get("source_verdict") or "") or " ".join(
        row["statement"] for row in layers[:2]
    )
    body = (
        conclusion(verdict)
        + f'<p class="note">{_e(ABLATION_NOTE)}</p>'
        + prov
        + _sub(
            "Конфигурации: слой за слоем",
            "∆MAE — разница с конфигурацией A; минус — ошибка меньше.",
        )
        + results.ablation_table(ablations)
        + f'<p class="note tiny">{_e(" ".join(part for part in (footnote, basis) if part))}</p>'
    )
    if meta.get("news_note"):
        body += f'<p class="note"><b>{_e(meta["news_note"])}</b></p>'
    top = results.scale_top(layers, alone)
    if layers:
        body += (
            _sub(
                "Вклад слоёв",
                "Средняя по рядам разность ошибок и устойчивость знака по окнам проверки.",
            )
            + results.contribution_legend()
            + results.layers_html(layers, top)
        )
    else:
        statements = "".join(f"<li>{_e(text)}</li>" for text in contribution_statements(dm))
        body += f'<ul class="notes">{statements}</ul>' if statements else ""
    if alone:
        body += (
            _sub("Каждый источник по отдельности", SOURCES_ALONE_NOTE)
            + results.source_tiles(alone)
            + results.sources_diagram(alone, top)
        )
    if layers or alone:
        body += (
            '<p class="note tiny">Шкала у диаграмм одна: полоса во всю половину — '
            f"{_e(format_number(top, 0))} ₽.</p>"
        )
    section = Section(
        "contribution",
        "Абляции и вклад источников",
        "Абляции",
        "Абляции: что реально помогает",
        body,
    )
    return Group(
        "Вклад частей", "Вклад частей решения", "что меняется, если часть отключить", [section]
    )


def _history_months(story: Mapping[str, Any]) -> int | None:
    """Длина панели в месяцах — столько точек истории у ряда."""
    spending = (story.get("facts") or {}).get("spending") or {}
    if not spending.get("first") or not spending.get("last"):
        return None
    return (pd.Period(spending["last"], "M") - pd.Period(spending["first"], "M")).n + 1


def _cases_group(meta: Mapping[str, Any], story: Mapping[str, Any]) -> Group:
    cases = meta.get("cases") or {}
    sections: list[Section] = []

    def anchor() -> str:
        return "cases" if not sections else f"case-{len(sections) + 1}"

    for case in cases.get("national", []):
        if not case.get("ds"):
            continue
        events = case.get("events", [])
        found = sum(1 for event in events if event.get("detected"))
        method = DETECTOR_NAMES.get(str(case.get("method")))
        named = f" «{method}»" if method else ""
        lead = (
            f"Офлайн-метод{named} со штрафом, откалиброванным на бенчмарке, находит {found} из "
            f"{len(events)} размеченных шоков в пределах ±2 месяцев и ставит "
            f"{counted(int(case.get('alarms') or 0), 'тревогу', 'тревоги', 'тревог')}."
        )
        if case.get("coverage") is not None:
            lead += (
                " Окна допуска вокруг этих тревог накрывают "
                f"{round(100 * float(case['coverage']))}% ряда: с такой вероятностью «найденной» "
                "оказалась бы случайная дата."
            )
        years = f"{str(case['ds'][0])[:4]}–{str(case['ds'][-1])[:4]}"
        sections.append(
            Section(
                anchor(),
                f"Национальные «{case['series']}»",
                f"национальный ряд · {years}",
                f"Национальные расходы «{case['series']}»",
                _lead(lead, "case-lead") + results.national_case_html(case),
                kind="case",
                extra={"tag": f"Офлайн-метод{named}, штраф откалиброван на бенчмарке"},
            )
        )
    municipal = cases.get("municipal")
    if municipal and municipal.get("ds"):
        place = municipal.get("mo") or municipal.get("region")
        detector = DETECTOR_NAMES.get(str((meta.get("live") or {}).get("detector")), "")
        lead = (
            f"Крупнейший остаток прогноза в потоке живого детектора ({municipal.get('region')}, "
            f"|z| = {format_number(municipal.get('max_abs_z'), 1)}). Это не шок, а устойчивый рост "
            "категории, за которым модель с 24 точками истории не успевает. Повышения ключевой "
            "ставки, попавшие в окно, объяснением не являются: рост начался раньше и шёл ровно."
        )
        years = f"{str(municipal['ds'][0])[:4]}"
        sections.append(
            Section(
                anchor(),
                str(place),
                f"ряд МО · {years}",
                f"{place} · {municipal.get('category')}",
                _lead(lead, "case-lead")
                + results.municipal_case_html(municipal, _history_months(story)),
                kind="case",
                extra={
                    "tag": f"Живой детектор{f' {detector}' if detector else ''} по остаткам ансамбля",
                    "sub": str(municipal.get("region") or ""),
                },
            )
        )
    national = [case for case in cases.get("national", []) if case.get("ds")]
    intro = ""
    if sections:
        kinds = []
        if national:
            rows = counted(
                len(national), "национальный ряд", "национальных ряда", "национальных рядов"
            )
            kinds.append(f"{rows} с размеченными шоками")
        if municipal and municipal.get("ds"):
            kinds.append("ряд МО с крупнейшим остатком прогноза")
        intro = _lead(f"Настоящие ряды: {' и '.join(kinds)}.", "lead mid")
    if national:
        intro += results.cases_legend(int(national[0].get("margin", 2)))
    return Group(
        "Разборы", "Разборы реальных примеров", "как методы ведут себя на настоящих рядах", sections,
        intro=intro, pill_anchor="cases",
    )  # fmt: skip


def _reference_group(story: Mapping[str, Any], meta: Mapping[str, Any]) -> Group:
    sections = []
    limitations = meta.get("limitation_items") or meta.get("limitations")
    if isinstance(limitations, str):
        limitations = [limitations]
    if limitations:
        areas = any(isinstance(item, Mapping) and item.get("area") for item in limitations)
        count = counted(len(limitations), "ограничение", "ограничения", "ограничений")
        note = (
            f'<p class="note">{count}, которые стоит держать в голове, читая результаты. Метка '
            "показывает, к какой части решения относится ограничение: прогноз, сдвиги или "
            "данные.</p>"
            if areas
            else ""
        )
        sections.append(
            Section(
                "limits",
                "Ограничения",
                "Ограничения",
                "Ограничения",
                note + results.limits_html(limitations),
            )
        )
    if story.get("criteria"):
        # Таблицу заполняет `render`: ссылки ставятся только на разделы, которые есть на странице.
        sections.append(
            Section(
                "criteria",
                "Критерии конкурса",
                "Для жюри",
                "Где смотреть по критериям конкурса",
                "",
            )
        )
    citations = story.get("citations") or []
    body = (
        results.commands_html()
        + _sub("Источник данных и обязательные подписи")
        + '<p class="p15">Источник данных — <b>Данные СберИндекса</b> (sberindex.ru).'
        + (" Обязательные подписи источников:" if citations else "")
        + "</p>"
        + (results.citations_html(citations) if citations else "")
    )
    sections.append(
        Section(
            "reproduce",
            "Данные и воспроизведение",
            "Данные и воспроизведение",
            "Воспроизвести решение",
            body,
        )
    )
    return Group(
        "Справка", "Справка", "ограничения, критерии конкурса, данные и воспроизведение", sections
    )


# ── Каркас страницы ──────────────────────────────────────────────────────────────────────────


def _pills(anchor: str, criteria: Sequence[Mapping[str, Any]]) -> str:
    """Критерии конкурса, которые закрывает раздел: по ссылкам из карты критериев."""
    return "".join(
        f'<span class="pill">Критерий {_e(row["code"])} · {row["weight"]}%</span>'
        for row in criteria
        if row.get("code") and any(link["href"] == f"#{anchor}" for link in row["links"])
    )


def _section_html(number: str, section: Section, criteria: Sequence[Mapping[str, Any]]) -> str:
    pills = _pills(section.anchor, criteria)
    if section.kind == "case":
        sub = f"<small>{_e(section.extra['sub'])}</small>" if section.extra.get("sub") else ""
        return (
            f'<section id="{section.anchor}" class="case"><div class="case-head"><div>'
            f'<div class="eyebrow">Разбор {number} · {_e(section.eyebrow)}</div>'
            f"<h2>{_e(section.title)}</h2>{sub}</div>"
            f'<span class="method-tag">{_e(section.extra.get("tag", ""))}</span></div>'
            f"{section.body}</section>"
        )
    if section.kind == "fold":
        codes = [
            str(row["code"])
            for row in criteria
            if row.get("code")
            and any(link["href"] == f"#{section.anchor}" for link in row["links"])
        ]
        tag = " · ".join([section.eyebrow, *codes])
        return (
            f'<section id="{section.anchor}" class="sec"><details class="fold"><summary><span>'
            f"<b>{number} · {_e(section.title)}</b><small>{_e(section.extra.get('summary', ''))}</small>"
            f"</span><em>{_e(tag)}</em></summary><div>{section.body}</div></details></section>"
        )
    return (
        f'<section id="{section.anchor}" class="sec"><div class="sec-eyebrow">'
        f'<span class="eyebrow">{number} · {_e(section.eyebrow)}</span>{pills}</div>'
        f'<h2 class="title">{_e(section.title)}</h2>{section.body}</section>'
    )


def _group_html(index: int, group: Group, criteria: Sequence[Mapping[str, Any]]) -> str:
    pill = _pills(group.pill_anchor, criteria) if group.pill_anchor else ""
    bar = (
        f'<div class="group-bar"><i>{index:02d}</i><b>{_e(group.name)}</b>'
        f"<span>{_e(group.caption)}</span>{pill}</div>"
    )
    parts = [bar, group.intro]
    for position, section in enumerate(group.sections, start=1):
        number = str(position) if section.kind == "case" else f"{index}.{position}"
        parts.append(_section_html(number, section, criteria))
    return "".join(parts)


def toc_html(
    groups: Sequence[Group], criteria: Sequence[Mapping[str, Any]], anchors: set[str]
) -> str:
    """Оглавление: разделы по группам и — отдельной вкладкой — критерии конкурса."""
    weights = {row["primary"]: row["weight"] for row in criteria if row.get("primary")}
    items = []
    if "chain" in anchors:
        items.append(
            '<a class="toc-top" href="#chain"><span>Обзор: данные → расчёт → результат</span></a>'
        )
    for index, group in enumerate(groups, start=1):
        links = "".join(
            f'<a href="#{section.anchor}"><span>{_e(section.toc)}</span>'
            + (f"<i>{weights[section.anchor]}%</i>" if section.anchor in weights else "")
            + "</a>"
            for section in group.sections
        )
        items.append(
            f'<div class="toc-group"><div><i>{index}</i>{_e(group.toc)}</div>{links}</div>'
        )
    note = (
        '<p class="toc-note">Проценты — вес критерия конкурса, который закрывает раздел.</p>'
        if weights
        else ""
    )
    sections_pane = f'<nav class="toc" aria-label="Оглавление" data-pane="sections">{"".join(items)}{note}</nav>'
    if not criteria:
        return (
            '<aside class="side"><button class="toc-toggle" type="button" aria-expanded="true" '
            'aria-controls="toc-body">Содержание</button><div class="toc-body" id="toc-body">'
            f'<p class="toc-title">Содержание</p>{sections_pane}</div></aside>'
        )
    top = max(int(row["weight"]) for row in criteria)
    total = sum(int(row["weight"]) for row in criteria)
    cards = []
    for row in criteria:
        links = "".join(
            f'<a href="{_e(link["href"])}">{_e(link["text"])}</a>'
            for link in row["links"]
            if link["href"][1:] in anchors
        )
        cards.append(
            '<div class="crit"><div class="crit-head">'
            f"<b><i>{_e(row.get('code', ''))}</i> {_e(row.get('short') or row['criterion'])}</b>"
            f"<span>{row['weight']}%</span></div>"
            f'<div class="meter"><i style="width:{100 * int(row["weight"]) / top:.0f}%"></i></div>'
            f'<div class="crit-links">{links}</div></div>'
        )
    criteria_pane = (
        '<nav class="toc" aria-label="Критерии конкурса" data-pane="criteria" hidden>'
        f'<p class="crit-lead">Где на странице смотреть каждый критерий. Сумма весов — {total}%.</p>'
        f"{''.join(cards)}</nav>"
    )
    tabs = (
        '<div class="tabs" role="group" aria-label="Вид оглавления">'
        '<button type="button" aria-pressed="true" data-tab="sections">Разделы</button>'
        '<button type="button" aria-pressed="false" data-tab="criteria">Критерии</button></div>'
    )
    return (
        '<aside class="side"><button class="toc-toggle" type="button" aria-expanded="true" '
        'aria-controls="toc-body">Содержание</button><div class="toc-body" id="toc-body">'
        f'<p class="toc-title">Содержание</p>{tabs}{sections_pane}{criteria_pane}</div></aside>'
    )


def _external_links(meta: Mapping[str, Any]) -> list[str]:
    links = []
    if meta.get("pdf"):
        links.append(f'<a href="{_e(meta["pdf"])}">Методологический отчёт, PDF</a>')
    if meta.get("repository"):
        links.append(f'<a href="{_e(meta["repository"])}">Репозиторий</a>')
    return links


def _header(meta: Mapping[str, Any]) -> str:
    contest = meta.get("contest")
    sub = f'<span class="top-sub">{_e(contest)}</span>' if contest else ""
    links = "".join([*_external_links(meta), '<a href="#reproduce">Как воспроизвести</a>'])
    return (
        '<header class="top"><div class="top-in">'
        f'<a class="brand" href="#top">{LOGO}<b>{BRAND}</b></a>{sub}'
        f'<nav class="top-nav" aria-label="Материалы">{links}</nav></div></header>'
    )


def _footer(meta: Mapping[str, Any]) -> str:
    links = "".join([*_external_links(meta), '<a href="#top">Наверх</a>'])
    return (
        '<footer><div class="foot">'
        f'<div class="foot-brand">{LOGO}<div><b>{BRAND}</b>'
        "<span>Источник данных — Данные СберИндекса</span></div></div>"
        f"<p>{inline_html(str(meta.get('footer', '')))}</p>"
        f'<nav aria-label="Ссылки">{links}</nav></div></footer>'
    )


def _hero(meta: Mapping[str, Any], chain: str) -> str:
    eyebrow = f'<p class="eyebrow">{_e(meta["contest"])}</p>' if meta.get("contest") else ""
    lead = meta.get("panel_line") or meta.get("subtitle", "")
    return (
        f'<div class="hero" id="top">{eyebrow}<h1>{_e(meta.get("title", DEFAULT_TITLE))}</h1>'
        f'<p class="hero-lead">{_e(lead)}</p>{chain}</div>'
    )


def keep_numbers_together(markup: str) -> str:
    """Неразрывный пробел между разрядами числа: «13 140» не переносится на две строки.

    Меняется только текст между тегами; атрибуты (координаты графиков, стили) не затрагиваются.
    """
    return re.sub(
        r">[^<]+<",
        lambda match: re.sub(r"(?<=\d) (?=\d{3}(?!\d))", "\u00a0", match.group(0)),
        markup,
    )


def render(data: Mapping[str, Any], out_path: Path, curve: pd.DataFrame | None = None) -> Path:
    """Собирает статический HTML лендинга из данных отчёта.

    Порядок страницы — «данные → как считали → результаты»: под шапкой три колонки со ссылками,
    ниже разделы в том же порядке, слева — оглавление. Блок, для которого нет данных,
    пропускается вместе со ссылками на него.
    """
    meta = data.get("meta", {})
    story = data.get("storyline") or {}
    points = (
        curve.to_dict("records") if curve is not None and len(curve) else data.get("curve") or []
    )
    candidates = [
        _data_group(story, meta),
        _method_group(story, meta),
        _forecast_group(data, meta, story),
        _shifts_group(data, meta, story, points),
        _contribution_group(data, meta, story),
        _cases_group(meta, story),
        _reference_group(story, meta),
    ]
    groups = [group for group in candidates if group.sections]
    anchors = {section.anchor for group in groups for section in group.sections}

    chain = ""
    if story.get("chain"):
        kept = {
            key: {
                **column,
                "lines": [line for line in column["lines"] if line["href"][1:] in anchors],
            }
            for key, column in story["chain"].items()
        }
        kept = {key: column for key, column in kept.items() if column["lines"]}
        if kept:
            chain = blocks.chain_html(kept, meta, str(story.get("source_verdict") or ""))
            anchors.add("chain")
    criteria = [
        {**row, "links": [link for link in row["links"] if link["href"][1:] in anchors]}
        for row in story.get("criteria") or []
    ]
    for group in groups:
        for section in group.sections:
            if section.anchor == "criteria":
                section.body = (
                    '<p class="note">Та же навигация есть в оглавлении, во вкладке «Критерии».</p>'
                    + results.criteria_html(criteria)
                )
    main = _hero(meta, chain) + "".join(
        _group_html(index, group, criteria) for index, group in enumerate(groups, start=1)
    )
    title = meta.get("title", DEFAULT_TITLE)
    body = keep_numbers_together(
        f"{_header(meta)}\n"
        f'<div class="shell">\n{toc_html(groups, criteria, anchors)}\n<main id="main">\n{main}\n</main>\n</div>\n'
        f"{_footer(meta)}\n"
    )
    page = (
        '<!doctype html>\n<html lang="ru">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_e(title)}</title>\n"
        f'<meta name="description" content="{_e(meta.get("subtitle", ""))}">\n'
        f'<link rel="icon" href="{ICON}">\n'
        f"<style>\n{assets.stylesheet()}\n</style>\n</head>\n<body>\n"
        '<a class="skip" href="#main">К содержанию</a>\n'
        f"{body}"
        f"<script>\n{assets.script()}\n</script>\n</body>\n</html>\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    (out_path.parent / "report_data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    return out_path
