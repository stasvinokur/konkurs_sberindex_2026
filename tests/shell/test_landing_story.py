"""Лендинг как цепочка «данные → расчёт → результат»: что обязано быть на странице.

Проверяется содержание и связность страницы, а не оформление: блоки на месте, числа взяты из
данных, каждая ссылка ведёт к существующему разделу, необязательный блок без данных пропущен.
"""

from __future__ import annotations

import html as html_lib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from sbx.core.folds import BacktestConfig, HorizonFolds, TargetSpec, make_folds
from sbx.core.storyline import (
    chain_summary,
    contribution_view,
    criteria_map,
    detector_demo,
    fold_timeline,
    foundation_view,
    news_timeline,
    source_inventory,
    source_verdict,
    source_view,
    timing_rows,
)
from sbx.shell.render import assets, blocks, landing
from tests.core.test_report import ENTITY
from tests.core.test_storyline import DM, FACTS, HORIZON_ROWS, WEIGHTS, WINNERS, _benchmark

CITATIONS = [
    "Потребительские безналичные расходы на уровне муниципальных образований по категориям "
    "трат. СберИндекс. Данные доступны по адресу https://sberindex.ru/ru/research/x (данные "
    "скачаны 2026-10-02).",
    "Данные СберИндекса (https://sberindex.ru).",
    "The GDELT Project (https://www.gdeltproject.org/).",
]
CONFIG = BacktestConfig(
    status="test",
    checked_at="2026-10-02",
    target=TargetSpec("d", "MS", "l"),
    folds=(
        HorizonFolds(horizon=1, origins=("2023-12-01", "2024-09-01")),
        HorizonFolds(horizon=6, origins=("2024-06-01",)),
        HorizonFolds(horizon=12, origins=("2023-12-01",)),
    ),
    primary_metric="mae",
    mae_aggregation="micro",
)
CONTRAST = {
    "detector": "cusum",
    "f1_residual": 0.221,
    "f1_raw": 0.188,
    "delay_residual": 2.275,
    "false_alarms_residual": 0.889,
    "best_raw": {"detector": "page_hinkley", "f1": 0.219, "delay": 3.242, "false_alarms": 0.9},
}


def _story() -> dict[str, Any]:
    folds = make_folds(CONFIG)
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    series, labels, seasonal = _benchmark()
    news = pd.DataFrame(
        {"region_code": "45", "ds": months, "country_news_events": np.arange(100, 124)}
    )
    events = [
        {
            "date": "2024-07-26",
            "kind": "key_rate_hike",
            "region_code": "all",
            "description": "Ключевая ставка повышена до 18%",
        },
    ]
    lags = {"spending": 35, "weekly": 7}
    titles = {"spending": "Потребительские расходы", "weekly": "Недельная инфляция"}
    return {
        "sources": source_inventory(FACTS, DM),
        "chain": chain_summary(FACTS, len(folds), 4, WINNERS, CONTRAST, DM),
        "timing": {"origin": "2024-09", "rows": timing_rows(months[20], lags, titles)},
        "contributions": contribution_view(DM),
        "foundation": foundation_view(HORIZON_ROWS, WEIGHTS),
        "citations": CITATIONS,
        "criteria": criteria_map(),
        "facts": {**FACTS, "benchmark": {"series": 1500, "breaks": 1131}},
        "numbers": {"folds": len(folds), "models": 4, "evaluation_series": 1201},
        "folds": fold_timeline(folds, months),
        "news_timeline": news_timeline(news, events),
        "detector_demo": detector_demo(series, labels, seasonal, "page_hinkley", 5.0),
    }


def _data(story: dict[str, Any] | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "meta": {
            "title": "Прогноз расходов МО",
            "subtitle": "Панель: 13 140 рядов.",
            "cards": [{"value": "580,0", "label": "MAE на горизонте 1 месяц, руб."}],
            "winners": WINNERS,
            "horizon_coverage": {1: {"models": 6, "observations": 4431}},
            "limitations": ["История каждого ряда — 24 месяца.", "Реальных шоков почти нет."],
            "footer": "Сформировано автоматически.",
            "detector_contrast": CONTRAST,
            "contributions": {"diebold_mariano": DM},
            "entity_summary": ["Панель построена по официальным кодам: 2 190 территорий."],
        },
        "leaderboard": [],
        "leaderboard_view": [{"Модель": "Ensemble", "MAE, руб.": 600.0}],
        "horizon_leaderboard": HORIZON_ROWS,
        "detectors": [
            {
                "detector": "page_hinkley",
                "kind": "online",
                "input": "residual",
                "f1": 0.213,
                "precision": 0.14,
                "recall": 0.42,
                "mean_delay": 3.1,
                "false_alarms_per_series_year": 0.94,
            },
        ],
        "ablations": [
            {"name": "A", "mae": 700.0, "delta_mae": 0.0, "models": "LightGBM", "f1": 0.2}
        ],
    }
    if story is not None:
        data["storyline"] = story
    return data


def _page(path: Path) -> str:
    """Собранная страница с обычными пробелами вместо неразрывных — так её сверяют тесты."""
    return path.read_text(encoding="utf-8").replace("\u00a0", " ")


@pytest.fixture(scope="module")
def page(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("landing") / "index.html"
    return _page(landing.render(_data(_story()), out))


def test_digit_groups_of_a_number_are_not_split_across_lines(tmp_path: Path) -> None:
    """«13 140» в узкой колонке переносилось бы после «13»: между разрядами — неразрывный пробел."""
    glue = landing.keep_numbers_together
    assert glue("<p>1 201 ряд, 24 месяца, 3 011 689 событий</p>") == (
        "<p>1\u00a0201 ряд, 24 месяца, 3\u00a0011\u00a0689 событий</p>"
    )
    assert glue('<svg viewBox="0 0 440 380"><text>1 000</text></svg>') == (
        '<svg viewBox="0 0 440 380"><text>1\u00a0000</text></svg>'
    ), "атрибуты не тронуты"
    raw = landing.render(_data(_story()), tmp_path / "nbsp.html").read_text(encoding="utf-8")
    assert "13\u00a0140 рядов" in raw and "2\u00a0190 МО" in raw
    assert 'viewBox="0 0 740 586"' in raw
    script = raw[raw.rindex("<script>") :]
    assert "\u00a0" not in script, "сценарий страницы не изменён"
    json.loads(re.search(r'<script type="application/json">(.*?)</script>', raw).group(1))


def _anchors(html: str) -> set[str]:
    return set(re.findall(r'\sid="([^"]+)"', html))


def _links(html: str) -> list[str]:
    return re.findall(r'href="#([^"]+)"', html)


def _block(html: str, anchor: str) -> str:
    """Разметка раздела с данным якорем — до следующего раздела."""
    start = html.index(f'id="{anchor}"')
    following = [m.start() for m in re.finditer(r"<section ", html) if m.start() > start]
    return html[start : following[0] if following else len(html)]


def _text(markup: str) -> str:
    """Видимый текст разметки: без тегов, с раскрытыми сущностями и одиночными пробелами."""
    return re.sub(r"\s+", " ", html_lib.unescape(re.sub(r"<[^>]+>", " ", markup))).strip()


def test_chain_sits_under_the_header_and_every_line_leads_to_a_section(page: str) -> None:
    assert page.index('id="chain"') < page.index('id="data"'), "цепочка — сразу под шапкой"
    chain = _block(page, "chain")
    for number, name in (("1", "Данные"), ("2", "Расчёт"), ("3", "Результат")):
        assert f"<i>{number}</i><h2>{name}</h2>" in chain
    targets = _links(chain)
    assert len(targets) >= 9 and set(targets) <= _anchors(page)
    assert "<span>Расходы МО</span><span>2 190 МО × 6 категорий × 24 месяца</span>" in chain
    assert "MAE ниже, чем у Prophet, на 4 горизонтах из 4" in chain


def test_result_column_shows_the_leaders_next_to_the_reference(page: str) -> None:
    """Колонка результата — сами числа: лучшая модель горизонта рядом с эталоном."""
    chain = _block(page, "chain")
    rows = re.findall(r"<tr><td>(\d+) мес\.</td>(.*?)</tr>", chain)
    assert [horizon for horizon, _ in rows] == ["1", "3", "6", "12"]
    assert "Chronos-2 с ковариатами" in rows[0][1] and "Ансамбль" in rows[3][1]
    assert '<td class="best">580,0</td><td class="ref">683,0</td>' in rows[0][1]
    assert "Детектор по остаткам прогноза" in chain and "<b>0,221</b>" in chain
    assert "по сырому ряду — 0,219 и 3,2 мес." in chain, "без эталона — строка о детекторах"
    assert "Точнее Prophet" not in chain, "счёта окон нет — нет и столбца"


def test_every_internal_link_leads_to_an_existing_anchor(page: str) -> None:
    missing = sorted(set(_links(page)) - _anchors(page))
    assert not missing, f"ссылки в никуда: {missing}"
    assert 'aria-label="Оглавление"' in page
    toc = page[page.index('aria-label="Оглавление"') : page.index('id="main"')]
    assert {"chain", "data", "protocol", "horizons", "changepoints", "criteria"} <= set(_links(toc))


def test_sections_follow_the_order_data_method_result(page: str) -> None:
    order = [
        "chain",
        "data",
        "territories",
        "timing",
        "pipeline",
        "protocol",
        "detector-demo",
        "horizons",
        "foundation",
        "metrics",
        "changepoints",
        "contribution",
        "limits",
        "criteria",
        "reproduce",
    ]
    positions = [page.index(f'<section id="{anchor}"') for anchor in order]
    assert positions == sorted(positions)
    bars = re.findall(r'<div class="group-bar"><i>(\d\d)</i><b>([^<]+)</b>', page)
    assert bars == [
        ("01", "Данные"),
        ("02", "Как считали"),
        ("03", "Результаты · прогноз"),
        ("04", "Результаты · сдвиги"),
        ("05", "Вклад частей решения"),
        ("06", "Справка"),
    ], "группа без разделов пропущена, нумерация остаётся сплошной"


def test_data_section_lists_every_source_with_its_terms_and_citations(page: str) -> None:
    data = _block(page, "data")
    for row in source_inventory(FACTS, DM):
        for cell in (row["title"], row["volume"], row["tag"], row["license"]):
            assert html_lib.escape(cell) in data, cell
        for cell in (row["provider"], row["note"]):
            assert not cell or html_lib.escape(cell) in data, cell
    assert "CC BY-SA 4.0" in data
    assert "Согласование по времени" in data and "Что дал" in data
    assert '<span class="dark-pill">цель прогноза</span>' in data
    assert "−60,8 руб., лучше в 9 окнах из 10 — устойчиво" in data, "вклад слоя — из абляции"
    assert "6 источников: цель прогноза, справочник территорий и 4 внешних" in data
    groups = re.findall(r'<tr class="lvl"><th colspan="5" scope="colgroup">([^<]+)</th>', data)
    assert groups == ["Основа панели · уровень МО", "Уровень страны", "Уровень региона и страны"]
    territories = _block(page, "territories")
    assert "Панель построена по официальным кодам: 2 190 территорий." in territories
    reproduce = _block(page, "reproduce")
    assert "Данные СберИндекса" in reproduce
    # Подпись читается дословно; адрес в ней — ссылка, поэтому сверяется видимый текст.
    visible = re.sub(r"<[^>]+>", "", reproduce)
    for citation in CITATIONS:
        assert citation.replace("&", "&amp;") in visible


def test_method_section_defines_mae_and_draws_every_validation_window(page: str) -> None:
    protocol = _block(page, "protocol")
    assert "MAE — средний модуль разности факта и прогноза" in protocol
    assert "ниже — лучше" in protocol.lower()
    assert protocol.count('class="folds-row window"') == 4, "по строке на окно проверки"
    assert protocol.count('class="train"') == 4 and protocol.count('class="test"') == 4
    assert "обучение" in protocol and "проверка" in protocol
    assert "1 201" in protocol and "13 140" in protocol, "оценочные ряды и вся панель"
    assert "Модели проверяем на 4 окнах" in protocol
    assert "Каждую модель" not in protocol, "не каждая модель строится в каждом окне"
    pipeline = _block(page, "pipeline")
    assert pipeline.count('<div class="node') == 10 and "Детектор сдвигов" in pipeline
    assert "Functional Core / Imperative Shell" in pipeline
    assert "2 190 МО по 6 категориям" in pipeline and "13 140 рядов «МО × категория»" in pipeline


def test_timing_block_states_the_rule_and_what_a_forecast_sees(page: str) -> None:
    timing = _block(page, "timing")
    assert "Момент прогноза — конец последнего наблюдаемого месяца" in timing
    assert "На всех шагах" in timing and "целевой месяц" in timing
    assert "Что видит прогноз, сделанный в конце сентября 2024 года" in timing
    assert "сентябрь 2024" in timing, "последний известный период — у источников без лага"
    assert "Момент прогноза · 30 сентября 2024" in timing
    for row in _story()["timing"]["rows"]:
        assert row["tag"] in timing and row["known"] in timing
    assert timing.count("<i style=") == 24, "по столбцу на месяц новостей"
    assert "Ключевая ставка повышена до 18%" in timing


def test_timing_bars_stop_where_the_publication_lag_cuts_them(page: str) -> None:
    """Источник с лагом 35 дней на конец сентября известен по июль: дальше — штриховка."""
    timing = _block(page, "timing")
    row = timing[timing.index("Потребительские расходы") :].split('<div class="sees-row">')[0]
    # Столбцы сетки: источник, май…сентябрь (2–6), три целевых месяца, последний известный период.
    assert '<i class="bar-known" style="grid-column:2 / 5"></i>' in row, "май — июль известны"
    assert '<i class="bar-wait" style="grid-column:5 / 7"></i>' in row, "август и сентябрь — нет"
    assert "июль 2024" in row
    assert 'style="--cols:8;--now:6"' in timing, "целевые месяцы начинаются после сентября"
    calendar = timing[timing.index("Производственный календарь") :].split('<div class="sees-row">')[
        0
    ]
    assert '<i class="bar-known" style="grid-column:2 / 10"></i>' in calendar, "известен заранее"


def test_detector_walkthrough_marks_the_break_and_the_alarm(page: str) -> None:
    demo = _block(page, "detector-demo")
    assert demo.count("<svg") == 1 and demo.count('class="panel"') == 3, (
        "ряд и прогноз, ошибка прогноза, статистика детектора"
    )
    assert "ноябрь 2023" in demo, "месяц внесённого слома"
    assert "Тревога" in demo and "порог" in demo
    assert 'class="dia"' in demo and 'class="tri"' in demo and 'class="thr"' in demo
    assert demo.count('class="win"') == 1, "месяцы от слома до тревоги выделены полосой"
    payload = json.loads(
        re.search(r'<script type="application/json">(.*?)</script>', demo).group(1)
    )
    assert len(payload["months"]) == 24 and payload["forecast"][0] is None
    assert set(payload) == {"months", "y", "forecast", "z", "statistic", "alarms"}


def test_foundation_models_have_their_own_block_with_a_measured_conclusion(page: str) -> None:
    block = _block(page, "foundation")
    assert "Chronos-2 с ковариатами" in block and "TimesFM-2.5" in block
    assert "Лучшая из остальных одиночных моделей" in block
    assert "amazon/chronos-2" in block and "apache-2.0" in block and "29ec3766" in block
    assert "лучше остальных одиночных моделей на горизонтах 1 мес." in block
    assert "Prophet они превосходят на всех горизонтах" in block
    cards = re.findall(r'<div class="hcard-head"><b>([^<]+)</b><span[^>]*>([^<]+)</span>', block)
    assert cards == [
        ("1 месяц", "foundation точнее"),
        ("6 месяцев", "остальные точнее"),
        ("12 месяцев", "остальные точнее"),
    ]


def test_contribution_diagram_puts_direction_and_windows_in_words(page: str) -> None:
    block = _block(page, "contribution")
    assert "Календарь и национальные ряды" in block
    rows = [_text(row) for row in block.split('<div class="div-row">')[1:]]
    assert "−60,8 ₽ 9 из 10 помог" in rows[0] and "+6,7 ₽ 6 из 10 неясно" in rows[1]
    assert '<i class="better" style="left:' in block and '<i class="none" style="left:' in block
    assert "ошибка меньше" in block and "знак меняется от окна к окну" in block
    assert "значимо" not in block, (
        "вывод делается по окнам, а не по тесту, считающему ряды независимыми"
    )


def test_result_sections_say_where_their_numbers_come_from(page: str) -> None:
    for anchor, report in (
        ("horizons", "раздел 4"),
        ("foundation", "раздел 6"),
        ("changepoints", "раздел 5"),
        ("contribution", "раздел 9"),
    ):
        block = _block(page, anchor)
        assert '<div class="concl"><span>Вывод</span>' in block, anchor
        assert '<dl class="prov">' in block, anchor
        line = block[block.index('<dl class="prov">') :].split("</dl>")[0]
        assert (
            "<dt>Данные</dt>" in line and "<dt>Метод</dt>" in line and "<dt>Подробнее</dt>" in line
        )
        assert f"Отчёт, {report}" in line and "configs/" in line
        assert "Диболда" not in line, "вывод строится по окнам, тест по рядам его не держит"
    horizons = _block(page, "horizons")
    assert "MAE; с Prophet сверено по окнам проверки" in horizons
    assert (
        "На 1 и 3 месяца точнее всех Chronos-2 с ковариатами, на 6 и 12 месяцев — ансамбль. У "
        "Prophet, эталона из условий конкурса, ошибка выше на каждом из 4 горизонтов." in horizons
    )
    contribution = _block(page, "contribution")
    assert "остальное не меняется; вывод — по окнам проверки" in contribution
    assert (
        "В столбце «F1 обнаружения» у A–E стоит лучший офлайн-метод по сырому ряду, у F — "
        "онлайн-детектор по остаткам прогноза." in contribution
    )


def test_sections_carry_the_criteria_they_answer(page: str) -> None:
    metrics = _block(page, "metrics")
    assert "<em>Справка · К2 · К6</em>" in metrics, "у свёрнутого блока критерии — в метке"
    horizons = _block(page, "horizons")
    assert "Критерий К2 · 20%" in horizons and "Критерий К6 · 10%" in horizons
    assert "Критерий К5 · 15%" in _block(page, "timing")
    assert "Критерий К4 · 15%" in _block(page, "foundation")
    assert "Критерий" not in _block(page, "territories").split("</h2>")[0]


def test_table_of_contents_has_a_tab_for_the_contest_criteria(page: str) -> None:
    toc = page[page.index('<aside class="side">') : page.index('id="main"')]
    assert 'data-tab="sections"' in toc and 'data-tab="criteria"' in toc
    sections, criteria = toc.split('data-pane="criteria"')
    assert '<a href="#horizons"><span>Лучшая модель по горизонтам</span><i>20%</i></a>' in sections
    assert '<a href="#data"><span>Источники</span></a>' in sections, "вес — у основного раздела"
    assert "hidden" in criteria.split(">")[0], "вкладка критериев открывается по нажатию"
    assert re.findall(r"<b><i>(К\d)</i>", criteria) == [f"К{i}" for i in range(1, 8)]
    assert "Сумма весов — 100%" in criteria
    assert set(_links(criteria)) <= _anchors(page) and "cases" not in _links(criteria)


def test_criteria_table_maps_all_seven_criteria_to_sections(page: str) -> None:
    block = _block(page, "criteria")
    assert block.count("<tr>") == 8, "заголовок и семь критериев"
    for weight in ("10%", "20%", "15%"):
        assert weight in block
    assert 'href="#foundation"' in block and 'href="#timing"' in block and 'href="#chain"' in block
    assert "artifacts/metrics/leaderboard_by_horizon.csv" in block


def test_limitations_are_separate_cards(page: str) -> None:
    block = _block(page, "limits")
    assert block.count('<article class="limit">') == 2 and "Реальных шоков почти нет." in block


def test_limitations_with_titles_show_the_area_and_do_not_repeat_the_title(
    tmp_path: Path,
) -> None:
    data = _data(_story())
    data["meta"]["limitation_items"] = [
        {"title": "Короткая история рядов", "area": "прогноз", "text": "История — 24 месяца."},
        {
            "title": "Календарь и национальные ряды",
            "area": "данные",
            "text": "Календарь и национальные ряды: устойчивого изменения нет — MAE ниже.",
        },
    ]
    block = _block(_page(landing.render(data, tmp_path / "i.html")), "limits")
    assert "<h3>Короткая история рядов</h3><em>прогноз</em>" in block
    assert "2 ограничения, которые стоит держать в голове, читая результаты." in block
    assert "<p>Устойчивого изменения нет — MAE ниже.</p>" in block
    assert "Реальных шоков почти нет." not in block, "вместо строк — пункты с заголовками"


def test_page_is_one_file_without_external_resources(page: str) -> None:
    """Страница открывается с диска и из контейнера без сети: шрифты, стили и сценарий внутри."""
    assert not re.findall(r'<(?:script|img|link)[^>]+(?:src|href)="https?:', page)
    assert '<link rel="stylesheet"' not in page and "@import" not in page
    assert set(re.findall(r"url\(([a-z]+):", page)) == {"data"}
    assert page.count("@font-face") == len(assets.FACES)
    assert "plotly" not in page.lower()
    assert page.count("<script>") == 1, "сценарий страницы один, остальные блоки — данные"


def test_charts_follow_the_colour_scheme_of_the_page(page: str) -> None:
    """Графики — встроенный SVG с цветами из переменных страницы: в тёмной теме они не остаются
    светлыми прямоугольниками."""
    assert "@media (prefers-color-scheme: dark)" in page
    markup = re.sub(r"<(style|script)\b.*?</\1>", "", page, flags=re.S)
    charts_markup = "".join(re.findall(r'<svg class="ch".*?</svg>', markup, flags=re.S))
    assert charts_markup.count("<svg") == 2, "пример детектора и MAE по горизонтам"
    assert not re.search(r'(?:fill|stroke)="#', charts_markup), "цвет задаётся классом, а не числом"
    assert "style=" not in charts_markup


def test_optional_blocks_are_skipped_without_their_data(tmp_path: Path) -> None:
    """Нет архива новостей, бенчмарка и панели — нет блоков, и ни одна ссылка не повисает."""
    story = _story()
    for key in ("news_timeline", "detector_demo", "folds"):
        del story[key]
    story["contributions"] = []
    story["foundation"] = {}
    data = _data(story)
    del data["detectors"], data["ablations"]
    html = _page(landing.render(data, tmp_path / "bare.html"))
    anchors = _anchors(html)
    assert "detector-demo" not in anchors and "foundation" not in anchors
    assert "contribution" not in anchors and "changepoints" not in anchors
    assert not set(_links(html)) - anchors
    assert "MAE — средний модуль" in html, "определение метрики остаётся и без диаграммы окон"
    assert "Что видит прогноз" in html and 'class="news"' not in html
    # Сценарий страницы сам содержит разметку графика выбора ряда — его в счёт не берём.
    markup = re.sub(r"<(style|script)\b.*?</\1>", "", html, flags=re.S)
    assert 'class="folds"' not in markup
    assert markup.count('<svg class="ch"') == 1, "из графиков остался один — MAE по горизонтам"


def test_page_without_the_storyline_still_renders(tmp_path: Path) -> None:
    html = _page(landing.render(_data(None), tmp_path / "plain.html"))
    assert "Лучшая модель зависит от горизонта прогноза" in html
    assert 'id="chain"' not in html and not set(_links(html)) - _anchors(html)
    assert "Данные СберИндекса" in html
    assert 'data-tab="criteria"' not in html, "без карты критериев вкладки нет"


def test_changepoint_section_puts_the_best_raw_detector_next_to_the_residual_one(page: str) -> None:
    """Сравнение с «тем же детектором по сырому ряду» одно выглядело бы выигрышнее, чем есть."""
    block = _block(page, "changepoints")
    assert (
        "Лучший онлайн-детектор по остаткам прогноза — CUSUM: F1 0,221, задержка 2,3 мес." in block
    )
    assert "Лучший по сырому ряду — Page–Hinkley: F1 0,219, задержка 3,2 мес." in block
    assert "Тот же CUSUM по сырому ряду — F1 0,188." in block
    assert "Главное сравнение" not in block and "важнее" not in block


def test_changepoint_section_shows_the_schedule_reference(tmp_path: Path) -> None:
    """Эталон в таблице назван по-русски, а рядом сказано, сколько методов его превосходят."""
    data = _data(_story())
    data["detectors"][0]["param"] = 3.0
    data["detectors"].append(
        {
            "detector": "periodic",
            "kind": "baseline",
            "input": "none",
            "param": 5.0,
            "f1": 0.197,
            "precision": 0.118,
            "recall": 0.604,
            "mean_delay": 2.0,
            "false_alarms_per_series_year": 1.68,
        }
    )
    html = _page(landing.render(data, tmp_path / "index.html"))
    block = _block(html, "changepoints")
    row = block[block.index('<tr class="reference">') :].split("</tr>")[0]
    assert ">Тревога по расписанию</th>" in row
    assert '<td><span class="tag-ref">эталон</span></td>' in row
    assert ">раз в 5 месяцев</td>" in row and '<i class="ref"' in row
    assert ">periodic<" not in block and ">baseline<" not in block and ">none<" not in block
    assert (
        "Эталон сравнения — тревога по расписанию, каждые 5 месяцев независимо от данных: "
        "F1 0,197. Выше эталона по F1 — 1 из 1 вариантов «метод × вход»; лучший превосходит его на 0,016."
        in block
    )
    # Строк больше, чем помещается в таблицу, а эталон слабее всех: он всё равно показан.
    method = data["detectors"][0]
    data["detectors"] = [{**method, "f1": 0.5 - 0.01 * i} for i in range(13)]
    data["detectors"].append({**method, "detector": "periodic", "kind": "baseline", "f1": 0.01})
    crowded = _page(landing.render(data, tmp_path / "crowded.html"))
    table = _block(crowded, "changepoints")
    table = table[table.index('<table class="variants">') :].split("</table>")[0]
    assert ">Тревога по расписанию</th>" in table
    assert table.count(">Page–Hinkley</th>") == 12


def test_variant_table_marks_the_three_detectors_the_conclusion_speaks_about(
    tmp_path: Path,
) -> None:
    data = _data(_story())
    base = {"kind": "online", "mean_delay": 2.0, "false_alarms_per_series_year": 0.9}
    data["detectors"] = [
        {**base, "detector": "cusum", "input": "residual", "f1": 0.221},
        {**base, "detector": "page_hinkley", "input": "raw", "f1": 0.219},
        {**base, "detector": "conformal", "input": "raw", "f1": 0.21},
        {**base, "detector": "cusum", "input": "raw", "f1": 0.188},
    ]
    html = _page(landing.render(data, tmp_path / "marks.html"))
    table = _block(html, "changepoints")
    table = table[table.index('<table class="variants">') :].split("</table>")[0]
    marks = re.findall(r'<span class="letter">(\w)</span></td><th scope="row">([^<]+)</th>', table)
    assert marks == [("A", "CUSUM"), ("B", "Page–Hinkley"), ("C", "CUSUM")]
    assert table.count('<td class="resid">остатки прогноза</td>') == 1
    assert table.count('<i class="acc"') == 1, "точка входа «остатки прогноза» — цветом"


def test_changepoint_section_compares_the_detector_with_an_equally_noisy_schedule(
    tmp_path: Path,
) -> None:
    data = _data(_story())
    data["detectors"] = [
        {"detector": "cusum", "kind": "online", "input": "residual", "param": 3.0, "f1": 0.221},
        {"detector": "periodic", "kind": "baseline", "input": "none", "param": 6.0, "f1": 0.227},
    ]
    data["meta"]["detector_contrast"] = {
        **CONTRAST,
        "alarmed_breaks": 0.804,
        "alarmed_controls": 0.714,
        "reference": {"f1": 0.227, "period": 6, "delay": 2.18},
    }
    data["meta"]["schedule_match"] = {
        "detector": "cusum",
        "period": 8,
        "detector_false_alarms": 0.889,
        "schedule_false_alarms": 0.872,
        "detector_recall": 0.430,
        "schedule_recall": 0.344,
        "detector_delay": 2.275,
        "schedule_delay": 3.082,
        "recall_diff": 0.086,
        "recall_diff_low": 0.031,
        "recall_diff_high": 0.142,
        "level": 0.95,
    }
    html = _page(landing.render(data, tmp_path / "index.html"))
    block = _block(html, "changepoints")
    assert "Расписание с ближайшей частотой ложных тревог — раз в 8 месяцев" in block
    assert "95%-й интервал бутстрепа по рядам — от 3,1 до 14,2 п.п." in block
    assert "На рядах без внесённого слома CUSUM поднимает тревогу у 71% рядов" in block
    card = _text(block[block.index('<div class="match">') :].split('<div class="findings">')[0])
    assert "CUSUM по остаткам прогноза 0,89 ложной тревоги на ряд в год 43% сломов" in card
    assert "Расписание: тревога раз в 8 месяцев 0,87 ложной тревоги на ряд в год 34% сломов" in card
    assert "+8,6 процентного пункта" in card
    assert "Интервал от 3,1 до 14,2 п.п. ноль не включает." in card
    conclusion = block[block.index('<div class="concl">') :].split("</div>")[0]
    assert "По F1 детектор эталон не превосходит. При равной частоте ложных тревог" in conclusion
    chain = _block(html, "chain")
    assert "Тревога по расписанию" in chain and "<b>0,227</b>" in chain, "эталон — на первом экране"

    data["meta"]["schedule_match"]["recall_diff_low"] = -0.02
    unsure = _page(landing.render(data, tmp_path / "zero.html"))
    assert "включает ноль: перевес не доказан." in _text(_block(unsure, "changepoints"))

    del data["meta"]["schedule_match"]
    data["meta"]["detector_contrast"] = CONTRAST
    plain = _block(_page(landing.render(data, tmp_path / "plain.html")), "changepoints")
    assert "Расписание с ближайшей частотой" not in plain
    assert "На рядах без внесённого слома" not in plain
    assert '<div class="match">' not in plain


def test_national_case_names_the_method_it_was_built_with(tmp_path: Path) -> None:
    data = _data(_story())
    data["meta"]["cases"] = {
        "national": [
            {
                "series": "Всего",
                "method": "window_l2",
                "penalty": 0.5,
                "alarms": 9,
                "coverage": 0.473,
                "margin": 2,
                "events": [
                    {"event": "шок спроса", "date": "2020-04-01", "detected": True},
                    {"event": "ставка 20%", "date": "2022-03-01", "detected": False},
                ],
                "ds": ["2020-03", "2020-04", "2020-05"],
                "values": [1.0, 2.0, 3.0],
                "breaks": ["2020-04"],
            }
        ]
    }
    html = _page(landing.render(data, tmp_path / "index.html"))
    block = _block(html, "cases")
    assert (
        "Офлайн-метод «скользящее окно (L2)» со штрафом, откалиброванным на бенчмарке, находит "
        "1 из 2 размеченных шоков в пределах ±2 месяцев и ставит 9 тревог. Окна допуска вокруг "
        "этих тревог накрывают 47% ряда: с такой вероятностью «найденной» оказалась бы случайная "
        "дата." in block
    )
    assert "сломы PELT" not in block
    assert "годится" not in block, "вывода о пригодности метода данные не дают"
    events = [_text(item) for item in block.split("<li>")[1:]]
    assert events[0].startswith("1 апрель 2020 шок спроса найден")
    assert events[1].startswith("2 март 2022 ставка 20% не найден"), "события названы, а не «шок»"
    assert "<b>1 из 2</b>" in block and "<b>47%</b>" in block
    assert 'href="#cases"' in html and "Разборы реальных примеров" in html
    bar = html[html.index("<b>Разборы реальных примеров</b>") :].split("</div>")[0]
    assert "Критерий К7 · 10%" in bar, "критерий — в шапке группы разборов"
    intro = html[html.index("<b>Разборы реальных примеров</b>") :].split("<section ")[0]
    assert "Настоящие ряды: 1 национальный ряд с размеченными шоками." in intro
    assert "окно допуска ±2 месяца вокруг тревоги" in intro


def test_municipal_case_counts_the_alarm_months_and_draws_them(tmp_path: Path) -> None:
    data = _data(_story())
    data["meta"]["live"] = {"detector": "cusum"}
    data["meta"]["cases"] = {
        "municipal": {
            "unique_id": "04-0708__marketplaces",
            "region": "Красноярский край",
            "mo": "Мотыгинский муниципальный район",
            "category": "Маркетплейсы",
            "max_abs_z": 18.279,
            "alarms": ["2024-01-01", "2024-03-01"],
            "ds": ["2024-01", "2024-02", "2024-03"],
            "y": [3160.0, 5462.0, 5952.0],
            "y_hat": [2301.7, 2512.3, 2391.8],
            "q_lo": [1298.3, 1267.1, 991.4],
            "q_hi": [3640.7, 4248.8, 4516.8],
        }
    }
    html = _page(landing.render(data, tmp_path / "municipal.html"))
    block = _block(html, "cases")
    assert "Разбор 1 · ряд МО · 2024" in block and "Красноярский край" in block
    assert "<h2>Мотыгинский муниципальный район · Маркетплейсы</h2>" in block
    assert "Живой детектор CUSUM по остаткам ансамбля" in block
    tiles = [_text(tile) for tile in re.split(r'<div class="tile(?: \w+)?">', block)[1:]]
    assert tiles[0] == "2 из 3 месяцев с тревогой детектора"
    assert "|z| = 18,3 крупнейший остаток прогноза в потоке живого детектора" in tiles[1]
    assert "24 месяца истории у ряда" in tiles[2], "длина панели — из собранных фактов"
    assert block.count('class="tri"') == 2 and block.count('class="ring"') == 2
    assert "Маркетплейсы, ₽ в месяц" in block
    assert (
        "окно допуска"
        not in html[html.index("<b>Разборы реальных примеров</b>") :].split("<section ")[0]
    ), "без национальных рядов легенды окон допуска нет"


def test_inline_code_in_texts_is_rendered_as_code(tmp_path: Path) -> None:
    data = _data(_story())
    data["meta"]["entity_summary"] = ["Лаги публикации — `configs/features.yaml`."]
    data["meta"]["footer"] = "Сформировано командой `sbx report build` <из артефактов>."
    html = _page(landing.render(data, tmp_path / "code.html"))
    assert "<code>configs/features.yaml</code>" in html
    assert "<code>sbx report build</code> &lt;из артефактов&gt;" in html, "остальное экранируется"
    assert "`configs/" not in html and "`sbx" not in html


def test_territory_check_stands_on_its_own() -> None:
    """Блок сверки не отсылает к другому документу: проверка описана в нём самом."""
    block = blocks.territories_html(ENTITY)
    assert "Как проверили" in block
    assert "<code>" not in block and ".md" not in block


def test_criteria_artifacts_are_files_of_the_repository() -> None:
    """Артефакт критерия — файл репозитория либо результат прогона из `artifacts/`."""
    root = Path(__file__).resolve().parents[2]
    paths = [row["artifact"] for row in criteria_map()]
    missing = [p for p in paths if not p.startswith("artifacts/") and not (root / p).exists()]
    assert not missing, f"таблица критериев отсылает к файлам, которых нет: {missing}"


def test_citation_addresses_are_links(page: str) -> None:
    block = _block(page, "reproduce")
    assert (
        '<a href="https://sberindex.ru/ru/research/x">https://sberindex.ru/ru/research/x</a>'
        in block
    )
    assert '(<a href="https://www.gdeltproject.org/">https://www.gdeltproject.org/</a>)' in block
    assert "<i>[1]</i>" in block and "<i>[3]</i>" in block, "подписи пронумерованы"


def test_reproduce_section_offers_the_commands_with_copy_buttons(page: str) -> None:
    block = _block(page, "reproduce")
    commands = re.findall(
        r'<div class="cmd"><code>([^<]+)</code><button type="button" data-copy', block
    )
    assert commands == ["make all", "docker compose run --rm pipeline", "make report"]
    assert "<code>README.md</code>" in block and "<code>DATA_LICENSES.md</code>" in block


def test_ablation_section_says_on_which_rows_the_configurations_are_compared(
    tmp_path: Path,
) -> None:
    data = _data(_story())
    data["ablations"] = [
        {"name": "A", "mae": 636.7, "delta_mae": 0.0, "models": "LightGBM", "rows": 31238},
        {
            "name": "D",
            "mae": 605.6,
            "delta_mae": -31.1,
            "models": "LightGBM (B: +макро), Chronos-2 (covariates)",
            "rows": 31238,
        },
        {
            "name": "E",
            "mae": 605.6,
            "delta_mae": -31.1,
            "models": "LightGBM (B: +макро), Chronos-2 (covariates)",
            "rows": 31238,
        },
    ]
    html = _page(landing.render(data, tmp_path / "rows.html"))
    block = _block(html, "contribution")
    assert "Все конфигурации измерены на одних и тех же 31 238 наблюдениях." in block
    assert "−31,1" in block, "разница в таблице — со знаком минуса"
    assert "F1 обнаружения" not in block, "столбца без данных нет, как и сноски к нему"
    assert "<i>D</i><span>Набор моделей с foundation-моделями</span>" in block
    assert '<td class="muted">LightGBM (B: +макро), Chronos-2 с ковариатами</td>' in block
    assert '<td class="muted">Как в D</td>' in block, "тот же набор моделей не повторяется"


SOURCES = [
    {
        "block": "neighbours",
        "title": "Расходы соседних МО",
        "against": "A",
        "mean_diff": -12.3,
        "p_value": 0.0004,
        "folds": 10,
        "folds_better": 9,
        "folds_worse": 1,
        "sign_p": 0.021,
    },
    {
        "block": "neighbours",
        "title": "Расходы соседних МО",
        "against": "B",
        "mean_diff": 1.2,
        "p_value": 0.41,
        "folds": 10,
        "folds_better": 5,
        "folds_worse": 5,
        "sign_p": 1.0,
    },
]


def test_contribution_section_shows_every_source_on_its_own(tmp_path: Path) -> None:
    story = _story()
    story["source_contributions"] = source_view(SOURCES)
    story["source_verdict"] = source_verdict(SOURCES)
    html = _page(landing.render(_data(story), tmp_path / "sources.html"))
    block = _block(html, "contribution")
    assert "Каждый источник по отдельности" in block
    assert "Устойчиво снижают ошибку: Расходы соседних МО (к базовой модели)." in block
    group = block[block.index('<div class="div-group">') :]
    assert group.startswith('<div class="div-group"><b>Расходы соседних МО</b>')
    rows = [_text(row) for row in group.split('<div class="div-row">')[1:3]]
    assert rows[0].startswith("к базовой модели") and "−12,3 ₽ 9 из 10 помог" in rows[0]
    assert rows[1].startswith("поверх календаря и национальных рядов")
    assert "+1,2 ₽ 5 из 10 неясно" in rows[1]
    tiles = _text(block[block.index('<div class="tiles">') :].split('<div class="div-head">')[0])
    assert "1 сравнений: источник устойчиво помог" in tiles and "1 из 2 неясно" in tiles
    assert "полоса во всю половину — 65 ₽" in block, "шкала общая с диаграммой слоёв: до 60,8"
    assert "Устойчиво снижают ошибку" in _block(html, "data"), "тот же вывод — у таблицы источников"


def test_contribution_section_without_source_rows_has_no_empty_diagram(page: str) -> None:
    assert "Каждый источник по отдельности" not in _block(page, "contribution")


def test_horizon_section_says_in_how_many_windows_the_leader_beats_prophet(
    tmp_path: Path,
) -> None:
    """Среднее по всем окнам может создать одно окно: рядом с MAE стоит счёт окон."""
    data = _data(_story())
    data["meta"]["significance"] = {
        "by_horizon": {
            "1": {
                "best_model": "Chronos-2 (covariates)",
                "dm": [
                    {
                        "pair": "Chronos-2 (covariates) vs Prophet",
                        "mean_diff": -97.5,
                        "folds": 4,
                        "folds_better": 1,
                        "folds_worse": 3,
                        "sign_p": 0.625,
                    }
                ],
            }
        }
    }
    html = _page(landing.render(data, tmp_path / "windows.html"))
    block = _block(html, "horizons")
    assert "По окнам проверки лучшая модель точнее Prophet: 1 месяц — в 1 окне из 4." in block
    assert "средний перевес создан меньшинством окон" in block
    leaders = block[block.index('<tr class="leaders">') :].split("</tr>")[0]
    assert 'style="--n:1;--of:4"' in leaders and "1 из 4" in leaders
    chain = _block(html, "chain")
    assert "Точнее Prophet" in chain and 'style="--n:1;--of:4"' in chain
    assert "На горизонте 1 месяц средний перевес создан меньшинством окон" in chain


def test_changepoint_section_says_what_the_alarms_on_real_residuals_mean(tmp_path: Path) -> None:
    data = _data(_story())
    data["meta"]["live"] = {
        "detector": "cusum",
        "series": 1111,
        "alarms": 3067,
        "alarms_per_series_year": 2.77,
        "conformal_alarms": 444,
        "series_with_alarm": 935,
        "above_forecast_share": 0.705,
    }
    html = _page(landing.render(data, tmp_path / "live.html"))
    block = _block(html, "changepoints")
    assert "CUSUM поднимает 3 067 тревог — 2,77 на ряд в год" in block
    assert "часть тревог отражает систематическое отставание прогноза" in block


SHOWCASE = {
    "unique_id": "77-2348__total",
    "region": "Чукотский автономный округ",
    "mo": "Анадырский муниципальный район",
    "category": "Все категории",
    "ds": ["2024-01", "2024-02", "2024-03"],
    "y": [49892.0, 53049.0, 63556.0],
    "y_hat": [59094.0, 50145.0, 59452.0],
    "q_lo": [49761.0, 48253.0, 45116.0],
    "q_hi": [72800.0, 78670.0, 81325.0],
}


def test_forecast_examples_and_the_series_picker_use_the_same_stream(tmp_path: Path) -> None:
    data = _data(_story())
    data["showcase"] = [SHOWCASE]
    data["meta"]["live"] = {"detector": "cusum", "above_forecast_share": 0.704}
    series = {
        key: SHOWCASE[key] for key in ("region", "mo", "category", "ds", "y", "y_hat", "q_hi")
    }
    series |= {"id": "77-2348__total", "q_lo": [49761.0, float("nan"), 45116.0]}
    series |= {"alarms": ["2024-03"], "band": [[0, 1], [2, 3]]}
    data["explorer"] = {
        "regions": [SHOWCASE["region"]],
        "categories": ["Все категории"],
        "series": [series],
    }
    html = _page(landing.render(data, tmp_path / "series.html"))
    examples = _block(html, "forecast-examples")
    assert (
        "Прогноз чаще ниже факта: на рядах оценочной выборки факт выше прогноза в 70,4% месяцев."
        in examples
    )
    assert "<b>Анадырский муниципальный район</b>" in examples
    assert "<span>Чукотский АО · все категории · тыс. ₽</span>" in examples, (
        "подпись — в одну строку"
    )
    assert "<p>Факт выше прогноза в 2 месяцах из 3</p>" in examples
    explorer = _block(html, "explorer")
    assert '<select id="ex-region"></select>' in explorer and '<label for="ex-region">' in explorer
    assert "тревоги детектора CUSUM по остаткам прогноза" in explorer
    payload = re.search(r'<script type="application/json">(.*?)</script>', explorer).group(1)
    assert "NaN" not in payload, "пропуск интервала — null: иначе сценарий не разберёт данные"
    parsed = json.loads(payload)
    assert parsed["series"][0]["q_lo"] == [49761.0, None, 45116.0]
    assert parsed["series"][0]["alarms"] == ["2024-03"]
