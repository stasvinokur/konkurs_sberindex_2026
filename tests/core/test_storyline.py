"""Сборщики блоков лендинга «данные → расчёт → результат»: чистые функции над данными отчёта."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sbx.core.changepoint.online import cusum, cusum_path, page_hinkley, page_hinkley_path
from sbx.core.folds import BacktestConfig, HorizonFolds, TargetSpec, make_folds
from sbx.core.storyline import (
    chain_summary,
    contribution_view,
    criteria_map,
    detector_demo,
    fold_timeline,
    foundation_view,
    month_label,
    news_timeline,
    source_inventory,
    source_statement,
    source_verdict,
    source_view,
    timing_rows,
)

CONFIG = BacktestConfig(
    status="test",
    checked_at="2026-10-02",
    target=TargetSpec("d", "MS", "l"),
    folds=(
        HorizonFolds(horizon=1, origins=("2023-12-01", "2024-09-01")),
        HorizonFolds(horizon=12, origins=("2023-10-01", "2023-12-01")),
    ),
    primary_metric="mae",
    mae_aggregation="micro",
)
MONTHS = pd.date_range("2023-01-01", periods=24, freq="MS")


def test_month_label_is_russian() -> None:
    assert month_label(pd.Timestamp("2024-05-01")) == "май 2024"
    assert month_label(pd.Timestamp("2022-11-01")) == "ноябрь 2022"


def test_fold_timeline_places_training_and_test_months_on_the_panel_axis() -> None:
    timeline = fold_timeline(make_folds(CONFIG), MONTHS)
    assert timeline["months"][0] == "2023-01" and timeline["months"][-1] == "2024-12"
    rows = {row["fold"]: row for row in timeline["rows"]}
    first = rows["h1_2023-12"]
    assert first["train"] == [0, 11] and first["test"] == [12, 12], (
        "обучение по декабрь, тест — январь"
    )
    long = rows["h12_2023-10"]
    assert long["train"] == [0, 9] and long["test"] == [10, 21]
    assert [row["horizon"] for row in timeline["rows"]] == [1, 1, 12, 12]
    summary = {group["horizon"]: group for group in timeline["horizons"]}
    assert summary[1] == {"horizon": 1, "folds": 2, "train_months": [12, 21], "test_months": 1}
    assert summary[12]["train_months"] == [10, 12] and summary[12]["test_months"] == 12


def test_fold_timeline_refuses_a_fold_outside_the_panel() -> None:
    with pytest.raises(ValueError, match="за пределами панели"):
        fold_timeline(make_folds(CONFIG), MONTHS[:20])


def test_timing_rows_say_what_a_forecast_from_a_month_sees() -> None:
    """Таблица на странице считается той же функцией, что и признаки в расчёте."""
    lags = {"weekly-inflation": 7, "key-rate": 7, "spending": 35, "wages": 60}
    titles = {
        "weekly-inflation": "Недельная инфляция",
        "key-rate": "Реальная ключевая ставка",
        "spending": "Потребительские расходы",
        "wages": "Медианная зарплата",
    }
    rows = timing_rows(pd.Timestamp("2024-06-01"), lags, titles)
    assert [(r["lag"], r["known"]) for r in rows] == [
        ("в день события", "июнь 2024"),
        ("по дате публикации события", "опубликованное до 30 июня 2024"),
        ("7 дней", "май 2024"),
        ("35 дней", "апрель 2024"),
        ("60 дней", "апрель 2024"),
        ("известен заранее", "целевой месяц"),
    ]
    assert rows[2]["source"] == "Недельная инфляция; Реальная ключевая ставка"
    assert rows[0]["source"].startswith("Новости GDELT")
    assert rows[-1]["source"].startswith("Производственный календарь")
    # В невисокосном году январские зарплаты к концу марта ещё не вышли.
    march = timing_rows(pd.Timestamp("2023-03-01"), {"wages": 60}, titles)
    assert [r["known"] for r in march if r["lag"] == "60 дней"] == ["декабрь 2022"]


def _signal() -> np.ndarray:
    rng = np.random.default_rng(4)
    return np.concatenate([rng.normal(size=30), rng.normal(loc=3.0, size=30)])


@pytest.mark.parametrize(
    ("detector", "path", "params"),
    [
        (page_hinkley, page_hinkley_path, {"threshold": 6.0, "delta": 0.05}),
        (cusum, cusum_path, {"threshold": 5.0, "drift": 0.5}),
    ],
)
def test_detector_path_is_the_statistic_behind_the_alarms(detector, path, params) -> None:
    """Трасса статистики для графика и тревоги детектора — один расчёт, а не два похожих."""
    signal = _signal()
    statistic, alarms = path(signal, **params)
    assert alarms == detector(signal, **params) and alarms, "сдвиг в середине ряда найден"
    assert len(statistic) == len(signal)
    over = [t for t, value in enumerate(statistic) if value > params["threshold"]]
    assert over == alarms, "порог превышен ровно в моменты тревог"
    assert statistic[alarms[0] + 1] < params["threshold"], "после тревоги статистика сброшена"


def test_detector_path_skips_missing_values() -> None:
    statistic, alarms = page_hinkley_path([np.nan, 0.0, 0.1, np.nan, 9.0], threshold=3.0)
    assert np.isnan(statistic[0]) and np.isnan(statistic[3]) and alarms == [4]


def _benchmark() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Три ряда: скачок уровня (+60% с десятого месяца), малый скачок и ряд без слома."""
    rng = np.random.default_rng(11)
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    series, labels = [], []
    for uid, kind, tau, delta in (
        ("synth_00001", "level_shift", 10, 0.6),
        ("synth_00002", "level_shift", 14, 0.02),
        ("synth_00003", "none", -1, 0.0),
    ):
        values = 1000.0 + rng.normal(0, 8, size=24)
        if tau >= 0:
            values[tau:] *= 1 + delta
        series.append(
            pd.DataFrame(
                {
                    "unique_id": uid,
                    "source_id": f"src-{uid}",
                    "ds": months,
                    "y": values,
                    "category": "Все категории",
                }
            )
        )
        labels.append(
            {
                "unique_id": uid,
                "source_id": f"src-{uid}",
                "kind": kind,
                "tau": tau,
                "delta": delta,
                "n_obs": 24,
            }
        )
    seasonal = pd.DataFrame({"category": "Все категории", "month": range(1, 13), "index": 1.0})
    return pd.concat(series, ignore_index=True), pd.DataFrame(labels), seasonal


def test_detector_demo_shows_a_series_with_a_known_break_and_the_alarm_after_it() -> None:
    series, labels, seasonal = _benchmark()
    demo = detector_demo(series, labels, seasonal, detector="page_hinkley", threshold=5.0)
    assert demo["unique_id"] == "synth_00001", "самый крупный слом, найденный без ложных тревог"
    assert demo["tau"] == 10 and demo["break_month"] == "2023-11"
    assert demo["alarms"] and demo["alarms"][0] >= demo["tau"]
    assert demo["delay"] == demo["alarms"][0] - 10 and demo["delay"] <= 2
    assert len(demo["months"]) == len(demo["y"]) == len(demo["forecast"]) == 24
    assert len(demo["z"]) == len(demo["statistic"]) == 24
    # Прогноз момента t — предыдущее значение (индекс сезонности здесь ровный).
    assert demo["forecast"][5] == pytest.approx(demo["y"][4])
    assert demo["forecast"][0] is None and demo["z"][0] is None
    alarm = demo["alarms"][0]
    assert demo["statistic"][alarm] > demo["threshold"] == 5.0
    assert all(v is None or v <= 5.0 for v in demo["statistic"][:alarm])
    assert demo["delta"] == pytest.approx(0.6) and demo["kind"] == "level_shift"


def test_detector_demo_takes_only_the_allowed_series_and_admits_having_none() -> None:
    series, labels, seasonal = _benchmark()
    assert detector_demo(series, labels, seasonal, "page_hinkley", 5.0, ids=["synth_00003"]) == {}
    assert detector_demo(series, labels, seasonal, "page_hinkley", 1e9) == {}, "ничего не найдено"
    with pytest.raises(ValueError, match="детектор"):
        detector_demo(series, labels, seasonal, "bocpd", 0.3)


HORIZON_ROWS = [
    {"Модель": "Chronos-2 (covariates)", "mae h=1": 580.0, "mae h=6": 640.0, "mae h=12": 1140.0},
    {"Модель": "TimesFM-2.5", "mae h=1": 660.0, "mae h=6": 790.0, "mae h=12": 1110.0},
    {"Модель": "AutoTheta", "mae h=1": 640.0, "mae h=6": 650.0, "mae h=12": 960.0},
    {"Модель": "LightGBM", "mae h=1": 860.0, "mae h=6": 600.0, "mae h=12": float("nan")},
    {"Модель": "Prophet", "mae h=1": 680.0, "mae h=6": 1350.0, "mae h=12": 4600.0},
    {"Модель": "Ensemble", "mae h=1": 817.0, "mae h=6": 488.0, "mae h=12": 793.0},
]
WEIGHTS = {
    "chronos2": {
        "repo_id": "amazon/chronos-2",
        "revision": "29ec3766d36d6f73f0696f85560a422f50e8498c",
        "weights_license": "apache-2.0",
    },
    "timesfm25": {
        "repo_id": "google/timesfm-2.5-200m-pytorch",
        "revision": "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
        "weights_license": "apache-2.0",
    },
}


def test_foundation_view_sets_foundation_models_against_the_rest() -> None:
    view = foundation_view(HORIZON_ROWS, WEIGHTS)
    assert view["horizons"] == [1, 6, 12]
    rows = {row["model"]: row for row in view["rows"]}
    assert rows["Chronos-2 (covariates)"]["role"] == "foundation"
    assert rows["Prophet"]["role"] == "reference" and rows["Ensemble"]["role"] == "ensemble"
    best_other = next(row for row in view["rows"] if row["role"] == "best_other")
    assert best_other["values"] == [640.0, 600.0, 960.0]
    assert best_other["names"] == ["AutoTheta", "LightGBM", "AutoTheta"], (
        "у каждого горизонта свой лучший"
    )
    assert "LightGBM" not in rows, "остальные модели сведены в одну строку"
    assert view["wins"] == [1], "foundation-модель лучше остальных только на горизонте 1 месяц"
    assert view["beats_reference"] == [1, 6, 12]
    assert view["weights"][0] == {
        "repo": "amazon/chronos-2",
        "revision": "29ec3766",
        "license": "apache-2.0",
    }
    assert foundation_view([], WEIGHTS) == {}


def test_contribution_view_orders_layers_and_marks_direction() -> None:
    def windows(better: int, sign_p: float) -> dict:
        return {"folds": 10, "folds_better": better, "folds_worse": 10 - better, "sign_p": sign_p}

    dm = {
        "C_vs_B": {"mean_diff": 6.7, "p_value": 0.31, **windows(4, 0.754)},
        "B_vs_A": {"mean_diff": -60.8, "p_value": 2.0e-9, **windows(9, 0.021)},
        "D_vs_B": {"mean_diff": 12.4, "p_value": 0.004, **windows(1, 0.021)},
    }
    rows = contribution_view(dm)
    assert [row["pair"] for row in rows] == ["B_vs_A", "C_vs_B", "D_vs_B"]
    assert [row["direction"] for row in rows] == ["better", "none", "worse"]
    assert rows[0]["delta"] == -60.8 and rows[0]["significant"] is True
    assert rows[0]["subject"] == "Календарь и национальные ряды"
    assert [(row["agree"], row["folds"]) for row in rows] == [(9, 10), (6, 10), (9, 10)], (
        "окна, согласные со знаком средней разности"
    )
    assert rows[1]["statement"].startswith(
        "Новости и календарь событий поверх них: устойчивого изменения нет"
    )
    assert contribution_view({}) == []


FACTS = {
    "spending": {
        "rows": 303126,
        "territories": 2190,
        "series": 13140,
        "first": "2023-01",
        "last": "2024-12",
        "categories": 6,
    },
    "reference": {"territories": 2190, "year": 2024, "with_coordinates": 1943},
    "national": [
        {
            "title": "Потребительские расходы",
            "frequency": "месяц",
            "first": "2013-01",
            "last": "2026-08",
            "lag_days": 35,
        },
        {
            "title": "Недельная инфляция",
            "frequency": "неделя",
            "first": "2021-01",
            "last": "2026-09",
            "lag_days": 7,
        },
        {
            "title": "Медианная зарплата",
            "frequency": "месяц",
            "first": "2019-01",
            "last": "2026-06",
            "lag_days": 60,
        },
    ],
    "national_features": 61,
    "calendar": {"first_year": 2018, "last_year": 2026},
    "news": {
        "events": 3011689,
        "regions": 82,
        "first": "2022-10",
        "last": "2024-12",
        "columns": 46,
    },
    "events": {"count": 23, "first": "2020-03", "last": "2024-10"},
}
DM = {
    "B_vs_A": {
        "mean_diff": -60.8,
        "p_value": 2.0e-9,
        "folds": 10,
        "folds_better": 9,
        "folds_worse": 1,
        "sign_p": 0.021,
    },
    "C_vs_B": {
        "mean_diff": 6.7,
        "p_value": 0.31,
        "folds": 10,
        "folds_better": 4,
        "folds_worse": 6,
        "sign_p": 0.754,
    },
}


def test_source_inventory_describes_every_source_from_the_collected_facts() -> None:
    rows = {row["key"]: row for row in source_inventory(FACTS, DM)}
    assert list(rows) == ["spending", "reference", "national", "calendar", "news", "events"]
    assert rows["spending"]["volume"] == (
        "303 126 строк: 2 190 МО, 13 140 рядов; январь 2023 — декабрь 2024"
    )
    assert rows["spending"]["level"] == "МО" and rows["spending"]["gave"] == "цель прогноза"
    assert rows["national"]["volume"] == "3 ряда; с января 2013 по сентябрь 2026; месяц и неделя"
    assert rows["national"]["timing"] == "лаг публикации 7–60 дней от конца периода"
    assert rows["national"]["what"].startswith("61 макропризнак; сезонный индекс")
    many = source_inventory({**FACTS, "national_features": 25}, DM)
    assert {r["key"]: r for r in many}["national"]["what"].startswith("25 макропризнаков; ")
    assert rows["national"]["gave"] == "−60,8 руб., лучше в 9 окнах из 10 — устойчиво"
    assert rows["calendar"]["timing"] == "известен заранее: берётся на целевой месяц"
    assert rows["news"]["volume"] == "3 011 689 событий, 82 региона; октябрь 2022 — декабрь 2024"
    assert rows["news"]["gave"] == "+6,7 руб., хуже в 6 окнах из 10 — неустойчиво"
    assert rows["events"]["volume"] == "23 события; март 2020 — октябрь 2024"
    assert all(row["source"] and row["what"] and row["license"] for row in rows.values())


def test_source_inventory_skips_absent_sources_and_admits_unmeasured_contribution() -> None:
    """Без архива новостей строки о нём нет; без абляций вклад не выдумывается."""
    facts = {k: v for k, v in FACTS.items() if k != "news"}
    rows = {row["key"]: row for row in source_inventory(facts, {})}
    assert "news" not in rows
    assert rows["national"]["gave"] == "не измерено в этом прогоне"


WINNERS = [
    {"horizon": 1, "model": "Chronos-2 (covariates)", "mae": 580.0, "reference_mae": 683.0},
    {"horizon": 3, "model": "Chronos-2 (covariates)", "mae": 590.0, "reference_mae": 835.0},
    {"horizon": 6, "model": "Ensemble", "mae": 488.0, "reference_mae": 1353.0},
    {"horizon": 12, "model": "Ensemble", "mae": 793.0, "reference_mae": 4600.0},
]


def test_chain_summary_links_every_line_to_a_section() -> None:
    chain = chain_summary(
        FACTS,
        folds=14,
        models=15,
        winners=WINNERS,
        contrast={
            "detector": "cusum",
            "f1_residual": 0.221,
            "f1_raw": 0.188,
            "delay_residual": 2.275,
            "best_raw": {"detector": "page_hinkley", "f1": 0.219, "delay": 3.242},
        },
        dm=DM,
    )
    assert list(chain) == ["data", "method", "result"]
    for column in chain.values():
        assert column["title"] and column["lines"]
        assert all(line["text"] and line["href"].startswith("#") for line in column["lines"])
    data = [line["text"] for line in chain["data"]["lines"]]
    assert data[0] == "Расходы МО: 2 190 МО × 6 категорий × 24 месяца"
    assert "Новости GDELT: 3 011 689 событий" in data
    method = [line["text"] for line in chain["method"]["lines"]]
    assert method[0] == "Проверка: 14 окон, горизонты 1, 3, 6 и 12 месяцев, метрика MAE"
    result = [line["text"] for line in chain["result"]["lines"]]
    assert result[0] == "MAE ниже, чем у Prophet, на 4 горизонтах из 4"
    assert result[1] == "Лидеры: 1 и 3 мес. — Chronos-2 (covariates); 6 и 12 мес. — Ensemble"
    assert result[2] == (
        "Сдвиги по остаткам прогноза: F1 0,221, задержка 2,3 мес.; "
        "по сырому ряду — 0,219 и 3,2 мес."
    ), "рядом с остатками — лучший детектор по сырому ряду, а не удобный для сравнения"


def test_chain_summary_does_not_claim_a_win_that_did_not_happen() -> None:
    lost = [{"horizon": 1, "model": "Prophet", "mae": 600.0, "reference_mae": 600.0}, WINNERS[2]]
    chain = chain_summary(FACTS, folds=5, models=3, winners=lost, contrast={}, dm={})
    result = [line["text"] for line in chain["result"]["lines"]]
    assert result[0] == "MAE ниже, чем у Prophet, на 1 горизонте из 2"
    assert not any("F1" in text for text in result), "нет сравнения детекторов — нет и строки"


def test_criteria_map_covers_every_criterion_and_links_only_to_existing_sections() -> None:
    full = criteria_map()
    assert [row["weight"] for row in full] == [10, 20, 20, 15, 15, 10, 10]
    assert sum(row["weight"] for row in full) == 100
    assert all(row["links"] and row["report"] and row["artifact"] for row in full)
    present = {"horizons", "protocol", "data"}
    partial = criteria_map(present)
    for row in partial:
        assert all(link["href"].lstrip("#") in present for link in row["links"])
    assert any(not row["links"] for row in partial), "раздела нет на странице — нет и ссылки"


def test_news_timeline_pairs_monthly_counts_with_dated_events() -> None:
    months = pd.to_datetime(["2024-06-01", "2024-07-01", "2024-08-01"])
    news = pd.DataFrame(
        {
            "region_code": ["45", "46"] * 3,
            "ds": np.repeat(months, 2),
            "country_news_events": np.repeat([100, 150, 120], 2),
        }
    )
    events = [
        {
            "date": "2024-07-26",
            "kind": "key_rate_hike",
            "region_code": "all",
            "description": "Ключевая ставка повышена до 18%",
        },
        {"date": "2021-01-01", "kind": "flood", "region_code": "53", "description": "вне периода"},
    ]
    timeline = news_timeline(news, events)
    assert timeline["months"] == ["2024-06", "2024-07", "2024-08"]
    assert timeline["counts"] == [100, 150, 120]
    assert timeline["events"] == [
        {
            "month": "2024-07",
            "date": "2024-07-26",
            "kind": "key_rate_hike",
            "description": "Ключевая ставка повышена до 18%",
            "scope": "вся страна",
        }
    ]
    assert news_timeline(news.iloc[:0], events) == {}


def _source(block: str, title: str, against: str, diff: float, better: int) -> dict:
    """Строка вклада источника: разность и число окон из десяти, где с источником точнее."""
    return {
        "block": block,
        "title": title,
        "against": against,
        "mean_diff": diff,
        "p_value": 1e-9,
        "folds": 10,
        "folds_better": better,
        "folds_worse": 10 - better,
        "sign_p": 0.021 if better in (0, 1, 9, 10) else 0.754,
    }


NEIGHBOURS_TITLE = "Расходы соседних МО по автодорогам"
SOURCES = [
    _source("macro", "Национальные ряды СберИндекса", "A", -25.0, better=9),
    _source("neighbours", NEIGHBOURS_TITLE, "A", -12.3, better=9),
    _source("neighbours", NEIGHBOURS_TITLE, "B", 1.2, better=5),
    _source("gdelt", "Новости GDELT", "A", 9.0, better=1),
    _source("gdelt", "Новости GDELT", "B", 8.0, better=0),
]
SPATIAL = {
    "territories": 2190,
    "with_market_access": 2170,
    "with_neighbours": 2172,
    "neighbours": 5,
    "highway_pairs": 2353366,
    "median_distance_km": 57.5,
    "cross_region_share": 0.147,
}


def test_source_statement_gives_the_block_alone_and_on_top_of_the_best_set() -> None:
    assert source_statement(SOURCES, "neighbours") == (
        "к базовой модели: −12,3 руб., лучше в 9 окнах из 10 — устойчиво; "
        "поверх календаря и национальных рядов: +1,2 руб., хуже в 5 окнах из 10 — неустойчиво"
    )
    assert source_statement(SOURCES, "macro") == (
        "к базовой модели: −25,0 руб., лучше в 9 окнах из 10 — устойчиво"
    )
    assert source_statement(SOURCES, "access") == "", "не измерено — не выдумывается"
    assert source_statement([], "macro") == ""


def test_source_inventory_prefers_the_contribution_measured_for_the_source_itself() -> None:
    rows = {row["key"]: row for row in source_inventory({**FACTS, "spatial": SPATIAL}, DM, SOURCES)}
    assert list(rows) == [
        "spending",
        "reference",
        "national",
        "calendar",
        "news",
        "events",
        "access",
        "neighbours",
    ]
    assert rows["national"]["gave"] == source_statement(SOURCES, "macro")
    assert rows["news"]["gave"] == source_statement(SOURCES, "gdelt")
    assert rows["calendar"]["gave"] == "входит в сравнение с национальными рядами", "не измерен"
    access, near = rows["access"], rows["neighbours"]
    assert access["level"] == "МО" and near["level"] == "МО"
    assert access["volume"] == "2 170 из 2 190 МО панели; расчёт организаторов за 2024 год"
    assert "позже части моментов прогноза" in access["timing"]
    assert access["gave"] == "не измерено в этом прогоне"
    assert near["volume"] == (
        "2 353 366 пар МО панели по автодорогам; 5 ближайших соседей есть у 2 172 МО, медианное "
        "расстояние 58 км, 15% соседей — в другом регионе"
    )
    assert near["timing"] == "расходы соседей — по месяц прогноза включительно, как и свой ряд"
    assert near["gave"] == source_statement(SOURCES, "neighbours")
    assert access["license"] == "CC BY-SA 4.0" and near["license"] == "CC BY-SA 4.0"


def test_source_inventory_without_spatial_facts_has_no_spatial_rows() -> None:
    keys = [row["key"] for row in source_inventory(FACTS, DM, SOURCES)]
    assert "access" not in keys and "neighbours" not in keys


def test_source_view_gives_a_bar_for_every_measured_pair() -> None:
    rows = source_view(SOURCES)
    assert [row["pair"] for row in rows] == [
        "macro_on_A",
        "neighbours_on_A",
        "neighbours_on_B",
        "gdelt_on_A",
        "gdelt_on_B",
    ]
    assert rows[1]["subject"] == f"{NEIGHBOURS_TITLE} — к базовой модели"
    assert [row["direction"] for row in rows] == ["better", "better", "none", "worse", "worse"]
    assert rows[2]["statement"] == (
        f"{NEIGHBOURS_TITLE} — поверх календаря и национальных рядов: устойчивого изменения "
        "нет — MAE выше на 1,2 руб. в среднем, хуже в 5 окнах из 10."
    )
    assert rows[1]["delta"] == -12.3 and rows[1]["significant"] is True
    assert rows[1]["windows"] == "в 9 окнах из 10" and rows[2]["windows"] == "в 5 окнах из 10"
    assert (rows[1]["agree"], rows[1]["folds"]) == (9, 10)
    assert rows[1]["title"] == NEIGHBOURS_TITLE and rows[1]["base"] == "к базовой модели"
    assert rows[2]["base"] == "поверх календаря и национальных рядов"
    assert source_view([]) == []


def test_timing_rows_cover_the_sources_of_the_municipal_level() -> None:
    rows = timing_rows(pd.Timestamp("2024-06-01"), {"spending": 35}, {}, spatial=True)
    by_source = {row["source"]: row for row in rows}
    near = by_source["Расходы соседних МО (набор организаторов)"]
    assert near == {
        "source": "Расходы соседних МО (набор организаторов)",
        "lag": "как собственный ряд",
        "known": "июнь 2024",
        "tag": "как свой ряд",
        "kind": "published",
        "known_month": "2024-06",
    }
    access = by_source["Доступность рынков и расстояния между МО (набор организаторов)"]
    assert access["lag"] == "постоянная характеристика"
    assert access["known"] == "расчёт организаторов за 2024 год"
    plain = timing_rows(pd.Timestamp("2024-06-01"), {"spending": 35}, {})
    assert not any("соседних МО" in row["source"] for row in plain)


def test_chain_names_the_municipal_level_tables_when_they_are_used() -> None:
    chain = chain_summary(
        {**FACTS, "spatial": SPATIAL}, folds=14, models=15, winners=WINNERS, contrast={}, dm=DM
    )
    data = [line["text"] for line in chain["data"]["lines"]]
    assert "Таблицы организаторов уровня МО: доступность рынков и 5 ближайших соседей" in data
    plain = chain_summary(FACTS, folds=14, models=15, winners=WINNERS, contrast={}, dm=DM)
    assert not any("уровня МО" in line["text"] for line in plain["data"]["lines"])


WEATHER = {
    "cells": 1343,
    "territories": 2190,
    "lag_days": 7,
    "first": "2022-01",
    "last": "2024-12",
    "norm_years": [2001, 2020],
}


def test_source_inventory_describes_the_weather_source() -> None:
    measured = [_source("weather", "Погода NASA POWER", "A", 2.0, better=4)]
    rows = {
        row["key"]: row for row in source_inventory({**FACTS, "weather": WEATHER}, DM, measured)
    }
    weather = rows["weather"]
    assert weather["level"] == "МО"
    assert weather["volume"] == (
        "1 343 ячейки сетки 0,5° × 0,625° для 2 190 МО; январь 2022 — декабрь 2024, "
        "норма — 2001–2020 годы"
    )
    assert weather["timing"] == (
        "лаг публикации 7 дней: к концу месяца известен предыдущий; норма известна заранее"
    )
    assert weather["gave"] == "к базовой модели: +2,0 руб., хуже в 6 окнах из 10 — неустойчиво"
    assert "NASA" in weather["source"] and "источник" in weather["license"]
    assert "weather" not in {row["key"] for row in source_inventory(FACTS, DM)}


def test_contribution_view_calls_a_layer_steady_only_when_the_windows_agree() -> None:
    """Столбец диаграммы окрашен, только если знак разности держится по окнам проверки."""
    rows = {row["pair"]: row for row in contribution_view(DM)}
    assert rows["B_vs_A"]["significant"] is True and rows["B_vs_A"]["direction"] == "better"
    assert rows["B_vs_A"]["windows"] == "в 9 окнах из 10"
    assert rows["C_vs_B"]["significant"] is False and rows["C_vs_B"]["direction"] == "none"
    by_series_only = {"B_vs_A": {"mean_diff": -60.8, "p_value": 2.0e-9}}
    row = contribution_view(by_series_only)[0]
    assert row["significant"] is False and row["windows"] == "", "p по рядам столбец не красит"
    assert row["agree"] is None and row["folds"] is None


def test_source_verdict_names_what_holds_across_the_windows() -> None:
    """Одна фраза над диаграммой: что устойчиво помогает, что устойчиво вредит."""
    assert source_verdict(SOURCES) == (
        "Устойчиво снижают ошибку: Национальные ряды СберИндекса (к базовой модели), "
        f"{NEIGHBOURS_TITLE} (к базовой модели). Устойчиво повышают: Новости GDELT (к базовой "
        "модели), Новости GDELT (поверх календаря и национальных рядов)."
    )


def test_source_verdict_says_so_when_nothing_holds() -> None:
    shaky = [
        _source("calendar", "Производственный календарь", "A", -58.7, better=8),
        _source("macro", "Национальные ряды СберИндекса", "A", 18.9, better=5),
        _source("gdelt", "Новости GDELT", "B", -20.4, better=7),
    ]
    assert source_verdict(shaky) == (
        "Ни один источник не изменил ошибку устойчиво: знак разности меняется от окна к окну. "
        "Ближе всех — Производственный календарь (к базовой модели): лучше в 8 окнах из 10."
    )
    assert source_verdict([]) == ""


POPULATION = {
    "territories": 2190,
    "matched": 2019,
    "share": 0.9219,
    "threshold": 0.9,
    "used": True,
    "year": 2023,
    "published": "2023-09-04",
}


def test_source_inventory_describes_the_population_and_how_much_of_it_was_matched() -> None:
    measured = [_source("population", "Численность населения (Росстат)", "A", -3.0, better=5)]
    facts = {**FACTS, "population": POPULATION, "first_forecast": "2023-10-31"}
    row = {r["key"]: r for r in source_inventory(facts, DM, measured)}["population"]
    assert row["level"] == "МО" and "Росстат" in row["source"]
    assert row["volume"] == (
        "2 019 из 2 190 МО панели (92,2%); привязка по названию внутри региона через бюллетень "
        "2024 года"
    )
    assert row["timing"] == (
        "постоянная характеристика: оценка на 1 января 2023 года в редакции, опубликованной "
        "к 4 сентября 2023 года, — до первого момента прогноза (31 октября 2023 года)"
    )
    assert row["gave"] == "к базовой модели: −3,0 руб., лучше в 5 окнах из 10 — неустойчиво"


def test_population_published_after_a_forecast_moment_is_said_to_be_late() -> None:
    """Сравнение дат делает расчёт: редакция, вышедшая после первого момента прогноза, не
    может быть названа известной заранее."""
    facts = {**FACTS, "population": POPULATION, "first_forecast": "2023-08-31"}
    row = {r["key"]: r for r in source_inventory(facts, DM)}["population"]
    assert row["timing"].endswith(
        "опубликованной к 4 сентября 2023 года, — позже части моментов прогноза (первый — "
        "31 августа 2023 года)"
    )
    same_day = {**facts, "first_forecast": "2023-09-04"}
    late = {r["key"]: r for r in source_inventory(same_day, DM)}["population"]["timing"]
    assert "позже части моментов прогноза" in late, "в день прогноза ещё не «до»"
    undated = {**FACTS, "population": {**POPULATION, "published": None}}
    plain = {r["key"]: r for r in source_inventory(undated, DM)}["population"]
    assert plain["timing"] == "постоянная характеристика: оценка на 1 января 2023 года"
    no_folds = {**FACTS, "population": POPULATION}
    known = {r["key"]: r for r in source_inventory(no_folds, DM)}["population"]["timing"]
    assert known.endswith("в редакции, опубликованной к 4 сентября 2023 года")


def test_population_below_the_threshold_is_shown_as_a_reference_figure() -> None:
    """Привязано меньше порога — источник в прогнозе не используется, и строка говорит это."""
    facts = {**FACTS, "population": {**POPULATION, "matched": 1900, "share": 0.8676, "used": False}}
    row = {r["key"]: r for r in source_inventory(facts, DM)}["population"]
    assert row["volume"].startswith("1 900 из 2 190 МО панели (86,8%)")
    assert row["gave"] == (
        "привязано меньше 90% территорий — в прогнозе не используется, справочный показатель"
    )


def test_timing_rows_name_the_population_when_it_is_used() -> None:
    rows = timing_rows(pd.Timestamp("2024-06-01"), {"spending": 35}, {}, population=POPULATION)
    row = next(r for r in rows if "Росстат" in r["source"])
    assert row == {
        "source": "Численность населения МО (Росстат)",
        "lag": "постоянная характеристика",
        "known": "оценка на 1 января 2023 года, опубликована к 4 сентября 2023 года",
        "tag": "постоянная",
        "kind": "static",
        "known_month": None,
    }
    undated = timing_rows(
        pd.Timestamp("2024-06-01"), {"spending": 35}, {}, population={"year": 2023}
    )
    assert undated[-1]["known"] == "оценка на 1 января 2023 года"
    plain = timing_rows(pd.Timestamp("2024-06-01"), {"spending": 35}, {})
    assert not any("Росстат" in r["source"] for r in plain)


def test_chain_summary_names_the_municipal_sources_that_were_connected() -> None:
    """Колонка «Данные» перечисляет все подключённые источники, а не только первые."""
    facts = {**FACTS, "spatial": SPATIAL, "weather": WEATHER, "population": POPULATION}
    chain = chain_summary(facts, folds=14, models=15, winners=WINNERS, contrast={}, dm=DM)
    data = [line["text"] for line in chain["data"]["lines"]]
    assert "Погода NASA POWER: 1 343 ячейки сетки" in data
    assert "Население МО (Росстат): 2 019 из 2 190 МО" in data
    assert data[-1] == "Все источники, лицензии и подписи"
    plain = chain_summary(FACTS, folds=14, models=15, winners=WINNERS, contrast={}, dm=DM)
    assert not any("Погода" in line["text"] for line in plain["data"]["lines"])
    assert not any("Население" in line["text"] for line in plain["data"]["lines"])
    unused = {**facts, "population": {**POPULATION, "used": False}}
    reference = chain_summary(unused, folds=14, models=15, winners=WINNERS, contrast={}, dm=DM)
    assert not any("Население" in line["text"] for line in reference["data"]["lines"]), (
        "справочный показатель в прогноз не идёт и в цепочку данных не входит"
    )


# --- Поля для страницы в оформлении редизайна: те же факты, разложенные по местам. ---


def test_source_inventory_splits_a_source_into_title_provider_and_timing_tag() -> None:
    """Таблица страницы показывает название, поставщика и метку согласования отдельно; строки
    отчёта (`source`, `timing`, `gave`) при этом остаются прежними."""
    facts = {**FACTS, "spatial": SPATIAL, "weather": WEATHER, "population": POPULATION}
    rows = {row["key"]: row for row in source_inventory(facts, DM, SOURCES)}
    spending = rows["spending"]
    assert spending["title"] == "Расходы на уровне МО"
    assert spending["provider"] == "набор организаторов, Лаборатория СберИндекс"
    assert spending["tag"] == "по месяц прогноза"
    assert spending["note"] == "прогноз от месяца использует ряд по этот месяц включительно"
    assert spending["source"].startswith("Расходы на уровне МО — набор организаторов")
    assert rows["reference"]["tag"] == "не зависит от времени" and rows["reference"]["note"] == ""
    assert rows["national"]["tag"] == "лаг 7–60 дней"
    assert rows["national"]["note"] == "лаг публикации от конца периода"
    assert rows["calendar"]["tag"] == "известен заранее"
    assert rows["calendar"]["note"] == "берётся на целевой месяц"
    assert rows["news"]["tag"] == "лаг 0" and rows["events"]["tag"] == "по дате публикации"
    assert rows["events"]["title"] == "Календарь событий"
    assert rows["events"]["provider"] == "составлен авторами по официальным источникам"
    assert rows["access"]["tag"] == "постоянная" and rows["neighbours"]["tag"] == "как свой ряд"
    assert (
        rows["weather"]["tag"] == "лаг 7 дней" and rows["weather"]["provider"] == "реанализ MERRA-2"
    )
    assert rows["population"]["tag"] == "постоянная"
    assert rows["population"]["note"].startswith("оценка на 1 января 2023 года")
    single = {**FACTS, "national": FACTS["national"][:1]}
    assert {r["key"]: r for r in source_inventory(single, DM)}["national"]["tag"] == "лаг 35 дней"


def test_source_inventory_groups_sources_by_the_level_they_describe() -> None:
    facts = {**FACTS, "spatial": SPATIAL, "weather": WEATHER, "population": POPULATION}
    groups: dict[str, list[str]] = {}
    for row in source_inventory(facts, DM, SOURCES):
        groups.setdefault(row["group"], []).append(row["key"])
    assert groups == {
        "Основа панели · уровень МО": ["spending", "reference"],
        "Уровень страны": ["national", "calendar"],
        "Уровень региона и страны": ["news", "events"],
        "Уровень МО": ["access", "neighbours", "weather", "population"],
    }


def test_source_inventory_carries_the_measured_effect_as_numbers() -> None:
    """Число, счёт окон и вывод — отдельными полями: страница рисует их, а не разбирает строку."""
    rows = {row["key"]: row for row in source_inventory({**FACTS, "spatial": SPATIAL}, DM, SOURCES)}
    assert rows["spending"]["effects"] == [] and rows["reference"]["effects"] == []
    assert rows["neighbours"]["effects"] == [
        {
            "base": "к базовой модели",
            "delta": -12.3,
            "agree": 9,
            "folds": 10,
            "direction": "better",
        },
        {
            "base": "поверх календаря и национальных рядов",
            "delta": 1.2,
            "agree": 5,
            "folds": 10,
            "direction": "none",
        },
    ]
    assert rows["news"]["effects"][0]["direction"] == "worse", "хуже в 9 окнах из 10 — устойчиво"
    assert rows["access"]["effects"] == [], "не измерено — эффекта нет, строка говорит об этом"
    assert rows["access"]["gave"] == "не измерено в этом прогоне"


def test_timing_rows_say_how_far_each_source_is_known() -> None:
    """Полосы «что видит прогноз» строятся из месяца, а не из его названия."""
    lags = {"weekly": 7, "spending": 35, "wages": 60}
    rows = timing_rows(pd.Timestamp("2024-09-01"), lags, {}, spatial=True, population=POPULATION)
    view = [(r["tag"], r["kind"], r["known_month"]) for r in rows]
    assert view == [
        ("лаг 0", "published", "2024-09"),
        ("по дате публикации", "published", "2024-09"),
        ("7 дней", "published", "2024-08"),
        ("35 дней", "published", "2024-07"),
        ("60 дней", "published", "2024-07"),
        ("известен заранее", "ahead", None),
        ("как свой ряд", "published", "2024-09"),
        ("постоянная", "static", None),
        ("постоянная", "static", None),
    ]


def test_chain_lines_carry_a_label_and_a_value() -> None:
    chain = chain_summary(FACTS, folds=14, models=13, winners=WINNERS, contrast={}, dm=DM)
    data = chain["data"]["lines"]
    assert (data[0]["label"], data[0]["value"]) == (
        "Расходы МО",
        "2 190 МО × 6 категорий × 24 месяца",
    )
    assert (data[2]["label"], data[2]["value"]) == ("Новости GDELT", "3 011 689 событий")
    assert data[-1]["label"] == "Все источники, лицензии и подписи" and data[-1]["value"] == ""
    method = [(line["label"], line["value"]) for line in chain["method"]["lines"]]
    assert method == [
        ("Проверка", "14 окон, горизонты 1, 3, 6 и 12 месяцев, метрика MAE"),
        ("Модели", "13 моделей и ансамбль по прошлым ошибкам"),
        ("Внешние данные", "только известные на момент прогноза"),
        ("Сдвиги", "ищем в ошибках прогноза, а не в самом ряде"),
    ]
    # Прежний текст строки не меняется: на него опираются отчёт и ссылки.
    assert chain["method"]["lines"][1]["text"] == "13 моделей и ансамбль по прошлым ошибкам"


def test_time_alignment_criterion_points_to_the_publication_lags() -> None:
    artifacts = {row["code"]: row["artifact"] for row in criteria_map()}
    assert artifacts["К5"] == "configs/features.yaml", "лаги публикации источников — в конфиге"


def test_criteria_have_a_code_a_short_name_and_a_primary_section() -> None:
    rows = criteria_map()
    assert [row["code"] for row in rows] == ["К1", "К2", "К3", "К4", "К5", "К6", "К7"]
    assert [row["primary"] for row in rows] == [
        "pipeline",
        "horizons",
        "changepoints",
        "foundation",
        "timing",
        "protocol",
        "limits",
    ]
    assert rows[0]["short"] == "Объяснение методологии"
    assert all(row["short"] and len(row["short"]) < len(row["criterion"]) + 1 for row in rows)
    for row in rows:
        anchors = {link["href"].lstrip("#") for link in row["links"]}
        assert row["primary"] in anchors, "основной раздел — один из тех, куда ведут ссылки"
    without = criteria_map({"horizons"})
    assert without[0]["primary"] is None, "основного раздела нет на странице — нет и отметки"
