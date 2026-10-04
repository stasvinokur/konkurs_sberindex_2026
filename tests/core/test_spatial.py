"""Признаки из таблиц организаторов: доступность рынков и соседние МО по автодорогам."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sbx.core.features.spatial import access_features, nearest_neighbours, neighbour_features

# Официальные коды → территории панели; 9 в панели нет.
TERRITORIES = pd.DataFrame(
    {"official_id": [1, 2, 3, 4], "territory_id": ["01-0001", "01-0002", "02-0003", "02-0004"]}
)
# Среди строк: путь по железной дороге, территория вне панели, пара «сама с собой» и пара,
# записанная второй раз в обратную сторону с другим расстоянием.
CONNECTION = pd.DataFrame(
    {
        "territory_id_x": [1, 1, 2, 1, 1, 3, 2, 2],
        "territory_id_y": [2, 3, 3, 9, 2, 4, 2, 1],
        "distance": [10.0, 30.0, 25.0, 1.0, 5.0, 500.0, 0.0, 12.0],
        "type": [
            "highway",
            "highway",
            "highway",
            "highway",
            "railway",
            "highway",
            "highway",
            "highway",
        ],
    }
)


def test_nearest_neighbours_are_symmetric_ranked_and_inside_the_panel() -> None:
    """В таблице расстояние записано в одну сторону (x → y); обратный путь тот же."""
    near = nearest_neighbours(CONNECTION, TERRITORIES, k=2)
    pairs = {
        territory: list(zip(group["neighbour"], group["rank"], group["distance"], strict=True))
        for territory, group in near.groupby("territory_id")
    }
    assert pairs["01-0001"] == [("01-0002", 1, 10.0), ("02-0003", 2, 30.0)], (
        "ближайший по автодороге; территория 9 не в панели, железная дорога не в счёт"
    )
    assert pairs["01-0002"] == [("01-0001", 1, 10.0), ("02-0003", 2, 25.0)], (
        "обратное направление; сама себе территория не сосед; из двух записей пары — меньшая"
    )
    assert pairs["02-0003"] == [("01-0002", 1, 25.0), ("01-0001", 2, 30.0)]
    assert pairs["02-0004"] == [("02-0003", 1, 500.0)], "соседей меньше k — сколько есть"


def test_access_features_are_logs_and_miss_where_the_source_misses() -> None:
    market = pd.DataFrame({"territory_id": [1, 2, 9], "market_access": [100.0, 1000.0, 300.0]})
    near = nearest_neighbours(CONNECTION, TERRITORIES, k=2)
    frame = access_features(market, near, TERRITORIES).set_index("territory_id")
    assert list(frame.columns) == ["market_access_log", "neighbour_distance_log"]
    assert frame.loc["01-0001", "market_access_log"] == pytest.approx(np.log(100.0))
    assert np.isnan(frame.loc["02-0003", "market_access_log"]), "индекса у территории нет"
    assert frame.loc["01-0001", "neighbour_distance_log"] == pytest.approx(np.log1p(20.0))
    assert frame.loc["02-0004", "neighbour_distance_log"] == pytest.approx(np.log1p(500.0))
    assert len(frame) == 4, "строка есть у каждой территории панели"


def _panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    months = pd.date_range("2024-01-01", periods=5, freq="MS")
    levels = {
        "01-0001": [100, 110, 121, 133.1, 146.41],
        "01-0002": [200, 200, 200, 200, 200],
        "02-0003": [50, 60, 72, 86.4, 103.68],
    }
    rows = [
        {
            "unique_id": f"{territory}__{slug}",
            "territory_id": territory,
            "category": category,
            "ds": month,
            "y": value * scale,
        }
        for territory, values in levels.items()
        for slug, category, scale in (
            ("total", "Все категории", 1.0),
            ("food", "Продовольствие", 0.5),
        )
        for month, value in zip(months, values, strict=True)
    ]
    panel = pd.DataFrame(rows)
    static = panel.drop_duplicates("unique_id")[["unique_id", "territory_id", "category"]]
    return panel, static


def test_neighbour_features_average_the_same_category_of_the_neighbours() -> None:
    panel, static = _panel()
    near = nearest_neighbours(CONNECTION, TERRITORIES, k=2)
    frame = neighbour_features(panel, static, near).set_index(["unique_id", "origin"])
    march = pd.Timestamp("2024-03-01")
    row = frame.loc[("01-0001__total", march)]
    # Соседи 01-0001 — 01-0002 (ровный ряд) и 02-0003 (растёт на 20% в месяц).
    assert row["nb_growth_1"] == pytest.approx((np.log(1.0) + np.log(1.2)) / 2)
    assert row["nb_level_gap"] == pytest.approx(np.log(121) - (np.log(200) + np.log(72)) / 2)
    april = frame.loc[("01-0001__total", pd.Timestamp("2024-04-01"))]
    assert april["nb_growth_3"] == pytest.approx((np.log(1.0) + np.log(86.4 / 50)) / 2)
    assert np.isnan(row["nb_growth_3"]), "три месяца назад данных ещё нет"
    food = frame.loc[("01-0001__food", march)]
    assert food["nb_growth_1"] == pytest.approx(row["nb_growth_1"]), "своя категория соседей"
    assert set(frame.columns) == {"nb_growth_1", "nb_growth_3", "nb_level_gap"}


def test_neighbour_features_of_an_origin_ignore_later_months() -> None:
    """Строка origin считается только по значениям не позже origin."""
    panel, static = _panel()
    near = nearest_neighbours(CONNECTION, TERRITORIES, k=2)
    cutoff = pd.Timestamp("2024-03-01")
    full = neighbour_features(panel, static, near)
    known = neighbour_features(panel[panel["ds"] <= cutoff], static, near)
    pd.testing.assert_frame_equal(
        full[full["origin"] <= cutoff].reset_index(drop=True), known.reset_index(drop=True)
    )


def test_a_series_without_neighbours_gets_no_neighbour_features() -> None:
    panel, static = _panel()
    alone = pd.DataFrame({"territory_id": [], "neighbour": [], "rank": [], "distance": []})
    frame = neighbour_features(panel, static, alone)
    assert frame[["nb_growth_1", "nb_growth_3", "nb_level_gap"]].isna().all().all()


def test_a_neighbour_without_a_series_of_the_category_is_skipped() -> None:
    panel, static = _panel()
    gone = "02-0003__food"
    near = nearest_neighbours(CONNECTION, TERRITORIES, k=2)
    frame = neighbour_features(
        panel[panel["unique_id"] != gone], static[static["unique_id"] != gone], near
    ).set_index(["unique_id", "origin"])
    food = frame.loc[("01-0001__food", pd.Timestamp("2024-03-01"))]
    assert food["nb_growth_1"] == pytest.approx(np.log(1.0)), "остался один сосед — ровный ряд"
    assert food["nb_level_gap"] == pytest.approx(np.log(60.5) - np.log(100))


def test_a_zero_value_of_a_neighbour_is_a_gap_not_minus_infinity() -> None:
    panel, static = _panel()
    march = pd.Timestamp("2024-03-01")
    panel.loc[(panel["unique_id"] == "02-0003__total") & (panel["ds"] == march), "y"] = 0.0
    near = nearest_neighbours(CONNECTION, TERRITORIES, k=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # логарифм нуля не берётся вовсе
        frame = neighbour_features(panel, static, near).set_index(["unique_id", "origin"])
    row = frame.loc[("01-0001__total", march)]
    assert row["nb_growth_1"] == pytest.approx(np.log(1.0)), "нулевой месяц соседа — пропуск"
    assert row["nb_level_gap"] == pytest.approx(np.log(121) - np.log(200))
    april = frame.loc[("01-0001__total", pd.Timestamp("2024-04-01"))]
    assert april["nb_growth_1"] == pytest.approx(np.log(1.0)), "прирост от нуля не считается"


def test_a_series_has_rows_only_for_the_months_it_was_observed() -> None:
    """Ряд, начавшийся позже, не получает строк за месяцы до своего начала: иначе набор строк
    origin зависел бы от того, что случилось после него."""
    panel, static = _panel()
    april = pd.Timestamp("2024-04-01")
    panel = panel[~((panel["unique_id"] == "01-0002__total") & (panel["ds"] < april))]
    near = nearest_neighbours(CONNECTION, TERRITORIES, k=2)
    frame = neighbour_features(panel, static, near)
    rows = frame[frame["unique_id"] == "01-0002__total"]
    assert list(rows["origin"]) == list(pd.date_range(april, periods=2, freq="MS"))
    cutoff = pd.Timestamp("2024-03-01")
    known = neighbour_features(panel[panel["ds"] <= cutoff], static, near)
    pd.testing.assert_frame_equal(
        frame[frame["origin"] <= cutoff].reset_index(drop=True), known.reset_index(drop=True)
    )
