"""Сопоставление рядов публичной выгрузки с рядами официального набора и сверка эвристики."""

from __future__ import annotations

import pandas as pd
import pytest

from sbx.core.crosswalk import (
    CrosswalkError,
    crosswalk_table,
    heuristic_agreement,
    match_blocks,
    rekey,
    to_official_ids,
)

MONTHS = ["2023-01", "2023-02", "2023-03"]


def _raw(series: list[tuple[str, str, list[int]]]) -> pd.DataFrame:
    """Выгрузка: ряды идут блоками строк, территория задана только названием."""
    rows = [
        {"period": f"{month}-01", "mo": name, "category_15": category, "value": float(value)}
        for name, category, values in series
        for month, value in zip(MONTHS, values, strict=True)
    ]
    return pd.DataFrame(rows)


def _official(series: list[tuple[int, str, list[int]]]) -> pd.DataFrame:
    rows = [
        {"date": month, "territory_id": official_id, "category": category, "value": value}
        for official_id, category, values in series
        for month, value in zip(MONTHS, values, strict=True)
    ]
    return pd.DataFrame(rows)


def test_match_blocks_pairs_same_name_series_by_their_values() -> None:
    """Два одноимённых района: по названию их не различить, по значениям — однозначно."""
    raw = _raw(
        [
            ("Каменский район", "Все категории", [100, 110, 120]),
            ("Каменский район", "Все категории", [900, 910, 920]),
            ("Каменский район", "Продовольствие", [50, 55, 60]),
        ]
    )
    official = _official(
        [
            (20, "Все категории", [900, 910, 920]),
            (10, "Продовольствие", [50, 55, 60]),
            (10, "Все категории", [100, 110, 120]),
        ]
    )
    pairs = match_blocks(raw, official).set_index("block_id")
    assert pairs["official_id"].to_dict() == {1: 10, 2: 20, 3: 10}
    assert pairs.loc[3, "category"] == "Продовольствие"


def test_match_blocks_tells_categories_apart_when_the_values_coincide() -> None:
    """Одинаковые значения в двух категориях — разные ряды: категория входит в подпись."""
    raw = _raw([("А", "Здоровье", [5, 5, 5]), ("А", "Транспорт", [5, 5, 5])])
    official = _official([(7, "Транспорт", [5, 5, 5]), (7, "Здоровье", [5, 5, 5])])
    pairs = match_blocks(raw, official)
    assert pairs["category"].tolist() == ["Здоровье", "Транспорт"]
    assert pairs["official_id"].tolist() == [7, 7]


def test_match_blocks_tells_series_apart_by_the_order_of_months() -> None:
    """Те же значения в другом порядке месяцев — другой ряд: месяц входит в подпись."""
    raw = _raw([("А", "Здоровье", [1, 2, 3]), ("Б", "Здоровье", [3, 2, 1])])
    official = _official([(8, "Здоровье", [3, 2, 1]), (9, "Здоровье", [1, 2, 3])])
    assert match_blocks(raw, official)["official_id"].tolist() == [9, 8]


def test_match_blocks_refuses_anything_but_a_one_to_one_match() -> None:
    raw = _raw([("А", "Все категории", [1, 2, 3]), ("Б", "Все категории", [7, 8, 9])])
    with pytest.raises(CrosswalkError, match="не нашлись"):
        match_blocks(raw, _official([(1, "Все категории", [1, 2, 3])]))
    twins = _official([(1, "Все категории", [1, 2, 3]), (2, "Все категории", [1, 2, 3])])
    with pytest.raises(CrosswalkError, match="одинаков"):
        match_blocks(raw.iloc[:3], twins)
    extra = _official(
        [
            (1, "Все категории", [1, 2, 3]),
            (2, "Все категории", [7, 8, 9]),
            (3, "Здоровье", [4, 4, 4]),
        ]
    )
    with pytest.raises(CrosswalkError, match="нет в выгрузке"):
        match_blocks(raw, extra)


def test_match_blocks_refuses_an_export_that_repeats_a_series() -> None:
    """Два ряда выгрузки с одной подписью легли бы на один официальный ряд."""
    raw = _raw([("А", "Все категории", [1, 2, 3]), ("Б", "Все категории", [1, 2, 3])])
    with pytest.raises(CrosswalkError):
        match_blocks(raw, _official([(1, "Все категории", [1, 2, 3])]))


def test_crosswalk_table_names_each_series_in_both_identifications() -> None:
    pairs = pd.DataFrame(
        {"block_id": [1, 2], "category": ["Все категории", "Здоровье"], "official_id": [10, 20]}
    )
    mapping = pd.DataFrame(
        {"block_id": [1, 2], "territory_id": ["01-a", "01-a"], "raw_mo": ["А район", "А район"]}
    )
    heuristic = pd.DataFrame({"territory_id": ["01-a"], "region_code": ["01"]})
    official = pd.DataFrame(
        {
            "official_id": [10, 20],
            "territory_id": ["01-0010", "95-0020"],
            "region_code": ["01", "95"],
        }
    )
    table = crosswalk_table(pairs, mapping, heuristic, official)
    assert table["unique_id"].tolist() == ["01-0010__total", "95-0020__health"]
    assert table["legacy_unique_id"].tolist() == ["01-a__total", "01-a__health"]
    assert table["heuristic_territory"].tolist() == ["01-a", "01-a"]
    assert table["heuristic_region"].tolist() == ["01", "01"]
    assert table["official_region"].tolist() == ["01", "95"]
    assert table["raw_mo"].tolist() == ["А район", "А район"]


def _pairs(rows: list[tuple[str, str, str, int, str]]) -> pd.DataFrame:
    columns = [
        "heuristic_territory",
        "raw_mo",
        "heuristic_region",
        "official_id",
        "official_region",
    ]
    return pd.DataFrame(rows, columns=columns)


def test_heuristic_agreement_counts_series_and_territories() -> None:
    pairs = _pairs(
        [
            # Территория А собрана верно.
            ("a", "А", "01", 1, "01"),
            ("a", "А", "01", 1, "01"),
            # Территория Б: один ряд на самом деле из другого региона и другой территории.
            ("b", "Б", "01", 2, "01"),
            ("b", "Б", "01", 3, "95"),
            # Территория В целиком записана не в тот регион.
            ("c", "В", "56", 4, "70"),
            ("c", "В", "56", 4, "70"),
        ]
    )
    assert heuristic_agreement(pairs) == {
        "series": 6,
        "series_in_wrong_region": 3,
        "territories": 3,
        "territories_with_foreign_series": 2,
        "territories_entirely_in_wrong_region": 1,
        "territories_mixing_same_name_series": 1,
        "territories_glued_from_several_names": 0,
        "official_territories": 4,
    }


def test_heuristic_agreement_tells_a_deliberate_glue_from_a_mix_up() -> None:
    """Склейка «преемник продолжает ряд» — два названия в одной территории по решению, а не
    по ошибке; путаницей считается только смешение рядов одного названия."""
    pairs = _pairs(
        [
            ("46-new", "округ Новый", "46", 30, "46"),
            ("46-new", "город Старый", "46", 31, "46"),
        ]
    )
    report = heuristic_agreement(pairs)
    assert report["territories_glued_from_several_names"] == 1
    assert report["territories_mixing_same_name_series"] == 0
    assert report["series_in_wrong_region"] == 0


def test_to_official_ids_translates_a_frozen_list_and_splits_a_glued_series() -> None:
    table = pd.DataFrame(
        {
            "legacy_unique_id": ["01-a__total", "61-s__total", "61-s__total", "02-b__total"],
            "unique_id": ["01-0010__total", "61-1847__total", "61-3101__total", "02-0007__total"],
        }
    )
    translated = to_official_ids(["61-s__total", "01-a__total"], table)
    assert translated.to_dict("records") == [
        {"unique_id": "01-0010__total", "legacy_unique_id": "01-a__total"},
        {"unique_id": "61-1847__total", "legacy_unique_id": "61-s__total"},
        {"unique_id": "61-3101__total", "legacy_unique_id": "61-s__total"},
    ]
    with pytest.raises(CrosswalkError, match="нет в таблице"):
        to_official_ids(["99-x__total"], table)


def test_rekey_moves_identifiers_to_official_and_refuses_a_glued_series() -> None:
    """Источник ряда бенчмарка — один реальный ряд; склейку двух МО перевести нельзя."""
    table = pd.DataFrame(
        {
            "legacy_unique_id": ["01-a__total", "61-s__total", "61-s__total"],
            "unique_id": ["01-0010__total", "61-1847__total", "61-3101__total"],
        }
    )
    frame = pd.DataFrame({"source_id": ["01-a__total", "01-a__total"], "y": [1.0, 2.0]})
    moved = rekey(frame, "source_id", table)
    assert moved["source_id"].tolist() == ["01-0010__total", "01-0010__total"]
    assert moved["y"].tolist() == [1.0, 2.0] and frame["source_id"].iloc[0] == "01-a__total"
    with pytest.raises(CrosswalkError, match="несколько"):
        rekey(pd.DataFrame({"source_id": ["61-s__total"]}), "source_id", table)
    with pytest.raises(CrosswalkError, match="нет в таблице"):
        rekey(pd.DataFrame({"source_id": ["99-x__total"]}), "source_id", table)
