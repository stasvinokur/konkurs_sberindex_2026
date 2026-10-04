"""Бюллетени Росстата о численности населения МО: разбор листа и привязка к территориям."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sbx.core.rosstat import bridge_names, parse_bulletin, population_features, population_table

# Лист бюллетеня на 1 января 2024: коды записаны по-разному в разных регионах.
CODED = [
    ["", "Выбрать субъект: -->  К перечню субъектов РФ", "", "", ""],
    ["Коды территорий", "", "население", "городское", "сельское"],
    ["ТЕРСОН-МО", "", "(человек)", "население", "население"],
    ["7900000000", "Республика Адыгея", "500591", "243610", "256981", ""],
    ["7970100000", "Городской округ - Город Майкоп", "161898", "137965", "23933", ""],
    ["797010000011003", "г Майкоп", "137965", "137965", "0", ""],
    ["7960500000", "Гиагинский  муниципальный район", "31872", "0", "31872", ""],
    ["7960540200", "Айрюмовское сельское поселение", "2834", "0", "2834", ""],
    ["0100000000", "Алтайский край", "2115308", "1237124", "878184", ""],
    ["01701000 0 0", "Городской округ г Барнаул", "687601", "638173", "49428", ""],
    ["01701000001 1 0 0 2", "г Барнаул", "620419", "620419", "0", ""],
    ["", "в том числе внутригородские районы", "", "", "", ""],
    ["01601000 0 0", "Гиагинский муниципальный район", "9000", "0", "9000", ""],
    ["9800000010", "Республика Саха (Якутия)", "1001664", "677004", "324660", ""],
    ["9870100010", "Городской округ г Якутск", "384979", "367667", "17312", ""],
]
# Лист бюллетеня на 1 января 2023: кодов нет, порядок тот же.
NAMED = [
    ["", "Все", "в том числе:", ""],
    ["", "население", "городское", "сельское"],
    ["Республика Адыгея", "497985", "243944", "254041"],
    ["Городской округ - Город Майкоп", "163766", "139687", "24079"],
    ["г Майкоп", "139687", "139687", "0"],
    ["Гиагинский муниципальный район", "31933", "0", "31933"],
    ["Айрюмовское сельское поселение", "2841", "0", "2841"],
    ["Алтайский край", "2130950", "1241693", "889257"],
    ["Городской округ г Барнаул", "695003", "644852", "50151"],
    ["Гиагинский муниципальный район", "9100", "0", "9100"],
    ["Республика Саха (Якутия)", "997565", "671965", "325600"],
    ["Городской округ город Якутск", "380195", "362815", "17380"],
]


def test_parse_reads_the_oktmo_from_every_code_layout() -> None:
    """Первые восемь цифр кода — ОКТМО, как бы ни был записан остаток: «00», « 0 0», «10»."""
    frame = parse_bulletin(CODED)
    upper = frame[frame["level"] == "municipality"]
    assert upper["oktmo"].tolist() == ["79701000", "79605000", "01701000", "01601000", "98701000"]
    assert frame[frame["level"] == "region"]["name"].tolist() == [
        "Республика Адыгея",
        "Алтайский край",
        "Республика Саха (Якутия)",
    ]
    barnaul = frame.set_index("oktmo").loc["01701000"].iloc[0]
    assert (barnaul["total"], barnaul["urban"], barnaul["rural"]) == (687601.0, 638173.0, 49428.0)
    assert len(frame) == 11, "заголовки и строки без чисел в таблицу не идут"
    assert frame[frame["level"] == "other"]["name"].tolist() == [
        "г Майкоп",
        "Айрюмовское сельское поселение",
        "г Барнаул",
    ], "города и поселения — не муниципальные образования верхнего уровня"
    assert upper["name"].iloc[1] == "Гиагинский муниципальный район", "лишние пробелы убраны"
    assert upper["region"].tolist() == ["Республика Адыгея"] * 2 + ["Алтайский край"] * 2 + [
        "Республика Саха (Якутия)"
    ]


def test_parse_of_a_sheet_without_codes_keeps_the_names_and_numbers() -> None:
    frame = parse_bulletin(NAMED)
    assert len(frame) == 10 and frame["oktmo"].isna().all() and frame["level"].isna().all()
    assert frame["name"].iloc[0] == "Республика Адыгея" and frame["total"].iloc[0] == 497985.0


def test_bridge_gives_a_named_row_the_code_of_the_same_name_in_the_same_region() -> None:
    """В бюллетене на 1 января 2023 кодов нет: код берётся у строки с тем же названием в том
    же регионе бюллетеня 2024 года. Одноимённые районы разных регионов не путаются."""
    bridged = bridge_names(parse_bulletin(CODED), parse_bulletin(NAMED))
    got = bridged.set_index("oktmo")["total"].to_dict()
    assert got == {
        "79701000": 163766.0,
        "79605000": 31933.0,
        "01701000": 695003.0,
        "01601000": 9100.0,
    }
    assert "98701000" not in got, "название изменилось («г Якутск» → «город Якутск») — не привязан"
    assert set(bridged.columns) >= {"oktmo", "region", "name", "total", "urban", "rural"}


def test_bridge_drops_a_name_that_is_not_unique_inside_the_region() -> None:
    """Два одноимённых МО в одном регионе по названию не различить: ни одно не привязывается."""
    coded = parse_bulletin(
        [*CODED, ["79606000 0 0", "Гиагинский муниципальный район", "1", "0", "1", ""]]
    )
    # Строка добавлена в конец листа — в блок Якутии; там такое название одно.
    twice = [
        *CODED[:8],
        ["7960600000", "Гиагинский муниципальный район", "5", "0", "5", ""],
        *CODED[8:],
    ]
    ambiguous = bridge_names(parse_bulletin(twice), parse_bulletin(NAMED))
    assert "79605000" not in set(ambiguous["oktmo"]) and "79606000" not in set(ambiguous["oktmo"])
    assert "79701000" in set(ambiguous["oktmo"])
    assert "79606000" not in set(bridge_names(coded, parse_bulletin(NAMED))["oktmo"])

    repeated = [*NAMED[:6], NAMED[5], *NAMED[6:]]
    doubled = bridge_names(parse_bulletin(CODED), parse_bulletin(repeated))
    assert "79605000" not in set(doubled["oktmo"]), "название дважды в блоке региона 2023 года"


TERRITORIES = pd.DataFrame(
    {
        "official_id": [1, 2, 3],
        "territory_id": ["79-0001", "01-0002", "98-0003"],
        "oktmo": ["79701000", "01701000", "98701000"],
        "mo_name": ["городской округ город Майкоп", "городской округ город Барнаул", "Якутск"],
    }
)


def test_population_table_lists_the_panel_territories_that_were_matched() -> None:
    bridged = bridge_names(parse_bulletin(CODED), parse_bulletin(NAMED))
    table = population_table(bridged, TERRITORIES)
    assert table["official_id"].tolist() == [1, 2], "Якутск не привязан — строки о нём нет"
    assert list(table.columns) == ["official_id", "oktmo", "name", "total", "urban", "rural"]
    assert table["total"].tolist() == [163766, 695003] and table["urban"].tolist() == [
        139687,
        644852,
    ]
    assert table["name"].iloc[1] == "Городской округ г Барнаул", "название — как у Росстата"


def test_population_features_are_the_log_of_the_population_and_the_urban_share() -> None:
    bridged = bridge_names(parse_bulletin(CODED), parse_bulletin(NAMED))
    table = population_table(bridged, TERRITORIES)
    frame = population_features(table, TERRITORIES).set_index("territory_id")
    assert list(frame.columns) == ["pop_log", "pop_urban_share"]
    assert frame.loc["79-0001", "pop_log"] == pytest.approx(np.log(163766))
    assert frame.loc["79-0001", "pop_urban_share"] == pytest.approx(139687 / 163766)
    assert np.isnan(frame.loc["98-0003", "pop_log"]), "не привязана — пропуск, а не ноль"
    assert len(frame) == 3, "строка есть у каждой территории панели"
