"""Таблица территорий из официального справочника организаторов."""

from __future__ import annotations

import pandas as pd
import pytest

from sbx.core.territories import (
    TerritoryContractError,
    build_territories,
    canonical_versions,
    kind_changes,
    normalize_oktmo,
    reference_frame,
)

HEADER = [
    "municipal_district_name_short",
    "oktmo",
    "municipal_district_name",
    "municipal_district_type",
    "municipal_district_status",
    "shape",
    "shape_linked_oktmo",
    "municipal_district_center",
    "source_rosstat",
    "year_from",
    "year_to",
    "territory_id",
    "change_id_from",
    "change_id_to",
    "region_code",
    "region_name",
    "municipal_district_center_lat",
    "municipal_district_center_lon",
]


def _row(
    official_id: int,
    name: str,
    kind: str,
    oktmo: str,
    years: tuple[int, int] = (2018, 9999),
    sber_region: str = "1",
    region: str = "Республика Адыгея",
    center: tuple[str, str] | None = ("44.6", "40.1"),
) -> list[str]:
    row = [
        name.split()[0],
        oktmo,
        name,
        kind,
        "",
        "1",
        "",
        "г Центр",
        "data",
        str(years[0]),
        str(years[1]),
        str(official_id),
        "",
        "",
        sber_region,
        region,
    ]
    # В настоящем файле у строк без координат двух последних ячеек нет вовсе.
    return row + list(center) if center else row


def _reference(*rows: list[str]) -> pd.DataFrame:
    return reference_frame([HEADER, *rows])


def test_normalize_oktmo_keeps_digits_only() -> None:
    assert normalize_oktmo("79-701-000-000") == "79701000000"
    assert normalize_oktmo(" 71 871 000 ") == "71871000"


def test_reference_frame_pads_short_rows_and_types_the_keys() -> None:
    frame = _reference(
        _row(
            7,
            "Гиагинский муниципальный район",
            "муниципальный район",
            "79-605-000-000",
            center=None,
        )
    )
    assert list(frame.columns) == HEADER
    assert frame.loc[0, "territory_id"] == 7 and frame.loc[0, "year_to"] == 9999
    assert frame.loc[0, "municipal_district_center_lat"] == ""


def test_reference_frame_rejects_a_file_without_the_expected_columns() -> None:
    with pytest.raises(TerritoryContractError, match="territory_id"):
        reference_frame([["oktmo", "region_name"], ["79-701-000-000", "Республика Адыгея"]])


def test_canonical_version_is_the_one_valid_in_the_reference_year() -> None:
    """Район стал округом в 2024 году: панель за 2023–2024 подписывается состоянием 2024 года."""
    reference = _reference(
        _row(
            208,
            "Беломорский муниципальный район",
            "муниципальный район",
            "86-604-000-000",
            (2018, 2024),
        ),
        _row(
            208,
            "Беломорский муниципальный округ",
            "муниципальный округ",
            "86-504-000-000",
            (2024, 9999),
        ),
        _row(5, "Старый район", "муниципальный район", "79-610-000-000", (2018, 2020)),
        _row(5, "Нынешний округ", "муниципальный округ", "79-510-000-000", (2020, 9999)),
    )
    picked = canonical_versions(reference, year=2024).set_index("territory_id")
    assert picked.loc[208, "municipal_district_name"] == "Беломорский муниципальный округ"
    assert picked.loc[5, "municipal_district_name"] == "Нынешний округ"
    assert len(picked) == 2


def test_canonical_version_does_not_take_a_state_that_begins_after_the_reference_year() -> None:
    reference = _reference(
        _row(9, "Район", "муниципальный район", "79-620-000-000", (2018, 2025)),
        _row(9, "Округ будущего", "муниципальный округ", "79-520-000-000", (2025, 9999)),
    )
    picked = canonical_versions(reference, year=2024)
    assert picked["municipal_district_name"].tolist() == ["Район"]


def test_canonical_version_falls_back_to_the_latest_earlier_state() -> None:
    reference = _reference(
        _row(3, "Упразднённый район", "муниципальный район", "79-630-000-000", (2018, 2022))
    )
    assert canonical_versions(reference, year=2024)["municipal_district_name"].tolist() == [
        "Упразднённый район"
    ]


def test_build_territories_derives_our_region_code_from_dashed_oktmo() -> None:
    reference = _reference(
        _row(1, "городской округ город Майкоп", "городской округ", "79-701-000-000"),
        _row(
            2594,
            "городской округ Сургут",
            "городской округ",
            "71-876-000-000",
            sber_region="81",
            region="Ханты-Мансийский автономный округ - Югра",
        ),
        _row(
            40,
            "муниципальный округ Арбат",
            "внутригородская территория города федерального значения",
            "45-374-000-000",
            sber_region="77",
            region="г. Москва",
            center=None,
        ),
    )
    table = build_territories(reference, official_ids=[1, 40, 2594], year=2024).set_index(
        "official_id"
    )

    assert table.loc[1, "territory_id"] == "79-0001"
    assert table.loc[1, "region_code"] == "79" and table.loc[1, "oktmo"] == "79701000"
    assert table.loc[1, "mo_kind"] == "urban_okrug"
    assert table.loc[1, "mo_name"] == "городской округ город Майкоп"
    # Автономный округ в составе области различается третьей цифрой ОКТМО.
    assert table.loc[2594, "territory_id"] == "718-2594" and table.loc[2594, "region_code"] == "718"
    assert table.loc[40, "mo_kind"] == "intracity"
    assert pd.isna(table.loc[40, "center_lat"]) and table.loc[1, "center_lat"] == 44.6


def test_territory_ids_keep_regions_contiguous_when_sorted() -> None:
    reference = _reference(
        _row(
            300,
            "район А",
            "муниципальный район",
            "11-605-000-000",
            sber_region="29",
            region="Архангельская область",
        ),
        _row(
            10,
            "район Б",
            "муниципальный район",
            "11-811-000-000",
            sber_region="83",
            region="Ненецкий автономный округ",
        ),
        _row(
            20,
            "район В",
            "муниципальный район",
            "12-605-000-000",
            sber_region="30",
            region="Астраханская область",
        ),
        _row(
            5,
            "район Г",
            "муниципальный район",
            "11-610-000-000",
            sber_region="29",
            region="Архангельская область",
        ),
    )
    table = build_territories(reference, official_ids=[5, 10, 20, 300], year=2024)
    ordered = table.sort_values("territory_id")["region_code"].tolist()
    assert ordered == ["11", "11", "118", "12"]


def test_build_territories_refuses_what_it_cannot_identify() -> None:
    reference = _reference(
        _row(1, "городской округ город Майкоп", "городской округ", "79-701-000-000")
    )
    with pytest.raises(TerritoryContractError, match="нет в справочнике"):
        build_territories(reference, official_ids=[1, 2], year=2024)

    unknown_type = _reference(_row(1, "сельское поселение", "сельское поселение", "79-701-000-000"))
    with pytest.raises(TerritoryContractError, match="неизвестный тип"):
        build_territories(unknown_type, official_ids=[1], year=2024)

    unknown_region = _reference(_row(1, "район", "муниципальный район", "00-701-000-000"))
    with pytest.raises(TerritoryContractError, match="регион"):
        build_territories(unknown_region, official_ids=[1], year=2024)


def test_build_territories_refuses_a_region_mapping_that_is_not_one_to_one() -> None:
    """Код региона справочника и наш код — разные нумерации, но обязаны соответствовать
    один к одному: иначе часть территорий получила чужой регион."""
    reference = _reference(
        _row(1, "район А", "муниципальный район", "79-605-000-000", sber_region="1"),
        _row(2, "район Б", "муниципальный район", "03-605-000-000", sber_region="1"),
    )
    with pytest.raises(TerritoryContractError, match="один к одному"):
        build_territories(reference, official_ids=[1, 2], year=2024)


def test_kind_changes_counts_territories_whose_type_differs_from_the_previous_year() -> None:
    """Тип МО берётся из редакции одного года; для окон проверки от месяцев предыдущего года это
    сведение из следующей редакции. Сколько территорий это затрагивает, считается по справочнику."""
    reference = _reference(
        _row(208, "Беломорский район", "муниципальный район", "86-604-000-000", (2018, 2023)),
        _row(208, "Беломорский округ", "муниципальный округ", "86-504-000-000", (2024, 9999)),
        _row(5, "Старый район", "муниципальный район", "79-610-000-000", (2018, 2020)),
        _row(5, "Нынешний округ", "муниципальный округ", "79-510-000-000", (2020, 9999)),
        _row(7, "Новый округ", "муниципальный округ", "79-520-000-000", (2024, 9999)),
        _row(8, "Район", "муниципальный район", "79-630-000-000", (2018, 2023)),
        _row(8, "Район с новым именем", "муниципальный район ", "79-630-000-000", (2024, 9999)),
    )
    assert kind_changes(reference, [208, 5, 7, 8], 2024) == 1, "только Беломорский сменил тип"
    assert kind_changes(reference, [5, 7, 8], 2024) == 0, "переименование и новая территория — нет"
    assert kind_changes(reference, [208, 5], 2020) == 1, "пятая сменила тип в 2020 году"
    assert kind_changes(reference, [], 2024) == 0
