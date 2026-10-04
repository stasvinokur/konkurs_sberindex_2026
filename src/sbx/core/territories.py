"""Таблица территорий из официального справочника организаторов конкурса.

Справочник «Границы и изменения МО» даёт каждой территории постоянный `territory_id` — тот же,
что стоит в официальном наборе расходов. У территории может быть несколько строк: название,
тип и код ОКТМО менялись при реорганизациях. Панель покрывает 2023–2024 годы, поэтому берётся
состояние последнего года панели.

Код региона остаётся нашим — префикс ОКТМО, как в `sbx.core.regions`: по нему присоединяются
новости и календарь событий. Собственная нумерация регионов в справочнике другая.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from sbx.core.regions import region_of_oktmo

REQUIRED_COLUMNS = (
    "territory_id",
    "oktmo",
    "municipal_district_name",
    "municipal_district_name_short",
    "municipal_district_type",
    "year_from",
    "year_to",
    "region_code",
    "region_name",
    "municipal_district_center_lat",
    "municipal_district_center_lon",
)
KIND_BY_TYPE = {
    "муниципальный район": "municipal_district",
    "городской округ": "urban_okrug",
    "муниципальный округ": "municipal_okrug",
    "внутригородская территория города федерального значения": "intracity",
}
# Число знаков id в `territory_id`: идентификаторы справочника не превышают 3101.
ID_WIDTH = 4


class TerritoryContractError(ValueError):
    """Справочник или набор территорий не соответствует ожиданиям шага."""


def normalize_oktmo(code: str) -> str:
    """Код ОКТМО без разделителей: «79-701-000-000» → «79701000000»."""
    return "".join(ch for ch in str(code) if ch.isdigit())


def reference_frame(rows: Sequence[Sequence[str]]) -> pd.DataFrame:
    """Строки листа справочника (первая — заголовок) → таблица с типизированными ключами.

    Excel не пишет пустые ячейки в конце строки, поэтому короткие строки дополняются.
    """
    header = [str(name) for name in rows[0]]
    missing = [name for name in REQUIRED_COLUMNS if name not in header]
    if missing:
        raise TerritoryContractError(f"в справочнике нет колонок {missing}")
    width = len(header)
    body = [[*map(str, row), *[""] * (width - len(row))][:width] for row in rows[1:]]
    frame = pd.DataFrame(body, columns=header)
    for column in ("territory_id", "year_from", "year_to"):
        frame[column] = frame[column].astype(int)
    return frame


def canonical_versions(reference: pd.DataFrame, year: int) -> pd.DataFrame:
    """Одна строка на территорию: состояние, действующее в `year`.

    Если в этом году действуют два состояния (реорганизация посреди года), берётся более новое.
    Если ни одно не действует, берётся последнее из более ранних. Состояния, начавшиеся позже
    `year`, не используются: подписывать ими данные прошлых лет нельзя.
    """
    known = reference[reference["year_from"] <= year].copy()
    known["_valid"] = known["year_to"] >= year
    ordered = known.sort_values(["territory_id", "_valid", "year_from"], kind="stable")
    picked = ordered.drop_duplicates("territory_id", keep="last")
    return picked.drop(columns="_valid").reset_index(drop=True)


def kind_changes(reference: pd.DataFrame, official_ids: Sequence[int], year: int) -> int:
    """Сколько территорий сменили тип между состоянием прошлого года и состоянием `year`.

    Тип МО идёт в признаки модели из редакции `year`. Для моментов прогноза из предыдущего года
    это сведение из следующей редакции; счёт показывает, скольких территорий это касается.
    """
    wanted = {int(i) for i in official_ids}
    column = "municipal_district_type"
    now = canonical_versions(reference, year).set_index("territory_id")[column].str.strip()
    before = canonical_versions(reference, year - 1).set_index("territory_id")[column].str.strip()
    common = sorted(wanted & set(now.index) & set(before.index))
    return int((now.loc[common] != before.loc[common]).sum())


def _coordinate(value: str) -> float:
    return float(value) if str(value).strip() else float("nan")


def build_territories(
    reference: pd.DataFrame, official_ids: Sequence[int], year: int
) -> pd.DataFrame:
    """Таблица территорий набора расходов: наш `territory_id`, регион, название, тип, центр.

    `territory_id` — «код региона-официальный id»: читается в артефактах и сохраняет регионы
    смежными при сортировке. `official_id` остаётся отдельной колонкой для стыковки с другими
    таблицами организаторов.
    """
    canonical = canonical_versions(reference, year).set_index("territory_id")
    wanted = sorted({int(i) for i in official_ids})
    absent = [i for i in wanted if i not in canonical.index]
    if absent:
        raise TerritoryContractError(f"территорий нет в справочнике: {absent[:10]}")
    if wanted and wanted[-1] >= 10**ID_WIDTH:
        raise TerritoryContractError(f"официальный id {wanted[-1]} не помещается в формат id")

    rows = []
    for official_id in wanted:
        row = canonical.loc[official_id]
        oktmo = normalize_oktmo(row["oktmo"])
        region = region_of_oktmo(oktmo)
        if region is None:
            raise TerritoryContractError(
                f"территория {official_id}: по ОКТМО {row['oktmo']!r} не определён регион"
            )
        kind = KIND_BY_TYPE.get(str(row["municipal_district_type"]).strip())
        if kind is None:
            raise TerritoryContractError(
                f"территория {official_id}: неизвестный тип {row['municipal_district_type']!r}"
            )
        rows.append(
            {
                "territory_id": f"{region.code}-{official_id:0{ID_WIDTH}d}",
                "official_id": official_id,
                "region_code": region.code,
                "mo_name": str(row["municipal_district_name"]).strip(),
                "mo_name_short": str(row["municipal_district_name_short"]).strip(),
                "mo_kind": kind,
                "oktmo": oktmo[:8],
                "center_lat": _coordinate(row["municipal_district_center_lat"]),
                "center_lon": _coordinate(row["municipal_district_center_lon"]),
                "_reference_region": str(row["region_code"]).strip(),
            }
        )
    table = pd.DataFrame(rows)
    _check_region_mapping(table)
    return (
        table.drop(columns="_reference_region").sort_values("territory_id").reset_index(drop=True)
    )


def _check_region_mapping(table: pd.DataFrame) -> None:
    """Нумерация регионов справочника и наша обязаны соответствовать один к одному."""
    pairs = table[["_reference_region", "region_code"]].drop_duplicates()
    split = pairs[pairs.duplicated("_reference_region", keep=False)]
    merged = pairs[pairs.duplicated("region_code", keep=False)]
    if len(split) or len(merged):
        broken = pd.concat([split, merged]).drop_duplicates().to_dict("records")
        raise TerritoryContractError(
            f"коды регионов справочника и наши не соответствуют один к одному: {broken[:6]}"
        )
