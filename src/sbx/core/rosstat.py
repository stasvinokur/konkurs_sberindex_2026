"""Численность населения МО из бюллетеней Росстата: разбор листа и привязка к территориям.

Бюллетень «Численность населения Российской Федерации по муниципальным образованиям» — лист
«Численность_по_МО»: регион, под ним муниципальные образования верхнего уровня (городские и
муниципальные округа, муниципальные районы), под ними поселения и города.

В бюллетене на 1 января 2024 года у строк есть код, и его первые восемь цифр — ОКТМО; остаток
в разных регионах записан по-разному («00», « 0 0», «10»). В бюллетене на 1 января 2023 года
кодов нет. Признаком служит численность на 1 января 2023 года — в редакции бюллетеня,
опубликованной до первого момента прогноза (какой файл брать, решает оболочка), — поэтому её
строки получают код через бюллетень 2024 года: по совпадению названия внутри региона.
Название, которое в регионе не единственное, не привязывается.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd

SHEET = "Численность_по_МО"
REGION, MUNICIPALITY, OTHER = "region", "municipality", "other"
_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
_OKTMO_DIGITS = 8
# Код длиннее десяти цифр — уровень поселения или населённого пункта.
_UPPER_CODE_LENGTH = 10


def _level(token: str) -> str:
    oktmo = token[:_OKTMO_DIGITS]
    if oktmo.endswith("000000"):
        return REGION
    if len(token) <= _UPPER_CODE_LENGTH and oktmo.endswith("000"):
        return MUNICIPALITY
    return OTHER


def parse_bulletin(rows: Sequence[Sequence[object]]) -> pd.DataFrame:
    """Лист бюллетеня → таблица строк с числами: `oktmo`, `level`, `region`, `name`, `total`,
    `urban`, `rural`.

    Строка идёт в таблицу, если в ней есть название и три числа (всё население, городское,
    сельское). Код, если он есть, стоит в первой ячейке; у листа без кодов `oktmo`, `level` и
    `region` пусты — их даёт `bridge_names`.
    """
    out = []
    for row in rows:
        cells = [str(cell).strip() for cell in row]
        token = cells[0].split(" ")[0] if cells and cells[0] else ""
        coded = token.isdigit() and len(token) >= _OKTMO_DIGITS
        rest = [cell for cell in (cells[1:] if coded else cells) if cell]
        names = [cell for cell in rest if not _NUMBER.match(cell)]
        numbers = [float(cell) for cell in rest if _NUMBER.match(cell)]
        if not names or len(numbers) < 3:
            continue
        out.append(
            {
                "oktmo": token[:_OKTMO_DIGITS] if coded else None,
                "level": _level(token) if coded else None,
                "name": " ".join(names[0].split()),
                "total": numbers[0],
                "urban": numbers[1],
                "rural": numbers[2],
            }
        )
    frame = pd.DataFrame(out, columns=["oktmo", "level", "name", "total", "urban", "rural"])
    frame["region"] = frame["name"].where(frame["level"] == REGION).ffill()
    return frame


def bridge_names(coded: pd.DataFrame, named: pd.DataFrame) -> pd.DataFrame:
    """Строкам бюллетеня без кодов даёт ОКТМО по названию внутри региона.

    Регионы листа без кодов опознаются по названиям регионов листа с кодами; дальше идут блоки
    «регион и всё, что под ним». Привязывается только название, единственное и среди
    муниципальных образований региона в листе с кодами, и в блоке региона листа без кодов.
    """
    regions = set(coded.loc[coded["level"] == REGION, "name"])
    named = named.assign(region=named["name"].where(named["name"].isin(regions)).ffill())
    units = coded[coded["level"] == MUNICIPALITY]
    keys = units.groupby(["region", "name"])["oktmo"].agg(list)
    unique = keys[keys.str.len() == 1].str[0].rename("oktmo").reset_index()
    candidates = named[~named["name"].isin(regions)].drop(columns=["oktmo", "level"])
    candidates = candidates[~candidates.duplicated(["region", "name"], keep=False)]
    bridged = candidates.merge(unique, on=["region", "name"])
    return bridged[["oktmo", "region", "name", "total", "urban", "rural"]].reset_index(drop=True)


def population_table(bridged: pd.DataFrame, territories: pd.DataFrame) -> pd.DataFrame:
    """Численность территорий панели: по строке на привязанную территорию.

    Территория находит свою строку по ОКТМО редакции справочника, по которой построена панель.
    """
    table = territories[["official_id", "oktmo"]].merge(bridged, on="oktmo")
    for column in ("total", "urban", "rural"):
        table[column] = table[column].round().astype(int)
    columns = ["official_id", "oktmo", "name", "total", "urban", "rural"]
    return table[columns].sort_values("official_id").reset_index(drop=True)


def population_features(table: pd.DataFrame, territories: pd.DataFrame) -> pd.DataFrame:
    """Постоянные характеристики территории: логарифм численности населения и доля городского
    населения. Территория без строки в таблице получает пропуск."""
    merged = territories[["territory_id", "official_id"]].merge(table, on="official_id", how="left")
    total = merged["total"].to_numpy(dtype=float)
    urban = merged["urban"].to_numpy(dtype=float)
    positive = np.where(total > 0, total, np.nan)
    out = pd.DataFrame(
        {
            "territory_id": merged["territory_id"].to_numpy(),
            "pop_log": np.log(positive),
            "pop_urban_share": urban / positive,
        }
    )
    return out.sort_values("territory_id").reset_index(drop=True)
