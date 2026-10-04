"""Сопоставление публичной выгрузки расходов с официальным набором организаторов.

В выгрузке территория задана названием, в официальном наборе — кодом, а наблюдения одни и те
же. Ряд однозначно узнаётся по своим значениям: набор «категория, месяц, значение» у каждого
ряда свой. Это даёт точную разметку рядов выгрузки официальными кодами — без названий — и
позволяет измерить, насколько ошибалась идентификация по названиям.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import pandas as pd

from sbx.core.entity import series_blocks
from sbx.core.panel import CATEGORY_SLUGS


class CrosswalkError(ValueError):
    """Ряды выгрузки и официального набора не сопоставляются один к одному."""


def _signatures(frame: pd.DataFrame, key: str, category: str, month: str, value: str) -> dict:
    """Подпись ряда — категория и все его пары (месяц, значение); ключ ряда → подпись."""
    months = frame[month].astype(str).str.slice(0, 7)
    values = frame[value].round().astype("int64")
    table = pd.DataFrame(
        {
            "key": frame[key].to_numpy(),
            "category": frame[category].to_numpy(),
            "m": months,
            "v": values,
        }
    )
    return {
        series_key: (
            group["category"].iloc[0],
            tuple(sorted(zip(group["m"], group["v"], strict=True))),
        )
        for series_key, group in table.groupby("key", sort=False)
    }


def match_blocks(raw: pd.DataFrame, consumption: pd.DataFrame) -> pd.DataFrame:
    """Ряды выгрузки (блоки строк) ↔ официальные ряды: `block_id`, `category`, `official_id`.

    Сопоставление обязано быть взаимно однозначным; иначе выгрузка и набор — разные данные,
    и продолжать нельзя.
    """
    blocks = raw.assign(block_id=series_blocks(raw).to_numpy())
    public = _signatures(blocks, "block_id", "category_15", "period", "value")
    official_frame = consumption.assign(
        series=list(zip(consumption["territory_id"], consumption["category"], strict=True))
    )
    official = _signatures(official_frame, "series", "category", "date", "value")

    repeated = [sig for sig, n in Counter(official.values()).items() if n > 1]
    if repeated:
        raise CrosswalkError(f"у {len(repeated)} официальных рядов одинаковые значения")
    by_signature = {signature: series for series, signature in official.items()}
    unmatched = [block for block, signature in public.items() if signature not in by_signature]
    if unmatched:
        raise CrosswalkError(f"{len(unmatched)} рядов выгрузки не нашлись в официальном наборе")
    used = {by_signature[signature] for signature in public.values()}
    if len(used) != len(official) or len(public) != len(official):
        raise CrosswalkError(f"{len(official) - len(used)} официальных рядов нет в выгрузке")

    rows = [
        {
            "block_id": int(block),
            "category": signature[0],
            "official_id": int(by_signature[signature][0]),
        }
        for block, signature in public.items()
    ]
    return pd.DataFrame(rows).sort_values("block_id").reset_index(drop=True)


def crosswalk_table(
    pairs: pd.DataFrame,
    mapping: pd.DataFrame,
    heuristic_territories: pd.DataFrame,
    territories: pd.DataFrame,
) -> pd.DataFrame:
    """По строке на ряд выгрузки: кто это по официальному набору и кем его считала эвристика.

    `pairs` — результат `match_blocks`; `mapping` и `heuristic_territories` — результат
    идентификации по названиям; `territories` — официальная таблица территорий.
    """
    official = territories.set_index("official_id")
    # Регион территории эвристики берётся так же, как его брали статические признаки панели.
    heuristic = heuristic_territories.drop_duplicates("territory_id", keep="last").set_index(
        "territory_id"
    )
    table = pairs.merge(
        mapping[["block_id", "territory_id", "raw_mo"]].rename(
            columns={"territory_id": "heuristic_territory"}
        ),
        on="block_id",
        how="left",
        validate="one_to_one",
    )
    slug = table["category"].map(CATEGORY_SLUGS)
    table["territory_id"] = table["official_id"].map(official["territory_id"])
    table["unique_id"] = table["territory_id"] + "__" + slug
    table["official_region"] = table["official_id"].map(official["region_code"])
    table["legacy_unique_id"] = table["heuristic_territory"] + "__" + slug
    table["heuristic_region"] = table["heuristic_territory"].map(heuristic["region_code"])
    columns = [
        "block_id",
        "category",
        "raw_mo",
        "official_id",
        "territory_id",
        "unique_id",
        "official_region",
        "heuristic_territory",
        "legacy_unique_id",
        "heuristic_region",
    ]
    return table[columns].sort_values("block_id").reset_index(drop=True)


def to_official_ids(legacy_ids: Iterable[str], table: pd.DataFrame) -> pd.DataFrame:
    """Список рядов в идентификаторах эвристики → те же ряды в официальных идентификаторах.

    Ряд-склейка эвристики (два названия в одной территории) — это несколько официальных
    рядов, и в переводе остаются все.
    """
    wanted = set(legacy_ids)
    unknown = sorted(wanted - set(table["legacy_unique_id"]))
    if unknown:
        raise CrosswalkError(f"рядов нет в таблице сопоставления: {unknown[:5]}")
    picked = table.loc[table["legacy_unique_id"].isin(wanted), ["unique_id", "legacy_unique_id"]]
    return picked.sort_values(["unique_id"]).reset_index(drop=True)


def rekey(frame: pd.DataFrame, column: str, table: pd.DataFrame) -> pd.DataFrame:
    """Заменяет в колонке идентификаторы рядов эвристики официальными.

    Годится только для рядов, которым соответствует ровно один официальный ряд: ряд-склейку
    двух МО одним идентификатором не назвать.
    """
    wanted = set(frame[column])
    known = table[table["legacy_unique_id"].isin(wanted)]
    unknown = sorted(wanted - set(known["legacy_unique_id"]))
    if unknown:
        raise CrosswalkError(f"рядов нет в таблице сопоставления: {unknown[:5]}")
    counts = known.groupby("legacy_unique_id")["unique_id"].nunique()
    glued = sorted(counts[counts > 1].index)
    if glued:
        raise CrosswalkError(f"рядам соответствует несколько официальных: {glued[:5]}")
    official = known.drop_duplicates("legacy_unique_id").set_index("legacy_unique_id")["unique_id"]
    return frame.assign(**{column: frame[column].map(official)})


def heuristic_agreement(pairs: pd.DataFrame) -> dict[str, int]:
    """Счётчики расхождения идентификации по названиям с официальной разметкой.

    `pairs` — по строке на ряд: территория, название и регион по эвристике
    (`heuristic_territory`, `raw_mo`, `heuristic_region`) и официальные `official_id`,
    `official_region`.
    """
    wrong = pairs["heuristic_region"].astype(str) != pairs["official_region"].astype(str)
    by_territory = wrong.groupby(pairs["heuristic_territory"])
    # Путаница — когда ряды ОДНОГО названия внутри территории принадлежат разным МО.
    per_name = pairs.groupby(["heuristic_territory", "raw_mo"])["official_id"].nunique()
    mixing = (per_name > 1).groupby(level="heuristic_territory").any()
    names = pairs.groupby("heuristic_territory")["raw_mo"].nunique()
    return {
        "series": int(len(pairs)),
        "series_in_wrong_region": int(wrong.sum()),
        "territories": int(pairs["heuristic_territory"].nunique()),
        "territories_with_foreign_series": int(by_territory.any().sum()),
        "territories_entirely_in_wrong_region": int(by_territory.all().sum()),
        "territories_mixing_same_name_series": int(mixing.sum()),
        "territories_glued_from_several_names": int((names > 1).sum()),
        "official_territories": int(pairs["official_id"].nunique()),
    }
