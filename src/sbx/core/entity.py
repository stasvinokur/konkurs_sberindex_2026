"""Однозначная идентификация муниципальных образований (МО) в данных расходов.

В исходном датасете МО задано только названием. Здесь чистыми функциями решаются три задачи:

1. Регион каждого вхождения МО в публичный список фильтра сайта (список сгруппирован по
   регионам): кандидаты по справочнику ОКТМО + сглаживание динамическим программированием
   (регионы идут непрерывными сегментами).
2. Для 49 неоднозначных названий — разбиение блоков строк датасета (каждый блок = один ряд
   «территория × категория») на территории и сопоставление территорий с регионами.
3. Реорганизации МО из курируемого списка — связывание рядов в одну территорию с датой
   административного разрыва.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from sbx.core.regions import Region, region_by_code, region_of_oktmo

TOTAL_CATEGORY = "Все категории"
_NO_REGION = Region("", "", "")
FEDERAL_CITIES = {"40", "45", "67"}

_TYPE_PATTERNS = [
    ("intracity", r"внутригородская территория (?:города федерального значения|городского округа)"),
    ("urban_okrug", r"городской округ"),
    ("municipal_okrug", r"муниципальный округ"),
    ("municipal_district", r"муниципальный район"),
    ("municipal_district", r"\b(?:район|улус|кожуун|аймак)\b"),
    ("settlement", r"\bпоселение\b"),
]
_NOISE = [
    r"внутригородская территория (?:города федерального значения|городского округа)",
    r"\(.*?\)",
    r"городской округ",
    r"муниципальный округ",
    r"муниципальный район",
    r"национальный",
    r"\bрайон\b",
    r"\bулус\b",
    r"\bкожуун\b",
    r"\bаймак\b",
    r"\bокруг\b",
    r"\bгород-курорт\b",
    r"\bгород\b",
    r"\bпоселок городского типа\b",
    r"\bрабочий поселок\b",
    r"\bпоселок\b",
    r"\bпоселение\b",
    r"\bсельское\b",
    r"\bзато\b",
    r"\bг\.",
    r"№",
]


def normalize_name(name: str) -> tuple[str, str]:
    """Возвращает (корень названия, тип МО) для сопоставления разных написаний."""
    text = str(name).lower().replace("ё", "е").replace('"', " ").replace("«", " ").replace("»", " ")
    kind = "other"
    for k, pattern in _TYPE_PATTERNS:
        if re.search(pattern, text):
            kind = k
            break
    root = text
    for pattern in _NOISE:
        root = re.sub(pattern, " ", root)
    root = re.sub(r"[^0-9a-zа-я\- ]", " ", root)
    root = re.sub(r"\s+", " ", root).strip(" -")
    return root, kind


def stem(root: str) -> str:
    """Грубая основа прилагательного/существительного: «кунгурский» ~ «кунгур»."""
    words = []
    for word in root.split():
        w = re.sub(
            r"(ский|цкий|ской|ская|ское|ский|ный|ная|ное|ий|ой|ая|ое|ы|и|а|о|е|у)$", "", word
        )
        words.append(w if len(w) >= 3 else word)
    return " ".join(words)


def reference_index(reference: pd.DataFrame) -> dict[str, dict[str, set[tuple[str, str]]]]:
    """Индексы справочника (label, oktmo): точный корень и основа → {(код региона, тип МО)}."""
    index: dict[str, dict[str, set[tuple[str, str]]]] = {"exact": {}, "stem": {}}
    for label, oktmo in zip(reference["label"], reference["oktmo"], strict=True):
        region = region_of_oktmo(oktmo)
        if region is None:
            continue
        root, kind = normalize_name(label)
        if not root:
            continue
        index["exact"].setdefault(root, set()).add((region.code, kind))
        index["stem"].setdefault(stem(root), set()).add((region.code, kind))
    return index


def candidate_regions(
    names: Sequence[str], index: Mapping[str, Mapping[str, set[tuple[str, str]]]]
) -> list[set[str]]:
    """Регионы-кандидаты: сначала точный корень, затем основа.

    Если кандидатов несколько и ровно один регион назван так же, как МО (город Тамбов →
    Тамбовская область), остаётся только он.
    """
    out = []
    for name in names:
        root, kind = normalize_name(name)
        hits = set(index["exact"].get(root, set())) or set(index["stem"].get(stem(root), set()))
        regions = {r for r, _ in hits}
        # Во внутригородских территориях — только города федерального значения, и наоборот.
        regions = (
            regions & FEDERAL_CITIES if kind == "intracity" else regions - FEDERAL_CITIES
        ) or regions
        if len(regions) > 1 and root:
            eponymous = {
                r
                for r in regions
                if stem(region_by_code(r).name.lower()).split()[0] == stem(root).split()[0]
            }
            if len(eponymous) == 1:
                regions = eponymous
        out.append(regions)
    return out


def smooth_regions(
    candidates: Sequence[set[str]],
    switch_cost: float = 0.45,
    miss_cost: float = 1.0,
    unknown_cost: float = 0.2,
) -> list[str | None]:
    """Витерби: каждой позиции списка — регион; регионы идут сегментами, кандидаты — «наблюдения»."""
    states = sorted({c for cands in candidates for c in cands})
    if not states:
        return [None] * len(candidates)
    n, k = len(candidates), len(states)
    pos = {s: i for i, s in enumerate(states)}

    def emission(i: int) -> np.ndarray:
        cands = candidates[i]
        if not cands:
            return np.full(k, unknown_cost)
        cost = np.full(k, miss_cost)
        for c in cands:
            cost[pos[c]] = 0.0
        return cost

    dp = emission(0)
    back = np.zeros((n, k), dtype=int)
    for i in range(1, n):
        best_prev = int(np.argmin(dp))
        stay = dp
        switch = dp[best_prev] + switch_cost
        choose_switch = switch < stay
        back[i] = np.where(choose_switch, best_prev, np.arange(k))
        dp = np.minimum(stay, switch) + emission(i)
    path = [int(np.argmin(dp))]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    path.reverse()
    return [states[j] for j in path]


def _segments(labels: Sequence[str | None]) -> list[tuple[int, int, str | None]]:
    """Непрерывные сегменты одинаковых меток: (начало, конец не включая, метка)."""
    out = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((start, i, labels[start]))
            start = i
    return out


def ordered_prefix_end(labels: Sequence[str | None], long_segment: int = 5) -> int:
    """Конец упорядоченной по регионам части списка.

    В конец списка фильтра дописаны новые МО вне порядка регионов; упорядоченная часть
    заканчивается последним «длинным» сегментом.
    """
    segs = [s for s in _segments(labels) if s[1] - s[0] >= long_segment]
    return segs[-1][1] if segs else len(labels)


def fix_sandwiched_segments(
    labels: Sequence[str | None], end: int, max_len: int = 3
) -> tuple[list[str | None], set[int]]:
    """Короткий сегмент между двумя сегментами одного региона получает этот регион."""
    result = list(labels)
    changed: set[int] = set()
    while True:
        segs = _segments(result[:end])
        fixed = False
        for (_, _, prev), (s, e, lab), (_, _, nxt) in zip(segs, segs[1:], segs[2:], strict=False):
            if e - s <= max_len and prev == nxt and prev is not None and lab != prev:
                for i in range(s, e):
                    result[i] = prev
                    changed.add(i)
                fixed = True
                break
        if not fixed:
            return result, changed


def attach_stray_singletons(
    labels: Sequence[str | None], end: int, max_len: int = 2
) -> tuple[list[str | None], set[int]]:
    """Короткий сегмент региона, у которого есть длинный сегмент в другом месте упорядоченной
    части, присоединяется к следующему сегменту (список региона начинается с его центра)."""
    result = list(labels)
    changed: set[int] = set()
    segs = _segments(result[:end])
    longest: dict[str | None, int] = {}
    for s_, e_, lab in segs:
        longest[lab] = max(longest.get(lab, 0), e_ - s_)
    for (s_, e_, lab), (_, _, nxt) in zip(segs, segs[1:], strict=False):
        if e_ - s_ <= max_len and longest[lab] > max_len and nxt is not None and nxt != lab:
            for i in range(s_, e_):
                result[i] = nxt
                changed.add(i)
    return result, changed


def resolve_filter_regions(
    filter_names: Sequence[str],
    reference: pd.DataFrame,
    overrides: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Регион для каждой позиции списка фильтра + источник и уверенность.

    `overrides` — курируемые исправления «название → код региона» для однозначных названий,
    которые нельзя разрешить по справочнику и порядку списка.
    """
    overrides = dict(overrides or {})
    index = reference_index(reference)
    cands = candidate_regions(filter_names, index)
    smoothed = smooth_regions(cands)
    end = ordered_prefix_end(smoothed)
    smoothed, sandwiched = fix_sandwiched_segments(smoothed, end)
    smoothed, strays = attach_stray_singletons(smoothed, end)
    sandwiched |= strays
    rows = []
    for i, (name, cand, region) in enumerate(zip(filter_names, cands, smoothed, strict=True)):
        if name in overrides:
            region, source, confidence = overrides[name], "manual_override", 0.95
        elif i in sandwiched:
            source, confidence = "filter_order_neighbors", 0.8
        elif region is not None and cand == {region}:
            source, confidence = "oktmo_reference", 1.0
        elif region is not None and region in cand:
            source, confidence = "oktmo_reference+filter_order", 0.9
        elif region is not None and not cand:
            source, confidence = "filter_order", 0.6
        else:
            source, confidence = "filter_order_conflict", 0.3
        rows.append(
            {
                "filter_pos": i,
                "raw_mo": name,
                "region_code": region,
                "candidates": ",".join(sorted(cand)),
                "region_source": source,
                "region_confidence": confidence,
            }
        )
    return pd.DataFrame(rows)


def series_blocks(raw: pd.DataFrame) -> pd.DataFrame:
    """Номер непрерывного блока строк (mo, category) в исходном порядке датасета."""
    key = raw["mo"].astype(str) + "\x1f" + raw["category_15"].astype(str)
    period = raw["period"].astype(str)
    # Новый блок начинается при смене ключа или «откате» периода (два одноимённых ряда подряд).
    starts = (key != key.shift()) | (period <= period.shift())
    return starts.cumsum().rename("block_id")


@dataclass(frozen=True)
class BlockSummary:
    block_id: int
    mo: str
    category: str
    periods: frozenset[str]
    values: Mapping[str, float]


def summarize_blocks(raw: pd.DataFrame) -> list[BlockSummary]:
    df = raw.assign(block_id=series_blocks(raw).to_numpy())
    out = []
    for block_id, g in df.groupby("block_id", sort=True):
        periods = [str(p) for p in g["period"]]
        out.append(
            BlockSummary(
                int(block_id),
                str(g["mo"].iloc[0]),
                str(g["category_15"].iloc[0]),
                frozenset(periods),
                dict(zip(periods, g["value"].astype(float), strict=True)),
            )
        )
    return out


def typical_category_share(raw: pd.DataFrame, exclude_names: Iterable[str]) -> dict[str, float]:
    """Медиана log(категория / «Все категории») по однозначным МО."""
    clean = raw[~raw["mo"].isin(set(exclude_names))]
    wide = clean.pivot_table(index=["mo", "period"], columns="category_15", values="value")
    shares = {}
    for cat in wide.columns:
        ratio = np.log(wide[cat] / wide[TOTAL_CATEGORY]).replace([np.inf, -np.inf], np.nan)
        shares[cat] = float(np.nanmedian(ratio)) if cat != TOTAL_CATEGORY else 0.0
    return shares


def _block_cost(total: BlockSummary, other: BlockSummary, share: float) -> float:
    common = sorted(total.periods & other.periods)
    union = total.periods | other.periods
    coverage = 1.0 - len(common) / max(len(union), 1)
    if not common:
        return 10.0 + coverage
    logs = [math.log(max(other.values[p], 1.0) / max(total.values[p], 1.0)) for p in common]
    level = abs(float(np.median(logs)) - share)
    exceed = sum(other.values[p] > total.values[p] * 1.001 for p in common) / len(common)
    return 4.0 * coverage + level + 3.0 * exceed


def group_ambiguous_blocks(
    blocks: Sequence[BlockSummary], shares: Mapping[str, float]
) -> dict[int, int]:
    """Разбивает блоки одного названия на территории: block_id → номер территории.

    Якоря — блоки «Все категории»; блоки остальных категорий назначаются им венгерским
    алгоритмом по совпадению покрытия периодов и типичному уровню доли категории.
    """
    anchors = sorted((b for b in blocks if b.category == TOTAL_CATEGORY), key=lambda b: b.block_id)
    assignment = {b.block_id: i for i, b in enumerate(anchors)}
    by_cat: dict[str, list[BlockSummary]] = {}
    for b in blocks:
        if b.category != TOTAL_CATEGORY:
            by_cat.setdefault(b.category, []).append(b)
    for cat, cat_blocks in by_cat.items():
        cost = np.array(
            [[_block_cost(a, b, shares.get(cat, 0.0)) for a in anchors] for b in cat_blocks]
        )
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols, strict=True):
            assignment[cat_blocks[r].block_id] = int(c)
        for r in set(range(len(cat_blocks))) - set(rows):
            assignment[cat_blocks[r].block_id] = -1
    return assignment


def assign_territories_to_regions(
    territory_levels: Sequence[float],
    territory_starts: Sequence[str],
    occurrence_regions: Sequence[str | None],
    region_levels: Mapping[str, float],
    region_starts: Mapping[str, str],
) -> list[int]:
    """Сопоставляет территории (по уровню и началу ряда) вхождениям списка фильтра (регионам)."""
    cost = np.zeros((len(territory_levels), len(occurrence_regions)))
    for i, (level, start) in enumerate(zip(territory_levels, territory_starts, strict=True)):
        for j, region in enumerate(occurrence_regions):
            ref = region_levels.get(region) if region else None
            level_cost = abs(level - ref) if ref is not None else 1.0
            start_cost = 0.0 if region is None or region_starts.get(region, start) <= start else 0.5
            cost[i, j] = level_cost + start_cost
    rows, cols = linear_sum_assignment(cost)
    out = [-1] * len(territory_levels)
    for r, c in zip(rows, cols, strict=True):
        out[r] = int(c)
    return out


def squash_spaces(name: str) -> str:
    """Схлопывает повторные пробелы: в сырых данных встречаются двойные пробелы в названиях."""
    return re.sub(r"\s+", " ", str(name)).strip()


def slugify(name: str) -> str:
    table = str.maketrans(
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
        "abvgdeejzijklmnoprstufhccss_y_eua",
    )
    s = str(name).lower().translate(table)
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def build_entity_mapping(
    raw: pd.DataFrame,
    filter_names: Sequence[str],
    reference: pd.DataFrame,
    reorganizations: Sequence[Mapping[str, str]] = (),
    region_overrides: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Полная таблица соответствия блоков строк датасета территориям.

    Возвращает (mapping по блокам, таблица территорий).
    """
    raw = raw.assign(mo=raw["mo"].map(squash_spaces))
    filter_names = [squash_spaces(n) for n in filter_names]
    region_overrides = {squash_spaces(k): v for k, v in (region_overrides or {}).items()}
    reorganizations = [
        {
            **r,
            "successor": squash_spaces(r["successor"]),
            "predecessor": squash_spaces(r["predecessor"]),
        }
        for r in reorganizations
    ]
    filter_regions = resolve_filter_regions(filter_names, reference, region_overrides)
    blocks = summarize_blocks(raw)
    counts = pd.Series(list(filter_names)).value_counts()
    ambiguous = set(counts[counts > 1].index)

    # Уровни регионов по однозначным МО: медиана log «Все категории».
    fr_unique = filter_regions[~filter_regions["raw_mo"].isin(ambiguous)].set_index("raw_mo")
    total = raw[(raw["category_15"] == TOTAL_CATEGORY) & ~raw["mo"].isin(ambiguous)]
    mo_level = np.log(total.groupby("mo")["value"].median())
    mo_start = total.groupby("mo")["period"].min()
    region_of_mo = fr_unique["region_code"]
    region_levels = mo_level.groupby(region_of_mo.reindex(mo_level.index)).median().to_dict()
    region_starts = mo_start.groupby(region_of_mo.reindex(mo_start.index)).min().to_dict()
    shares = typical_category_share(raw, ambiguous)

    records = []
    territories = []
    for name in sorted({b.mo for b in blocks}):
        name_blocks = [b for b in blocks if b.mo == name]
        occ = filter_regions[filter_regions["raw_mo"] == name].sort_values("filter_pos")
        if name not in ambiguous:
            row = occ.iloc[0] if len(occ) else None
            region = row["region_code"] if row is not None else None
            tid = f"{region or 'xx'}-{slugify(name)}"
            territories.append(
                {
                    "territory_id": tid,
                    "raw_mo": name,
                    "region_code": region,
                    "region_source": row["region_source"] if row is not None else "none",
                    "region_confidence": row["region_confidence"] if row is not None else 0.0,
                    "resolution_source": "unique_name",
                    "confidence": row["region_confidence"] if row is not None else 0.0,
                }
            )
            records += [{"block_id": b.block_id, "territory_id": tid} for b in name_blocks]
            continue

        groups = group_ambiguous_blocks(name_blocks, shares)
        anchors = sorted(
            (b for b in name_blocks if b.category == TOTAL_CATEGORY), key=lambda b: b.block_id
        )
        levels = [float(np.log(np.median(list(a.values.values())))) for a in anchors]
        starts = [min(a.periods) for a in anchors]
        occ_regions = list(occ["region_code"])
        match = assign_territories_to_regions(
            levels, starts, occ_regions, region_levels, region_starts
        )
        tids = []
        for i in range(len(anchors)):
            j = match[i]
            region = occ_regions[j] if j >= 0 else None
            distinct = len({r for r in occ_regions if r}) == len(occ_regions)
            tid = f"{region or 'xx'}-{slugify(name)}"
            if tid in tids or not distinct:
                tid = f"{tid}-{i + 1}"
            tids.append(tid)
            territories.append(
                {
                    "territory_id": tid,
                    "raw_mo": name,
                    "region_code": region,
                    "region_source": occ.iloc[j]["region_source"] if j >= 0 else "none",
                    "region_confidence": occ.iloc[j]["region_confidence"] if j >= 0 else 0.0,
                    "resolution_source": "ambiguous_name:block_grouping+level_matching",
                    "confidence": 0.5 if distinct else 0.3,
                }
            )
        for b in name_blocks:
            g = groups.get(b.block_id, -1)
            records.append({"block_id": b.block_id, "territory_id": tids[g] if g >= 0 else None})

    terr = pd.DataFrame(territories)
    terr["oktmo"] = [
        match_oktmo(name, region, reference)
        for name, region in zip(terr["raw_mo"], terr["region_code"], strict=True)
    ]
    conflicts = terr["oktmo"].notna() & terr["oktmo"].duplicated(keep=False)
    terr.loc[conflicts, "oktmo"] = None
    terr, renames = apply_reorganizations(terr, reorganizations)
    mapping = pd.DataFrame(records)
    mapping["territory_id"] = mapping["territory_id"].map(lambda t: renames.get(t, t))
    block_info = pd.DataFrame(
        [{"block_id": b.block_id, "raw_mo": b.mo, "category": b.category} for b in blocks]
    )
    columns = [
        "raw_mo",
        "region_code",
        "oktmo",
        "resolution_source",
        "confidence",
        "admin_break_at",
    ]
    mapping = mapping.merge(block_info, on="block_id", how="left").merge(
        terr[columns], on="raw_mo", how="left", suffixes=("", "_t")
    )
    mapping = mapping.drop_duplicates("block_id")
    return mapping, terr


def match_oktmo(name: str, region_code: str | None, reference: pd.DataFrame) -> str | None:
    """Код ОКТМО территории: единственное совпадение в пределах региона.

    Сначала точный корень названия, затем основа — но только среди объектов того же типа МО
    (иначе «Бузулукский район» совпал бы с «городом Бузулук»).
    """
    if region_code is None:
        return None
    root, kind = normalize_name(name)
    codes = reference["oktmo"].astype(str)
    in_region = reference[
        codes.map(lambda c: (region_of_oktmo(c) or _NO_REGION).code == region_code)
    ]
    parsed = in_region["label"].map(normalize_name)
    roots = parsed.map(lambda x: x[0])
    kinds = parsed.map(lambda x: x[1])
    exact = in_region.loc[roots == root, "oktmo"].unique()
    if len(exact) == 1:
        return str(exact[0])
    compatible = kinds.map(lambda k: _kinds_compatible(kind, k))
    by_stem = in_region.loc[(roots.map(stem) == stem(root)) & compatible, "oktmo"].unique()
    return str(by_stem[0]) if len(by_stem) == 1 else None


def _kinds_compatible(a: str, b: str) -> bool:
    district_like = {"municipal_district", "municipal_okrug"}
    return a == b or (a in district_like and b in district_like)


def apply_reorganizations(
    territories: pd.DataFrame, reorganizations: Sequence[Mapping[str, str]]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Связывает ряды реорганизованных МО.

    Запись: {successor, predecessor, relation, effective: YYYY-MM-DD, source}.
    - `continuation` — преемник продолжает территорию предшественника: оба ряда получают один
      `territory_id` (преемника), в дату `effective` стоит административный разрыв;
    - `merge` — предшественник влит в преемника: ряды остаются разными, предшественнику
      проставляются `successor_territory_id` и `admin_break_at`.

    Возвращает (таблица территорий, словарь переименования territory_id).
    """
    terr = territories.copy()
    terr["admin_break_at"] = None
    terr["successor_territory_id"] = None
    renames: dict[str, str] = {}
    for reorg in reorganizations:
        succ = terr.index[terr["raw_mo"] == reorg["successor"]]
        pred = terr.index[terr["raw_mo"] == reorg["predecessor"]]
        if len(succ) != 1 or len(pred) != 1:
            raise ValueError(f"reorganization names must be unique in data: {reorg}")
        if reorg["relation"] not in {"continuation", "merge"}:
            raise ValueError(f"unknown relation: {reorg['relation']}")
        succ_id = terr.loc[succ[0], "territory_id"]
        terr.loc[succ[0], "admin_break_at"] = reorg["effective"]
        terr.loc[pred[0], "admin_break_at"] = reorg["effective"]
        terr.loc[pred[0], "successor_territory_id"] = succ_id
        if reorg["relation"] == "continuation":
            renames[terr.loc[pred[0], "territory_id"]] = succ_id
            terr.loc[pred[0], "territory_id"] = succ_id
    return terr, renames


@dataclass(frozen=True)
class RegionOverride:
    name: str
    region_code: str
    source: str


@dataclass(frozen=True)
class Reorganization:
    successor: str
    predecessor: str
    relation: str
    effective: str
    source: str


@dataclass(frozen=True)
class EntityReference:
    filter_order: str
    oktmo: str


@dataclass(frozen=True)
class EntityConfig:
    reference: EntityReference
    region_overrides: tuple[RegionOverride, ...] = ()
    reorganizations: tuple[Reorganization, ...] = ()


def mapping_report(raw: pd.DataFrame, mapping: pd.DataFrame, territories: pd.DataFrame) -> dict:
    """Сводка разрешения для документации и проверок."""
    rows = raw.assign(block_id=series_blocks(raw).to_numpy()).merge(
        mapping, on="block_id", how="left"
    )
    dup = rows.dropna(subset=["territory_id"]).duplicated(["territory_id", "category_15", "period"])
    return {
        "raw_rows": int(len(raw)),
        "rows_mapped": int(rows["territory_id"].notna().sum()),
        "rows_unmapped": int(rows["territory_id"].isna().sum()),
        "duplicate_keys_after": int(dup.sum()),
        "territories": int(territories["territory_id"].nunique()),
        "raw_names": int(raw["mo"].nunique()),
        "ambiguous_territories": int(
            territories["resolution_source"].str.startswith("ambiguous").sum()
        ),
        "region_source_counts": territories["region_source"].value_counts().to_dict(),
        "admin_breaks": territories.dropna(subset=["admin_break_at"])[
            ["territory_id", "raw_mo", "admin_break_at", "successor_territory_id"]
        ].to_dict("records"),
    }
