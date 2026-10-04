import math

import pandas as pd

from sbx.core.entity import (
    attach_stray_singletons,
    build_entity_mapping,
    fix_sandwiched_segments,
    group_ambiguous_blocks,
    mapping_report,
    normalize_name,
    smooth_regions,
    summarize_blocks,
)

PERIODS = [f"2023-{m:02d}-01" for m in range(1, 13)]
CATS = ["Все категории", "Продовольствие", "Здоровье"]
SHARE = {"Все категории": 1.0, "Продовольствие": 0.45, "Здоровье": 0.05}


def _series(mo: str, level: float, cats=CATS, periods=PERIODS) -> list[dict]:
    rows = []
    for cat in cats:
        for i, p in enumerate(periods):
            rows.append(
                {
                    "period": p,
                    "mo": mo,
                    "category_15": cat,
                    "value": level * SHARE[cat] * (1 + i / 100),
                }
            )
    return rows


def _fixture() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Два региона (Дагестан 82, КЧР 91), в каждом свой «Ногайский муниципальный район».

    Уровень расходов Дагестана ниже, чем КЧР; блоки строк датасета перемешаны.
    """
    dag = [
        ("город Махачкала", 30000),
        ("Бабаюртовский район", 15000),
        ("Кумторкалинский район", 16000),
    ]
    kchr = [("город Черкесск", 40000), ("Абазинский район", 32000), ("Урупский район", 31000)]
    rows = []
    for name, level in dag + kchr:
        rows += _series(name, level)
    nog_dag = _series("Ногайский муниципальный район", 14000)
    nog_kchr = _series("Ногайский муниципальный район", 33000)
    # Перемешиваем блоки: категории двух «Ногайских» идут вперемешку с другими рядами.
    blocks_dag = [nog_dag[i : i + 12] for i in range(0, len(nog_dag), 12)]
    blocks_kchr = [nog_kchr[i : i + 12] for i in range(0, len(nog_kchr), 12)]
    rows = blocks_kchr[1] + rows[:36] + blocks_dag[0] + blocks_kchr[0] + rows[36:] + blocks_dag[2]
    rows += blocks_kchr[2] + blocks_dag[1]
    raw = pd.DataFrame(rows)
    filter_names = [
        "город Махачкала",
        "Бабаюртовский район",
        "Ногайский муниципальный район",
        "Кумторкалинский район",
        "город Черкесск",
        "Абазинский район",
        "Ногайский муниципальный район",
        "Урупский район",
    ]
    reference = pd.DataFrame(
        {
            "label": [
                "Махачкала (городской округ)",
                "Бабаюртовский район",
                "Кумторкалинский район",
                "Ногайский район",
                "Черкесск (городской округ)",
                "Абазинский район",
                "Урупский район",
                "Ногайский район",
            ],
            "oktmo": [
                "82701000",
                "82606000",
                "82627000",
                "82640000",
                "91701000",
                "91603000",
                "91620000",
                "91625000",
            ],
        }
    )
    return raw, filter_names, reference


def test_normalize_name_extracts_root_and_kind() -> None:
    assert normalize_name("Гиагинский муниципальный район") == ("гиагинский", "municipal_district")
    assert normalize_name("городской округ город-курорт Кисловодск") == (
        "кисловодск",
        "urban_okrug",
    )
    assert normalize_name("Абыйский улус") == ("абыйский", "municipal_district")
    root, kind = normalize_name(
        "внутригородская территория города федерального значения поселение Щаповское"
    )
    assert (root, kind) == ("щаповское", "intracity")


def test_smoothing_prefers_contiguous_regions() -> None:
    cands = [{"82"}, {"82", "91"}, set(), {"82"}, {"91"}, {"91", "82"}, {"91"}]
    assert smooth_regions(cands) == ["82", "82", "82", "82", "91", "91", "91"]


def test_sandwich_and_stray_fixes() -> None:
    labels = ["70"] * 6 + ["45"] + ["70"] * 6 + ["05"] + ["33"] * 6 + ["05"] * 6
    fixed, changed = fix_sandwiched_segments(labels, end=len(labels))
    assert fixed[6] == "70" and changed == {6}
    fixed2, changed2 = attach_stray_singletons(fixed, end=len(fixed))
    assert fixed2[13] == "33" and changed2 == {13}


def test_ambiguous_blocks_are_grouped_by_territory() -> None:
    raw, _, _ = _fixture()
    blocks = [b for b in summarize_blocks(raw) if b.mo == "Ногайский муниципальный район"]
    assert len(blocks) == 6
    shares = {c: math.log(SHARE[c]) for c in CATS}
    groups = group_ambiguous_blocks(blocks, shares)
    by_group: dict[int, set[float]] = {}
    for b in blocks:
        level = next(iter(b.values.values())) / SHARE[b.category]
        by_group.setdefault(groups[b.block_id], set()).add(round(level))
    assert sorted(len(v) for v in by_group.values()) == [1, 1]


def test_nogaisky_split_into_two_regions_without_duplicates() -> None:
    raw, filter_names, reference = _fixture()
    mapping, territories = build_entity_mapping(raw, filter_names, reference)
    nog = territories[territories["raw_mo"] == "Ногайский муниципальный район"]
    assert set(nog["region_code"]) == {"82", "91"}
    report = mapping_report(raw, mapping, territories)
    assert report["rows_unmapped"] == 0 and report["duplicate_keys_after"] == 0
    # Низкий уровень расходов — Дагестан.
    blocks = {b.block_id: b for b in summarize_blocks(raw)}
    low_block = next(
        bid
        for bid, b in blocks.items()
        if b.mo == "Ногайский муниципальный район"
        and b.category == "Все категории"
        and min(b.values.values()) < 20000
    )
    tid = mapping.set_index("block_id").loc[low_block, "territory_id"]
    assert tid.startswith("82-")


def test_reorganization_continuation_and_merge() -> None:
    raw, filter_names, reference = _fixture()
    reorgs = [
        {
            "successor": "город Черкесск",
            "predecessor": "Абазинский район",
            "relation": "continuation",
            "effective": "2024-01-01",
            "source": "test",
        },
        {
            "successor": "город Черкесск",
            "predecessor": "Урупский район",
            "relation": "merge",
            "effective": "2024-01-01",
            "source": "test",
        },
    ]
    raw = raw[~((raw["mo"] == "Абазинский район") & (raw["period"] > "2023-06-01"))]
    raw = raw[~((raw["mo"] == "город Черкесск") & (raw["period"] <= "2023-06-01"))]
    mapping, territories = build_entity_mapping(raw, filter_names, reference, reorgs)
    t = territories.set_index("raw_mo")
    assert t.loc["Абазинский район", "territory_id"] == t.loc["город Черкесск", "territory_id"]
    assert t.loc["город Черкесск", "admin_break_at"] == "2024-01-01"
    assert (
        t.loc["Урупский район", "successor_territory_id"] == t.loc["город Черкесск", "territory_id"]
    )
    assert t.loc["Урупский район", "territory_id"] != t.loc["город Черкесск", "territory_id"]
    assert mapping_report(raw, mapping, territories)["duplicate_keys_after"] == 0


def test_mapping_is_deterministic() -> None:
    raw, filter_names, reference = _fixture()
    m1, t1 = build_entity_mapping(raw, filter_names, reference)
    m2, t2 = build_entity_mapping(raw, filter_names, reference)
    pd.testing.assert_frame_equal(m1, m2)
    pd.testing.assert_frame_equal(t1, t2)
