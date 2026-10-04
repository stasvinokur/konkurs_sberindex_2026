"""Шаг идентификации МО: официальные коды организаторов и сверка прежней эвристики с ними.

Территорию ряда задаёт официальный `territory_id` набора организаторов конкурса, а название,
тип, ОКТМО и координаты — справочник территорий. Идентификация по названиям
(`sbx.core.entity`), на которой решение стояло раньше, осталась как независимая сверка: по
ней измеряется, насколько названия вводили в заблуждение, и через неё замороженные выборки
переведены в официальные коды.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd

from sbx.core.crosswalk import crosswalk_table, heuristic_agreement, match_blocks
from sbx.core.entity import EntityConfig, build_entity_mapping, mapping_report
from sbx.core.territories import build_territories, kind_changes
from sbx.shell.download import official
from sbx.shell.io import CONFIG_DIR, DATA_DIR, ROOT, load_config

MAIN_SLUG = "potrebitelskie-beznalicnye-rashody-na-urovne-munizipalnyh-obrazovanij"
PROCESSED = DATA_DIR / "processed"
# Название, тип и ОКТМО территории берутся в редакции справочника, действующей в последний
# год панели: у части территорий они менялись.
REFERENCE_YEAR = 2024


def load_raw_spending() -> pd.DataFrame:
    """Публичная выгрузка расходов: те же наблюдения, но территория задана названием."""
    return pd.read_parquet(DATA_DIR / "raw" / MAIN_SLUG / f"{MAIN_SLUG}.parquet")


def official_territories(consumption: pd.DataFrame) -> pd.DataFrame:
    """Таблица территорий набора организаторов: по строке на официальный код."""
    ids = sorted(int(i) for i in consumption["territory_id"].unique())
    return build_territories(official.load_reference(), ids, REFERENCE_YEAR)


def heuristic_mapping(
    raw: pd.DataFrame, config_path: Path = CONFIG_DIR / "entity.yaml"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Идентификация по названиям: (соответствие блоков строк территориям, таблица территорий)."""
    cfg = load_config(EntityConfig, config_path)
    filter_names = (ROOT / cfg.reference.filter_order).read_text(encoding="utf-8").splitlines()
    reference = pd.read_csv(ROOT / cfg.reference.oktmo, dtype=str)
    return build_entity_mapping(
        raw,
        filter_names,
        reference,
        reorganizations=[dataclasses.asdict(r) for r in cfg.reorganizations],
        region_overrides={o.name: o.region_code for o in cfg.region_overrides},
    )


def heuristic_crosswalk(
    consumption: pd.DataFrame,
    territories: pd.DataFrame,
    config_path: Path = CONFIG_DIR / "entity.yaml",
) -> tuple[pd.DataFrame, dict]:
    """Ряды выгрузки в обеих идентификациях и сводка самой эвристики."""
    raw = load_raw_spending()
    mapping, heuristic = heuristic_mapping(raw, config_path)
    table = crosswalk_table(match_blocks(raw, consumption), mapping, heuristic, territories)
    return table, mapping_report(raw, mapping, heuristic)


def run(config_path: Path = CONFIG_DIR / "entity.yaml", out_dir: Path = PROCESSED) -> dict:
    consumption = official.load_consumption()
    territories = official_territories(consumption)
    table, mapping_summary = heuristic_crosswalk(consumption, territories, config_path)
    names = territories["mo_name"].value_counts()
    report = {
        "source": str(official.ARCHIVE.relative_to(ROOT)),
        "reference": str(official.REFERENCE_CSV.relative_to(ROOT)),
        "reference_year": REFERENCE_YEAR,
        "rows": int(len(consumption)),
        "territories": int(len(territories)),
        "series": int(len(table)),
        "regions": int(territories["region_code"].nunique()),
        "territories_by_kind": territories["mo_kind"].value_counts().to_dict(),
        "kind_changed_since_previous_year": kind_changes(
            official.load_reference(), territories["official_id"].tolist(), REFERENCE_YEAR
        ),
        "territories_with_coordinates": int(territories["center_lat"].notna().sum()),
        "same_names": int((names > 1).sum()),
        "same_name_territories": int(names[names > 1].sum()),
        "heuristic_check": {**heuristic_agreement(table), "mapping": mapping_summary},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    territories.to_parquet(out_dir / "territories.parquet", index=False)
    table.to_parquet(out_dir / "heuristic_crosswalk.parquet", index=False)
    (out_dir / "entity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return report
