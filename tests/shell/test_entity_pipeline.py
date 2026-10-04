"""Идентификация МО на реальных данных: официальные коды и сверка прежней эвристики с ними."""

import pandas as pd
import pytest

from sbx.core.regions import region_by_code
from sbx.shell.io import DATA_DIR
from sbx.shell.pipelines import entity


@pytest.fixture(scope="module")
def report(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("processed")
    rep = entity.run(out_dir=out)
    rep["out"] = out
    return rep


@pytest.fixture(scope="module")
def heuristic() -> tuple[pd.DataFrame, pd.DataFrame]:
    return entity.heuristic_mapping(entity.load_raw_spending())


def test_territories_are_the_organisers_territories(report: dict) -> None:
    t = pd.read_parquet(report["out"] / "territories.parquet")
    assert len(t) == 2190 and t["official_id"].is_unique and t["territory_id"].is_unique
    assert (report["rows"], report["territories"], report["series"]) == (303126, 2190, 13140)
    assert report["reference_year"] == 2024 and report["kind_changed_since_previous_year"] == 84
    assert t["region_code"].notna().all() and t["mo_name"].str.len().gt(0).all()
    expected = t["region_code"] + "-" + t["official_id"].map("{:04d}".format)
    assert (t["territory_id"] == expected).all()


def test_same_name_municipalities_stay_apart(report: dict) -> None:
    """49 названий встречаются в нескольких регионах — всего 121 территория, у каждой свой код."""
    t = pd.read_parquet(report["out"] / "territories.parquet")
    counts = t["mo_name"].value_counts()
    assert (counts > 1).sum() == 49 and counts[counts > 1].sum() == 121
    kuybyshev = t[t["mo_name"] == "Куйбышевский муниципальный район"]
    assert sorted(kuybyshev["region_code"]) == ["29", "50"], "Калужская и Новосибирская области"
    assert report["same_name_territories"] == 121


def test_every_series_of_the_export_is_one_official_series(report: dict) -> None:
    crosswalk = pd.read_parquet(report["out"] / "heuristic_crosswalk.parquet")
    assert len(crosswalk) == 13140
    assert crosswalk["block_id"].is_unique and crosswalk["unique_id"].is_unique
    assert crosswalk["legacy_unique_id"].nunique() == 13128, "две склейки — по шесть категорий"


def test_heuristic_check_measures_how_often_names_misled(report: dict) -> None:
    check = report["heuristic_check"]
    assert check["series"] == 13140 and check["series_in_wrong_region"] == 327
    assert check["territories"] == 2188
    assert check["territories_with_foreign_series"] == 101
    assert check["territories_entirely_in_wrong_region"] == 14
    assert check["territories_mixing_same_name_series"] == 91
    assert check["territories_glued_from_several_names"] == 2
    assert check["mapping"]["rows_mapped"] == check["mapping"]["raw_rows"] == 303126
    assert check["mapping"]["duplicate_keys_after"] == 0


def test_region_codes_stay_the_ones_news_and_events_are_joined_by(report: dict) -> None:
    """Новости и календарь событий стыкуются по коду региона. Переход на официальные коды
    территорий не должен менять набор кодов регионов и оставлять регион без соответствия."""
    crosswalk = pd.read_parquet(report["out"] / "heuristic_crosswalk.parquet")
    official_regions = set(crosswalk["official_region"])
    assert official_regions == set(crosswalk["heuristic_region"]), "набор кодов регионов прежний"
    assert len(official_regions) == report["regions"] == 77
    for code in official_regions:
        assert region_by_code(code).name
    fips = pd.read_csv(DATA_DIR / "reference" / "fips_adm1_ru.csv", dtype=str)
    assert official_regions <= set(fips["region_code"]), "у региона нет соответствия в GDELT"


def test_heuristic_resolves_ambiguous_names_to_distinct_regions(
    heuristic: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    _, t = heuristic
    amb = t[t["resolution_source"].str.startswith("ambiguous")]
    assert amb["raw_mo"].nunique() == 49
    assert len(amb) == 121
    assert (amb.groupby("raw_mo")["region_code"].nunique() == amb.groupby("raw_mo").size()).all()
    nog = amb[amb["raw_mo"] == "Ногайский муниципальный район"]
    assert set(nog["region_code"]) == {"82", "91"}


def test_heuristic_gives_every_territory_a_region_and_flags_breaks(
    heuristic: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    _, t = heuristic
    assert t["region_code"].notna().all()
    breaks = t.dropna(subset=["admin_break_at"])
    assert set(breaks["raw_mo"]) >= {
        "городской округ Павлово-Посадский",
        "городской округ Павловский Посад",
    }
