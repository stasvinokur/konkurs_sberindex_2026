"""Панель расходов на реальных данных: строится из набора организаторов по официальным кодам."""

from pathlib import Path

import pandas as pd
import pytest

from sbx.shell.download import official
from sbx.shell.pipelines import entity, panel


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("processed")
    entity.run(out_dir=out)
    summary = panel.run(out_dir=out, figures=out / "figures")
    return {"out": out, "summary": summary}


def _read(built: dict, name: str) -> pd.DataFrame:
    return pd.read_parquet(Path(built["out"]) / name)


def test_panel_has_the_organisers_territories_and_series(built: dict) -> None:
    summary = built["summary"]
    assert (summary["rows"], summary["territories"], summary["series"]) == (303126, 2190, 13140)
    assert summary["series_with_admin_break"] == 0, "официальный код — МО в постоянных границах"
    frame = _read(built, "panel.parquet")
    consumption = official.load_consumption()
    assert frame["y"].sum() == pytest.approx(float(consumption["value"].sum()), rel=1e-12)
    assert frame["ds"].min() == pd.Timestamp("2023-01-01")
    assert frame["ds"].max() == pd.Timestamp("2024-12-01")


def test_every_series_is_in_the_region_of_its_official_territory(built: dict) -> None:
    static = _read(built, "static.parquet")
    territories = _read(built, "territories.parquet").set_index("official_id")
    assert static["official_id"].notna().all()
    assert (static["region_code"] == static["official_id"].map(territories["region_code"])).all()
    assert (static["mo_name"] == static["official_id"].map(territories["mo_name"])).all()
    assert set(static["mo_kind"]) == {
        "municipal_district",
        "urban_okrug",
        "municipal_okrug",
        "intracity",
    }
    assert static["region_name"].notna().all()


def test_territory_size_is_known_at_the_first_forecast_moment(built: dict) -> None:
    """Размер территории — признак LightGBM на все окна. Он оценён по месяцам не позже самого
    раннего момента прогноза; у территорий, чей ряд начался позже, размера нет."""
    from sbx.core.folds import make_folds
    from sbx.core.panel import static_features
    from sbx.shell.pipelines.backtest import load_backtest_config

    first = min(fold.cutoff for fold in make_folds(load_backtest_config()))
    assert panel.first_forecast_month() == first == pd.Timestamp("2023-10-01")
    static, frame = _read(built, "static.parquet"), _read(built, "panel.parquet")
    total = frame[frame["category"] == "Все категории"]
    known = set(total.loc[total["ds"] <= first, "territory_id"])
    sized = static["territory_id"].isin(known)
    assert static.loc[sized, ["size_level", "size_group"]].notna().all().all()
    assert static.loc[~sized, ["size_level", "size_group"]].isna().all().all()
    assert static.loc[~sized, "territory_id"].nunique() == 30, "ряд «Все категории» начался позже"
    summary = built["summary"]
    assert summary["size_level_until"] == "2023-10" and summary["territories_without_size"] == 30
    assert set(static["size_group"].dropna().astype(int)) == {1, 2, 3, 4, 5}

    territories = _read(built, "territories.parquet")
    past = static_features(frame[frame["ds"] <= first], territories, level_until=first)
    merged = past.merge(static, on="unique_id", suffixes=("_past", ""))
    assert len(merged) == len(past) and len(past) > 12_000
    assert (merged["size_level_past"] == merged["size_level"]).all(), "будущее на размер не влияет"
    assert (merged["size_group_past"] == merged["size_group"]).all()


def test_same_name_district_is_shown_in_its_own_region(built: dict) -> None:
    """Куйбышевский район Новосибирской области эвристика относила к Калужской."""
    static = _read(built, "static.parquet")
    rows = static[static["mo_name"] == "Куйбышевский муниципальный район"]
    by_region = rows.groupby("region_name")["unique_id"].count().to_dict()
    assert by_region == {"Калужская область": 6, "Новосибирская область": 6}


def test_static_with_names_needs_no_extra_table(built: dict, monkeypatch) -> None:
    monkeypatch.setattr(panel, "PROCESSED", Path(built["out"]))
    assert "mo_name" in panel.load_static().columns
    assert not hasattr(panel, "load_static_with_names"), "название МО уже есть в static.parquet"


def test_static_seasonal_index_is_estimated_on_data_published_before_the_panel(
    built: dict,
) -> None:
    """Статический индекс читают детекторы на всём периоде панели, поэтому в нём не должно
    быть ни одного месяца, опубликованного позже первого месяца панели."""
    from sbx.core.seasonality import national_seasonal_index

    seasonal = _read(built, "seasonal_index.parquet")
    last_known = pd.Timestamp("2022-11-01")  # ноябрь вышел 4 января; декабрь — 4 февраля 2023
    expected = national_seasonal_index(panel.load_national_spending(), end=last_known)
    columns = ["category", "month", "index"]
    pd.testing.assert_frame_equal(seasonal[columns], expected[columns])
    assert built["summary"]["seasonal_index_period"] == ["2019-01", "2022-11"]


def test_panel_step_stamps_its_result_with_the_inputs_it_read(built: dict) -> None:
    """Отметка панели называет таблицу территорий и архив организаторов: по ней отчёт видит,
    что территории пересобраны, а панель — нет."""
    from sbx.shell import stamps

    out = Path(built["out"])
    recorded = stamps.read(out / "panel_inputs.json")
    assert list(recorded) == panel.stamp_inputs(out) and len(recorded) == 5
    assert "file:configs/backtest.yaml" in recorded, (
        "окна проверки задают момент, по который оценивается размер территории"
    )
    assert stamps.stale(out / "panel_inputs.json") == []
    territories = out / "territories.parquet"
    original = territories.read_bytes()
    try:
        territories.write_bytes(original + b" ")
        assert stamps.stale(out / "panel_inputs.json") == [stamps.file_key(territories)]
    finally:
        territories.write_bytes(original)
