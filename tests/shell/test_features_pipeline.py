"""Шаг внешних признаков: блоки и их привязка ко времени на настоящих данных СберИндекса."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from sbx.core.features.asof import KNOWN_FUTURE, PAST_ONLY, STATIC, latest_published_month
from sbx.core.features.national import build_national_features
from sbx.core.panel import build_official_panel, static_features
from sbx.shell.download import official
from sbx.shell.io import CONFIG_DIR
from sbx.shell.pipelines import entity, features

ORIGINS = pd.date_range("2023-01-01", "2024-12-01", freq="MS")
JUNE = pd.Timestamp("2024-06-01")


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("features")
    months = pd.date_range(ORIGINS[0], ORIGINS[-1] + pd.DateOffset(months=6), freq="MS")
    features.build_calendar(months, out)
    features.build_national(ORIGINS, out)
    features.build_events(ORIGINS, ["53", "46"], out)
    return out


def test_tables_are_keyed_by_their_time_rule(built: Path) -> None:
    calendar = pd.read_parquet(built / "calendar.parquet")
    national = pd.read_parquet(built / "national.parquet")
    events = pd.read_parquet(built / "events.parquet")
    assert "ds" in calendar and "origin" not in calendar, "календарь известен заранее"
    assert "month" not in calendar, "месяц цели уже есть среди признаков модели"
    assert "origin" in national and "ds" not in national
    assert national["origin"].tolist() == list(ORIGINS)
    assert {"origin", "region_code"} <= set(events.columns) and "ds" not in events
    assert len(events) == len(ORIGINS) * 2


def test_national_table_holds_the_latest_published_month_of_every_source(built: Path) -> None:
    """Сверка с сырыми рядами: в строке июня 2024 — апрель у месячных рядов (лаг 35 дней) и
    май у недельных (лаг 7 дней)."""
    national = pd.read_parquet(built / "national.parquet").set_index("origin")
    lags = features.load_publication_lags()
    groups = build_national_features(features._load_national_frames(), lags)
    assert set(groups) == {7, 35, 45, 60}
    for lag, frame in groups.items():
        period = latest_published_month(JUNE, lag)
        expected = frame.set_index("ds").loc[period].drop("published_at")
        got = national.loc[JUNE, [f"nat{lag}__{c}" for c in expected.index]]
        got.index = expected.index
        pd.testing.assert_series_equal(got.astype(float), expected.astype(float), check_names=False)
    assert latest_published_month(JUNE, 35) == pd.Timestamp("2024-04-01")
    assert latest_published_month(JUNE, 7) == pd.Timestamp("2024-05-01")


def test_event_table_of_an_origin_knows_the_decisions_made_by_its_end(built: Path) -> None:
    """Ключевая ставка повышена 26 июля 2024: в строке июня этого ещё нет, в строке июля есть."""
    events = pd.read_parquet(built / "events.parquet").set_index(["region_code", "origin"])
    july = pd.Timestamp("2024-07-01")
    assert events.loc[("46", JUNE), "event_key_rate_hike"] == 0.0
    assert events.loc[("46", july), "event_key_rate_hike"] == 1.0


def test_blocks_carry_their_kind_and_news_are_those_of_the_origin_month(
    built: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    news = pd.DataFrame(
        {
            "region_code": ["46", "46"],
            "ds": [JUNE, pd.Timestamp("2024-07-01")],
            "news_events": [120, 999],
        }
    )
    news.to_parquet(news_dir / "region_month_features.parquet", index=False)
    monkeypatch.setattr(features, "FEATURES_DIR", built)
    monkeypatch.setattr(features, "NEWS_DIR", news_dir)

    blocks = features.load_feature_blocks()

    assert {name: block.kind for name, block in blocks.items()} == {
        "calendar": KNOWN_FUTURE,
        "macro": PAST_ONLY,
        "news": PAST_ONLY,
        "gdelt": PAST_ONLY,
        "events": PAST_ONLY,
    }
    # Те же новости и события — отдельными блоками, чтобы измерить вклад каждого источника.
    assert (
        "news_events" in blocks["gdelt"].columns
        and "event_key_rate_hike" in blocks["events"].columns
    )
    assert not set(blocks["gdelt"].columns) & set(blocks["events"].columns)
    assert set(blocks["gdelt"].columns) | set(blocks["events"].columns) == set(
        blocks["news"].columns
    )
    row = blocks["news"].frame.set_index(["region_code", "origin"]).loc[("46", JUNE)]
    assert row["news_events"] == 120, "к концу июня известны новости июня, но не июля"
    assert row["event_key_rate_hike"] == 0.0

    monkeypatch.setattr(features, "NEWS_DIR", tmp_path / "no-archive")
    without_archive = features.load_feature_blocks()
    assert "news_events" not in without_archive["news"].frame.columns
    assert "event_key_rate_hike" in without_archive["news"].frame.columns
    assert "gdelt" not in without_archive, "нет архива — нет и блока новостей GDELT"


def test_every_block_named_in_the_ablation_config_exists(
    built: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(features, "FEATURES_DIR", built)
    monkeypatch.setattr(features, "NEWS_DIR", tmp_path / "no-archive")
    config = yaml.safe_load((CONFIG_DIR / "ablations.yaml").read_text(encoding="utf-8"))
    blocks = features.load_feature_blocks()
    for name, variant in config["feature_variants"].items():
        missing = set(variant["blocks"]) - set(blocks)
        assert not missing, f"вариант {name}: нет блоков {sorted(missing)}"
    # Блоки источников: новости GDELT без архива и таблицы организаторов без шага признаков
    # отсутствуют, остальные обязаны существовать.
    optional = {"gdelt", "access", "neighbours", "weather", "population"}
    assert set(config["sources"]["blocks"]) - optional <= set(blocks)
    weather = config["sources"]["blocks"]["weather"]
    assert weather["blocks"] == ["weather", "climate"], "источник из двух блоков"
    assert config["feature_variants"]["B"]["blocks"] == ["calendar", "macro"]
    assert config["feature_variants"]["C"]["blocks"] == ["calendar", "macro", "news"]


def test_feature_step_stamps_its_result_with_the_inputs_it_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sbx.shell import stamps

    panel = pd.DataFrame({"unique_id": "a", "ds": ORIGINS, "y": 1.0})
    static = pd.DataFrame({"unique_id": ["a"], "region_code": ["46"]})
    monkeypatch.setattr(features, "load_panel", lambda: panel)
    monkeypatch.setattr(features, "load_static", lambda: static)
    monkeypatch.setattr(features, "news_step", lambda months: "архива нет — блок пропущен")
    territories = pd.DataFrame({"official_id": [1], "territory_id": ["46-0001"]})
    seen = {}

    def spatial(panel_, static_, territories_, out_dir):
        seen.update(panel=panel_, territories=territories_, out_dir=out_dir)
        return {"with_neighbours": 1}

    monkeypatch.setattr(features, "load_territories", lambda: territories)
    monkeypatch.setattr(features, "build_spatial", spatial)
    monkeypatch.setattr(
        features, "build_weather", lambda origins, months, territories_, out_dir: {"cells": 1}
    )
    monkeypatch.setattr(features, "build_population", lambda territories_, out_dir: {"matched": 1})
    summary = features.run(out_dir=tmp_path)
    assert summary["origins"] == 24 and summary["national_columns"] == 61
    assert summary["spatial"] == {"with_neighbours": 1} and summary["weather"] == {"cells": 1}
    assert summary["population"] == {"matched": 1}
    assert seen["panel"] is panel and seen["territories"] is territories
    assert seen["out_dir"] == tmp_path
    recorded = stamps.read(tmp_path / "inputs.json")
    assert list(recorded) == features.stamp_inputs()
    assert {stamps.PANEL, stamps.STATIC, "file:configs/events.yaml"} <= set(recorded)
    # Таблицы организаторов и таблица территорий — тоже входы шага.
    assert "file:data/raw/sberindex-municipal/hackathonlicence.zip" in recorded
    assert "file:data/processed/territories.parquet" in recorded
    assert "file:data/reference/weather/power_monthly.parquet" in recorded
    assert "file:configs/weather.yaml" in recorded
    assert "file:data/reference/rosstat/population_2023.csv" in recorded
    assert len(recorded) == 12 + len(features.NATIONAL_SLUGS)


@pytest.fixture(scope="module")
def spatial(tmp_path_factory) -> tuple[Path, dict, pd.DataFrame]:
    """Таблицы доступности рынков и соседей на настоящем наборе организаторов."""
    consumption = official.load_consumption()
    territories = entity.official_territories(consumption)
    panel = build_official_panel(consumption, territories)
    static = static_features(panel, territories)
    out = tmp_path_factory.mktemp("spatial")
    return out, features.build_spatial(panel, static, territories, out), panel


def test_spatial_tables_cover_the_panel(spatial: tuple[Path, dict, pd.DataFrame]) -> None:
    out, summary, panel = spatial
    access = pd.read_parquet(out / "access.parquet")
    assert len(access) == 2190 and access["territory_id"].is_unique
    assert list(access.columns) == ["territory_id", "market_access_log", "neighbour_distance_log"]
    neighbours = pd.read_parquet(out / "neighbours.parquet")
    assert len(neighbours) == len(panel), "строка на каждое наблюдение панели"
    assert set(neighbours.columns) == {
        "unique_id",
        "origin",
        "nb_growth_1",
        "nb_growth_3",
        "nb_level_gap",
    }
    pairs = pd.read_parquet(out / "neighbour_pairs.parquet")
    assert pairs.groupby("territory_id").size().max() == 5
    assert summary == {
        "territories": 2190,
        "with_market_access": 2170,
        "with_neighbours": 2172,
        "neighbours": 5,
        "highway_pairs": 2353366,
        "median_distance_km": pytest.approx(57.5, abs=0.5),
        "cross_region_share": pytest.approx(0.147, abs=0.005),
    }


def test_spatial_blocks_carry_their_kind_and_keys(
    spatial: tuple[Path, dict, pd.DataFrame], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(features, "FEATURES_DIR", spatial[0])
    monkeypatch.setattr(features, "NEWS_DIR", tmp_path / "no-archive")
    blocks = features.load_feature_blocks()
    assert blocks["access"].kind == STATIC and blocks["access"].keys == ["territory_id"]
    assert blocks["access"].columns == ["market_access_log", "neighbour_distance_log"]
    assert blocks["neighbours"].kind == PAST_ONLY
    assert blocks["neighbours"].keys == ["origin", "unique_id"]
    assert blocks["neighbours"].columns == ["nb_growth_1", "nb_growth_3", "nb_level_gap"]


WEATHER_TABLE = pd.DataFrame(
    {
        "lat": 56.0,
        "lon": 37.5,
        "ds": pd.date_range("2023-11-01", periods=5, freq="MS"),
        "t2m": [-1.0, -5.0, -12.0, -3.0, 4.0],
        "prectot": [2.0, 2.0, 3.0, 1.0, 2.0],
        "t2m_norm": [-2.0, -6.0, -8.0, -7.0, -1.0],
        "prectot_norm": 2.0,
    }
)
MOSCOW_TERRITORIES = pd.DataFrame(
    {
        "territory_id": ["45-0001", "45-0002"],
        "region_code": "45",
        "center_lat": [float("nan"), 55.9],
        "center_lon": [float("nan"), 37.4],
    }
)


def test_weather_step_writes_what_is_known_at_each_origin_and_the_climate_norms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(features.power, "load_table", lambda: WEATHER_TABLE)
    origins = pd.DatetimeIndex(["2024-02-01", "2024-03-01"])
    months = pd.date_range("2024-02-01", periods=3, freq="MS")
    summary = features.build_weather(origins, months, MOSCOW_TERRITORIES, tmp_path)
    assert summary == {
        "cells": 1,
        "territories": 2,
        "lag_days": 7,
        "first": "2023-11",
        "last": "2024-03",
        "norm_years": [2001, 2020],
    }
    weather = pd.read_parquet(tmp_path / "weather.parquet").set_index(["territory_id", "origin"])
    assert weather.loc[("45-0001", pd.Timestamp("2024-02-01")), "wx_t_anom"] == -4.0, (
        "к концу февраля известен январь; центр территории без координат — центр города"
    )
    climate = pd.read_parquet(tmp_path / "climate.parquet")
    assert len(climate) == 6 and list(climate.columns) == [
        "territory_id",
        "ds",
        "wx_t_norm",
        "wx_p_norm",
    ]

    monkeypatch.setattr(features, "FEATURES_DIR", tmp_path)
    monkeypatch.setattr(features, "NEWS_DIR", tmp_path / "no-archive")
    blocks = features.load_feature_blocks()
    assert blocks["weather"].kind == PAST_ONLY
    assert blocks["weather"].keys == ["origin", "territory_id"]
    assert blocks["climate"].kind == KNOWN_FUTURE and blocks["climate"].keys == [
        "ds",
        "territory_id",
    ]


def test_weather_step_is_skipped_without_the_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(features.power, "load_table", lambda: None)
    result = features.build_weather(ORIGINS, ORIGINS, MOSCOW_TERRITORIES, tmp_path)
    assert isinstance(result, str) and "sbx data weather" in result
    assert not (tmp_path / "weather.parquet").exists()


def test_weather_config_names_the_source_the_norm_and_the_lag() -> None:
    cfg = features.power.load_config()
    assert cfg["parameters"] == ["T2M", "PRECTOTCORR"] and cfg["norm_years"] == [2001, 2020]
    assert int(cfg["end_year"]) == 2024 and str(cfg["keep_from"]) == "2022-01-01"
    assert set(cfg["city_centers"]) == {"40", "45"}, "Санкт-Петербург и Москва"
    assert features.load_publication_lags()[features.WEATHER_SOURCE] == 7


POPULATION = pd.DataFrame(
    {
        "official_id": [1, 2],
        "oktmo": ["79701000", "79605000"],
        "name": ["Городской округ - Город Майкоп", "Гиагинский муниципальный район"],
        "total": [163766, 31933],
        "urban": [139687, 0],
        "rural": [24079, 31933],
    }
)


def _panel_territories(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "official_id": list(range(1, count + 1)),
            "territory_id": [f"79-{i:04d}" for i in range(1, count + 1)],
        }
    )


def test_population_block_is_built_when_enough_territories_are_matched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(features.rosstat, "load_table", lambda: POPULATION)
    summary = features.build_population(_panel_territories(2), tmp_path)
    assert summary == {
        "territories": 2,
        "matched": 2,
        "share": 1.0,
        "threshold": 0.9,
        "used": True,
        "year": 2023,
        "published": "2023-09-04",
    }
    monkeypatch.setattr(features, "FEATURES_DIR", tmp_path)
    monkeypatch.setattr(features, "NEWS_DIR", tmp_path / "no-archive")
    block = features.load_feature_blocks()["population"]
    assert block.kind == STATIC and block.keys == ["territory_id"]
    assert block.columns == ["pop_log", "pop_urban_share"]
    assert block.frame.set_index("territory_id").loc["79-0002", "pop_urban_share"] == 0.0


def test_population_stays_a_reference_figure_when_too_few_territories_are_matched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Правило задачи: привязано меньше 90% территорий — в прогноз население не идёт."""
    monkeypatch.setattr(features.rosstat, "load_table", lambda: POPULATION)
    (tmp_path / "population.parquet").write_bytes(b"left over from an earlier run")
    summary = features.build_population(_panel_territories(3), tmp_path)
    assert summary["matched"] == 2 and summary["share"] == pytest.approx(2 / 3)
    assert summary["used"] is False
    assert not (tmp_path / "population.parquet").exists(), "прежний блок не должен остаться"
    monkeypatch.setattr(features, "FEATURES_DIR", tmp_path)
    monkeypatch.setattr(features, "NEWS_DIR", tmp_path / "no-archive")
    assert "population" not in features.load_feature_blocks()


def test_population_is_used_at_exactly_the_threshold_and_counts_only_panel_territories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """«Меньше 90%» — не идёт в прогноз; ровно 90% — идёт. Строки таблицы о территориях вне
    панели в долю не входят."""
    table = pd.DataFrame(
        {
            "official_id": [*range(1, 10), 99],
            "oktmo": [f"7970{i}000" for i in range(10)],
            "name": [f"район {i}" for i in range(10)],
            "total": [1000] * 10,
            "urban": [400] * 10,
            "rural": [600] * 10,
        }
    )
    monkeypatch.setattr(features.rosstat, "load_table", lambda: table)
    summary = features.build_population(_panel_territories(10), tmp_path)
    assert summary["matched"] == 9, "территория 99 не из панели"
    assert summary["share"] == 0.9 and summary["used"] is True
    assert (tmp_path / "population.parquet").exists()


def test_population_step_is_skipped_without_the_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(features.rosstat, "load_table", lambda: None)
    result = features.build_population(_panel_territories(2), tmp_path)
    assert isinstance(result, str) and "sbx data population" in result
    assert features.population_threshold() == 0.9, "порог записан в configs/features.yaml"
