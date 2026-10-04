"""Сборка внешних признаков: календарь, национальные ряды, новости, события, таблицы МО.

У каждой таблицы — своё правило привязки ко времени (`sbx.core.features.asof`):

- календарь известен заранее и хранится по месяцу `ds`, к прогнозу он присоединяется по
  целевому месяцу;
- национальные ряды и календарь событий хранятся по `origin`: строка содержит то, что
  опубликовано к концу этого месяца;
- новости GDELT хранятся по месяцу события (так их читает модель вероятности шока); блок для
  прогнозных моделей получается из них с лагом публикации 0;
- расходы соседних МО хранятся по `origin` и ряду: строка считается по значениям не позже
  этого месяца — так же, как признаки собственного ряда;
- доступность рынков и расстояние до соседей — постоянные характеристики территории.

Проверка — в `tests/test_leakage.py` и `tests/shell/test_features_pipeline.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from sbx.core.features.asof import (
    KNOWN_FUTURE,
    PAST_ONLY,
    STATIC,
    ExogenousBlock,
    as_of_origin,
)
from sbx.core.features.calendar import calendar_features, parse_calendar
from sbx.core.features.national import build_national_features, national_as_of
from sbx.core.features.spatial import (
    NEIGHBOURS,
    access_features,
    nearest_neighbours,
    neighbour_features,
)
from sbx.core.news import (
    NEWS_LAG_DAYS,
    add_novelty,
    event_features_as_of,
    map_regions,
    monthly_news_features,
    unmapped_share,
)
from sbx.core.rosstat import population_features
from sbx.core.weather import climate_norms, territory_cells, weather_as_of
from sbx.shell import stamps
from sbx.shell.download import official, power, rosstat
from sbx.shell.io import ARTIFACTS_DIR, CONFIG_DIR, DATA_DIR, ROOT, publication_lags, read_yaml
from sbx.shell.pipelines.entity import PROCESSED
from sbx.shell.pipelines.panel import load_panel, load_static

FEATURES_DIR = ARTIFACTS_DIR / "features"
NEWS_DIR = ARTIFACTS_DIR / "news"
# Имя источника погоды в таблице лагов публикации (`configs/features.yaml`).
WEATHER_SOURCE = "nasa-power-weather"
NATIONAL_SLUGS = (
    "consumer-spending",
    "consumer-spending-growth",
    "consumper-spending-index-sa",
    "median-wages",
    "oboroty-biznesa",
    "izmenenie-obema-fot",
    "real-key-interest-rate",
    "nedelnaa-inflazia-v-razreze-analiticeskih-komponentov",
    "ver-izmenenie-trat-po-kategoriyam",
    "potrebitelskaya-aktivnost-po-kategoriyam-tovarov-v-razreze-vozrastov",
)


def load_publication_lags(path: Path | None = None) -> dict[str, int]:
    """Лаги публикации по источникам с обоснованием — из конфига, а не из констант в коде."""
    return publication_lags(path)


def _load_national_frames() -> dict[str, pd.DataFrame]:
    frames = {}
    for slug in NATIONAL_SLUGS:
        path = DATA_DIR / "raw" / slug / f"{slug}.parquet"
        if path.exists():
            frames[slug] = pd.read_parquet(path)
    return frames


def build_calendar(months: pd.DatetimeIndex, out_dir: Path | None = None) -> pd.DataFrame:
    """Производственный календарь по месяцам — известен заранее на весь горизонт."""
    out_dir = FEATURES_DIR if out_dir is None else out_dir
    calendar_raw = json.loads(
        (ROOT / "data" / "reference" / "ru_calendar.json").read_text(encoding="utf-8")
    )
    # Номер целевого месяца у модели уже есть, повторять его колонкой календаря незачем.
    calendar = calendar_features(months, parse_calendar(calendar_raw)).drop(columns=["month"])
    out_dir.mkdir(parents=True, exist_ok=True)
    calendar.to_parquet(out_dir / "calendar.parquet", index=False)
    return calendar


def build_national(origins: pd.DatetimeIndex, out_dir: Path | None = None) -> pd.DataFrame:
    """Национальные ряды, известные к концу каждого месяца origin.

    У каждого источника свой лаг публикации, поэтому признаки идут группами: к концу месяца
    недельные индикаторы известны за предыдущий месяц, месячные ряды СберИндекса — за
    позапрошлый, зарплаты отстают ещё сильнее.
    """
    out_dir = FEATURES_DIR if out_dir is None else out_dir
    groups = build_national_features(_load_national_frames(), load_publication_lags())
    national = national_as_of(groups, origins)
    out_dir.mkdir(parents=True, exist_ok=True)
    national.to_parquet(out_dir / "national.parquet", index=False)
    return national


def build_news(months: pd.DatetimeIndex, out_dir: Path = NEWS_DIR) -> dict[str, Any]:
    """Новостные признаки GDELT на уровне «регион × месяц» и страны."""
    from sbx.shell.download import gdelt

    events = gdelt.load_archive(gdelt.load_manifest())
    reference = pd.read_csv(ROOT / "data" / "reference" / "fips_adm1_ru.csv", dtype=str)
    events = map_regions(events, reference)
    coverage = unmapped_share(events)

    regional = monthly_news_features(events, level="region")
    regional = add_novelty(regional, window=6)
    country = monthly_news_features(events, level="country")
    country = add_novelty(country, window=6, group_col="__none__")
    features = regional.merge(country, on="ds", how="left")
    features = features[features["ds"].isin(months)]

    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_dir / "region_month_features.parquet", index=False)
    coverage.to_csv(out_dir / "unmapped_adm1.csv", index=False)
    summary = {
        "events": int(len(events)),
        "months": int(features["ds"].nunique()),
        "regions": int(features["region_code"].nunique()),
        "unmapped_share": float(coverage["share"].sum()) if len(coverage) else 0.0,
        "columns": int(features.shape[1]),
    }
    (out_dir / "news_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def news_step(months: pd.DatetimeIndex, out_dir: Path = NEWS_DIR) -> dict[str, Any] | str:
    """Новостной блок: без архива пропускается, а неполный архив — ошибка `ArchiveMismatch`.

    Признаки по части архива выглядели бы как настоящие, но были бы другими, поэтому
    недокачанный архив не считается «почти готовым».
    """
    from sbx.shell.download.gdelt import ArchiveAbsent

    try:
        return build_news(months, out_dir)
    except ArchiveAbsent as exc:
        return f"{exc} — новостной блок пропущен (скачать архив: make gdelt)"


def build_events(
    origins: pd.DatetimeIndex, regions: list[str], out_dir: Path | None = None
) -> pd.DataFrame:
    """Курируемый календарь событий: что было известно к концу каждого месяца origin."""
    out_dir = FEATURES_DIR if out_dir is None else out_dir
    cfg = yaml.safe_load((CONFIG_DIR / "events.yaml").read_text(encoding="utf-8"))
    features = event_features_as_of(cfg["events"], origins, regions)
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_dir / "events.parquet", index=False)
    return features


def load_territories() -> pd.DataFrame:
    """Таблица территорий панели: официальный код организаторов и наш `territory_id`."""
    return pd.read_parquet(PROCESSED / "territories.parquet")


def build_spatial(
    panel: pd.DataFrame, static: pd.DataFrame, territories: pd.DataFrame, out_dir: Path
) -> dict[str, Any]:
    """Признаки из таблиц организаторов: доступность рынков и соседи по автодорогам.

    Прежние внешние данные — страновые и региональные, эти различаются между МО. Пишутся три
    таблицы: пары «территория — сосед», постоянные характеристики территории и динамика
    расходов соседей на каждый месяц.
    """
    connection = official.load_connection()
    pairs = nearest_neighbours(connection, territories)
    access = access_features(official.load_market_access(), pairs, territories)
    neighbours = neighbour_features(panel, static, pairs)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out_dir / "neighbour_pairs.parquet", index=False)
    access.to_parquet(out_dir / "access.parquet", index=False)
    neighbours.to_parquet(out_dir / "neighbours.parquet", index=False)
    inside = set(territories["official_id"])
    roads = connection[
        (connection["type"] == "highway")
        & connection["territory_id_x"].isin(inside)
        & connection["territory_id_y"].isin(inside)
    ]
    region = territories.set_index("territory_id")["region_code"]
    crossing = pairs["territory_id"].map(region) != pairs["neighbour"].map(region)
    return {
        "territories": int(len(territories)),
        "with_market_access": int(access["market_access_log"].notna().sum()),
        "with_neighbours": int(pairs["territory_id"].nunique()),
        "neighbours": NEIGHBOURS,
        "highway_pairs": int(len(roads)),
        "median_distance_km": float(pairs["distance"].median()),
        "cross_region_share": float(crossing.mean()),
    }


def build_weather(
    origins: pd.DatetimeIndex, months: pd.DatetimeIndex, territories: pd.DataFrame, out_dir: Path
) -> dict[str, Any] | str:
    """Погода NASA POWER по территориям: что известно к концу месяца origin и нормы.

    Таблица «ячейка × месяц» лежит в репозитории; без неё блок пропускается. Территория
    получает погоду ячейки сетки, в которую попадает её центр.
    """
    table = power.load_table()
    if table is None:
        return "таблицы погоды нет — блок пропущен (скачать: sbx data weather)"
    cfg = power.load_config()
    cells = territory_cells(territories, cfg["city_centers"])
    lag = load_publication_lags()[WEATHER_SOURCE]
    out_dir.mkdir(parents=True, exist_ok=True)
    weather_as_of(table, cells, list(origins), lag).to_parquet(
        out_dir / "weather.parquet", index=False
    )
    climate_norms(table, cells, list(months)).to_parquet(out_dir / "climate.parquet", index=False)
    return {
        "cells": int(len(cells[["lat", "lon"]].drop_duplicates())),
        "territories": int(len(cells)),
        "lag_days": int(lag),
        "first": f"{table['ds'].min():%Y-%m}",
        "last": f"{table['ds'].max():%Y-%m}",
        "norm_years": [int(year) for year in cfg["norm_years"]],
    }


def population_threshold() -> float:
    """Доля привязанных территорий, ниже которой население в прогноз не идёт."""
    return float(read_yaml(CONFIG_DIR / "features.yaml")["population"]["min_match_share"])


def build_population(territories: pd.DataFrame, out_dir: Path) -> dict[str, Any] | str:
    """Численность населения МО (Росстат) как постоянная характеристика территории.

    Таблица привязки лежит в репозитории; без неё блок пропускается. Блок строится, только
    если привязано не меньше пороговой доли территорий панели: иначе население остаётся
    справочным показателем, и прежний файл блока удаляется, чтобы модель его не подхватила.
    """
    table = rosstat.load_table()
    if table is None:
        return "таблицы населения нет — блок пропущен (скачать: sbx data population)"
    matched = int(table["official_id"].isin(set(territories["official_id"])).sum())
    share, threshold = matched / len(territories), population_threshold()
    summary = {
        "territories": int(len(territories)),
        "matched": matched,
        "share": float(share),
        "threshold": threshold,
        "used": bool(share >= threshold),
        "year": rosstat.VALUES_YEAR,
        "published": rosstat.VALUES_PUBLISHED,
    }
    path = out_dir / "population.parquet"
    if summary["used"]:
        out_dir.mkdir(parents=True, exist_ok=True)
        population_features(table, territories).to_parquet(path, index=False)
    else:
        path.unlink(missing_ok=True)
    return summary


def load_feature_blocks() -> dict[str, ExogenousBlock]:
    """Блоки внешних признаков для моделей и абляций — каждый со своим правилом времени.

    Новости и календарь событий есть и одним блоком `news` (конфигурация C), и по
    отдельности — `gdelt` и `events`: так измеряется вклад каждого источника.
    """
    blocks: dict[str, ExogenousBlock] = {}
    calendar = FEATURES_DIR / "calendar.parquet"
    if calendar.exists():
        blocks["calendar"] = ExogenousBlock("calendar", pd.read_parquet(calendar), KNOWN_FUTURE)
    national = FEATURES_DIR / "national.parquet"
    if national.exists():
        blocks["macro"] = ExogenousBlock("macro", pd.read_parquet(national), PAST_ONLY)

    events_path = FEATURES_DIR / "events.parquet"
    news_path = NEWS_DIR / "region_month_features.parquet"
    frames = []
    if news_path.exists():
        news = pd.read_parquet(news_path)
        # GDELT публикует событие в день события: к концу месяца его новости известны целиком.
        origins = sorted(news["ds"].unique())
        frames.append(as_of_origin(news, origins, NEWS_LAG_DAYS))
        blocks["gdelt"] = ExogenousBlock("gdelt", frames[-1], PAST_ONLY)
    if events_path.exists():
        frames.append(pd.read_parquet(events_path))
        blocks["events"] = ExogenousBlock("events", frames[-1], PAST_ONLY)
    if frames:
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on=["region_code", "origin"], how="outer")
        blocks["news"] = ExogenousBlock("news", merged, PAST_ONLY)

    access = FEATURES_DIR / "access.parquet"
    if access.exists():
        blocks["access"] = ExogenousBlock("access", pd.read_parquet(access), STATIC)
    neighbours = FEATURES_DIR / "neighbours.parquet"
    if neighbours.exists():
        blocks["neighbours"] = ExogenousBlock("neighbours", pd.read_parquet(neighbours), PAST_ONLY)
    weather = FEATURES_DIR / "weather.parquet"
    if weather.exists():
        blocks["weather"] = ExogenousBlock("weather", pd.read_parquet(weather), PAST_ONLY)
    climate = FEATURES_DIR / "climate.parquet"
    if climate.exists():
        blocks["climate"] = ExogenousBlock("climate", pd.read_parquet(climate), KNOWN_FUTURE)
    population = FEATURES_DIR / "population.parquet"
    if population.exists():
        blocks["population"] = ExogenousBlock("population", pd.read_parquet(population), STATIC)
    return blocks


def run(out_dir: Path | None = None) -> dict[str, Any]:
    out_dir = FEATURES_DIR if out_dir is None else out_dir
    panel, static = load_panel(), load_static()
    origins = pd.DatetimeIndex(sorted(panel["ds"].unique()))
    horizon_months = pd.date_range(
        origins.min(), origins.max() + pd.DateOffset(months=6), freq="MS"
    )
    regions = sorted(static["region_code"].dropna().unique())
    # Новости — первыми: неполный архив останавливает шаг до того, как что-либо записано.
    news = news_step(horizon_months)
    calendar = build_calendar(horizon_months, out_dir)
    national = build_national(origins, out_dir)
    events = build_events(origins, regions, out_dir)
    territories = load_territories()
    spatial = build_spatial(panel, static, territories, out_dir)
    weather = build_weather(origins, horizon_months, territories, out_dir)
    population = build_population(territories, out_dir)
    summary: dict[str, Any] = {
        "calendar_columns": int(calendar.shape[1] - 1),
        "national_columns": int(national.shape[1] - 1),
        "event_columns": int(events.shape[1] - 2),
        "origins": int(len(origins)),
        "months": int(len(horizon_months)),
        "news": news,
        "spatial": spatial,
        "weather": weather,
        "population": population,
    }
    (out_dir / "features_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    stamps.write(out_dir / "inputs.json", stamp_inputs())
    return summary


def stamp_inputs() -> list[str]:
    """Входы шага для отметки рядом с результатом (см. `sbx.shell.stamps`).

    Панель и статические признаки задают месяцы и регионы; остальное — сырые ряды, календарь,
    календарь событий, лаги публикации, манифест архива новостей, архив организаторов (таблицы
    доступности рынков и связей МО) и таблица территорий.
    """
    return [
        stamps.PANEL,
        stamps.STATIC,
        stamps.file_key(PROCESSED / "territories.parquet"),
        stamps.file_key(official.ARCHIVE),
        stamps.file_key(power.TABLE_PATH),
        stamps.file_key(power.CONFIG_PATH),
        stamps.file_key(rosstat.TABLE_PATH),
        stamps.file_key(CONFIG_DIR / "features.yaml"),
        stamps.file_key(CONFIG_DIR / "events.yaml"),
        stamps.file_key(ROOT / "data" / "reference" / "ru_calendar.json"),
        stamps.file_key(ROOT / "data" / "reference" / "fips_adm1_ru.csv"),
        stamps.file_key(DATA_DIR / "gdelt_manifest.yaml"),
        *(stamps.file_key(DATA_DIR / "raw" / slug / f"{slug}.parquet") for slug in NATIONAL_SLUGS),
    ]
