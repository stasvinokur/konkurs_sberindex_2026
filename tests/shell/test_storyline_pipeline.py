"""Сбор фактов для блоков лендинга: манифесты, конфиги и сырые ряды из репозитория."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sbx.shell.pipelines import storyline
from sbx.shell.pipelines.features import NATIONAL_SLUGS, load_publication_lags


def test_facts_about_tracked_sources_are_read_from_the_files() -> None:
    facts = storyline.tracked_facts()
    national = facts["national"]
    assert len(national) == len(NATIONAL_SLUGS) == 10
    lags = load_publication_lags()
    by_slug = {item["slug"]: item for item in national}
    assert by_slug["consumer-spending"]["title"] == "Потребительские расходы"
    assert by_slug["consumer-spending"]["frequency"] == "месяц"
    assert by_slug["consumer-spending"]["lag_days"] == lags["consumer-spending"] == 35
    weekly = by_slug["nedelnaa-inflazia-v-razreze-analiticeskih-komponentov"]
    assert weekly["frequency"] == "неделя" and weekly["lag_days"] == 7
    assert all(item["first"] < item["last"] for item in national)
    assert facts["calendar"] == {"first_year": 2018, "last_year": 2026}
    assert facts["events"] == {"count": 23, "first": "2020-03", "last": "2024-10"}


def test_facts_about_computed_sources_are_absent_until_the_steps_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Нет панели и архива новостей — нет и фактов о них: блок на странице пропускается."""
    monkeypatch.setattr(storyline, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(storyline, "NEWS_DIR", tmp_path / "news")
    monkeypatch.setattr(storyline, "FEATURES_DIR", tmp_path / "features")
    facts = storyline.collect_facts()
    assert "spending" not in facts and "news" not in facts and "reference" not in facts
    assert "spatial" not in facts
    assert "national" in facts and "calendar" in facts and "events" in facts


def test_computed_facts_follow_the_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    processed, news = tmp_path / "processed", tmp_path / "news"
    processed.mkdir()
    news.mkdir()
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    pd.DataFrame({"ds": months, "category": "Все категории"}).to_parquet(
        processed / "panel.parquet", index=False
    )
    entity = {
        "rows": 48,
        "territories": 2,
        "series": 4,
        "reference_year": 2024,
        "territories_with_coordinates": 1,
    }
    (processed / "entity_report.json").write_text(json.dumps(entity), encoding="utf-8")
    (news / "news_summary.json").write_text(
        json.dumps({"events": 500, "regions": 3, "columns": 46}), encoding="utf-8"
    )
    spatial = {"territories": 2, "with_market_access": 1, "with_neighbours": 2, "neighbours": 5}
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    (features_dir / "features_summary.json").write_text(
        json.dumps({"national_columns": 61, "spatial": spatial, "weather": "таблицы погоды нет"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(storyline, "PROCESSED", processed)
    monkeypatch.setattr(storyline, "NEWS_DIR", news)
    monkeypatch.setattr(storyline, "FEATURES_DIR", features_dir)
    facts = storyline.collect_facts()
    assert facts["spatial"] == spatial and facts["national_features"] == 61
    assert "weather" not in facts, "шаг признаков пропустил погоду — факта о ней нет"
    assert facts["spending"] == {
        "rows": 48,
        "territories": 2,
        "series": 4,
        "first": "2023-01",
        "last": "2024-12",
        "categories": 1,
    }
    assert facts["reference"] == {"territories": 2, "year": 2024, "with_coordinates": 1}
    assert facts["news"]["events"] == 500 and facts["news"]["first"] == "2022-10"
    assert facts["news"]["last"] == "2024-12", "период архива — из манифеста GDELT"


def test_build_passes_the_measured_source_contributions_to_the_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Вклад источника из `sources.csv` доходит и до таблицы источников, и до диаграммы."""
    import json

    features_dir = tmp_path / "features"
    features_dir.mkdir()
    spatial = {
        "territories": 2190,
        "with_market_access": 2170,
        "with_neighbours": 2172,
        "neighbours": 5,
        "highway_pairs": 2353366,
        "median_distance_km": 57.5,
        "cross_region_share": 0.147,
    }
    population = {
        "territories": 2190,
        "matched": 2019,
        "share": 0.9219,
        "threshold": 0.9,
        "used": True,
        "year": 2023,
        "published": "2023-09-04",
    }
    weather = {
        "cells": 1343,
        "territories": 2190,
        "lag_days": 7,
        "first": "2022-01",
        "last": "2024-12",
        "norm_years": [2001, 2020],
    }
    (features_dir / "features_summary.json").write_text(
        json.dumps({"spatial": spatial, "weather": weather, "population": population}),
        encoding="utf-8",
    )
    for name in ("PROCESSED", "NEWS_DIR", "CPD_DIR"):
        monkeypatch.setattr(storyline, name, tmp_path / name.lower())
    monkeypatch.setattr(storyline, "FEATURES_DIR", features_dir)
    sources = [
        {
            "block": "neighbours",
            "title": "Расходы соседних МО",
            "against": "A",
            "mean_diff": -12.3,
            "p_value": 0.0004,
        }
    ]
    story = storyline.build([], {}, {}, [], None, sources)
    assert [row["pair"] for row in story["source_contributions"]] == ["neighbours_on_A"]
    assert story["source_verdict"].startswith("Ни один источник не изменил ошибку устойчиво")
    gave = {row["key"]: row["gave"] for row in story["sources"]}
    assert gave["neighbours"].startswith("к базовой модели: −12,3 руб.")
    assert gave["access"] == "не измерено в этом прогоне"
    assert any("соседних МО" in row["source"] for row in story["timing"]["rows"])
    week = next(row for row in story["timing"]["rows"] if row["lag"] == "7 дней")
    assert "Погода NASA POWER" in week["source"], "погода — в строке источников с тем же лагом"
    assert story["facts"]["weather"] == weather
    assert story["facts"]["population"] == population
    known = next(r["known"] for r in story["timing"]["rows"] if "Росстат" in r["source"])
    assert known == "оценка на 1 января 2023 года, опубликована к 4 сентября 2023 года"
    assert story["facts"]["first_forecast"] == "2023-10-31", "конец месяца самого раннего окна"
    timing = {row["key"]: row["timing"] for row in story["sources"]}
    assert timing["population"].endswith("до первого момента прогноза (31 октября 2023 года)")

    bare = storyline.build([], {}, {}, [], None)
    assert bare["source_contributions"] == []

    # Привязано меньше порога: население — справочный показатель, прогноз его не видит.
    unused = {**population, "matched": 1900, "share": 0.8676, "used": False}
    (features_dir / "features_summary.json").write_text(
        json.dumps({"spatial": spatial, "weather": weather, "population": unused}),
        encoding="utf-8",
    )
    reference = storyline.build([], {}, {}, [], None)
    assert not any("Росстат" in row["source"] for row in reference["timing"]["rows"])
    assert "population" in {row["key"] for row in reference["sources"]}


def test_citations_name_every_source_whose_data_the_run_uses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Подписи источников на странице: набор организаторов, СберИндекс, GDELT — и NASA POWER с
    Росстатом, раз их таблицы лежат в репозитории и идут в расчёт. NASA просит приводить две
    подписи: о проекте и о версии сервиса с датой обращения."""
    lines = storyline.citations()
    assert lines[0].startswith("Потребительские безналичные расходы на уровне муниципальных")
    assert "Данные СберИндекса (https://sberindex.ru)." in lines
    nasa = [line for line in lines if "POWER" in line]
    assert len(nasa) == 2 and nasa[0].startswith("The data was obtained from National Aeronautics")
    assert nasa[1].startswith("The data was obtained from the POWER Project's Monthly and Annual")
    assert lines[-1].startswith("Росстат. Численность населения Российской Федерации")
    assert len(lines) == len(set(lines)), "подпись не повторяется"

    monkeypatch.setattr(storyline.power, "MANIFEST_PATH", tmp_path / "no-weather.yaml")
    monkeypatch.setattr(storyline.rosstat, "MANIFEST_PATH", tmp_path / "no-population.yaml")
    bare = storyline.citations()
    assert not any("POWER" in line or "Росстат" in line for line in bare)
    assert bare[-2:] == list(storyline.EXTRA_CITATIONS)
