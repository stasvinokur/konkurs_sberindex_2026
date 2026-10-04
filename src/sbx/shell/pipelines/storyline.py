"""Данные для блоков лендинга «данные → расчёт → результат».

Здесь читаются манифесты, конфиги и артефакты; сами блоки собирают чистые функции
`sbx.core.storyline`. Необязательного артефакта нет — блока в результате нет, и страница его
пропускает.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from sbx.core.changepoint.synthetic import split_benchmark
from sbx.core.folds import make_folds
from sbx.core.storyline import (
    DETECTOR_PATHS,
    chain_summary,
    contribution_view,
    criteria_map,
    detector_demo,
    fold_timeline,
    foundation_view,
    news_timeline,
    source_inventory,
    source_verdict,
    source_view,
    timing_rows,
)
from sbx.shell.download import official, power, rosstat
from sbx.shell.io import CONFIG_DIR, DATA_DIR, publication_lags, read_yaml
from sbx.shell.pipelines import samples
from sbx.shell.pipelines.backtest import load_backtest_config, load_models_config
from sbx.shell.pipelines.cpd import CPD_DIR, load_cpd_config
from sbx.shell.pipelines.entity import PROCESSED
from sbx.shell.pipelines.features import FEATURES_DIR, NATIONAL_SLUGS, NEWS_DIR

# Подписи источников, которых нет в манифесте организаторов.
EXTRA_CITATIONS = (
    "Данные СберИндекса (https://sberindex.ru).",
    "The GDELT Project (https://www.gdeltproject.org/).",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _national() -> list[dict[str, Any]]:
    """Национальные ряды: название, период, частота и лаг публикации каждого."""
    manifest = read_yaml(DATA_DIR / "manifest.yaml")
    titles = {d["slug"]: d["title"] for d in manifest["datasets"]}
    lags = publication_lags()
    rows = []
    for slug in NATIONAL_SLUGS:
        path = DATA_DIR / "raw" / slug / f"{slug}.parquet"
        if not path.exists():
            continue
        periods = pd.to_datetime(pd.read_parquet(path, columns=["period"])["period"])
        dates = pd.Series(sorted(periods.unique()))
        step = dates.diff().dt.days.median()
        rows.append(
            {
                "slug": slug,
                "title": titles.get(slug, slug),
                "frequency": "неделя" if step <= 10 else "месяц",
                "first": f"{dates.iloc[0]:%Y-%m}",
                "last": f"{dates.iloc[-1]:%Y-%m}",
                "lag_days": lags[slug],
            }
        )
    return rows


def tracked_facts() -> dict[str, Any]:
    """Факты об источниках, лежащих в репозитории: они есть всегда."""
    calendar = json.loads((DATA_DIR / "reference" / "ru_calendar.json").read_text("utf-8"))
    events = yaml.safe_load((CONFIG_DIR / "events.yaml").read_text(encoding="utf-8"))["events"]
    dates = sorted(pd.Timestamp(e["date"]) for e in events)
    years = sorted(int(year) for year in calendar)
    return {
        "national": _national(),
        "calendar": {"first_year": years[0], "last_year": years[-1]},
        "events": {
            "count": len(events),
            "first": f"{dates[0]:%Y-%m}",
            "last": f"{dates[-1]:%Y-%m}",
        },
    }


def collect_facts() -> dict[str, Any]:
    """Все факты об источниках: из репозитория и из результатов шагов расчёта."""
    facts = tracked_facts()
    entity = _json(PROCESSED / "entity_report.json")
    panel = PROCESSED / "panel.parquet"
    if entity and panel.exists():
        frame = pd.read_parquet(panel, columns=["ds", "category"])
        facts["spending"] = {
            "rows": int(entity["rows"]),
            "territories": int(entity["territories"]),
            "series": int(entity["series"]),
            "first": f"{frame['ds'].min():%Y-%m}",
            "last": f"{frame['ds'].max():%Y-%m}",
            "categories": int(frame["category"].nunique()),
        }
        facts["reference"] = {
            "territories": int(entity["territories"]),
            "year": int(entity["reference_year"]),
            "with_coordinates": int(entity["territories_with_coordinates"]),
        }
    news = _json(NEWS_DIR / "news_summary.json")
    if news:
        archive = read_yaml(DATA_DIR / "gdelt_manifest.yaml")
        facts["news"] = {
            "events": int(news["events"]),
            "regions": int(news["regions"]),
            "columns": int(news["columns"]),
            "first": str(archive["start"]),
            "last": str(archive["end"]),
        }
    features = _json(FEATURES_DIR / "features_summary.json")
    if features.get("national_columns"):
        facts["national_features"] = int(features["national_columns"])
    if features.get("spatial"):
        facts["spatial"] = dict(features["spatial"])
    # Шаг признаков пишет строку вместо сводки, когда таблицы погоды нет.
    if isinstance(features.get("weather"), dict):
        facts["weather"] = dict(features["weather"])
    if isinstance(features.get("population"), dict):
        facts["population"] = dict(features["population"])
    labels = CPD_DIR / "benchmark_labels.parquet"
    if labels.exists():
        kinds = pd.read_parquet(labels, columns=["kind"])["kind"]
        facts["benchmark"] = {"series": int(len(kinds)), "breaks": int((kinds != "none").sum())}
    return facts


def _detector_demo(detectors: pd.DataFrame | None) -> dict[str, Any]:
    """Пример на ряду тестовой части бенчмарка для лучшего онлайн-детектора по остаткам."""
    series_path = CPD_DIR / "benchmark_series.parquet"
    labels_path = CPD_DIR / "benchmark_labels.parquet"
    seasonal_path = PROCESSED / "seasonal_index.parquet"
    if detectors is None or not all(p.exists() for p in (series_path, labels_path, seasonal_path)):
        return {}
    online = detectors[
        (detectors["kind"] == "online")
        & (detectors["input"] == "residual")
        & detectors["detector"].isin(list(DETECTOR_PATHS))
    ].sort_values("f1", ascending=False)
    if online.empty:
        return {}
    best = online.iloc[0]
    labels = pd.read_parquet(labels_path)
    # Порог подбирался на калибровочной половине; пример берётся из тестовой.
    _, test_ids = split_benchmark(labels, seed=int(load_cpd_config().get("split_seed", 1)))
    return detector_demo(
        pd.read_parquet(series_path),
        labels,
        pd.read_parquet(seasonal_path),
        detector=str(best["detector"]),
        threshold=float(best["param"]),
        ids=test_ids,
    )


def _news_timeline() -> dict[str, Any]:
    path = NEWS_DIR / "region_month_features.parquet"
    if not path.exists():
        return {}
    events = yaml.safe_load((CONFIG_DIR / "events.yaml").read_text(encoding="utf-8"))["events"]
    return news_timeline(pd.read_parquet(path), events)


def citations() -> list[str]:
    """Обязательные подписи источников: из манифеста организаторов, общие и — если таблицы
    погоды и населения подключены — подписи NASA POWER и Росстата из их манифестов."""
    lines: list[str] = []
    for spec in official.load_manifest().files:
        lines += [c for c in spec.citations if c not in lines]
    lines += EXTRA_CITATIONS
    if power.MANIFEST_PATH.exists():
        weather = read_yaml(power.MANIFEST_PATH)
        # NASA просит две подписи: о проекте и о версии сервиса с датой обращения.
        lines += [str(weather[key]) for key in ("citation", "data_reference") if weather.get(key)]
    if rosstat.MANIFEST_PATH.exists():
        lines.append(str(read_yaml(rosstat.MANIFEST_PATH)["citation"]))
    return lines


def build(
    winners: Sequence[Mapping[str, Any]],
    contrast: Mapping[str, Any],
    dm: Mapping[str, Mapping[str, Any]],
    horizon_rows: Sequence[Mapping[str, Any]],
    detectors: pd.DataFrame | None,
    sources: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Все блоки цепочки «данные → расчёт → результат» для страницы.

    `sources` — измеренный вклад каждого источника (`artifacts/ablations/sources.csv`).
    """
    facts = collect_facts()
    config = load_backtest_config()
    folds = make_folds(config)
    # Момент прогноза — конец месяца origin; самый ранний нужен постоянным характеристикам.
    facts["first_forecast"] = f"{min(f.cutoff for f in folds) + pd.offsets.MonthEnd(0):%Y-%m-%d}"
    models_cfg = load_models_config()
    titles = {item["slug"]: item["title"] for item in facts["national"]}
    lags = {item["slug"]: item["lag_days"] for item in facts["national"]}
    if facts.get("weather"):
        titles["weather"], lags["weather"] = "Погода NASA POWER", facts["weather"]["lag_days"]
    singles = [r for r in horizon_rows if not str(r["Модель"]).startswith("Ensemble")]

    story: dict[str, Any] = {
        "sources": source_inventory(facts, dm, sources),
        "source_contributions": source_view(sources),
        "source_verdict": source_verdict(sources),
        "chain": chain_summary(
            facts,
            folds=len(folds),
            models=len(singles),
            winners=winners,
            contrast=contrast,
            dm=dm,
            horizons=config.horizons,
        ),
        "timing": {
            "origin": f"{max(f.cutoff for f in folds):%Y-%m}",
            "rows": timing_rows(
                max(f.cutoff for f in folds),
                lags,
                titles,
                spatial="spatial" in facts,
                population=(
                    facts["population"] if (facts.get("population") or {}).get("used") else None
                ),
            ),
        },
        "contributions": contribution_view(dm),
        "foundation": foundation_view(horizon_rows, models_cfg.get("foundation_models", {})),
        "citations": citations(),
        "criteria": criteria_map(),
        "facts": facts,
        "numbers": {
            "folds": len(folds),
            "models": len(singles),
            "evaluation_series": len(samples.load_evaluation()),
        },
    }
    panel = PROCESSED / "panel.parquet"
    if panel.exists():
        months = sorted(pd.read_parquet(panel, columns=["ds"])["ds"].unique())
        story["folds"] = fold_timeline(folds, months)
    for key, block in (
        ("news_timeline", _news_timeline()),
        ("detector_demo", _detector_demo(detectors)),
    ):
        if block:
            story[key] = block
    return story
