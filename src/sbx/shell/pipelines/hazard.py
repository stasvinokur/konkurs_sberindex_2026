"""Модель вероятности структурного шока на 1–3 месяца вперёд и её сравнение с детекторами."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sbx.core.changepoint.hazard import (
    build_hazard_table,
    hazard_feature_columns,
    make_labels,
    series_invariant_columns,
)
from sbx.core.changepoint.metrics import score_detector
from sbx.core.changepoint.residuals import residual_stream
from sbx.shell import stamps
from sbx.shell.hazard_model import (
    best_threshold,
    evaluate_hazard,
    feature_importance,
    fit_hazard,
    predict_hazard,
    split_by_series,
    threshold_for_alarm_rate,
)
from sbx.shell.io import ARTIFACTS_DIR, CONFIG_DIR, read_yaml
from sbx.shell.pipelines.cpd import CPD_DIR
from sbx.shell.pipelines.entity import PROCESSED

HAZARD_DIR = ARTIFACTS_DIR / "hazard"


def load_hazard_config(path: Path = CONFIG_DIR / "hazard.yaml") -> dict[str, Any]:
    return dict(read_yaml(path))


def _exogenous_path(rel: str) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else ARTIFACTS_DIR.parent / rel


def stamp_inputs(cfg: Mapping[str, Any], cfg_path: Path) -> list[str]:
    """Входы шага для отметки рядом с результатом (см. `sbx.shell.stamps`)."""
    return [
        stamps.STATIC,
        stamps.SEASONAL,
        stamps.file_key(cfg_path),
        stamps.file_key(CPD_DIR / "benchmark_series.parquet"),
        stamps.file_key(CPD_DIR / "benchmark_labels.parquet"),
        stamps.file_key(CPD_DIR / "alarms.json"),
        *(stamps.file_key(Path(rel)) for rel in (cfg.get("exogenous") or {}).values()),
    ]


def _exogenous(
    series: pd.DataFrame, static: pd.DataFrame, cfg: Mapping[str, Any]
) -> pd.DataFrame | None:
    """Внешние признаки (новости, макро) на (ряд, t).

    Ряды бенчмарка синтетические, но каждый построен из реального ряда (`source_id`), поэтому
    регион берётся у исходной территории.
    """
    paths = cfg.get("exogenous", {})
    frames = []
    for name, rel in paths.items():
        path = _exogenous_path(rel)
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        keep = {"ds", "region_code"}
        frame = frame.rename(columns={c: f"{name}__{c}" for c in frame.columns if c not in keep})
        frames.append(frame)
    if not frames:
        return None

    index = (
        series[["unique_id", "source_id", "ds"]].drop_duplicates().sort_values(["unique_id", "ds"])
    )
    index["t"] = index.groupby("unique_id").cumcount()
    regions = static.set_index("unique_id")["region_code"]
    index["region_code"] = index["source_id"].map(regions)
    merged = index.drop(columns=["source_id"])
    for frame in frames:
        keys = [c for c in ("ds", "region_code") if c in frame.columns]
        merged = merged.merge(frame, on=keys, how="left")
    return merged.drop(columns=["ds", "region_code"])


def _mean_lead_time(
    breaks: Mapping[str, list[int]], alarms: Mapping[str, list[int]], ids: set[str]
) -> float:
    """Среднее упреждение: сколько месяцев до слома прозвучала последняя предшествующая тревога."""
    values = []
    for uid, taus in breaks.items():
        if uid not in ids:
            continue
        for tau in taus:
            earlier = [t for t in alarms.get(uid, []) if t < tau]
            values.append(tau - max(earlier) if earlier else np.nan)
    return float(np.nanmean(values)) if values else float("nan")


def run(cfg_path: Path = CONFIG_DIR / "hazard.yaml", out_dir: Path = HAZARD_DIR) -> dict[str, Any]:
    """Шаг расчёта: модель вероятности шока и отметка входов рядом с результатом."""
    cfg = load_hazard_config(cfg_path)
    results = train_and_compare(cfg, out_dir)
    stamps.write(out_dir / "inputs.json", stamp_inputs(cfg, cfg_path))
    return results


def train_and_compare(cfg: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    """Обучает модель вероятности шока на бенчмарке и сравнивает её с онлайн-детекторами."""
    series = pd.read_parquet(CPD_DIR / "benchmark_series.parquet")
    labels_raw = pd.read_parquet(CPD_DIR / "benchmark_labels.parquet")
    seasonal = pd.read_parquet(PROCESSED / "seasonal_index.parquet")

    residuals = residual_stream(series, seasonal)
    breaks = {r.unique_id: ([int(r.tau)] if r.tau >= 0 else []) for r in labels_raw.itertuples()}
    lengths = {r.unique_id: int(r.n_obs) for r in labels_raw.itertuples()}
    horizons = tuple(cfg.get("horizons", (1, 2, 3)))
    labels = make_labels(breaks, lengths, horizons)

    alarms_path = CPD_DIR / "alarms.json"
    detector_alarms = None
    detector_key = cfg.get("detector_alarms_key")
    if alarms_path.exists() and detector_key:
        dump = json.loads(alarms_path.read_text(encoding="utf-8"))
        detector_alarms = {k: list(map(int, v)) for k, v in dump.get(detector_key, {}).items()}

    table = build_hazard_table(
        residuals.rename(columns={"z": "z"})[["unique_id", "t", "z"]],
        labels,
        exogenous=_exogenous(series, pd.read_parquet(PROCESSED / "static.parquet"), cfg),
        detector_alarms=detector_alarms,
    )
    features = hazard_feature_columns(table)
    if cfg.get("drop_series_invariant", True):
        # Признаки, одинаковые для всех рядов в каждом t, на этом бенчмарке эквивалентны
        # позиции в ряду и дают модели выучить форму разметки вместо сигнала о шоке.
        invariant = series_invariant_columns(table, features)
        features = [c for c in features if c not in set(invariant)]
        results_invariant = invariant
    else:
        results_invariant = []
    train_ids, test_ids = split_by_series(
        table["unique_id"].unique(), float(cfg.get("train_share", 0.6)), int(cfg.get("seed", 0))
    )
    # Обучающая часть делится ещё раз: на подгонку и на калибровку порога. Порог, подобранный
    # по прогнозам модели на её же обучающих строках, бесполезен — LightGBM их запоминает,
    # in-sample F1 близка к единице, и выбранный порог не переносится на тестовую часть.
    fit_ids, valid_ids = split_by_series(
        train_ids, float(cfg.get("fit_share", 0.75)), int(cfg.get("seed", 0)) + 1
    )
    train = table[table["unique_id"].isin(set(fit_ids))]
    valid = table[table["unique_id"].isin(set(valid_ids))]
    test = table[table["unique_id"].isin(set(test_ids))]

    results: dict[str, Any] = {
        "features": len(features),
        "dropped_series_invariant": len(results_invariant),
        "dropped_examples": sorted(results_invariant)[:8],
        "fit_rows": len(train),
        "valid_rows": len(valid),
        "test_rows": len(test),
    }
    importances = {}
    probabilities = {}
    thresholds = {}
    for horizon in horizons:
        label = f"z_h{horizon}"
        booster = fit_hazard(
            train, features, label, cfg.get("params"), int(cfg.get("num_boost_round", 300))
        )
        # Порог тревоги подбирается на отложенной валидационной части: фиксированные 0,5 для
        # редкого события дают почти нулевую полноту, а подбор на строках подгонки даёт порог,
        # настроенный на запомненные моделью ответы.
        valid_probs = predict_hazard(booster, valid, features)
        threshold, valid_f1 = best_threshold(valid_probs, valid, label)
        thresholds[label] = {"threshold": threshold, "valid_f1": valid_f1}
        probs = predict_hazard(booster, test, features)
        results[f"h{horizon}"] = evaluate_hazard(probs, test, label, threshold)
        results[f"h{horizon}"]["threshold"] = threshold
        importances[label] = feature_importance(booster, features).head(20).to_dict("records")
        probabilities[label] = probs

    # Сравнение с онлайн-детектором: тревоги модели против тревог детектора на тех же рядах.
    test_labels = labels_raw[labels_raw["unique_id"].isin(set(test_ids))]
    horizon = horizons[-1]
    threshold = thresholds[f"z_h{horizon}"]["threshold"]
    probs = probabilities[f"z_h{horizon}"]
    model_alarms = {
        uid: sorted(g.loc[g["p"] >= threshold, "t"].astype(int))
        for uid, g in probs.groupby("unique_id")
    }
    results["model_as_detector"] = score_detector(
        test_labels, model_alarms, margin=int(cfg.get("margin", 1))
    )
    if detector_alarms:
        subset = {uid: detector_alarms.get(uid, []) for uid in test_labels["unique_id"]}
        results["reference_detector"] = score_detector(
            test_labels, subset, margin=int(cfg.get("margin", 1))
        )
        # Второй режим сравнения: модели даётся ровно столько же тревог, сколько у детектора.
        # Порог по F1 уводит модель в режим «тревога каждый второй месяц», и сравнение
        # превращается в сравнение удобства порога, а не качества сигнала.
        detector_rate = sum(len(v) for v in subset.values()) / max(len(test), 1)
        matched = threshold_for_alarm_rate(
            predict_hazard(
                fit_hazard(
                    train,
                    features,
                    f"z_h{horizon}",
                    cfg.get("params"),
                    int(cfg.get("num_boost_round", 300)),
                ),
                valid,
                features,
            ),
            detector_rate,
        )
        matched_alarms = {
            uid: sorted(g.loc[g["p"] >= matched, "t"].astype(int))
            for uid, g in probs.groupby("unique_id")
        }
        results["model_at_matched_alarm_rate"] = {
            "threshold": matched,
            "target_alarm_rate": detector_rate,
            **score_detector(test_labels, matched_alarms, margin=int(cfg.get("margin", 1))),
        }
        results["model_at_matched_alarm_rate"]["mean_lead_time"] = _mean_lead_time(
            breaks, matched_alarms, set(test_ids)
        )

    lead = []
    for uid, taus in breaks.items():
        if uid not in set(test_ids):
            continue
        for tau in taus:
            earlier = [t for t in model_alarms.get(uid, []) if t < tau]
            lead.append(
                {
                    "unique_id": uid,
                    "tau": tau,
                    "lead_time": tau - max(earlier) if earlier else np.nan,
                }
            )
    lead_frame = pd.DataFrame(lead)
    results["thresholds"] = thresholds
    results["mean_lead_time"] = (
        float(lead_frame["lead_time"].mean()) if len(lead_frame) else float("nan")
    )
    results["warned_share"] = (
        float(lead_frame["lead_time"].notna().mean()) if len(lead_frame) else float("nan")
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([p.assign(label=label) for label, p in probabilities.items()]).to_parquet(
        out_dir / "hazard_probabilities.parquet", index=False
    )
    lead_frame.to_csv(out_dir / "lead_times.csv", index=False)
    (out_dir / "hazard_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "hazard_importance.json").write_text(
        json.dumps(importances, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return results
