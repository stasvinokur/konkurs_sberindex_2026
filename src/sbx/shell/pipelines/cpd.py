"""Бенчмарк обнаружения структурных изменений: синтетика, калибровка, сравнение методов."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sbx.core.changepoint.metrics import (
    nearest_rate,
    recall_by,
    recall_gap_interval,
    score_detector,
)
from sbx.core.changepoint.offline import periodic_alarms
from sbx.core.changepoint.online import bocpd, conformal_alarm, cusum, page_hinkley
from sbx.core.changepoint.residuals import (
    causal_zscores,
    freshest_forecast,
    interval_exceedance,
    residual_frame,
    residual_stream,
)
from sbx.core.changepoint.synthetic import build_benchmark, split_benchmark
from sbx.shell import stamps
from sbx.shell.cpd_offline import OFFLINE_METHODS, adwin_alarms, detect_offline_panel
from sbx.shell.io import ARTIFACTS_DIR, CONFIG_DIR, ROOT, read_yaml
from sbx.shell.pipelines import samples
from sbx.shell.pipelines.backtest import OOF_DIR
from sbx.shell.pipelines.entity import PROCESSED
from sbx.shell.pipelines.panel import load_panel, load_static

CPD_DIR = ARTIFACTS_DIR / "cpd"
ONLINE_DETECTORS = ("cusum", "page_hinkley", "bocpd", "adwin", "conformal")


def load_cpd_config(path: Path = CONFIG_DIR / "cpd.yaml") -> dict[str, Any]:
    return dict(read_yaml(path))


def live_stream(oof: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Поток прогнозов, который видит живой детектор: месяц один раз, короткий горизонт.

    Один и тот же поток используют тревоги, разбор МО и графики лендинга — иначе позиции
    тревог указывали бы не на те месяцы.
    """
    cfg = load_cpd_config() if cfg is None else cfg
    return freshest_forecast(oof, max_horizon=cfg.get("live_max_horizon"))


def build(cfg: Mapping[str, Any], out_dir: Path = CPD_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Полусинтетический бенчмарк: замороженный (`frozen_benchmark`) или собранный заново.

    Замороженный — это бенчмарк опубликованного расчёта: те же ряды и те же внесённые сломы.
    Повторный отбор на другой панели дал бы другие ряды, и сравнение детекторов до и после
    изменений в расчёте шло бы на разных данных.
    """
    frozen = cfg.get("frozen_benchmark")
    if frozen:
        series, labels = samples.load_benchmark(ROOT / frozen)
        absent = sorted(set(labels["source_id"]) - set(load_static()["unique_id"]))
        if absent:
            raise ValueError(
                f"{len(absent)} исходных рядов замороженного бенчмарка ({frozen}) нет в панели, "
                f"например {absent[:3]}: бенчмарк составлен для другой панели"
            )
    else:
        panel, static = load_panel(), load_static()
        exclude = static.loc[static["admin_break_at"].notna(), "unique_id"].tolist()
        series, labels = build_benchmark(
            panel,
            static,
            n_series=int(cfg.get("n_series", 1500)),
            deltas=tuple(cfg.get("deltas", (0.05, 0.1, 0.2, 0.4))),
            control_share=float(cfg.get("control_share", 0.25)),
            seed=int(cfg.get("seed", 0)),
            exclude_ids=exclude,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    series.to_parquet(out_dir / "benchmark_series.parquet", index=False)
    labels.to_parquet(out_dir / "benchmark_labels.parquet", index=False)
    return series, labels


def _online_alarms(
    name: str, stream: Mapping[str, np.ndarray], params: Mapping[str, Any]
) -> dict[str, list[int]]:
    out = {}
    for uid, values in stream.items():
        clean = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
        if name == "cusum":
            out[uid] = cusum(clean, threshold=params["threshold"], drift=params.get("drift", 0.5))
        elif name == "page_hinkley":
            out[uid] = page_hinkley(
                clean, threshold=params["threshold"], delta=params.get("delta", 0.05)
            )
        elif name == "bocpd":
            out[uid] = bocpd(
                clean,
                hazard=params.get("hazard", 1 / 12),
                threshold=params["threshold"],
                confirm=int(params.get("confirm", 2)),
            )
        elif name == "adwin":
            out[uid] = adwin_alarms(clean, delta=params["threshold"])
        elif name == "conformal":
            exceed = (np.abs(clean) > params["threshold"]).astype(int)
            out[uid] = conformal_alarm(exceed, consecutive=int(params.get("consecutive", 2)))
        else:
            raise ValueError(name)
    return out


def _streams(series: pd.DataFrame, seasonal: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Два входных потока для онлайн-детекторов: остатки прогноза и сырой ряд.

    Оба нормируются только по прошлым значениям ряда: детектор в момент t не должен получать
    масштаб, оценённый по месяцам после t.
    """
    residuals = residual_stream(series, seasonal)
    by_residual = {uid: g["z"].to_numpy() for uid, g in residuals.groupby("unique_id")}
    raw = {
        uid: causal_zscores(g["y"].to_numpy(dtype=float))
        for uid, g in series.sort_values(["unique_id", "ds"]).groupby("unique_id")
    }
    return {"residual": by_residual, "raw": raw}


def calibrate_and_evaluate(
    series: pd.DataFrame, labels: pd.DataFrame, cfg: Mapping[str, Any], out_dir: Path = CPD_DIR
) -> pd.DataFrame:
    """Калибрует пороги на калибровочной части и оценивает методы на тестовой."""
    seasonal = pd.read_parquet(PROCESSED / "seasonal_index.parquet")
    calibration_ids, test_ids = split_benchmark(labels, seed=int(cfg.get("split_seed", 1)))
    margin = int(cfg.get("margin", 1))
    streams = _streams(series, seasonal)
    rows, alarms_dump = [], {}

    def evaluate(alarms: Mapping[str, list[int]], ids: Sequence[str]) -> dict[str, float]:
        subset = labels[labels["unique_id"].isin(set(ids))]
        picked = {uid: alarms.get(uid, []) for uid in subset["unique_id"]}
        return score_detector(subset, picked, margin=margin)

    for method in cfg.get("offline_methods", OFFLINE_METHODS):
        best = None
        for penalty in cfg.get("penalty_grid", [1, 3, 10, 30, 100, 300]):
            alarms = detect_offline_panel(
                series[series["unique_id"].isin(set(calibration_ids))], method, float(penalty)
            )
            score = evaluate(alarms, calibration_ids)
            if best is None or score["f1"] > best[1]["f1"]:
                best = (float(penalty), score)
        started = time.perf_counter()
        alarms = detect_offline_panel(
            series[series["unique_id"].isin(set(test_ids))], method, best[0]
        )
        elapsed = time.perf_counter() - started
        score = evaluate(alarms, test_ids)
        rows.append(
            {
                "detector": method,
                "kind": "offline",
                "input": "raw",
                "param": best[0],
                **score,
                "runtime_sec": round(elapsed, 2),
            }
        )
        alarms_dump[f"offline::{method}"] = alarms

    # Эталон: тревога по расписанию, не глядя на данные. Период подбирается так же, как
    # параметры методов, — по F1 на калибровочной части.
    periods = [int(period) for period in cfg.get("baseline_periods", [])]
    if periods:
        lengths = {r.unique_id: int(r.n_obs) for r in labels.itertuples()}

        def on_schedule(ids: Sequence[str], period: int) -> dict[str, list[int]]:
            return {uid: periodic_alarms(lengths[uid], period) for uid in ids}

        best = None
        for period in periods:
            score = evaluate(on_schedule(calibration_ids, period), calibration_ids)
            if best is None or score["f1"] > best[1]["f1"]:
                best = (period, score)
        alarms = on_schedule(test_ids, best[0])
        rows.append(
            {
                "detector": "periodic",
                "kind": "baseline",
                "input": "none",
                "param": float(best[0]),
                **evaluate(alarms, test_ids),
                "runtime_sec": 0.0,
            }
        )
        alarms_dump["baseline::periodic"] = alarms

    for detector in cfg.get("online_detectors", ONLINE_DETECTORS):
        grid = cfg.get("threshold_grids", {}).get(detector, [1, 2, 3, 5])
        for stream_name, stream in streams.items():
            calibration_stream = {k: v for k, v in stream.items() if k in set(calibration_ids)}
            test_stream = {k: v for k, v in stream.items() if k in set(test_ids)}
            best = None
            for threshold in grid:
                alarms = _online_alarms(
                    detector, calibration_stream, {"threshold": float(threshold)}
                )
                score = evaluate(alarms, calibration_ids)
                if best is None or score["f1"] > best[1]["f1"]:
                    best = (float(threshold), score)
            started = time.perf_counter()
            alarms = _online_alarms(detector, test_stream, {"threshold": best[0]})
            elapsed = time.perf_counter() - started
            score = evaluate(alarms, test_ids)
            rows.append(
                {
                    "detector": detector,
                    "kind": "online",
                    "input": stream_name,
                    "param": best[0],
                    **score,
                    "runtime_sec": round(elapsed, 2),
                }
            )
            alarms_dump[f"online::{detector}::{stream_name}"] = alarms

    results = pd.DataFrame(rows).sort_values(["f1"], ascending=False).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "detector_comparison.csv", index=False)
    (out_dir / "alarms.json").write_text(
        json.dumps(
            {k: {u: list(map(int, v)) for u, v in a.items()} for k, a in alarms_dump.items()},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    # Лучший метод — среди тех, что читают ряд: расписание в их число не входит.
    best_row = results[results["kind"] != "baseline"].iloc[0]
    best_key = (
        f"offline::{best_row['detector']}"
        if best_row["kind"] == "offline"
        else f"online::{best_row['detector']}::{best_row['input']}"
    )
    test_labels = labels[labels["unique_id"].isin(set(test_ids))]
    recalls = pd.concat(
        [
            recall_by(test_labels, alarms_dump[best_key], by).assign(cut=by)
            for by in ("kind", "delta")
        ]
    )
    recalls.to_csv(out_dir / "best_detector_recall.csv", index=False)

    # Разрез по типу и величине слома для каждого метода, а не только для лучшего: методы
    # почти не различаются по общему F1, и разница видна именно здесь.
    per_method = pd.concat(
        [
            recall_by(test_labels, alarms, by).assign(cut=by, detector=key)
            for key, alarms in alarms_dump.items()
            for by in ("kind", "delta")
        ]
    )
    per_method.to_csv(out_dir / "recall_by_break.csv", index=False)
    return results


def delay_vs_false_alarms(
    series: pd.DataFrame, labels: pd.DataFrame, cfg: Mapping[str, Any], out_dir: Path = CPD_DIR
) -> pd.DataFrame:
    """Кривая «задержка обнаружения ↔ частота ложных тревог» для онлайн-детекторов."""
    seasonal = pd.read_parquet(PROCESSED / "seasonal_index.parquet")
    streams = _streams(series, seasonal)
    rows = []
    for detector in cfg.get("online_detectors", ONLINE_DETECTORS):
        for threshold in cfg.get("threshold_grids", {}).get(detector, [1, 2, 3, 5]):
            alarms = _online_alarms(detector, streams["residual"], {"threshold": float(threshold)})
            score = score_detector(labels, alarms, margin=int(cfg.get("margin", 1)))
            rows.append({"detector": detector, "threshold": float(threshold), **score})
    # Линия отсчёта: расписание с каждым периодом сетки. Детектор лучше расписания, если при
    # той же частоте ложных тревог срабатывает раньше.
    lengths = {r.unique_id: int(r.n_obs) for r in labels.itertuples()}
    for period in cfg.get("baseline_periods", []):
        alarms = {uid: periodic_alarms(n, int(period)) for uid, n in lengths.items()}
        score = score_detector(labels, alarms, margin=int(cfg.get("margin", 1)))
        rows.append({"detector": "periodic", "threshold": float(period), **score})
    curve = pd.DataFrame(rows)
    curve.to_csv(out_dir / "delay_vs_false_alarms.csv", index=False)
    return curve


def schedule_match(
    series: pd.DataFrame,
    labels: pd.DataFrame,
    cfg: Mapping[str, Any],
    results: pd.DataFrame,
    out_dir: Path = CPD_DIR,
) -> dict[str, Any]:
    """Лучший онлайн-детектор по остаткам против расписания с той же частотой ложных тревог.

    F1 и задержка по отдельности детектор от расписания не отличают, поэтому сравнение идёт
    при равной частоте ложных тревог. Период расписания подбирается по частоте ложных тревог
    детектора на калибровочной половине, сравнение — на тестовой. Интервал разности полноты —
    парный бутстреп по рядам тестовой половины.
    """
    periods = [int(period) for period in cfg.get("baseline_periods", [])]
    if not periods or results.empty or "kind" not in results.columns:
        return {}
    online = results[(results["kind"] == "online") & (results["input"] == "residual")]
    if online.empty:
        return {}
    best = online.sort_values("f1", ascending=False).iloc[0]
    detector, threshold = str(best["detector"]), float(best["param"])
    seasonal = pd.read_parquet(PROCESSED / "seasonal_index.parquet")
    calibration_ids, test_ids = split_benchmark(labels, seed=int(cfg.get("split_seed", 1)))
    margin = int(cfg.get("margin", 1))
    alarms = _online_alarms(
        detector, _streams(series, seasonal)["residual"], {"threshold": threshold}
    )
    lengths = {r.unique_id: int(r.n_obs) for r in labels.itertuples()}

    def on_schedule(ids: Sequence[str], period: int) -> dict[str, list[int]]:
        return {uid: periodic_alarms(lengths[uid], period) for uid in ids}

    def score(ids: Sequence[str], by_series: Mapping[str, Sequence[int]]) -> dict[str, float]:
        part = labels[labels["unique_id"].isin(set(ids))]
        return score_detector(part, {uid: by_series.get(uid, []) for uid in ids}, margin=margin)

    rate = "false_alarms_per_series_year"
    period = nearest_rate(
        {p: score(calibration_ids, on_schedule(calibration_ids, p))[rate] for p in periods},
        score(calibration_ids, alarms)[rate],
    )
    scheduled = on_schedule(test_ids, period)
    own, schedule = score(test_ids, alarms), score(test_ids, scheduled)
    level = 0.95
    gap = recall_gap_interval(
        labels[labels["unique_id"].isin(set(test_ids))],
        alarms,
        scheduled,
        margin=margin,
        n_boot=int(cfg.get("bootstrap", 2000)),
        level=level,
        seed=int(cfg.get("seed", 0)),
    )
    match = {
        "detector": detector,
        "threshold": threshold,
        "period": period,
        "detector_recall": own["recall"],
        "detector_delay": own["mean_delay"],
        "detector_false_alarms": own[rate],
        "schedule_recall": schedule["recall"],
        "schedule_delay": schedule["mean_delay"],
        "schedule_false_alarms": schedule[rate],
        "recall_diff": gap["diff"],
        "recall_diff_low": gap["low"],
        "recall_diff_high": gap["high"],
        "level": level,
        "n": gap["n"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "schedule_match.json").write_text(
        json.dumps(match, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return match


def live_alarms(results: pd.DataFrame, out_dir: Path = CPD_DIR) -> dict[str, Any]:
    """Тревоги по настоящим остаткам ансамбля на реальных рядах МО.

    На бенчмарке детекторы работают по остаткам сезонно-наивного прогноза, а не ансамбля:
    ряды бенчмарка синтетические, их нет в бэктесте, и OOF-прогноза ансамбля для них не
    существует. Здесь механизм собирается целиком так, как он работал бы в эксплуатации:
    остатки ансамбля по реальным рядам плюс конформный детектор по его же интервалам.
    Меток на реальных рядах нет, поэтому F1 здесь не считается — только частота тревог.
    """
    ensemble_path = OOF_DIR / "ensemble.parquet"
    if not ensemble_path.exists() or results.empty or "kind" not in results.columns:
        return {}
    oof = pd.read_parquet(ensemble_path)
    model = "Ensemble (по категориям)"
    if model not in set(oof["model"]):
        model = str(oof["model"].iloc[0])
    # Горизонты перекрываются, а онлайн-детекторы накопительные: повторённый месяц создал бы
    # ложный дрейф. Поток — ряд без повторов с самым коротким горизонтом на каждый месяц.
    oof = live_stream(oof)
    residuals = residual_frame(oof, model)

    online = results[(results["kind"] == "online") & (results["input"] == "residual")]
    online = online.sort_values("f1", ascending=False)
    detector = str(online.iloc[0]["detector"])
    threshold = float(online.iloc[0]["param"])
    stream = {uid: g.sort_values("ds")["z"].to_numpy() for uid, g in residuals.groupby("unique_id")}
    alarms = _online_alarms(detector, stream, {"threshold": threshold})

    # Конформный детектор по настоящим интервалам ансамбля, а не по порогу на z.
    conformal = {}
    if {"q_0.1", "q_0.9"} <= set(oof.columns):
        for uid, g in (
            oof[oof["model"] == model].sort_values(["unique_id", "ds"]).groupby("unique_id")
        ):
            exceed = interval_exceedance(
                g["y"].to_numpy(), g["q_0.1"].to_numpy(), g["q_0.9"].to_numpy()
            )
            conformal[uid] = conformal_alarm(exceed, consecutive=2)

    months = float(len(residuals)) / max(len(stream), 1)
    summary = {
        "model": model,
        "detector": detector,
        "threshold": threshold,
        "series": len(stream),
        "months_per_series": round(months, 2),
        "alarms": int(sum(len(v) for v in alarms.values())),
        "alarms_per_series_year": round(
            sum(len(v) for v in alarms.values()) / max(len(stream) * months / 12, 1e-9), 3
        ),
        "conformal_alarms": int(sum(len(v) for v in conformal.values())),
        # Без этих двух чисел частота тревог читалась бы как частота сдвигов.
        "series_with_alarm": int(sum(1 for v in alarms.values() if len(v))),
        "above_forecast_share": round(float((residuals["residual"] > 0).mean()), 3),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "live_alarms.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "alarms": {u: list(map(int, v)) for u, v in alarms.items() if v},
                "conformal": {u: list(map(int, v)) for u, v in conformal.items() if v},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return summary


def best_offline(results: pd.DataFrame) -> tuple[str, float] | None:
    """Лучший по F1 офлайн-метод бенчмарка и его откалиброванный штраф."""
    if results.empty or "kind" not in results.columns:
        return None
    offline = results[results["kind"] == "offline"].sort_values("f1", ascending=False)
    if offline.empty:
        return None
    return str(offline.iloc[0]["detector"]), float(offline.iloc[0]["param"])


def pseudo_labels(results: pd.DataFrame, out_dir: Path = CPD_DIR) -> pd.DataFrame:
    """Псевдометки сломов по реальным рядам МО лучшим офлайн-методом.

    Реальных размеченных шоков почти нет, поэтому модель вероятности шока учится на синтетике.
    Псевдометки — второй источник обучающего сигнала и материал для разбора примеров: это
    догадки метода, а не факты, поэтому они пишутся отдельным артефактом со штрафом и именем
    метода, а не подмешиваются к разметке бенчмарка.
    """
    calibrated = best_offline(results)
    if calibrated is None:
        return pd.DataFrame()
    method, penalty = calibrated
    panel, static = load_panel(), load_static()
    # Административные разрывы — не экономические шоки, метить их незачем.
    exclude = set(static.loc[static["admin_break_at"].notna(), "unique_id"])
    subset = panel[~panel["unique_id"].isin(exclude)]
    alarms = detect_offline_panel(subset, method, penalty)
    dates = {uid: g.sort_values("ds")["ds"].to_numpy() for uid, g in subset.groupby("unique_id")}
    rows = [
        {
            "unique_id": uid,
            "tau": int(tau),
            "ds": dates[uid][tau],
            "method": method,
            "penalty": penalty,
        }
        for uid, taus in alarms.items()
        for tau in taus
        if uid in dates and tau < len(dates[uid])
    ]
    frame = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / "pseudo_labels.parquet", index=False)
    return frame


def stamp_inputs(cfg: Mapping[str, Any], cfg_path: Path) -> list[str]:
    """Входы шага для отметки рядом с результатом (см. `sbx.shell.stamps`)."""
    keys = [
        stamps.PANEL,
        stamps.STATIC,
        stamps.SEASONAL,
        stamps.file_key(cfg_path),
        stamps.file_key(OOF_DIR / "ensemble.parquet"),
    ]
    frozen = cfg.get("frozen_benchmark")
    if frozen:
        keys += [stamps.file_key(path) for path in samples.benchmark_paths(ROOT / frozen)]
    return keys


def run(cfg_path: Path = CONFIG_DIR / "cpd.yaml", out_dir: Path = CPD_DIR) -> pd.DataFrame:
    cfg = load_cpd_config(cfg_path)
    series, labels = build(cfg, out_dir)
    results = calibrate_and_evaluate(series, labels, cfg, out_dir)
    delay_vs_false_alarms(series, labels, cfg, out_dir)
    schedule_match(series, labels, cfg, results, out_dir)
    pseudo_labels(results, out_dir)
    live_alarms(results, out_dir)
    stamps.write(out_dir / "inputs.json", stamp_inputs(cfg, cfg_path))
    return results
