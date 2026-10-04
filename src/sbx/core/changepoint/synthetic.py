"""Полусинтетический бенчмарк структурных сломов.

В реальные ряды МО внедряются сломы с известными моментом, формой и величиной. Это единственный
способ честно сравнить методы обнаружения: реальных размеченных шоков на уровне МО за 2023–2024
почти нет (приграничных МО Курской области нет в данных, паводок в Орске в месячных расходах
не виден).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

BREAK_KINDS = (
    "level_shift",
    "slope_change",
    "variance_jump",
    "seasonality_change",
    "temporary_drop",
    "temporary_spike",
    "gradual_drift",
)


@dataclass(frozen=True)
class ShockLabel:
    unique_id: str
    tau: int  # индекс первого изменённого наблюдения
    kind: str
    delta: float
    source_id: str


def inject(
    values: np.ndarray,
    kind: str,
    tau: int,
    delta: float,
    rng: np.random.Generator,
    duration: int = 3,
) -> np.ndarray:
    """Вносит слом заданной формы начиная с позиции `tau` (включительно).

    `delta` — относительная величина (0.2 = 20%). До `tau` ряд не изменяется.
    """
    if kind not in BREAK_KINDS:
        raise ValueError(f"неизвестный тип слома: {kind}")
    if not 0 < tau < len(values):
        raise ValueError("tau должен быть внутри ряда")
    out = np.asarray(values, dtype=float).copy()
    n = len(out)
    after = np.arange(n - tau)
    if kind == "level_shift":
        out[tau:] *= 1.0 + delta
    elif kind == "slope_change":
        out[tau:] *= 1.0 + delta * (after + 1) / max(len(after), 1)
    elif kind == "variance_jump":
        history_std = float(np.nanstd(out[:tau])) if tau > 1 else 0.0
        # У постоянного ряда разброс нулевой: масштаб берём от уровня, иначе «слома» не будет.
        scale = np.abs(delta) * (history_std if history_std > 0 else float(np.nanmean(out)))
        out[tau:] += rng.normal(0.0, max(scale, 1e-9), size=len(after))
    elif kind == "seasonality_change":
        phase = np.pi * after / 6.0
        out[tau:] *= 1.0 + delta * np.sin(phase)
    elif kind == "temporary_drop":
        end = min(n, tau + duration)
        out[tau:end] *= 1.0 - abs(delta)
    elif kind == "temporary_spike":
        end = min(n, tau + duration)
        out[tau:end] *= 1.0 + abs(delta)
    elif kind == "gradual_drift":
        out[tau:] *= 1.0 + delta * np.minimum(1.0, (after + 1) / max(duration, 1))
    return np.clip(out, 0.0, None)


def build_benchmark(
    panel: pd.DataFrame,
    static: pd.DataFrame,
    n_series: int,
    kinds: Sequence[str] = BREAK_KINDS,
    deltas: Sequence[float] = (0.05, 0.1, 0.2, 0.4),
    min_gap: int = 4,
    control_share: float = 0.25,
    seed: int = 0,
    strata: Sequence[str] = ("category", "size_group"),
    exclude_ids: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Собирает бенчмарк: изменённые ряды и разметку сломов.

    Выборка стратифицирована по категории и размеру МО; `control_share` рядов остаются без
    инъекции (для оценки ложных тревог). Ряды из `exclude_ids` (административные разрывы,
    неполные ряды) не используются.
    """
    rng = np.random.default_rng(seed)
    complete = static[static["is_complete"]] if "is_complete" in static else static
    pool = complete[~complete["unique_id"].isin(set(exclude_ids))].copy()
    if pool.empty:
        raise ValueError("нет подходящих рядов для бенчмарка")
    pool["_stratum"] = pool[list(strata)].astype(str).agg("|".join, axis=1)
    shares = pool["_stratum"].value_counts(normalize=True)

    picked: list[str] = []
    for stratum, share in shares.items():
        candidates = np.sort(pool.loc[pool["_stratum"] == stratum, "unique_id"].to_numpy())
        take = min(len(candidates), int(round(share * n_series)))
        picked += list(rng.choice(candidates, size=take, replace=False))
    picked = sorted(picked)

    series = {
        uid: g.sort_values("ds")
        for uid, g in panel[panel["unique_id"].isin(picked)].groupby("unique_id")
    }
    rows, labels = [], []
    for i, uid in enumerate(picked):
        g = series[uid]
        values = g["y"].to_numpy(dtype=float)
        is_control = rng.random() < control_share
        if is_control or len(values) < 2 * min_gap + 2:
            new_values, kind, tau, delta = values, "none", -1, 0.0
        else:
            kind = str(rng.choice(list(kinds)))
            delta = float(rng.choice(list(deltas)))
            sign = 1.0 if rng.random() < 0.5 else -1.0
            tau = int(rng.integers(min_gap, len(values) - min_gap))
            new_values = inject(values, kind, tau, sign * delta, rng)
            delta = sign * delta
        new_id = f"synth_{i:05d}"
        rows.append(
            pd.DataFrame(
                {
                    "unique_id": new_id,
                    "source_id": uid,
                    "ds": g["ds"].to_numpy(),
                    "y": new_values,
                    "category": g["category"].to_numpy(),
                }
            )
        )
        labels.append(
            {
                "unique_id": new_id,
                "source_id": uid,
                "kind": kind,
                "tau": tau,
                "tau_ds": g["ds"].to_numpy()[tau] if tau >= 0 else pd.NaT,
                "delta": delta,
                "n_obs": len(values),
            }
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(labels)


def split_benchmark(
    labels: pd.DataFrame, calibration_share: float = 0.5, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Детерминированное деление бенчмарка на калибровочную и тестовую части."""
    rng = np.random.default_rng(seed)
    ids = np.sort(labels["unique_id"].to_numpy())
    mask = rng.random(len(ids)) < calibration_share
    return sorted(ids[mask]), sorted(ids[~mask])
