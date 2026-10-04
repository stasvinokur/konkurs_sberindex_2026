"""Классификатор вероятности структурного шока (LightGBM) и его оценка."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

DEFAULT_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
}


def fit_hazard(
    train: pd.DataFrame,
    features: Sequence[str],
    label: str,
    params: Mapping[str, Any] | None = None,
    rounds: int = 300,
):
    import lightgbm as lgb

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = lgb.Dataset(train[list(features)], label=train[label].astype(int))
        return lgb.train({**DEFAULT_PARAMS, **(params or {})}, dataset, num_boost_round=rounds)


def predict_hazard(booster, frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    out = frame[["unique_id", "t"]].copy()
    out["p"] = booster.predict(frame[list(features)])
    return out


def evaluate_hazard(
    probabilities: pd.DataFrame, labels: pd.DataFrame, label: str, threshold: float
) -> dict[str, float]:
    merged = probabilities.merge(
        labels[["unique_id", "t", label]], on=["unique_id", "t"], how="inner"
    )
    truth = merged[label].astype(int).to_numpy()
    scores = merged["p"].to_numpy()
    positive = truth.sum()
    return {
        "pr_auc": float(average_precision_score(truth, scores))
        if 0 < positive < len(truth)
        else float("nan"),
        "positives": int(positive),
        "n": int(len(truth)),
        "alarm_rate": float((scores >= threshold).mean()),
        "precision_at_threshold": float(truth[scores >= threshold].mean())
        if (scores >= threshold).any()
        else float("nan"),
        "recall_at_threshold": float(truth[scores >= threshold].sum() / positive)
        if positive
        else float("nan"),
    }


def feature_importance(booster, features: Sequence[str]) -> pd.DataFrame:
    gains = booster.feature_importance("gain")
    return (
        pd.DataFrame({"feature": list(features), "gain": gains})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )


def split_by_series(ids: Sequence[str], share: float, seed: int) -> tuple[list[str], list[str]]:
    """Деление по рядам: один и тот же инъецированный слом не попадает в обе части."""
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(ids)))
    mask = rng.random(len(unique)) < share
    return sorted(unique[mask]), sorted(unique[~mask])


def best_threshold(
    probabilities: pd.DataFrame,
    labels: pd.DataFrame,
    label: str,
    grid: Sequence[float] = tuple(np.arange(0.02, 0.61, 0.02)),
) -> tuple[float, float]:
    """Порог тревоги, максимизирующий F1 на обучающей части (на тесте не подбирается)."""
    merged = probabilities.merge(
        labels[["unique_id", "t", label]], on=["unique_id", "t"], how="inner"
    )
    truth = merged[label].astype(int).to_numpy()
    scores = merged["p"].to_numpy()
    best, best_f1 = float(grid[0]), -1.0
    for threshold in grid:
        predicted = scores >= threshold
        if not predicted.any():
            continue
        precision = float(truth[predicted].mean())
        recall = float(truth[predicted].sum() / max(truth.sum(), 1))
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best_f1:
            best, best_f1 = float(threshold), f1
    return best, best_f1


def threshold_for_alarm_rate(probabilities: pd.DataFrame, target_rate: float) -> float:
    """Порог, при котором доля тревог равна `target_rate`.

    Нужен для честного сравнения с онлайн-детектором: модель и детектор ставятся в одинаковые
    условия по числу тревог, и тогда сравнивать можно полноту и упреждение, а не удобство
    выбранного порога. Порог берётся как квантиль распределения вероятностей на валидации.
    """
    scores = probabilities["p"].to_numpy(dtype=float)
    if len(scores) == 0:
        return 1.0
    rate = min(max(float(target_rate), 0.0), 1.0)
    return float(np.quantile(scores, 1.0 - rate))
