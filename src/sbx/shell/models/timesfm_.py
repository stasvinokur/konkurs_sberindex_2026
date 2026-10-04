"""Вторая foundation-модель: TimesFM 2.5 200M (google/timesfm-2.5-200m-pytorch, Apache-2.0).

Веса 3.0 использовать нельзя (non-commercial лицензия, см. THIRD_PARTY_MODELS.md) — загружается
только класс `TimesFM_2p5_200M_torch` с закреплённой ревизией весов 2.5.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from sbx.core.folds import Fold
from sbx.shell.models.base import register

_MODEL_CACHE: dict[tuple[str, str, int, int], Any] = {}
QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def load_model(
    repo_id: str, revision: str, max_context: int, max_horizon: int, per_core_batch_size: int = 32
):
    """Загрузка и компиляция модели.

    `per_core_batch_size` по умолчанию в `ForecastConfig` равен 1 — на 1200 рядах это часы
    инференса, поэтому батч задаётся явно. Число потоков torch выставляется по числу ядер.
    """
    import timesfm
    import torch

    if "timesfm-3" in repo_id:
        raise ValueError("веса TimesFM 3.0 запрещены лицензией конкурса; используйте 2.5")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    key = (repo_id, revision, max_context, max_horizon, per_core_batch_size)
    if key not in _MODEL_CACHE:
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo_id, revision=revision)
        model.compile(
            timesfm.ForecastConfig(
                max_context=max_context,
                max_horizon=max_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                fix_quantile_crossing=True,
                infer_is_positive=True,
                per_core_batch_size=per_core_batch_size,
            )
        )
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


@register("timesfm25")
def forecast_timesfm(
    train: pd.DataFrame, fold: Fold, cfg: Mapping[str, Any], context: Mapping[str, Any]
) -> pd.DataFrame:
    spec = context["foundation_models"]["timesfm25"]
    model = load_model(
        spec["repo_id"],
        spec["revision"],
        int(cfg.get("max_context", 64)),
        max(int(cfg.get("max_horizon", 12)), fold.horizon),
        int(cfg.get("per_core_batch_size", 32)),
    )
    eval_ids = context.get("eval_ids")
    history = train[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])
    if eval_ids:
        history = history[history["unique_id"].isin(set(eval_ids))]

    ids, inputs = [], []
    for uid, g in history.groupby("unique_id", sort=True):
        ids.append(uid)
        inputs.append(g["y"].to_numpy(dtype=np.float32))

    batch_size = int(cfg.get("batch_size", 256))
    points, quantiles = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for start in range(0, len(inputs), batch_size):
            chunk = inputs[start : start + batch_size]
            point, quants = model.forecast(horizon=fold.horizon, inputs=chunk)
            points.append(np.asarray(point))
            quantiles.append(np.asarray(quants))
    point = np.concatenate(points, axis=0)
    quants = np.concatenate(quantiles, axis=0)

    future_ds = pd.date_range(
        fold.cutoff + pd.DateOffset(months=1), periods=fold.horizon, freq="MS"
    )
    rows = []
    wanted = list(cfg.get("quantiles", [0.1, 0.5, 0.9]))
    for i, uid in enumerate(ids):
        for h, ds in enumerate(future_ds):
            row = {"unique_id": uid, "ds": ds, "model": "TimesFM-2.5", "y_hat": float(point[i, h])}
            for q in wanted:
                if q in QUANTILE_LEVELS:
                    # В выходе квантильной головы первый столбец — среднее, далее 0.1…0.9.
                    idx = QUANTILE_LEVELS.index(q) + 1
                    if idx < quants.shape[-1]:
                        row[f"q_{q}"] = float(quants[i, h, idx])
            rows.append(row)
    out = pd.DataFrame(rows)
    for col in [c for c in out.columns if c == "y_hat" or c.startswith("q_")]:
        out[col] = out[col].clip(lower=0.0)
    return out
