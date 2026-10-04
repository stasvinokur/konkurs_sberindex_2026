"""Foundation-модель Chronos-2 (amazon/chronos-2, веса Apache-2.0).

Три режима сравнения:
- `univariate` — zero-shot по каждому ряду отдельно;
- `grouped` — общий проход с `cross_learning=True`: модель делит информацию между рядами
  одного пакета (group attention). Ряды подаются регион за регионом, поэтому в пакет попадают
  соседние МО одного региона, все категории вперемешку; отдельной группировки по категориям
  нет;
- `covariates` — добавлены известные будущие ковариаты: месяц и национальный сезонный индекс
  категории.

Ревизия весов закреплена в `configs/models.yaml` (см. THIRD_PARTY_MODELS.md).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from sbx.core.folds import Fold
from sbx.shell.models.base import register

_PIPELINE_CACHE: dict[tuple[str, str, str], Any] = {}


def load_pipeline(repo_id: str, revision: str, device: str):
    from chronos import Chronos2Pipeline

    key = (repo_id, revision, device)
    if key not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[key] = Chronos2Pipeline.from_pretrained(
            repo_id, revision=revision, device_map=device
        )
    return _PIPELINE_CACHE[key]


def _covariate_frames(
    train: pd.DataFrame, static: pd.DataFrame, seasonal: pd.DataFrame, fold: Fold
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Известные будущие ковариаты: месяц и национальный сезонный индекс категории."""
    index = seasonal.pivot_table(index="category", columns="month", values="index")
    cats = static.set_index("unique_id")["category"]

    def add(frame: pd.DataFrame) -> pd.DataFrame:
        category = cats.reindex(frame["unique_id"]).to_numpy()
        month = frame["ds"].dt.month.to_numpy()
        frame = frame.copy()
        frame["month_sin"] = np.sin(2 * np.pi * month / 12)
        frame["month_cos"] = np.cos(2 * np.pi * month / 12)
        frame["seasonal_index"] = [
            float(index.at[c, m]) if c in index.index else 1.0
            for c, m in zip(category, month, strict=True)
        ]
        return frame

    history = add(train[["unique_id", "ds", "y"]])
    future_ds = pd.date_range(
        fold.cutoff + pd.DateOffset(months=1), periods=fold.horizon, freq="MS"
    )
    future = pd.DataFrame(
        [{"unique_id": uid, "ds": d} for uid in train["unique_id"].unique() for d in future_ds]
    )
    return history, add(future).drop(columns=[c for c in ("y",) if c in future.columns])


def region_ordered(history: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    """Ряды в порядке «регион, ряд, месяц» — для совместного режима.

    Пакет Chronos-2 — это соседние ряды входа в порядке первого появления. Явная сортировка
    по региону не даёт составу пакетов зависеть от того, как устроен идентификатор ряда.
    """
    region = static.set_index("unique_id")["region_code"]
    keyed = history.assign(_region=history["unique_id"].map(region).astype(str))
    ordered = keyed.sort_values(["_region", "unique_id", "ds"], kind="stable")
    return ordered.drop(columns="_region").reset_index(drop=True)


def _predict(
    pipeline,
    history: pd.DataFrame,
    fold: Fold,
    quantiles: Sequence[float],
    future: pd.DataFrame | None,
    cross_learning: bool,
    batch_size: int,
) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds = pipeline.predict_df(
            history,
            future_df=future,
            id_column="unique_id",
            timestamp_column="ds",
            target="y",
            prediction_length=fold.horizon,
            quantile_levels=list(quantiles),
            batch_size=batch_size,
            cross_learning=cross_learning,
            freq="MS",
        )
    return preds


def finetune_pipeline(
    pipeline,
    history: pd.DataFrame,
    fold: Fold,
    cfg: Mapping[str, Any],
):
    """LoRA-дообучение Chronos-2 на обучающей части фолда.

    Дообучение включается только флагом `finetune.enabled` и использует исключительно данные
    до cutoff. На CPU полное дообучение 120M параметров нереально по времени, поэтому режим —
    LoRA с небольшим числом шагов.
    """
    from chronos.chronos2.preprocess import from_data_frame

    params = dict(cfg.get("finetune", {}))
    inputs = from_data_frame(
        history.rename(columns={"unique_id": "item_id", "ds": "timestamp", "y": "target"}),
        target_columns=["target"],
        prediction_length=fold.horizon,
        id_column="item_id",
        timestamp_column="timestamp",
    )
    return pipeline.fit(
        inputs,
        prediction_length=fold.horizon,
        finetune_mode=str(params.get("mode", "lora")),
        learning_rate=float(params.get("learning_rate", 1e-4)),
        num_steps=int(params.get("num_steps", 200)),
        batch_size=int(params.get("batch_size", 32)),
        output_dir=params.get("output_dir"),
        remove_printer_callback=True,
    )


@register("chronos2")
def forecast_chronos2(
    train: pd.DataFrame, fold: Fold, cfg: Mapping[str, Any], context: Mapping[str, Any]
) -> pd.DataFrame:
    models_cfg = context["foundation_models"]["chronos2"]
    pipeline = load_pipeline(
        models_cfg["repo_id"], models_cfg["revision"], str(cfg.get("device", "cpu"))
    )
    quantiles = list(cfg.get("quantiles", [0.1, 0.5, 0.9]))
    batch_size = int(cfg.get("batch_size", 64))
    static = context["static"]
    seasonal = context["seasonal_index"]
    eval_ids = context.get("eval_ids")

    history = train[["unique_id", "ds", "y"]]
    if eval_ids:
        history = history[history["unique_id"].isin(set(eval_ids))]
    # Chronos-2 требует регулярную сетку без пропусков.
    full = history.groupby("unique_id")["ds"].agg(["min", "max", "size"])
    regular = full["size"] == (
        (full["max"].dt.year - full["min"].dt.year) * 12
        + full["max"].dt.month
        - full["min"].dt.month
        + 1
    )
    history = history[history["unique_id"].isin(regular.index[regular])]

    finetuned = None
    if dict(cfg.get("finetune", {})).get("enabled"):
        finetuned = finetune_pipeline(pipeline, history, fold, cfg)

    out = []
    for mode in cfg.get("modes", ["univariate"]):
        if mode == "covariates":
            hist_cov, future_cov = _covariate_frames(history, static, seasonal, fold)
            preds = _predict(pipeline, hist_cov, fold, quantiles, future_cov, False, batch_size)
            name = "Chronos-2 (covariates)"
        elif mode == "finetuned":
            if finetuned is None:
                continue
            preds = _predict(finetuned, history, fold, quantiles, None, False, batch_size)
            name = "Chronos-2 (fine-tuned)"
        else:
            cross = mode == "grouped"
            series = region_ordered(history, static) if cross else history
            preds = _predict(pipeline, series, fold, quantiles, None, cross, batch_size)
            name = "Chronos-2" if mode == "univariate" else "Chronos-2 (grouped)"
        frame = preds.rename(columns={"predictions": "y_hat"})
        frame["model"] = name
        rename = {str(q): f"q_{q}" for q in quantiles}
        frame = frame.rename(columns=rename)
        if "y_hat" not in frame.columns and f"q_{0.5}" in frame.columns:
            frame["y_hat"] = frame[f"q_{0.5}"]
        cols = [
            "unique_id",
            "ds",
            "model",
            "y_hat",
            *[f"q_{q}" for q in quantiles if f"q_{q}" in frame],
        ]
        frame = frame[cols]
        for col in cols[3:]:
            frame[col] = frame[col].clip(lower=0.0)
        out.append(frame)
    return pd.concat(out, ignore_index=True)
