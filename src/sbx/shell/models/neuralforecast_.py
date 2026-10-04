"""Глобальные нейросетевые модели NHITS и NBEATSx (neuralforecast, Apache-2.0).

Обучение идёт сразу на всей панели (глобальная модель), вход — `input_size` последних месяцев.
NBEATSx дополнительно получает известные будущие ковариаты: гармоники месяца и национальный
сезонный индекс категории. Риск переобучения на 24 точках контролируется ранней остановкой по
валидационной части обучающего окна (данные после cutoff не используются).
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from sbx.core.folds import Fold
from sbx.shell.models.base import register

FUTR_EXOG = ["month_sin", "month_cos", "seasonal_index"]

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def _exog(frame: pd.DataFrame, static: pd.DataFrame, seasonal: pd.DataFrame) -> pd.DataFrame:
    index = seasonal.pivot_table(index="category", columns="month", values="index")
    cats = static.set_index("unique_id")["category"]
    out = frame.copy()
    month = out["ds"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    category = cats.reindex(out["unique_id"]).to_numpy()
    out["seasonal_index"] = [
        float(index.at[c, m]) if c in index.index else 1.0
        for c, m in zip(category, month, strict=True)
    ]
    return out


@register("neural")
def forecast_neural(
    train: pd.DataFrame, fold: Fold, cfg: Mapping[str, Any], context: Mapping[str, Any]
) -> pd.DataFrame:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS, NBEATSx

    static = context["static"]
    seasonal = context["seasonal_index"]
    input_size = int(cfg.get("input_size", 12))
    max_steps = int(cfg.get("max_steps", 300))
    # neuralforecast требует val_size ≥ горизонта (иначе ранняя остановка невозможна).
    val_size = max(int(cfg.get("val_size", 3)), fold.horizon)
    seed = int(cfg.get("seed", 0))

    history = train[["unique_id", "ds", "y"]].copy()
    # Нейросетям нужна регулярная сетка без пропусков и история длиннее окна входа.
    stats = history.groupby("unique_id")["ds"].agg(["min", "max", "size"])
    span = (
        (stats["max"].dt.year - stats["min"].dt.year) * 12
        + stats["max"].dt.month
        - stats["min"].dt.month
        + 1
    )
    regular = stats["size"] == span
    # На раннем фолде истории всего 12 месяцев: окно входа подгоняется под доступную длину,
    # иначе не осталось бы ни одного ряда для обучения.
    longest = int(stats.loc[regular, "size"].max()) if regular.any() else 0
    input_size = min(input_size, longest - fold.horizon - val_size)
    if input_size < int(cfg.get("min_input_size", 6)):
        return pd.DataFrame(columns=["unique_id", "ds", "model", "y_hat"])
    ok = regular & (stats["size"] >= input_size + fold.horizon + val_size)
    history = history[history["unique_id"].isin(ok.index[ok])]
    if history.empty:
        return pd.DataFrame(columns=["unique_id", "ds", "model", "y_hat"])
    history = _exog(history, static, seasonal)

    future_ds = pd.date_range(
        fold.cutoff + pd.DateOffset(months=1), periods=fold.horizon, freq="MS"
    )
    future = pd.DataFrame(
        [{"unique_id": uid, "ds": d} for uid in history["unique_id"].unique() for d in future_ds]
    )
    future = _exog(future, static, seasonal)

    common = {
        "h": fold.horizon,
        "input_size": input_size,
        "max_steps": max_steps,
        "early_stop_patience_steps": int(cfg.get("patience", 5)),
        "val_check_steps": int(cfg.get("val_check_steps", 25)),
        "scaler_type": "standard",
        "random_seed": seed,
        "enable_progress_bar": False,
        "logger": False,
        "accelerator": str(cfg.get("accelerator", "cpu")),
    }
    # NHITS понижает частоту внутри стеков; коэффициенты по умолчанию (2, 1, 1) требуют
    # горизонта не меньше самого большого из них, иначе библиотека падает с
    # «Horizon h=1 incompatible with seasonality or trend in stacks». На h=1 стеки
    # вырождаются в единичные — это не подбор гиперпараметра, а условие применимости.
    downsample = [max(1, min(factor, fold.horizon)) for factor in (2, 1, 1)]
    # NBEATSx по умолчанию собран из стеков identity + trend + seasonality. Стек сезонности
    # раскладывает горизонт по гармоникам и на h=1 неопределим: одной точкой период не задать.
    # Для h=1 берём только identity-стеки — это условие применимости, а не подбор архитектуры.
    stacks = ["identity"] * 3 if fold.horizon < 2 else ["identity", "trend", "seasonality"]
    models = [
        NHITS(n_freq_downsample=downsample, **common),
        NBEATSx(stack_types=stacks, futr_exog_list=FUTR_EXOG, **common),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nf = NeuralForecast(models=models, freq="MS")
        nf.fit(df=history, val_size=val_size, verbose=False)
        preds = nf.predict(futr_df=future)

    preds = preds.reset_index() if "unique_id" not in preds.columns else preds
    long = preds.melt(id_vars=["unique_id", "ds"], var_name="model", value_name="y_hat")
    long["model"] = long["model"].replace({"NHITS": "NHITS", "NBEATSx": "NBEATSx"})
    long["y_hat"] = long["y_hat"].clip(lower=0.0)
    eval_ids = context.get("eval_ids")
    if eval_ids:
        long = long[long["unique_id"].isin(set(eval_ids))]
    return long.dropna(subset=["y_hat"])
