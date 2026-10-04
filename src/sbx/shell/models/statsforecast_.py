"""Статистические бейзлайны через statsforecast (Naive, SeasonalNaive, ETS, Theta, ARIMA)."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import pandas as pd

from sbx.core.folds import Fold
from sbx.shell.models.base import register

MODELS = ("Naive", "SeasonalNaive", "AutoETS", "AutoTheta", "AutoARIMA")


def _build(names: tuple[str, ...], season_length: int):
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, Naive, SeasonalNaive

    factory = {
        "Naive": lambda: Naive(),
        "SeasonalNaive": lambda: SeasonalNaive(season_length=season_length),
        "AutoETS": lambda: AutoETS(season_length=season_length),
        "AutoTheta": lambda: AutoTheta(season_length=season_length),
        "AutoARIMA": lambda: AutoARIMA(season_length=season_length),
    }
    return [factory[n]() for n in names]


@register("stats")
def forecast_stats(
    train: pd.DataFrame, fold: Fold, cfg: Mapping[str, Any], context: Mapping[str, Any]
) -> pd.DataFrame:
    from statsforecast import StatsForecast

    names = tuple(cfg.get("models", MODELS))
    season_length = int(cfg.get("season_length", 12))
    df = train[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])
    eval_ids = context.get("eval_ids")
    if eval_ids:
        df = df[df["unique_id"].isin(set(eval_ids))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from statsforecast.models import Naive

        # У части рядов всего 6–11 месяцев истории: AutoETS/AutoTheta на таких рядах падают,
        # поэтому задан запасной Naive (statsforecast применяет его при ошибке модели).
        sf = StatsForecast(
            models=_build(names, season_length),
            freq="MS",
            n_jobs=int(cfg.get("n_jobs", -1)),
            fallback_model=Naive(),
        )
        fc = sf.forecast(df=df, h=fold.horizon)
    long = fc.melt(id_vars=["unique_id", "ds"], var_name="model", value_name="y_hat")
    long["y_hat"] = long["y_hat"].clip(lower=0.0)
    return long.dropna(subset=["y_hat"])
