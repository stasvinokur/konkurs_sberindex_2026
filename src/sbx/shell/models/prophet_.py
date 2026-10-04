"""Prophet как честный бейзлайн сравнения.

Prophet обучается отдельно на каждом ряду и стоит около 5 секунд на подгонку (Stan), поэтому
он считается на стратифицированной подвыборке рядов; остальные модели оцениваются на той же
подвыборке для сопоставимости (см. `sbx.core.forecasting.stratified_series_sample`).

`changepoint_prior_scale` подбирается внутренней кросс-валидацией **только по данным до cutoff**:
последние `inner_horizon` месяцев обучающей части используются как валидация.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from joblib import Parallel, delayed

from sbx.core.folds import Fold
from sbx.shell.models.base import register

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)


def _fit_predict_one(
    history: pd.DataFrame, horizon: int, params: Mapping[str, Any]
) -> pd.DataFrame:
    from prophet import Prophet

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Prophet(
            growth=str(params.get("growth", "linear")),
            seasonality_mode=params.get("seasonality_mode", "multiplicative"),
            yearly_seasonality=int(params.get("yearly_fourier_order", 4)),
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=float(params.get("changepoint_prior_scale", 0.05)),
            seasonality_prior_scale=float(params.get("seasonality_prior_scale", 10.0)),
            changepoint_range=float(params.get("changepoint_range", 0.8)),
            uncertainty_samples=int(params.get("uncertainty_samples", 0)),
        )
        if params.get("country_holidays"):
            model.add_country_holidays(country_name=str(params["country_holidays"]))
        model.fit(history[["ds", "y"]])
        future = model.make_future_dataframe(periods=horizon, freq="MS")
        forecast = model.predict(future)
    return forecast[["ds", "yhat"]].tail(horizon)


def _series_forecast(
    uid: str, history: pd.DataFrame, horizon: int, params: Mapping[str, Any]
) -> pd.DataFrame:
    out = _fit_predict_one(history, horizon, params)
    out.insert(0, "unique_id", uid)
    return out


def _inner_horizon(cfg: Mapping[str, Any], fold: Fold) -> int:
    """Горизонт внутренней валидации при подборе гиперпараметров.

    Берём горизонт фолда, но зажимаем с двух сторон. Верхняя граница — потому что при слишком
    длинной валидации на 12–24 точках истории ни один ряд не проходит отбор в `_tune`. Нижняя —
    потому что на h=1 валидация из одного месяца подбирает параметры под одну точку: прогон
    показал MAE 1439 против 853 на h=3, то есть более короткий горизонт получался «сложнее»
    длинного, чего быть не может.
    """
    floor = int(cfg.get("inner_horizon_min", 3))
    ceiling = int(cfg.get("inner_horizon_max", 6))
    return max(floor, min(ceiling, fold.horizon))


def _tune(
    train: pd.DataFrame,
    grid: Sequence[Mapping[str, Any]],
    base_params: Mapping[str, Any],
    n_series: int,
    inner_horizon: int,
    n_jobs: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Выбор гиперпараметров Prophet по внутренней валидации на данных до cutoff.

    Последние `inner_horizon` месяцев обучающей части играют роль валидации; данные после
    cutoff не используются. Сетка включает `growth` (в том числе flat), режим сезонности,
    порядок Фурье и changepoint_prior_scale: на 12–24 точках линейный тренд Prophet легко
    улетает, поэтому честное сравнение требует дать модели выбрать «плоский» тренд.
    """
    ids = sorted(train["unique_id"].unique())
    sample = pd.Series(ids).sample(n=min(n_series, len(ids)), random_state=seed)
    histories = [train[train["unique_id"] == uid].sort_values("ds") for uid in sample]

    # На ранних origin история короче горизонта плюс запас, и при длинном окне валидации
    # ни один ряд не проходит отбор. Молча вернуть первый элемент сетки нельзя: это
    # ненастроенный Prophet, а сравнение с ненастроенным Prophet ничего не доказывает.
    # Поэтому окно валидации сокращается, пока не найдётся хотя бы несколько рядов.
    jobs: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    while inner_horizon >= 1:
        jobs = [
            (hist.iloc[:-inner_horizon], hist.iloc[-inner_horizon:])
            for hist in histories
            if len(hist) > inner_horizon + 6
        ]
        if len(jobs) >= 5:
            break
        inner_horizon -= 1
    scores = []
    for candidate in grid:
        params = {**base_params, **candidate}
        preds = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_fit_predict_one)(inner_train, inner_horizon, params) for inner_train, _ in jobs
        )
        errors = []
        for (_, inner_test), pred in zip(jobs, preds, strict=True):
            merged = inner_test.merge(pred, on="ds", how="inner")
            errors.append(float((merged["y"] - merged["yhat"]).abs().mean()))
        scores.append({**candidate, "inner_mae": float(pd.Series(errors).mean())})
    best = min(scores, key=lambda row: row["inner_mae"])
    return {k: v for k, v in best.items() if k != "inner_mae"}, scores


@register("prophet")
def forecast_prophet(
    train: pd.DataFrame, fold: Fold, cfg: Mapping[str, Any], context: Mapping[str, Any]
) -> pd.DataFrame:
    base_params = dict(cfg.get("params", {}))
    grid = [dict(item) for item in cfg.get("grid", [])]
    n_jobs = int(cfg.get("n_jobs", -1))
    eval_ids = context.get("eval_ids")
    if len(grid) > 1:
        tune_pool = train[train["unique_id"].isin(set(eval_ids or train["unique_id"]))]
        best, scores = _tune(
            tune_pool,
            grid,
            base_params,
            int(cfg.get("tune_series", 40)),
            _inner_horizon(cfg, fold),
            n_jobs,
            int(cfg.get("seed", 0)),
        )
    else:
        best, scores = (grid[0] if grid else {}), []
    params = {**base_params, **best}
    context.setdefault("prophet_tuning", {})[fold.name] = {"best": best, "scores": scores}

    ids = (
        sorted(set(eval_ids) & set(train["unique_id"]))
        if eval_ids
        else sorted(train["unique_id"].unique())
    )
    histories = {uid: g.sort_values("ds") for uid, g in train.groupby("unique_id")}
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_series_forecast)(uid, histories[uid], fold.horizon, params) for uid in ids
    )
    out = pd.concat(results, ignore_index=True).rename(columns={"yhat": "y_hat"})
    out["model"] = "Prophet"
    out["y_hat"] = out["y_hat"].clip(lower=0.0)
    return out[["unique_id", "ds", "model", "y_hat"]]
