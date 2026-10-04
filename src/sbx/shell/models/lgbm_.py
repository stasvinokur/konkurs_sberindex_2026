"""Глобальная модель LightGBM на всей панели (прямой прогноз шага h).

Обучающая выборка — все (ряд, origin ≤ cutoff − h, шаг h) со всей панели: модель учится сразу
на тысячах рядов, что и требуется при 24 точках истории у отдельного МО. Цель — логарифм
отношения `y_{t+h}/y_t`, поэтому уровень МО не влияет на обучение.

Гиперпараметры подбираются Optuna по внутренней валидации: последние `inner_origins` origin-ов
до cutoff. Никакие данные после cutoff при этом не используются.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from sbx.core.features.asof import ExogenousBlock
from sbx.core.features.lags import feature_columns, make_features, predictions_from_ratio
from sbx.core.folds import Fold
from sbx.shell.models.base import register


def _exogenous_blocks(cfg: Mapping[str, Any], context: Mapping[str, Any]) -> list[ExogenousBlock]:
    """Блоки внешних признаков согласно конфигурации абляции (`feature_blocks`).

    Привязку ко времени задаёт сам блок: известное заранее берётся на целевой месяц, остальное
    — по состоянию на момент прогноза (`sbx.core.features.asof`).
    """
    available = context.get("feature_blocks", {})
    return [available[name] for name in cfg.get("feature_blocks", []) if name in available]


DEFAULT_PARAMS = {
    "objective": "l1",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
}


def training_origins(
    panel: pd.DataFrame, cutoff: pd.Timestamp, horizon: int, min_history: int
) -> list[pd.Timestamp]:
    months = pd.DatetimeIndex(sorted(panel["ds"].unique()))
    first = months[0] + pd.DateOffset(months=min_history - 1)
    return [m for m in months if first <= m <= cutoff - pd.DateOffset(months=horizon)]


def _fit(train_x: pd.DataFrame, train_y: np.ndarray, params: Mapping[str, Any], rounds: int):
    import lightgbm as lgb

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
        return lgb.train({**DEFAULT_PARAMS, **params}, dataset, num_boost_round=rounds)


def _tune(
    features: pd.DataFrame, cols: list[str], cutoff: pd.Timestamp, cfg: Mapping[str, Any]
) -> tuple[dict[str, Any], int, float]:
    """Подбор гиперпараметров по внутренней валидации (последние origin-ы до cutoff)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    inner_origins = int(cfg.get("inner_origins", 3))
    origins = sorted(features["origin"].unique())
    split_at = origins[-inner_origins] if len(origins) > inner_origins else origins[-1]
    train = features[features["origin"] < split_at]
    valid = features[features["origin"] >= split_at]
    if train.empty or valid.empty:
        return {}, int(cfg.get("num_boost_round", 400)), float("nan")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 400, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True),
        }
        rounds = trial.suggest_int("num_boost_round", 150, 700, step=50)
        booster = _fit(train[cols], train["target"].to_numpy(), params, rounds)
        preds = predictions_from_ratio(valid, booster.predict(valid[cols]))
        return float(np.mean(np.abs(valid["y"].to_numpy() - preds["y_hat"].to_numpy())))

    sampler = optuna.samplers.TPESampler(seed=int(cfg.get("seed", 0)))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=int(cfg.get("n_trials", 15)), show_progress_bar=False)
    best = dict(study.best_params)
    rounds = int(best.pop("num_boost_round"))
    return best, rounds, float(study.best_value)


@register("lgbm")
def forecast_lgbm(
    train: pd.DataFrame, fold: Fold, cfg: Mapping[str, Any], context: Mapping[str, Any]
) -> pd.DataFrame:
    static = context["static"]
    seasonal = context["seasonal_index"]
    min_history = int(cfg.get("min_history", 7))
    origins = training_origins(train, fold.cutoff, fold.horizon, min_history)
    exogenous = _exogenous_blocks(cfg, context)
    features = make_features(train, static, seasonal, origins, fold.horizon, exogenous=exogenous)
    if features.empty:
        # Direct-модель требует, чтобы цель y_{t+h} лежала внутри train: нужен обучающий origin
        # не позже cutoff − h. При h=12 и панели с 2023-01 таких origin нет вовсе.
        # Это не сбой прогона, а свойство данных: на 24 месяцах истории горизонт 12
        # для direct-модели неопределим. Возвращаем пустой прогноз, чтобы модель осталась
        # в сравнении на остальных горизонтах, а не исчезла из leaderboard целиком.
        context.setdefault("lgbm_skipped", []).append(
            {"fold": fold.name, "horizon": fold.horizon, "training_origins": len(origins)}
        )
        return pd.DataFrame(columns=["unique_id", "ds", "model", "y_hat"])
    cols = feature_columns(features)

    tuning = context.setdefault("lgbm_tuning", {})
    if cfg.get("tune", True):
        params, rounds, score = _tune(features, cols, fold.cutoff, cfg)
    else:
        params, rounds, score = {}, int(cfg.get("num_boost_round", 400)), float("nan")
    tuning[fold.name] = {"params": params, "num_boost_round": rounds, "inner_mae": score}

    booster = _fit(features[cols], features["target"].to_numpy(), params, rounds)
    context.setdefault("lgbm_importance", {})[fold.name] = dict(
        zip(cols, booster.feature_importance("gain").tolist(), strict=True)
    )

    eval_ids = context.get("eval_ids")
    predict_panel = train[train["unique_id"].isin(set(eval_ids))] if eval_ids else train
    future = make_features(
        predict_panel,
        static,
        seasonal,
        [fold.cutoff],
        fold.horizon,
        require_target=False,
        exogenous=exogenous,
    )
    future = future[future["ds"] > fold.cutoff]
    out = predictions_from_ratio(future, booster.predict(future[cols]))
    out["model"] = str(cfg.get("model_name", "LightGBM"))
    return out[["unique_id", "ds", "model", "y_hat"]]
