"""Контракт прогнозных моделей и подготовка OOF-таблицы (чистые функции)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from sbx.core.folds import Fold, step_of

FORECAST_COLUMNS = ["unique_id", "ds", "model", "y_hat"]


class ForecastContractError(ValueError):
    """Прогноз нарушает контракт."""


def validate_forecast(
    preds: pd.DataFrame, expected_ids: Sequence[str], fold: Fold, allow_missing: bool = False
) -> None:
    missing_cols = set(FORECAST_COLUMNS) - set(preds.columns)
    if missing_cols:
        raise ForecastContractError(f"нет колонок {sorted(missing_cols)}")
    if preds[FORECAST_COLUMNS].isna().any().any():
        raise ForecastContractError("пропуски в обязательных колонках прогноза")
    if not np.isfinite(preds["y_hat"]).all():
        raise ForecastContractError("нечисловые значения прогноза")
    if (preds["y_hat"] < 0).any():
        raise ForecastContractError("отрицательный прогноз расходов")
    if preds.duplicated(["unique_id", "ds", "model"]).any():
        raise ForecastContractError("дубли (unique_id, ds, model)")
    horizon_ok = (preds["ds"] > fold.cutoff) & (preds["ds"] <= fold.test_end)
    if not horizon_ok.all():
        raise ForecastContractError("прогноз вне горизонта фолда")
    if not allow_missing:
        missing = set(expected_ids) - set(preds["unique_id"])
        if missing:
            raise ForecastContractError(f"нет прогноза для {len(missing)} рядов")


def build_oof(
    preds: pd.DataFrame, test: pd.DataFrame, fold: Fold, static: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Склейка прогноза с фактом в OOF-таблицу протокола оценки."""
    oof = test.merge(preds, on=["unique_id", "ds"], how="inner")
    oof["fold"] = fold.name
    oof["cutoff"] = fold.cutoff
    # Горизонт фолда, а не шаг: модели сравниваются внутри горизонта, потому что при h=12
    # часть из них неопределима и общее выравнивание выбросило бы весь горизонт.
    oof["horizon"] = fold.horizon
    oof["step"] = step_of(oof["ds"], fold.cutoff)
    if static is not None:
        cols = [c for c in ("category", "size_group", "region_code") if c in static.columns]
        oof = oof.drop(columns=[c for c in cols if c in oof.columns]).merge(
            static[["unique_id", *cols]], on="unique_id", how="left"
        )
    keep = [
        "unique_id",
        "ds",
        "fold",
        "cutoff",
        "horizon",
        "step",
        "model",
        "y",
        "y_hat",
        *[c for c in oof.columns if c.startswith("q_")],
        *[c for c in ("category", "size_group", "region_code") if c in oof.columns],
    ]
    return oof[keep].sort_values(["model", "unique_id", "ds"]).reset_index(drop=True)


def stratified_series_sample(
    static: pd.DataFrame, n: int, seed: int, strata: Sequence[str] = ("category", "size_group")
) -> list[str]:
    """Стратифицированная подвыборка рядов: пропорционально стратам, детерминированно по seed."""
    if n >= len(static):
        return sorted(static["unique_id"])
    rng = np.random.default_rng(seed)
    df = static.copy()
    df["_stratum"] = df[list(strata)].astype(str).agg("|".join, axis=1)
    shares = df["_stratum"].value_counts(normalize=True)
    picked: list[str] = []
    for stratum, share in shares.items():
        pool = df.loc[df["_stratum"] == stratum, "unique_id"].sort_values().to_numpy()
        take = min(len(pool), int(round(share * n)))
        picked += list(rng.choice(pool, size=take, replace=False))
    return sorted(picked)
