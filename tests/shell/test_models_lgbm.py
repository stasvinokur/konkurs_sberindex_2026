"""Адаптер LightGBM на маленькой панели: что он принимает и что возвращает."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sbx.core.folds import Fold
from sbx.shell.models.lgbm_ import forecast_lgbm

CUTOFF = pd.Timestamp("2024-06-01")


def _panel(series: int = 60) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    rng = np.random.default_rng(0)
    rows = []
    for i in range(series):
        level = 1000.0 * (1 + i % 5)
        for t, month in enumerate(months):
            rows.append(
                {
                    "unique_id": f"t{i:02d}__total",
                    "ds": month,
                    "y": level * (1 + 0.01 * t) * (1 + 0.02 * rng.standard_normal()),
                }
            )
    panel = pd.DataFrame(rows)
    static = pd.DataFrame(
        {
            "unique_id": [f"t{i:02d}__total" for i in range(series)],
            "category": "Все категории",
            "category_slug": "total",
            "region_code": [f"{10 + i % 3}" for i in range(series)],
            "federal_district": "Центральный",
            "mo_kind": "municipal_district",
            "size_level": [np.log(1000.0 * (1 + i % 5)) for i in range(series)],
            "size_group": pd.array([1 + i % 5 for i in range(series)], dtype="Int64"),
        }
    )
    seasonal = pd.DataFrame({"category": "Все категории", "month": range(1, 13), "index": 1.0})
    return panel, static, seasonal


def test_series_without_a_size_is_still_forecast() -> None:
    """У территории, чей ряд начался после момента оценки размера, размера нет: модель обязана
    принять пропуск в признаке, а не потерять ряд или упасть."""
    panel, static, seasonal = _panel()
    sizeless = ["t00__total", "t07__total"]
    static.loc[static["unique_id"].isin(sizeless), "size_level"] = np.nan
    static.loc[static["unique_id"].isin(sizeless), "size_group"] = pd.NA
    fold = Fold("h3_2024-06", CUTOFF, 3, "main")
    context = {"static": static, "seasonal_index": seasonal}
    cfg = {"tune": False, "num_boost_round": 20, "seed": 0}
    out = forecast_lgbm(panel[panel["ds"] <= CUTOFF], fold, cfg, context)

    assert set(out["unique_id"]) == set(static["unique_id"]), "все ряды получили прогноз"
    assert out.groupby("unique_id")["ds"].count().eq(3).all(), "по три шага на ряд"
    assert np.isfinite(out["y_hat"]).all() and (out["y_hat"] > 0).all()
    assert out["ds"].min() > CUTOFF
    used = context["lgbm_importance"]["h3_2024-06"]
    assert "size_level" in used and "size_group" in used, "признак размера остаётся в модели"
