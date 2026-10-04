"""Метрики качества прогноза на единой OOF-таблице.

OOF-контракт: колонки `unique_id, ds, fold, cutoff, horizon, step, model, y, y_hat`, опционально
квантили `q_<level>` (например, `q_0.1`, `q_0.9`) и статические срезы (`category`, `size_group`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

OOF_COLUMNS = ["unique_id", "ds", "fold", "cutoff", "horizon", "step", "model", "y", "y_hat"]


def mae(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(y_hat))))


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(y_hat)) ** 2)))


def wape(y: np.ndarray, y_hat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    denom = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - np.asarray(y_hat))) / denom) if denom > 0 else float("nan")


def r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = np.sum((y - np.asarray(y_hat)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def pinball(y: np.ndarray, q_hat: np.ndarray, level: float) -> float:
    diff = np.asarray(y, dtype=float) - np.asarray(q_hat, dtype=float)
    return float(np.mean(np.maximum(level * diff, (level - 1.0) * diff)))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def mase_scales(
    train: pd.DataFrame, seasonal_index: pd.DataFrame, category_col: str = "category"
) -> pd.Series:
    """Масштаб MASE ряда: средняя ошибка «наивного прогноза с сезонной поправкой» на train.

    Прогноз: ŷ_t = y_{t-1} · S(c, m_t) / S(c, m_{t-1}), где S — национальный сезонный индекс
    категории c. У рядов МО всего 2 года, поэтому классический сезонный наивный масштаб
    (лаг 12) почти не определён.
    """
    s = seasonal_index.set_index(["category", "month"])["index"]
    df = train.sort_values(["unique_id", "ds"]).copy()
    df["month"] = df["ds"].dt.month
    df["s"] = [s.get((c, m), 1.0) for c, m in zip(df[category_col], df["month"], strict=True)]
    prev_y = df.groupby("unique_id")["y"].shift(1)
    prev_s = df.groupby("unique_id")["s"].shift(1)
    prev_ds = df.groupby("unique_id")["ds"].shift(1)
    consecutive = (df["ds"] - pd.DateOffset(months=1)) == prev_ds
    naive = prev_y * df["s"] / prev_s
    err = (df["y"] - naive).abs().where(consecutive)
    return err.groupby(df["unique_id"]).mean().rename("mase_scale")


def _metric_row(g: pd.DataFrame, quantiles: Sequence[float]) -> dict[str, float]:
    row = {
        "n": int(len(g)),
        "mae": mae(g["y"], g["y_hat"]),
        "rmse": rmse(g["y"], g["y_hat"]),
        "wape": wape(g["y"], g["y_hat"]),
        "r2_pooled": r2(g["y"], g["y_hat"]),
    }
    per_series = g.groupby("unique_id")
    series_mae = per_series.apply(lambda s: mae(s["y"], s["y_hat"]), include_groups=False)
    row["mae_macro"] = float(series_mae.mean())
    series_r2 = per_series.apply(
        lambda s: r2(s["y"], s["y_hat"]) if len(s) >= 3 else np.nan, include_groups=False
    )
    row["r2_per_series_median"] = (
        float(np.nanmedian(series_r2)) if series_r2.notna().any() else float("nan")
    )
    if "mase_scale" in g:
        ratio = np.abs(g["y"] - g["y_hat"]) / g["mase_scale"]
        row["mase"] = float(ratio[np.isfinite(ratio)].groupby(g["unique_id"]).mean().mean())
    for level in quantiles:
        col = f"q_{level}"
        if col in g and g[col].notna().all():
            row[f"pinball_{level}"] = pinball(g["y"], g[col], level)
    lo = f"q_{min(quantiles)}" if quantiles else ""
    hi = f"q_{max(quantiles)}" if quantiles else ""
    if quantiles and lo in g and hi in g and g[lo].notna().all() and g[hi].notna().all():
        row[f"coverage_{min(quantiles)}_{max(quantiles)}"] = coverage(g["y"], g[lo], g[hi])
    return row


def summarize(
    oof: pd.DataFrame,
    slices: Sequence[str] = ("category", "size_group", "step", "fold"),
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    """Таблица метрик: срез `overall` и срезы по каждой колонке из `slices`."""
    missing = set(OOF_COLUMNS) - set(oof.columns)
    if missing:
        raise ValueError(f"OOF без колонок {sorted(missing)}")
    rows = []
    for model, g in oof.groupby("model", sort=True):
        rows.append(
            {"model": model, "slice": "overall", "value": "all", **_metric_row(g, quantiles)}
        )
        for col in slices:
            if col not in g:
                continue
            for value, gg in g.groupby(col, sort=True):
                rows.append(
                    {
                        "model": model,
                        "slice": col,
                        "value": str(value),
                        **_metric_row(gg, quantiles),
                    }
                )
    return pd.DataFrame(rows)


def leaderboard(summary: pd.DataFrame) -> pd.DataFrame:
    """Срез overall, отсортированный по MAE."""
    board = summary[summary["slice"] == "overall"].drop(columns=["slice", "value"])
    return board.sort_values("mae").reset_index(drop=True)
