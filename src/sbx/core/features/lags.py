"""Прямые (direct) признаки глобальной модели: всё считается на момент origin.

Обучающая строка — (ряд, origin t, шаг h): признаки построены только по наблюдениям до t
включительно, цель — log(y_{t+h} / y_t). Логарифм отношения делает цель безразмерной, что важно
при разбросе уровней МО в сотни раз.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from sbx.core.features.asof import ExogenousBlock

LAGS = (1, 2, 3, 6, 12)
WINDOWS = (3, 6, 12)
STATIC_COLUMNS = (
    "category_slug",
    "region_code",
    "federal_district",
    "mo_kind",
    "size_group",
    "size_level",
)


def _series_matrix(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Широкая матрица «ряд × месяц» с пропусками для отсутствующих месяцев."""
    wide = panel.pivot(index="unique_id", columns="ds", values="y").sort_index(axis=1)
    return wide, wide.columns


def make_features(
    panel: pd.DataFrame,
    static: pd.DataFrame,
    seasonal_index: pd.DataFrame,
    origins: Sequence[pd.Timestamp],
    horizon: int,
    lags: Sequence[int] = LAGS,
    windows: Sequence[int] = WINDOWS,
    require_target: bool = True,
    exogenous: Sequence[ExogenousBlock] = (),
) -> pd.DataFrame:
    """Таблица признаков для всех (ряд, origin, шаг).

    `require_target=True` оставляет только строки с известным фактом `y_{t+h}` (обучение);
    `False` — строки для предсказания.

    `exogenous` — блоки внешних признаков. Блок, известный заранее, присоединяется по
    целевому месяцу; блок прошлого — по origin, то есть на всех шагах прогноза стоит то, что
    было известно к моменту прогноза (см. `sbx.core.features.asof`).
    """
    wide, months = _series_matrix(panel)
    month_pos = {m: i for i, m in enumerate(months)}
    values = wide.to_numpy(dtype=float)
    ids = wide.index.to_numpy()
    seasonal = seasonal_index.pivot_table(index="category", columns="month", values="index")
    cat_by_id = static.set_index("unique_id")["category"].reindex(ids)

    rows = []
    for origin in origins:
        if origin not in month_pos:
            continue
        t = month_pos[origin]
        base = values[:, t]
        history_ok = np.isfinite(base)
        feat = {
            "unique_id": ids,
            "origin": origin,
            "y_last": base,
        }
        for lag in lags:
            idx = t - lag
            feat[f"lag_{lag}"] = values[:, idx] / base if idx >= 0 else np.full(len(ids), np.nan)
        for window in windows:
            lo = max(0, t - window + 1)
            block = values[:, lo : t + 1]
            with np.errstate(invalid="ignore"):
                feat[f"roll_mean_{window}"] = np.nanmean(block, axis=1) / base
                feat[f"roll_std_{window}"] = np.nanstd(block, axis=1) / base
        with np.errstate(invalid="ignore", divide="ignore"):
            feat["mom"] = values[:, t - 1] / base if t >= 1 else np.full(len(ids), np.nan)
            feat["yoy"] = values[:, t - 12] / base if t >= 12 else np.full(len(ids), np.nan)
        feat["n_history"] = np.isfinite(values[:, : t + 1]).sum(axis=1)
        frame = pd.DataFrame(feat)
        frame = frame[history_ok & (frame["y_last"] > 0)]
        for step in range(1, horizon + 1):
            target_pos = t + step
            part = frame.copy()
            target_ds = origin + pd.DateOffset(months=step)
            part["ds"] = target_ds
            part["step"] = step
            part["month"] = target_ds.month
            cats = cat_by_id.reindex(part["unique_id"]).to_numpy()
            s_target = np.array(
                [seasonal.at[c, target_ds.month] if c in seasonal.index else 1.0 for c in cats]
            )
            s_origin = np.array(
                [seasonal.at[c, origin.month] if c in seasonal.index else 1.0 for c in cats]
            )
            part["seasonal_ratio"] = s_target / s_origin
            if require_target:
                if target_pos >= values.shape[1]:
                    continue
                target = (
                    pd.Series(values[:, target_pos], index=ids)
                    .reindex(part["unique_id"])
                    .to_numpy()
                )
                part["y"] = target
                part = part[np.isfinite(target) & (target >= 0)]
                part["target"] = np.log((part["y"] + 1.0) / (part["y_last"] + 1.0))
            rows.append(part)
    if not rows:
        return pd.DataFrame()
    features = pd.concat(rows, ignore_index=True)
    static_cols = [c for c in STATIC_COLUMNS if c in static.columns]
    features = features.merge(static[["unique_id", *static_cols]], on="unique_id", how="left")
    features = _join_blocks(features, static, exogenous)
    for col in ("category_slug", "region_code", "federal_district", "mo_kind"):
        if col in features:
            features[col] = features[col].astype("category")
    return features


def _join_blocks(
    features: pd.DataFrame, static: pd.DataFrame, blocks: Sequence[ExogenousBlock]
) -> pd.DataFrame:
    """Присоединяет блоки по их ключам; ключ ряда, которого нет среди признаков, берётся из
    статической таблицы и после присоединения убирается."""
    for block in blocks:
        if not isinstance(block, ExogenousBlock):
            raise TypeError(
                "внешние признаки передаются блоками ExogenousBlock: по голой таблице нельзя "
                "понять, присоединять её по целевому месяцу или по моменту прогноза"
            )
        clash = sorted(set(block.columns) & set(features.columns))
        if clash:
            raise ValueError(f"блок «{block.name}» повторяет колонки признаков: {clash}")
        borrowed = [k for k in block.keys if k not in features.columns]
        absent = [k for k in borrowed if k not in static.columns]
        if absent:
            raise ValueError(f"блок «{block.name}»: ключа {absent} нет в статических признаках")
        if borrowed:
            features = features.merge(static[["unique_id", *borrowed]], on="unique_id", how="left")
        features = features.merge(block.frame, on=block.keys, how="left")
        features = features.drop(columns=borrowed)
    return features


def feature_columns(features: pd.DataFrame) -> list[str]:
    drop = {"unique_id", "origin", "ds", "y", "target", "y_last"}
    return [c for c in features.columns if c not in drop]


def predictions_from_ratio(features: pd.DataFrame, ratio_pred: np.ndarray) -> pd.DataFrame:
    """Обратное преобразование цели: y_hat = (y_last + 1)·exp(pred) − 1."""
    y_hat = (features["y_last"].to_numpy() + 1.0) * np.exp(ratio_pred) - 1.0
    return pd.DataFrame(
        {
            "unique_id": features["unique_id"].to_numpy(),
            "ds": features["ds"].to_numpy(),
            "y_hat": np.clip(y_hat, 0.0, None),
        }
    )
