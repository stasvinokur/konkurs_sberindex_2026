"""Модель вероятности структурного шока: разметка и признаки.

Метка `z_t^(h) = 1`, если слом происходит в интервале (t, t+h]. Признаки на момент t строятся
только из данных, доступных к t: траектория остатков прогноза, сигналы онлайн-детекторов,
ширина квантильного интервала, макро- и новостные признаки региона.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def make_labels(
    breaks: Mapping[str, Sequence[int]],
    lengths: Mapping[str, int],
    horizons: Sequence[int] = (1, 2, 3),
) -> pd.DataFrame:
    """Таблица (ряд, t) с метками «слом в ближайшие h месяцев» для каждого h."""
    rows = []
    for uid, length in lengths.items():
        taus = sorted(t for t in breaks.get(uid, []) if t >= 0)
        for t in range(length):
            row = {"unique_id": uid, "t": t}
            for h in horizons:
                row[f"z_h{h}"] = int(any(t < tau <= t + h for tau in taus))
            rows.append(row)
    return pd.DataFrame(rows)


def residual_trajectory_features(
    z: Sequence[float], windows: Sequence[int] = (1, 3, 6)
) -> pd.DataFrame:
    """Признаки траектории стандартизованных остатков, известные на момент t."""
    series = pd.Series(np.asarray(z, dtype=float))
    out = pd.DataFrame({"t": np.arange(len(series))})
    out["resid_z"] = series.to_numpy()
    out["resid_abs"] = series.abs().to_numpy()
    for window in windows:
        out[f"resid_mean_{window}"] = series.rolling(window, min_periods=1).mean().to_numpy()
        out[f"resid_absmean_{window}"] = (
            series.abs().rolling(window, min_periods=1).mean().to_numpy()
        )
        out[f"resid_std_{window}"] = (
            series.rolling(window, min_periods=min(2, window)).std().to_numpy()
        )
    out["resid_trend"] = series.diff().to_numpy()
    out["resid_sign_run"] = (
        series.gt(0).astype(int).groupby((series.gt(0) != series.gt(0).shift()).cumsum()).cumcount()
        + 1
    ).to_numpy()
    return out


def interval_width_features(
    q_lo: Sequence[float], q_hi: Sequence[float], y_hat: Sequence[float]
) -> pd.DataFrame:
    """Относительная ширина квантильного интервала и её изменение (сигнал неуверенности FM)."""
    width = (np.asarray(q_hi, dtype=float) - np.asarray(q_lo, dtype=float)) / np.maximum(
        np.abs(np.asarray(y_hat, dtype=float)), 1e-9
    )
    frame = pd.DataFrame({"t": np.arange(len(width)), "interval_width": width})
    frame["interval_width_change"] = pd.Series(width).diff().to_numpy()
    return frame


def build_hazard_table(
    residuals: pd.DataFrame,
    labels: pd.DataFrame,
    exogenous: pd.DataFrame | None = None,
    detector_alarms: Mapping[str, Sequence[int]] | None = None,
) -> pd.DataFrame:
    """Собирает обучающую таблицу модели вероятности шока.

    `residuals` — колонки `unique_id, t, z` (и, если есть, `q_lo`, `q_hi`, `y_hat`).
    `exogenous` — признаки на (unique_id, t) или (region_code, t), уже сдвинутые point-in-time.
    """
    frames = []
    for uid, g in residuals.sort_values(["unique_id", "t"]).groupby("unique_id"):
        part = residual_trajectory_features(g["z"].to_numpy())
        part["unique_id"] = uid
        if {"q_lo", "q_hi", "y_hat"} <= set(g.columns):
            widths = interval_width_features(g["q_lo"], g["q_hi"], g["y_hat"])
            part = part.merge(widths, on="t", how="left")
        if detector_alarms is not None:
            alarms = set(detector_alarms.get(uid, []))
            part["detector_alarm"] = [int(t in alarms) for t in part["t"]]
            part["detector_alarm_recent"] = (
                pd.Series(part["detector_alarm"]).rolling(3, min_periods=1).max().to_numpy()
            )
        frames.append(part)
    features = pd.concat(frames, ignore_index=True)
    table = labels.merge(features, on=["unique_id", "t"], how="inner")
    if exogenous is not None:
        keys = [c for c in ("unique_id", "region_code", "t") if c in exogenous.columns]
        table = table.merge(exogenous, on=keys, how="left")
    return table


def hazard_feature_columns(table: pd.DataFrame) -> list[str]:
    drop = {"unique_id", "region_code", "t", "ds"} | {
        c for c in table.columns if c.startswith("z_h")
    }
    return [c for c in table.columns if c not in drop]


def series_invariant_columns(
    table: pd.DataFrame, features: Sequence[str], tolerance: float = 0.99
) -> list[str]:
    """Признаки, одинаковые для всех рядов внутри каждого момента t.

    На полусинтетическом бенчмарке все ряды живут на одном календаре, поэтому такой признак —
    это переобозначение позиции t. А доля положительных меток от позиции зависит: момент слома
    разыгран не ближе четырёх точек к краям, поэтому у краёв метка всегда нулевая. Модель,
    получив такой признак, выучит форму разметки вместо сигнала о шоке. Национальные новостные
    показатели попадают сюда целиком; региональные — нет, они различаются между рядами.
    """
    invariant = []
    for column in features:
        if column not in table.columns:
            continue
        # Пропуски не считаются отдельным значением: у части рядов регион не определился,
        # и признак уровня страны иначе выглядел бы «различающимся» из-за NaN.
        counts = table.groupby("t")[column].nunique(dropna=True)
        if len(counts) and float((counts <= 1).mean()) >= tolerance:
            invariant.append(column)
    return invariant


def alarms_from_probabilities(
    probabilities: pd.DataFrame, threshold: float, prob_col: str = "p", id_col: str = "unique_id"
) -> dict[str, list[int]]:
    """Тревоги там, где вероятность шока превышает порог."""
    hits = probabilities[probabilities[prob_col] >= threshold]
    return {uid: sorted(g["t"].astype(int)) for uid, g in hits.groupby(id_col)}


def lead_time_table(
    probabilities: pd.DataFrame, breaks: Mapping[str, Sequence[int]], threshold: float
) -> pd.DataFrame:
    """Насколько заранее модель подняла тревогу перед реальным сломом."""
    alarms = alarms_from_probabilities(probabilities, threshold)
    rows = []
    for uid, taus in breaks.items():
        for tau in sorted(t for t in taus if t >= 0):
            earlier = [a for a in alarms.get(uid, []) if a < tau]
            rows.append(
                {
                    "unique_id": uid,
                    "tau": tau,
                    "lead_time": tau - max(earlier) if earlier else np.nan,
                    "warned": bool(earlier),
                }
            )
    return pd.DataFrame(rows)
