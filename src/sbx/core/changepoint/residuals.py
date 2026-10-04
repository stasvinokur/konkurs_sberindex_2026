"""Стандартизованные остатки прогноза — вход онлайн-детекторов сдвигов.

Детектор, работающий по остаткам прогнозной модели, не путает предсказуемый декабрьский пик
со структурным сломом: сезонность уже объяснена прогнозом.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def freshest_forecast(oof: pd.DataFrame, max_horizon: int | None = None) -> pd.DataFrame:
    """Один прогноз на месяц ряда: из фолда с самым коротким горизонтом.

    Горизонты протокола перекрываются: один и тот же месяц предсказан в фолдах h=1, 3, 6 и 12.
    Онлайн-детекторы накопительные, и повторённый месяц создаёт ложный дрейф; на графике он же
    рисует пилу. Поэтому и детектор, и графики работают с рядом без повторов, а на каждый месяц
    берётся самый точный из доступных прогнозов — самый короткий горизонт, при равном горизонте
    самый поздний cutoff.

    `max_horizon` отбрасывает длинные горизонты целиком: в эксплуатации детектор получает
    прогнозы на ближайшие месяцы, а фолды h=6 и h=12 существуют для оценки точности. Без
    этого в поток попадают месяцы, предсказанные только из фолдов h=12 с 10–11 месяцами
    обучения, и их промахи выглядят сдвигами.
    """
    if max_horizon is not None and "horizon" in oof.columns:
        oof = oof[oof["horizon"] <= max_horizon]
    keys = ["unique_id", "ds", "model"]
    order = [*keys, "horizon"] if "horizon" in oof.columns else [*keys]
    ascending = [True] * len(order)
    if "cutoff" in oof.columns:
        order.append("cutoff")
        ascending.append(False)
    elif "fold" in oof.columns:
        order.append("fold")
        ascending.append(True)
    ordered = oof.sort_values(order, ascending=ascending, kind="stable")
    return ordered.drop_duplicates(keys, keep="first").reset_index(drop=True)


def standardized_residuals(
    y: np.ndarray, y_hat: np.ndarray, scale: np.ndarray | float | None = None, eps: float = 1e-9
) -> np.ndarray:
    """(y − ŷ) / масштаб. По умолчанию масштаб — робастная оценка по самим остаткам (MAD)."""
    resid = np.asarray(y, dtype=float) - np.asarray(y_hat, dtype=float)
    if scale is None:
        mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
        scale = 1.4826 * mad if mad > 0 else np.nanstd(resid)
    scale = np.asarray(scale, dtype=float)
    return resid / np.maximum(scale, eps)


def causal_zscores(
    values: np.ndarray, center: bool = True, min_history: int = 3, eps: float = 1e-9
) -> np.ndarray:
    """Стандартизация по расширяющемуся окну строго прошлых значений.

    Значение в момент t делится на робастный масштаб (MAD) значений до t и, если `center`,
    отсчитывается от их медианы. Само значение в свою оценку не входит: иначе сдвиг раздувал бы
    собственный масштаб, а нормировка по всему ряду ещё и видела бы будущее. Пока прошлых
    значений меньше `min_history`, оценки нет (NaN).
    """
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan)
    for t in range(len(values)):
        past = values[:t]
        past = past[np.isfinite(past)]
        if len(past) < min_history or not np.isfinite(values[t]):
            continue
        median = float(np.median(past))
        mad = float(np.median(np.abs(past - median)))
        scale = 1.4826 * mad if mad > 0 else float(np.std(past))
        out[t] = (values[t] - (median if center else 0.0)) / max(scale, eps)
    return out


def residual_frame(
    oof: pd.DataFrame, model: str, scale_col: str | None = "mase_scale"
) -> pd.DataFrame:
    """Остатки выбранной модели по рядам и месяцам в длинном формате."""
    part = oof[oof["model"] == model].sort_values(["unique_id", "ds"]).copy()
    scale = part[scale_col].to_numpy() if scale_col and scale_col in part else None
    part["residual"] = part["y"].to_numpy() - part["y_hat"].to_numpy()
    part["z"] = standardized_residuals(part["y"].to_numpy(), part["y_hat"].to_numpy(), scale)
    return part[["unique_id", "ds", "fold", "step", "y", "y_hat", "residual", "z"]]


def interval_exceedance(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """1, если факт вне квантильного интервала (для конформного детектора)."""
    y = np.asarray(y, dtype=float)
    return ((y < np.asarray(lo, dtype=float)) | (y > np.asarray(hi, dtype=float))).astype(int)


def seasonal_naive_residuals(
    values: np.ndarray, months: np.ndarray, seasonal_index: dict[int, float]
) -> np.ndarray:
    """Остатки одношагового наивного прогноза с сезонной поправкой.

    ŷ_t = y_{t-1}·S(m_t)/S(m_{t-1}). Прогноз на момент t использует только y_{t-1}, поэтому
    поток остатков причинный и годится для онлайн-детекторов. Первый элемент — NaN.
    """
    values = np.asarray(values, dtype=float)
    months = np.asarray(months, dtype=int)
    forecast = np.full(len(values), np.nan)
    for t in range(1, len(values)):
        s_now = seasonal_index.get(int(months[t]), 1.0)
        s_prev = seasonal_index.get(int(months[t - 1]), 1.0)
        forecast[t] = values[t - 1] * s_now / max(s_prev, 1e-9)
    return values - forecast


def residual_stream(
    panel: pd.DataFrame, seasonal: pd.DataFrame, value_col: str = "y"
) -> pd.DataFrame:
    """Причинный поток стандартизованных остатков по всем рядам панели.

    Причинны и остаток (прогноз момента t использует только y_{t-1}), и его масштаб: остаток
    делится на разброс остатков до t (`causal_zscores`).
    """
    index = seasonal.pivot_table(index="category", columns="month", values="index")
    frames = []
    for uid, g in panel.sort_values(["unique_id", "ds"]).groupby("unique_id"):
        category = g["category"].iloc[0] if "category" in g else None
        table = index.loc[category].to_dict() if category in index.index else {}
        resid = seasonal_naive_residuals(
            g[value_col].to_numpy(), g["ds"].dt.month.to_numpy(), table
        )
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": uid,
                    "t": np.arange(len(g)),
                    "ds": g["ds"].to_numpy(),
                    "residual": resid,
                    "z": causal_zscores(resid, center=False),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
