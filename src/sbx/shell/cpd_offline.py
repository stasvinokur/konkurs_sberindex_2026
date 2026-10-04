"""Адаптеры офлайн-методов обнаружения точек изменений (ruptures) и онлайн-детектора river."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from sbx.core.changepoint.offline import to_noise_units

OFFLINE_METHODS = ("pelt_l2", "pelt_rbf", "binseg_l2", "window_l2", "kernel_rbf")

# Слом допускается в любой точке ряда. Умолчание ruptures — 5: методам остаются только
# позиции, кратные пяти, то есть четыре месяца из двадцати четырёх.
JUMP = 1


def detect_offline(
    values: Sequence[float], method: str, penalty: float, min_size: int = 3
) -> list[int]:
    """Точки изменения по одному ряду; возвращает индексы начала новых сегментов.

    Ряд переводится в единицы собственного шума (`to_noise_units`), поэтому штраф означает
    одно и то же для рядов любого уровня: во сколько дисперсий шума должен окупиться разрез.
    """
    import ruptures as rpt

    signal = to_noise_units(values).reshape(-1, 1)
    if len(signal) < 2 * min_size:
        return []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "pelt_l2":
            algo = rpt.Pelt(model="l2", min_size=min_size, jump=JUMP).fit(signal)
            points = algo.predict(pen=penalty)
        elif method == "pelt_rbf":
            algo = rpt.Pelt(model="rbf", min_size=min_size, jump=JUMP).fit(signal)
            points = algo.predict(pen=penalty)
        elif method == "binseg_l2":
            algo = rpt.Binseg(model="l2", min_size=min_size, jump=JUMP).fit(signal)
            points = algo.predict(pen=penalty)
        elif method == "window_l2":
            width = max(2 * min_size, 6)
            algo = rpt.Window(width=width, model="l2", min_size=min_size, jump=JUMP).fit(signal)
            points = algo.predict(pen=penalty)
        elif method == "kernel_rbf":
            algo = rpt.KernelCPD(kernel="rbf", min_size=min_size).fit(signal)
            points = algo.predict(pen=penalty)
        else:
            raise ValueError(f"неизвестный метод: {method}")
    return [int(p) for p in points if 0 < p < len(signal)]


def detect_offline_panel(
    panel: pd.DataFrame, method: str, penalty: float, value_col: str = "y", min_size: int = 3
) -> dict[str, list[int]]:
    out = {}
    for uid, g in panel.sort_values(["unique_id", "ds"]).groupby("unique_id"):
        out[uid] = detect_offline(g[value_col].to_numpy(), method, penalty, min_size)
    return out


def adwin_alarms(values: Sequence[float], delta: float = 0.002) -> list[int]:
    """Онлайн-детектор ADWIN (river): тревога при обнаружении дрейфа."""
    from river.drift import ADWIN

    detector = ADWIN(delta=delta)
    alarms = []
    for t, value in enumerate(np.asarray(values, dtype=float)):
        if not np.isfinite(value):
            continue
        detector.update(float(value))
        if detector.drift_detected:
            alarms.append(t)
    return alarms


def adwin_panel(
    series: Mapping[str, Sequence[float]], delta: float = 0.002
) -> dict[str, list[int]]:
    return {uid: adwin_alarms(values, delta) for uid, values in series.items()}
