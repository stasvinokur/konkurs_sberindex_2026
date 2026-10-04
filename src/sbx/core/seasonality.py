"""Сезонные индексы по месяцам: национальный (длинные ряды) и панельный (ряды МО)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sbx.core.features.asof import latest_published_month

SEASONAL_START = "2019-01-01"

# Соответствие категорий национального ряда `consumer-spending` категориям МО (приближённое).
NATIONAL_TO_MO = {
    "Всего": ["Все категории"],
    "Продовольственные товары": ["Продовольствие"],
    "Общественное питание": ["Общественное питание"],
    "Непродовольственные товары": ["Здоровье", "Маркетплейсы"],
    "Услуги": ["Транспорт"],
}


def ratio_to_moving_average(series: pd.Series, window: int = 12) -> pd.Series:
    """Отношение к центрированному 2×12 скользящему среднему (классическая декомпозиция)."""
    s = series.sort_index().astype(float)
    ma = s.rolling(window, center=True).mean().rolling(2).mean().shift(-1)
    return s / ma


def seasonal_index_from_ratios(ratios: pd.Series) -> pd.Series:
    """Медиана отношений по месяцам, нормированная к среднему 1 (12 значений)."""
    by_month = ratios.dropna().groupby(ratios.dropna().index.month).median()
    by_month = by_month.reindex(range(1, 13))
    return by_month / by_month.mean()


def national_seasonal_index(
    national: pd.DataFrame, *, end: str | pd.Timestamp, start: str | pd.Timestamp = SEASONAL_START
) -> pd.DataFrame:
    """Сезонный индекс национальных категорий и его перенос на категории МО.

    `national` — long (period, type, value) из `consumer-spending` (млрд руб., номинал).
    Возвращает long (category, month, index, national_type).

    Конец окна оценки `end` обязателен: индекс, оценённый по всему ряду, знает о месяцах,
    которые на момент прогноза ещё не опубликованы.
    """
    df = national.assign(ds=pd.to_datetime(national["period"]))
    df = df[(df["ds"] >= start) & (df["ds"] <= end)]
    rows = []
    for nat_type, g in df.groupby("type"):
        idx = seasonal_index_from_ratios(ratio_to_moving_average(g.set_index("ds")["value"]))
        for mo_cat in NATIONAL_TO_MO.get(nat_type, []):
            for month, value in idx.items():
                rows.append(
                    {
                        "category": mo_cat,
                        "month": int(month),
                        "index": float(value),
                        "national_type": nat_type,
                    }
                )
    return pd.DataFrame(rows).sort_values(["category", "month"]).reset_index(drop=True)


def seasonal_index_as_of(
    national: pd.DataFrame,
    origin: pd.Timestamp,
    lag_days: int,
    start: str | pd.Timestamp = SEASONAL_START,
) -> pd.DataFrame:
    """Сезонный индекс по национальному ряду, опубликованному к концу месяца `origin`."""
    return national_seasonal_index(
        national, start=start, end=latest_published_month(origin, lag_days)
    )


def panel_seasonal_index(panel: pd.DataFrame) -> pd.DataFrame:
    """Медианный профиль МО: y / среднее по году ряда, медиана по рядам и годам, по месяцам."""
    full_years = panel.groupby(["unique_id", panel["ds"].dt.year])["ds"].transform("size") == 12
    p = panel[full_years].copy()
    p["year"] = p["ds"].dt.year
    p["rel"] = p["y"] / p.groupby(["unique_id", "year"])["y"].transform("mean")
    prof = p.groupby(["category", p["ds"].dt.month])["rel"].median().rename("index").reset_index()
    prof = prof.rename(columns={"ds": "month"})
    prof["index"] = prof["index"] / prof.groupby("category")["index"].transform("mean")
    return prof


def seasonal_amplitude(index: pd.DataFrame) -> pd.Series:
    """Размах сезонности (max/min индекса) по категориям."""
    return index.groupby("category")["index"].agg(lambda s: float(np.max(s) / np.min(s)))
