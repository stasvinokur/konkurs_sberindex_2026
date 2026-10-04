"""Погода NASA POWER на уровне МО: ячейки сетки, нормы и признаки на момент прогноза.

Источник — реанализ MERRA-2 в выдаче NASA POWER: месячные средние температуры воздуха на
высоте 2 м и осадков. Значение относится к ячейке сетки (0,5° по широте, 0,625° по долготе),
поэтому МО привязываются к ячейке, в которую попадает их центр, и запрос делается один на
ячейку.

Признак — не сама погода, а её отклонение от нормы ячейки для этого календарного месяца:
уровень температуры в Якутии и в Краснодаре несравним, а «на четыре градуса холоднее
обычного» означает одно и то же. Норма считается по годам до начала панели. Абсолютная
ошибка реанализа в норму входит так же, как в значение, и в отклонении сокращается.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from sbx.core.features.asof import as_of_origin

LAT_STEP, LON_STEP = 0.5, 0.625
LON_CELLS = round(360 / LON_STEP)
PARAMETERS = {"T2M": "t2m", "PRECTOTCORR": "prectot"}
CELL = ["lat", "lon"]
# Сколько последних опубликованных месяцев усредняется в «погоде за квартал».
WINDOW = 3


def grid_cell(lat: float, lon: float) -> tuple[float, float]:
    """Центр ячейки сетки MERRA-2, в которую попадает точка.

    Точка на границе относится к ячейке с большей координатой; долгота замыкается: ячейка
    за 179,375° — это ячейка −180°.
    """
    row = math.floor(lat / LAT_STEP + 0.5)
    column = math.floor((lon + 180.0) / LON_STEP + 0.5) % LON_CELLS
    return row * LAT_STEP, column * LON_STEP - 180.0


def territory_cells(
    territories: pd.DataFrame, city_centers: Mapping[str, Sequence[float]]
) -> pd.DataFrame:
    """Ячейка сетки каждой территории: `territory_id`, `lat`, `lon`.

    У внутригородских территорий городов федерального значения в справочнике нет координат
    центра; им присваивается центр города (`city_centers`: код региона → широта, долгота).
    """
    rows = []
    for item in territories.itertuples():
        lat, lon = item.center_lat, item.center_lon
        if pd.isna(lat) or pd.isna(lon):
            if str(item.region_code) not in city_centers:
                raise ValueError(
                    f"у территории {item.territory_id} нет координат центра, а для региона "
                    f"{item.region_code} не задан центр города"
                )
            lat, lon = city_centers[str(item.region_code)]
        rows.append((item.territory_id, *grid_cell(float(lat), float(lon))))
    return pd.DataFrame(rows, columns=["territory_id", *CELL])


def parse_power(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Ответ NASA POWER (месячные значения одной точки) → таблица `ds`, `t2m`, `prectot`.

    Ключ значения — «ГГГГММ»; «месяц» 13 — среднее за год, он не нужен. Пропуск у источника
    записан значением-заглушкой (`fill_value`) и становится NaN.
    """
    fill = float(payload["header"]["fill_value"])
    parameters = payload["properties"]["parameter"]
    keys = sorted(k for k in next(iter(parameters.values())) if 1 <= int(k[4:]) <= 12)
    frame = pd.DataFrame({"ds": [pd.Timestamp(int(k[:4]), int(k[4:]), 1) for k in keys]})
    for name, column in PARAMETERS.items():
        values = np.array([float(parameters[name][k]) for k in keys])
        frame[column] = np.where(values == fill, np.nan, values)
    return frame


def with_norms(table: pd.DataFrame, first_year: int, last_year: int) -> pd.DataFrame:
    """Добавляет норму ячейки для календарного месяца: среднее за годы `first_year…last_year`."""
    month = table["ds"].dt.month.rename("month")
    reference = table[table["ds"].dt.year.between(first_year, last_year)]
    norms = (
        reference.groupby([*CELL, reference["ds"].dt.month.rename("month")])[
            list(PARAMETERS.values())
        ]
        .mean()
        .add_suffix("_norm")
        .reset_index()
    )
    out = table.assign(month=month).merge(norms, on=[*CELL, "month"], how="left")
    return out.drop(columns="month")


def weather_as_of(
    table: pd.DataFrame, cells: pd.DataFrame, origins: Sequence[pd.Timestamp], lag_days: int
) -> pd.DataFrame:
    """Погода, известная к концу каждого месяца `origin`, по территориям.

    `wx_t_anom` — отклонение температуры последнего опубликованного месяца от нормы, градусы;
    `wx_p_anom` — относительное отклонение осадков; `wx_t_anom_3` — среднее отклонение
    температуры за три последних опубликованных месяца.
    """
    work = table.sort_values([*CELL, "ds"]).copy()
    work["wx_t_anom"] = work["t2m"] - work["t2m_norm"]
    work["wx_p_anom"] = work["prectot"] / work["prectot_norm"].where(work["prectot_norm"] > 0) - 1
    work["wx_t_anom_3"] = (
        work.groupby(CELL)["wx_t_anom"].rolling(WINDOW, min_periods=1).mean().to_numpy()
    )
    features = ["wx_t_anom", "wx_p_anom", "wx_t_anom_3"]
    known = as_of_origin(work[[*CELL, "ds", *features]], origins, lag_days)
    out = cells.merge(known, on=CELL)[["territory_id", "origin", *features]]
    return out.sort_values(["territory_id", "origin"]).reset_index(drop=True)


def climate_norms(
    table: pd.DataFrame, cells: pd.DataFrame, months: Sequence[pd.Timestamp]
) -> pd.DataFrame:
    """Климатическая норма территории для целевого месяца: `wx_t_norm`, `wx_p_norm`.

    Норма посчитана по годам до начала панели и от года не зависит, поэтому известна заранее
    на любой месяц. Календарного месяца, которого нет в таблице, нет и в норме.
    """
    norms = (
        table.assign(month=table["ds"].dt.month)
        .drop_duplicates([*CELL, "month"])[[*CELL, "month", "t2m_norm", "prectot_norm"]]
        .rename(columns={"t2m_norm": "wx_t_norm", "prectot_norm": "wx_p_norm"})
    )
    target = pd.DataFrame({"ds": [pd.Timestamp(m) for m in months]})
    target["month"] = target["ds"].dt.month
    out = cells.merge(target, how="cross").merge(norms, on=[*CELL, "month"], how="left")
    out = out[["territory_id", "ds", "wx_t_norm", "wx_p_norm"]]
    return out.sort_values(["territory_id", "ds"]).reset_index(drop=True)
