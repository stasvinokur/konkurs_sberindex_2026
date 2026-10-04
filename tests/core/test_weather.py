"""Погода NASA POWER: ячейки сетки, разбор ответа, нормы и признаки на момент прогноза."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sbx.core.weather import (
    climate_norms,
    grid_cell,
    parse_power,
    territory_cells,
    weather_as_of,
    with_norms,
)

MOSCOW, PETERSBURG = (55.7558, 37.6173), (59.9386, 30.3141)


def test_grid_cell_snaps_a_point_to_the_centre_of_its_merra2_cell() -> None:
    """Сетка MERRA-2: 0,5° по широте, 0,625° по долготе, отсчёт долготы от −180°."""
    assert grid_cell(*MOSCOW) == (56.0, 37.5)
    assert grid_cell(*PETERSBURG) == (60.0, 30.625)
    assert grid_cell(66.3, -179.12) == (66.5, -179.375)
    assert grid_cell(64.7, 179.9) == (64.5, -180.0), "за последней ячейкой сетка замыкается"
    assert grid_cell(55.25, 37.5) == (55.5, 37.5), "точка на границе — в верхнюю ячейку"


def test_territory_cells_use_the_city_centre_where_the_reference_has_no_coordinates() -> None:
    territories = pd.DataFrame(
        {
            "territory_id": ["45-0001", "50-0002", "40-0003"],
            "region_code": ["45", "50", "40"],
            "center_lat": [np.nan, 55.03, np.nan],
            "center_lon": [np.nan, 82.92, np.nan],
        }
    )
    cells = territory_cells(territories, {"45": MOSCOW, "40": PETERSBURG}).set_index("territory_id")
    assert cells.loc["45-0001", ["lat", "lon"]].tolist() == [56.0, 37.5]
    assert cells.loc["40-0003", ["lat", "lon"]].tolist() == [60.0, 30.625]
    assert cells.loc["50-0002", ["lat", "lon"]].tolist() == [55.0, 83.125]
    with pytest.raises(ValueError, match="координат"):
        territory_cells(territories, {"45": MOSCOW})


PAYLOAD = {
    "header": {"fill_value": -999.0},
    "properties": {
        "parameter": {
            "T2M": {"202301": -5.0, "202302": -999.0, "202313": 3.0},
            "PRECTOTCORR": {"202301": 1.5, "202302": 2.0, "202313": 1.7},
        }
    },
}


def test_parse_power_keeps_the_months_and_turns_fill_values_into_gaps() -> None:
    frame = parse_power(PAYLOAD)
    assert frame["ds"].tolist() == list(pd.to_datetime(["2023-01-01", "2023-02-01"])), (
        "тринадцатый «месяц» — среднее за год — в таблицу не идёт"
    )
    assert frame["t2m"].iloc[0] == -5.0 and np.isnan(frame["t2m"].iloc[1])
    assert frame["prectot"].tolist() == [1.5, 2.0]


def _history() -> pd.DataFrame:
    """Две ячейки; январи и июли трёх лет."""
    rows = []
    for lat, shift in ((56.0, 0.0), (60.0, -4.0)):
        for year, january, july in ((2019, -10.0, 18.0), (2020, -6.0, 20.0), (2023, 0.0, 25.0)):
            rows.append((lat, 37.5, f"{year}-01-01", january + shift, 1.0 + year % 2))
            rows.append((lat, 37.5, f"{year}-07-01", july + shift, 3.0))
    frame = pd.DataFrame(rows, columns=["lat", "lon", "ds", "t2m", "prectot"])
    return frame.assign(ds=pd.to_datetime(frame["ds"]))


def test_norms_are_calendar_month_means_over_the_reference_years_of_the_cell() -> None:
    table = with_norms(_history(), 2019, 2020).set_index(["lat", "ds"])
    january = table.loc[(56.0, pd.Timestamp("2023-01-01"))]
    assert january["t2m_norm"] == -8.0, "2023 год в норму не входит"
    assert january["prectot_norm"] == 1.5
    assert table.loc[(56.0, pd.Timestamp("2023-07-01")), "t2m_norm"] == 19.0
    assert table.loc[(60.0, pd.Timestamp("2023-01-01")), "t2m_norm"] == -12.0, "у ячейки своя норма"


def _monthly() -> tuple[pd.DataFrame, pd.DataFrame]:
    months = pd.date_range("2023-11-01", periods=5, freq="MS")
    table = pd.DataFrame(
        {
            "lat": 56.0,
            "lon": 37.5,
            "ds": months,
            "t2m": [-1.0, -5.0, -12.0, -3.0, 4.0],
            "prectot": [2.0, 2.0, 3.0, 1.0, 2.0],
            "t2m_norm": [-2.0, -6.0, -8.0, -7.0, -1.0],
            "prectot_norm": [2.0, 2.0, 2.0, 2.0, 2.0],
        }
    )
    cells = pd.DataFrame({"territory_id": ["a", "b"], "lat": 56.0, "lon": 37.5})
    return table, cells


def test_weather_of_an_origin_is_that_of_the_last_published_month() -> None:
    """К концу февраля опубликован январь, к концу марта — февраль: лаг источника 7 дней."""
    table, cells = _monthly()
    february, march = pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-01")
    out = weather_as_of(table, cells, [february, march], lag_days=7)
    assert set(out.columns) == {"territory_id", "origin", "wx_t_anom", "wx_p_anom", "wx_t_anom_3"}
    rows = out.set_index(["territory_id", "origin"])
    assert rows.loc[("a", february), "wx_t_anom"] == -4.0, "январь: −12 при норме −8"
    assert rows.loc[("a", february), "wx_p_anom"] == pytest.approx(0.5), (
        "осадков в полтора раза больше"
    )
    assert rows.loc[("a", february), "wx_t_anom_3"] == pytest.approx((1.0 + 1.0 - 4.0) / 3)
    assert rows.loc[("a", march), "wx_t_anom"] == 4.0, "февраль: −3 при норме −7"
    assert rows.loc[("b", march), "wx_t_anom"] == 4.0, "территории одной ячейки — одна погода"
    assert len(out) == 4


def test_weather_of_an_origin_does_not_change_when_later_months_are_removed() -> None:
    table, cells = _monthly()
    february = pd.Timestamp("2024-02-01")
    full = weather_as_of(table, cells, [february], lag_days=7)
    known = weather_as_of(table[table["ds"] <= february], cells, [february], lag_days=7)
    pd.testing.assert_frame_equal(full, known)
    late = weather_as_of(table, cells, [february], lag_days=40).set_index("territory_id")
    assert late.loc["a", "wx_t_anom"] == 1.0, "при лаге 40 дней к концу февраля известен декабрь"


def test_climate_norms_are_known_for_any_target_month() -> None:
    table, cells = _monthly()
    # Те же календарные месяцы годом раньше: норма одна, строка в результате тоже одна.
    earlier = table.assign(ds=table["ds"] - pd.DateOffset(years=1), t2m=0.0)
    table = pd.concat([earlier, table], ignore_index=True)
    months = [pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01"), pd.Timestamp("2024-07-01")]
    out = climate_norms(table, cells, months).set_index(["territory_id", "ds"])
    assert out.loc[("a", months[0]), "wx_t_norm"] == -8.0
    assert out.loc[("b", months[1]), "wx_t_norm"] == -8.0, "норма января — на любой январь"
    assert out.loc[("a", months[0]), "wx_p_norm"] == 2.0
    assert np.isnan(out.loc[("a", months[2]), "wx_t_norm"]), "нормы июля в таблице нет"
    assert len(out) == 6


def test_three_month_weather_of_a_cell_does_not_borrow_from_another_cell() -> None:
    table, cells = _monthly()
    north = table.assign(lat=60.0, t2m=table["t2m_norm"] + 10.0)
    cells = pd.DataFrame({"territory_id": ["a", "n"], "lat": [56.0, 60.0], "lon": 37.5})
    december = pd.Timestamp("2023-12-01")
    out = weather_as_of(pd.concat([table, north]), cells, [december], lag_days=7)
    rows = out.set_index("territory_id")
    assert rows.loc["n", "wx_t_anom_3"] == 10.0, "у северной ячейки известен один её ноябрь"
    assert rows.loc["a", "wx_t_anom_3"] == 1.0
