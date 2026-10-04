"""Графики лендинга — встроенный SVG: проверяется геометрия, а не оформление."""

from __future__ import annotations

import math
import re

import pytest

from sbx.shell.render import charts

GAP = float("nan")
SERIES = {
    "region": "Регион",
    "mo": "Округ",
    "category": "Все категории",
    "ds": ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"],
    "y": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    "y_hat": [10.0, 10.0, 12.0, 12.0, 14.0, 14.0],
    "q_lo": [8.0, 8.0, GAP, GAP, 12.0, 12.0],
    "q_hi": [12.0, 12.0, GAP, GAP, 16.0, 16.0],
}
HORIZONS = [
    {"Модель": "Chronos-2 (covariates)", "mae h=1": 580.0, "mae h=6": 640.0, "mae h=12": 1136.0},
    {"Модель": "LightGBM", "mae h=1": 678.0, "mae h=6": 422.0, "mae h=12": GAP},
    {"Модель": "Ensemble", "mae h=1": 811.0, "mae h=6": 485.0, "mae h=12": 794.0},
    {"Модель": "Prophet", "mae h=1": 683.0, "mae h=6": 1354.0, "mae h=12": 4596.0},
    {"Модель": "Naive", "mae h=1": 788.0, "mae h=6": 760.0, "mae h=12": 993.0},
]
WINNERS = [
    {"horizon": 1, "model": "Chronos-2 (covariates)"},
    {"horizon": 6, "model": "LightGBM"},
    {"horizon": 12, "model": "Ensemble"},
]
CURVE = [
    {
        "detector": "cusum",
        "threshold": 3.0,
        "mean_delay": 2.4,
        "false_alarms_per_series_year": 0.88,
    },
    {
        "detector": "cusum",
        "threshold": 5.0,
        "mean_delay": 2.4,
        "false_alarms_per_series_year": 0.42,
    },
    {"detector": "bocpd", "threshold": 0.2, "mean_delay": 4.4, "false_alarms_per_series_year": 0.5},
    {"detector": "bocpd", "threshold": 0.5, "mean_delay": 3.4, "false_alarms_per_series_year": 0.3},
    {"detector": "adwin", "threshold": 0.1, "mean_delay": GAP, "false_alarms_per_series_year": 0.0},
    {
        "detector": "periodic",
        "threshold": 3,
        "mean_delay": 1.0,
        "false_alarms_per_series_year": 3.1,
    },
    {
        "detector": "periodic",
        "threshold": 6,
        "mean_delay": 2.2,
        "false_alarms_per_series_year": 1.28,
    },
    {
        "detector": "periodic",
        "threshold": 8,
        "mean_delay": 3.0,
        "false_alarms_per_series_year": 0.87,
    },
]
CASE = {
    "series": "Всего",
    "margin": 2,
    "ds": [f"{2020 + i // 12}-{i % 12 + 1:02d}" for i in range(36)],
    "values": [100.0 + i for i in range(36)],
    "breaks": ["2020-04", "2021-10"],
    "events": [
        {"event": "шок", "date": "2020-04-01", "detected": True},
        {"event": "ставка", "date": "2022-03-01", "detected": False},
    ],
}
DEMO = {
    "detector": "cusum",
    "threshold": 3.0,
    "months": [f"{2023 + i // 12}-{i % 12 + 1:02d}" for i in range(12)],
    "y": [10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 6.0, 6.5, 6.0, 6.2, 6.1, 6.0],
    "forecast": [None, 10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 6.0, 6.5, 6.0, 6.2, 6.1],
    "z": [None, None, None, 0.5, -0.4, 0.3, -7.5, 0.6, -0.5, 0.2, -0.1, -0.1],
    "statistic": [None, None, None, 0.1, 0.2, 0.0, 7.1, 0.5, 0.4, 0.0, 0.0, 0.0],
    "tau": 6,
    "break_month": "2023-07",
    "alarms": [8],
    "delay": 2,
    "delta": -0.4,
}


def _numbers(path: str) -> list[float]:
    return [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", path)]


def _paths(svg: str, cls: str) -> list[str]:
    """Атрибуты `d` у контуров с данным классом."""
    return re.findall(rf'<path class="{re.escape(cls)}" d="([^"]+)"', svg)


def test_nice_ticks_cover_the_range_with_round_steps() -> None:
    assert charts.nice_ticks(47.2, 83.9) == [40, 50, 60, 70, 80, 90]
    assert charts.nice_ticks(0.0, 0.43, 4) == pytest.approx([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    assert charts.nice_ticks(1100, 1790, 7) == [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800]
    flat = charts.nice_ticks(5.0, 5.0)
    assert flat[0] < 5.0 < flat[-1], "у ряда из одинаковых значений шкала всё равно есть"


def test_scale_maps_the_domain_onto_the_range_and_log_keeps_ratios() -> None:
    linear = charts.Scale(0.0, 10.0, 100.0, 0.0)
    assert linear(0.0) == 100.0 and linear(10.0) == 0.0 and linear(2.5) == 75.0
    log = charts.Scale(100.0, 10_000.0, 0.0, 200.0, log=True)
    assert log(1000.0) == pytest.approx(100.0), "равные отношения — равные расстояния"
    assert log(200.0) - log(100.0) == pytest.approx(log(2000.0) - log(1000.0))


def test_line_breaks_on_a_missing_value_instead_of_bridging_it() -> None:
    path = charts.line_path([0, 10, 20, 30, 40], [1.0, 2.0, None, 4.0, GAP])
    assert path.count("M") == 2, "после пропуска линия начинается заново"
    assert _numbers(path) == [0, 1, 10, 2, 30, 4]
    assert "nan" not in path.lower()


def test_interval_band_is_drawn_only_where_the_interval_exists() -> None:
    """Пропуск внутри интервала не должен ломать полосу: до и после него — свои отрезки."""
    svg = charts.series_chart(SERIES)
    bands = _paths(svg, "band")
    assert len(bands) == 2
    assert "nan" not in svg.lower()
    first = _numbers(bands[0])
    xs = sorted(set(first[0::2]))
    assert len(xs) == 2, "первый отрезок полосы — два месяца, без клина к соседнему"
    assert all(band.endswith("Z") for band in bands)


def test_series_chart_draws_fact_points_and_marks_alarm_months() -> None:
    plain = charts.series_chart(SERIES)
    assert plain.count('class="pt"') == 6, "по точке на месяц факта"
    assert 'class="tri"' not in plain and 'class="ring"' not in plain
    alarmed = charts.series_chart(SERIES, alarms=["2024-03", "2024-05", "2030-01"])
    assert alarmed.count('class="tri"') == 2, "тревога вне ряда не рисуется"
    assert alarmed.count('class="ring"') == 2
    assert 'role="img"' in alarmed and "aria-label=" in alarmed


def test_series_chart_labels_months_and_scales_large_values_to_thousands() -> None:
    svg = charts.series_chart(SERIES, every_month=True)
    assert ">янв 2024<" in svg and ">июн<" in svg
    big = {**SERIES, "y": [v * 5000 for v in SERIES["y"]], "y_hat": [v * 5000 for v in SERIES["y"]]}
    big["q_lo"], big["q_hi"] = [], []
    assert charts.series_unit(big) == "тыс. ₽" and charts.series_unit(SERIES) == "₽"
    ticks = re.findall(
        r'<text class="s11" [^>]*text-anchor="end"[^>]*>([^<]+)</text>', charts.series_chart(big)
    )
    assert ticks and all(float(t.replace(" ", "").replace(",", ".")) <= 80 for t in ticks)


def test_horizon_chart_rings_each_leader_and_stops_a_line_where_the_model_is_undefined() -> None:
    svg = charts.horizon_chart(HORIZONS, WINNERS, reference="Prophet")
    assert svg.count('class="best"') == 3, "кольцо — у лучшей модели каждого горизонта"
    assert "nan" not in svg.lower()
    lgbm = _paths(svg, "ln s-amber")
    assert len(lgbm) == 1 and lgbm[0].count("L") == 1, "у LightGBM две точки: при h=12 её нет"
    assert len(_paths(svg, "ln s-ref dash")) == 1, "эталон — пунктиром"
    for label in ("Prophet", "LightGBM", "Ансамбль", "Наивный", "1 мес.", "12 мес."):
        assert f">{label}<" in svg, label
    assert "Chronos-2" in svg and "с ковариатами" in svg


def test_horizon_chart_uses_a_log_scale() -> None:
    """Иначе Prophet при h=12 сжимает остальные линии в одну."""
    svg = charts.horizon_chart(HORIZONS, WINNERS, reference="Prophet")
    prophet = _numbers(_paths(svg, "ln s-ref dash")[0])[1::2]
    ratio_step = (prophet[0] - prophet[1]) / math.log(1354.0 / 683.0)
    assert (prophet[1] - prophet[2]) / math.log(4596.0 / 1354.0) == pytest.approx(
        ratio_step, rel=0.02
    )


def test_delay_chart_draws_the_schedule_as_a_reference_line_clipped_to_the_detectors() -> None:
    svg = charts.delay_chart(CURVE, focus="cusum")
    assert len(_paths(svg, "ln s-ref dash")) == 1, "расписание — пунктирная линия отсчёта"
    assert len(_paths(svg, "ln s-acc")) == 1 and len(_paths(svg, "ln s-neutral")) == 1
    assert "ADWIN" not in svg, "детектор без единой тревоги на графике не показан"
    assert 'class="ref-area"' in svg, "область ниже расписания закрашена"
    # Частое расписание даёт в разы больше ложных тревог, чем любой детектор: ось обрезана по
    # детекторам, иначе их кривые сжались бы в левый край.
    right = max(_numbers(_paths(svg, "ln s-acc")[0])[0::2])
    schedule = _numbers(_paths(svg, "ln s-ref dash")[0])[0::2]
    assert max(schedule) > right, "расписание доходит до края оси"
    assert max(schedule) <= charts.DELAY_SIZE[0], "но не выходит за рисунок"
    assert ">тревога по расписанию<" not in svg and ">расписание<" in svg


def test_delay_chart_connects_the_points_compared_at_an_equal_false_alarm_rate() -> None:
    match = {"detector": "cusum", "threshold": 3.0, "period": 8}
    svg = charts.delay_chart(CURVE, focus="cusum", match=match)
    assert svg.count('class="match"') == 1
    assert 'class="match"' not in charts.delay_chart(CURVE, focus="cusum")
    methods_only = [row for row in CURVE if row["detector"] != "periodic"]
    assert 'class="ref-area"' not in charts.delay_chart(methods_only, focus="cusum")


def test_case_chart_marks_found_and_missed_events_and_alarm_windows() -> None:
    svg = charts.case_chart(CASE)
    assert svg.count('class="win"') == 2, "окно допуска вокруг каждой тревоги"
    assert svg.count('class="tri"') == 2
    bottom = charts.CASE_SIZE[1] - charts.CASE_PAD[2]
    tips = [float(y) for y in re.findall(r'<path class="tri" d="M[\d.]+ ([\d.]+)l', svg)]
    assert all(y > bottom for y in tips), "тревоги — под осью, а не поверх ряда"
    numbers = re.findall(r'<text class="s11 b c-event" [^>]*text-anchor="middle">(\d)</text>', svg)
    assert numbers == ["1", "2"], "номер события — над ромбом"
    assert svg.count('class="ev"') == 1 and svg.count('class="ev miss"') == 1
    assert svg.count('class="dia"') == 1 and svg.count('class="dia miss"') == 1
    assert ">2020<" in svg and ">2022<" in svg, "по оси — годы"
    window = re.search(r'<rect class="win" x="([\d.]+)" [^>]*width="([\d.]+)"', svg)
    assert window is not None
    step = (charts.CASE_SIZE[0] - charts.CASE_PAD[1] - charts.CASE_PAD[3]) / 35
    assert float(window.group(2)) == pytest.approx(4 * step, abs=0.2), "±2 месяца"


def test_demo_chart_has_three_panels_with_the_break_the_alarm_and_the_threshold() -> None:
    svg = charts.demo_chart(DEMO)
    assert svg.count("<svg") == 1 and svg.count('class="panel"') == 3
    assert 'role="group"' in svg and 'role="slider"' in svg, (
        "внутри рисунка — выбор месяца: роль «картинка» скрыла бы его от вспомогательных программ"
    )
    assert svg.count('class="win"') == 1, "одна полоса от слома до тревоги через все панели"
    assert svg.count('class="dia"') == 1 and svg.count('class="tri"') == 1, "ромб слома, тревога"
    assert svg.count('class="thr"') == 1 and "порог 3,0" in svg
    assert svg.count('class="pt alarm"') == 1, "статистика в месяц тревоги отмечена точкой"
    assert ">слом внесён<" in svg and ">тревога · через 2 месяца<" in svg
    assert "nan" not in svg.lower() and "None" not in svg
    geometry = re.search(r'data-x0="([\d.]+)" data-step="([\d.]+)"', svg)
    assert geometry is not None, "сценарию страницы нужна сетка месяцев"
    step = float(geometry.group(2))
    width = float(re.search(r'<rect class="win" [^>]*width="([\d.]+)"', svg).group(1))
    assert width == pytest.approx(3 * step, abs=0.2), "слом в июле, тревога в сентябре: три месяца"
    same_month = charts.demo_chart({**DEMO, "alarms": [6], "delay": 0})
    narrow = float(re.search(r'<rect class="win" [^>]*width="([\d.]+)"', same_month).group(1))
    assert narrow == pytest.approx(step, abs=0.2), "тревога в месяц слома — полоса в один месяц"
    assert ">тревога · тот же месяц<" in same_month
