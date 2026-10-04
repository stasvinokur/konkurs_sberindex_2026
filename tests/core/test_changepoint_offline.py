"""Вход офлайн-методов в единицах шума и эталон «тревога по расписанию»."""

from __future__ import annotations

import numpy as np
import pytest

from sbx.core.changepoint.offline import noise_scale, periodic_alarms, to_noise_units


def test_noise_scale_recovers_the_noise_level_not_the_series_level() -> None:
    rng = np.random.default_rng(0)
    values = 25_000.0 + rng.normal(0, 400.0, size=2_000)
    assert noise_scale(values) == pytest.approx(400.0, rel=0.1)


def test_noise_scale_is_not_inflated_by_the_break_itself() -> None:
    """Слом — то, что метод ищет: он не должен увеличивать оценку шума и прятать сам себя."""
    rng = np.random.default_rng(1)
    flat = 1_000.0 + rng.normal(0, 50.0, size=200)
    shifted = flat.copy()
    shifted[100:] += 400.0
    assert noise_scale(shifted) == pytest.approx(noise_scale(flat), rel=0.05)
    assert np.std(shifted) > 3 * noise_scale(shifted)


def test_noise_scale_is_not_inflated_by_a_seasonal_peak() -> None:
    """Декабрьский пик даёт две большие разности в году; медиана разностей их не замечает."""
    rng = np.random.default_rng(2)
    values = 1_000.0 + rng.normal(0, 20.0, size=24)
    peaked = values.copy()
    peaked[[11, 23]] += 300.0
    assert noise_scale(peaked) < 2 * noise_scale(values)
    assert np.std(np.diff(peaked)) > 5 * noise_scale(peaked)


def test_noise_scale_does_not_count_a_steady_trend_as_noise() -> None:
    rng = np.random.default_rng(5)
    noise = rng.normal(0, 10.0, size=400)
    trend = np.arange(400) * 50.0
    assert noise_scale(trend + noise) == pytest.approx(noise_scale(noise), rel=1e-6)


def test_noise_scale_of_a_noiseless_step_falls_back_to_the_spread_of_differences() -> None:
    """Все разности, кроме одной, нулевые: медианная оценка даёт ноль, а делить на ноль нельзя."""
    values = np.full(48, 1_000.0)
    values[15:] = 600.0
    assert noise_scale(values) == pytest.approx(np.std(np.diff(values)) / np.sqrt(2.0))


def test_noise_scale_treats_rounding_error_as_no_noise() -> None:
    """Ровная прямая: все разности равны с точностью до округления. «Шум» порядка 1e-17
    превратил бы ряд в единицах шума в числа порядка 1e17."""
    assert noise_scale(np.linspace(0.0, 1.0, 24)) == 1.0
    assert noise_scale(1_000_000.0 + np.arange(24.0) / 3.0) == 1.0


def test_noise_scale_of_a_constant_series_is_one() -> None:
    assert noise_scale(np.full(24, 500.0)) == 1.0
    assert noise_scale(np.array([7.0])) == 1.0


def test_to_noise_units_does_not_depend_on_the_units_of_the_series() -> None:
    rng = np.random.default_rng(3)
    values = 1_000.0 + rng.normal(0, 30.0, size=36)
    np.testing.assert_allclose(to_noise_units(values * 1_000.0), to_noise_units(values), atol=1e-9)
    np.testing.assert_allclose(to_noise_units(values + 5_000.0), to_noise_units(values), atol=1e-9)


def test_to_noise_units_gives_unit_noise_around_zero() -> None:
    rng = np.random.default_rng(4)
    scaled = to_noise_units(25_000.0 + rng.normal(0, 400.0, size=2_000))
    assert np.std(scaled) == pytest.approx(1.0, rel=0.1)
    assert np.median(scaled) == pytest.approx(0.0, abs=1e-9)


def test_to_noise_units_keeps_the_break_as_large_as_it_is_against_the_noise() -> None:
    """Сдвиг в восемь стандартных отклонений шума остаётся сдвигом в восемь единиц. Деление на
    разброс самого ряда сжало бы его до двух: чем крупнее слом, тем хуже он был бы виден."""
    rng = np.random.default_rng(6)
    values = 1_000.0 + rng.normal(0, 50.0, size=400)
    values[200:] += 400.0
    scaled = to_noise_units(values)
    assert np.median(scaled[200:]) - np.median(scaled[:200]) == pytest.approx(8.0, rel=0.15)


def test_periodic_alarms_fire_on_schedule_whatever_the_data() -> None:
    assert periodic_alarms(24, 5) == [5, 10, 15, 20]
    assert periodic_alarms(24, 12) == [12]
    assert periodic_alarms(24, 24) == []
    assert periodic_alarms(10, 12) == []


@pytest.mark.parametrize("period", [0, -3])
def test_periodic_alarms_reject_a_period_that_is_not_positive(period: int) -> None:
    with pytest.raises(ValueError, match="положительным"):
        periodic_alarms(24, period)
