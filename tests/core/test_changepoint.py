import numpy as np
import pandas as pd
import pytest

from sbx.core.changepoint.hazard import (
    alarms_from_probabilities,
    build_hazard_table,
    lead_time_table,
    make_labels,
    residual_trajectory_features,
)
from sbx.core.changepoint.metrics import (
    covering,
    detection_delays,
    f1_margin,
    false_alarm_rate,
    match_changepoints,
    nearest_rate,
    recall_by,
    recall_gap_interval,
    score_detector,
    window_coverage,
)
from sbx.core.changepoint.online import (
    bocpd,
    bocpd_run_length_probs,
    conformal_alarm,
    cusum,
    page_hinkley,
)
from sbx.core.changepoint.residuals import (
    freshest_forecast,
    interval_exceedance,
    standardized_residuals,
)
from sbx.core.changepoint.synthetic import BREAK_KINDS, build_benchmark, inject, split_benchmark


def _flat(n: int = 36, level: float = 100.0) -> np.ndarray:
    return np.full(n, level)


@pytest.mark.parametrize("kind", BREAK_KINDS)
def test_injection_changes_only_after_tau(kind: str) -> None:
    rng = np.random.default_rng(0)
    base = _flat()
    tau = 20
    out = inject(base, kind, tau, 0.3, rng)
    np.testing.assert_allclose(out[:tau], base[:tau])
    assert not np.allclose(out[tau:], base[tau:])
    assert (out >= 0).all()


def test_level_shift_magnitude_matches_delta() -> None:
    rng = np.random.default_rng(0)
    out = inject(_flat(), "level_shift", 12, 0.25, rng)
    assert out[12] == pytest.approx(125.0)
    assert out[-1] == pytest.approx(125.0)


def test_temporary_drop_recovers() -> None:
    rng = np.random.default_rng(0)
    out = inject(_flat(), "temporary_drop", 10, 0.5, rng, duration=3)
    assert out[10:13].max() == pytest.approx(50.0)
    assert out[13] == pytest.approx(100.0)


def test_injection_is_deterministic_for_seed() -> None:
    a = inject(_flat(), "variance_jump", 10, 0.3, np.random.default_rng(7))
    b = inject(_flat(), "variance_jump", 10, 0.3, np.random.default_rng(7))
    np.testing.assert_allclose(a, b)


def _panel(n_series: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    ds = pd.date_range("2023-01-01", periods=24, freq="MS")
    rows = []
    for i in range(n_series):
        for d in ds:
            rows.append(
                {
                    "unique_id": f"u{i}",
                    "ds": d,
                    "y": 1000.0 * (1 + i % 5) + 10 * d.month,
                    "category": "Все категории" if i % 2 else "Продовольствие",
                }
            )
    panel = pd.DataFrame(rows)
    static = panel.drop_duplicates("unique_id")[["unique_id", "category"]].assign(
        size_group=lambda d: (d.index % 3) + 1, is_complete=True
    )
    return panel, static


def test_benchmark_is_stratified_deterministic_and_has_controls() -> None:
    panel, static = _panel()
    series_a, labels_a = build_benchmark(panel, static, n_series=20, seed=3, control_share=0.3)
    series_b, labels_b = build_benchmark(panel, static, n_series=20, seed=3, control_share=0.3)
    pd.testing.assert_frame_equal(series_a, series_b)
    pd.testing.assert_frame_equal(labels_a, labels_b)
    assert (labels_a["kind"] == "none").any(), "нужны контрольные ряды без слома"
    assert (labels_a["kind"] != "none").any()
    injected = labels_a[labels_a["tau"] >= 0]
    assert injected["tau"].between(4, 19).all()
    calibration, test = split_benchmark(labels_a, seed=1)
    assert set(calibration) | set(test) == set(labels_a["unique_id"])
    assert not set(calibration) & set(test)


def test_benchmark_excludes_requested_series() -> None:
    panel, static = _panel()
    excluded = ["u0", "u1", "u2"]
    _, labels = build_benchmark(panel, static, n_series=20, seed=0, exclude_ids=excluded)
    assert not set(labels["source_id"]) & set(excluded)


def test_match_and_f1_with_margin() -> None:
    assert match_changepoints([10], [11], margin=1) == [(10, 11)]
    assert match_changepoints([10], [13], margin=1) == []
    score = f1_margin({"a": [10], "b": [5]}, {"a": [11], "b": [20]}, margin=1)
    assert score.recall == pytest.approx(0.5)
    assert score.precision == pytest.approx(0.5)
    assert score.f1 == pytest.approx(0.5)


def test_covering_is_one_for_exact_segmentation() -> None:
    assert covering([10], [10], 20) == pytest.approx(1.0)
    assert covering([10], [], 20) < 1.0


def test_delays_and_false_alarm_rate() -> None:
    delays = detection_delays({"a": 10, "b": 5}, {"a": [12], "b": []})
    assert delays.set_index("unique_id").loc["a", "delay"] == 2
    assert not delays.set_index("unique_id").loc["b", "detected"]
    # Ложные: 1 и 12 у «a» (вне окна ±1 вокруг tau=10) и 3 у контрольного «b» → 3 на 4 ряда-года.
    rate = false_alarm_rate({"a": [1, 12], "b": [3]}, {"a": 10, "b": -1}, {"a": 24, "b": 24})
    assert rate == pytest.approx(3 / 4)


def test_score_detector_and_recall_by() -> None:
    labels = pd.DataFrame(
        {
            "unique_id": ["a", "b", "c"],
            "tau": [10, 12, -1],
            "kind": ["level_shift", "level_shift", "none"],
            "delta": [0.2, 0.4, 0.0],
            "n_obs": [24, 24, 24],
        }
    )
    alarms = {"a": [10], "b": [20], "c": [4]}
    score = score_detector(labels, alarms)
    assert score["recall"] == pytest.approx(0.5)
    assert score["precision"] == pytest.approx(1 / 3)
    # Задержка считается по первой тревоге после слома без окна допуска: 0 у «a», 8 у «b».
    assert score["mean_delay"] == pytest.approx(4.0)
    assert score["false_alarms_per_series_year"] == pytest.approx(2 / 6)
    by_delta = recall_by(labels, alarms, "delta")
    assert by_delta.set_index("delta").loc[0.2, "recall"] == 1.0


def test_score_detector_tells_how_often_series_without_a_break_are_alarmed() -> None:
    """Метод, который поднимает тревогу на рядах без слома так же часто, как на рядах со
    сломом, сломов не находит — какой бы ни была его полнота в окне допуска."""
    labels = pd.DataFrame(
        {
            "unique_id": list("abcde"),
            "tau": [10, 12, 14, -1, -1],
            "kind": ["level_shift"] * 3 + ["none"] * 2,
            "delta": [0.2, 0.2, 0.2, 0.0, 0.0],
            "n_obs": [24] * 5,
        }
    )
    alarms = {"a": [10], "b": [], "c": [3, 20], "d": [5]}
    score = score_detector(labels, alarms)
    assert score["alarmed_break_share"] == pytest.approx(2 / 3)
    assert score["alarmed_control_share"] == pytest.approx(1 / 2), "у ряда «e» тревог нет"
    only_breaks = score_detector(labels[labels["tau"] >= 0], alarms)
    assert np.isnan(only_breaks["alarmed_control_share"])
    assert only_breaks["alarmed_break_share"] == pytest.approx(2 / 3)


def test_window_coverage_is_the_chance_that_a_random_date_counts_as_found() -> None:
    assert window_coverage(20, [5], margin=2) == pytest.approx(5 / 20)
    assert window_coverage(20, [5, 6], margin=2) == pytest.approx(6 / 20), "окна пересекаются"
    assert window_coverage(20, [0, 19], margin=2) == pytest.approx(6 / 20), "окна обрезаны краями"
    assert window_coverage(20, [5], margin=0) == pytest.approx(1 / 20)
    assert window_coverage(20, [], margin=2) == 0.0
    assert window_coverage(0, [3], margin=2) == 0.0


def test_recall_gap_interval_resamples_series_and_keeps_the_pairs() -> None:
    """Два набора тревог сравниваются на одних и тех же рядах: пересэмплируются ряды со
    сломом, и в каждой выборке полнота обоих считается заново. Второй набор находит
    подмножество рядов первого, поэтому разность шумит меньше, чем при независимых выборках."""
    labels = pd.DataFrame(
        {"unique_id": [f"s{i}" for i in range(200)], "tau": [10] * 150 + [-1] * 50, "n_obs": 24}
    )
    first = {f"s{i}": [10] for i in range(90)}
    second = {f"s{i}": [11] for i in range(60)}
    gap = recall_gap_interval(labels, first, second, margin=1, n_boot=500, seed=3)
    assert gap["recall_a"] == pytest.approx(0.6) and gap["recall_b"] == pytest.approx(0.4)
    assert gap["diff"] == pytest.approx(0.2) and gap["n"] == 150
    assert 0.1 < gap["low"] < 0.2 < gap["high"] < 0.3
    assert recall_gap_interval(labels, first, second, margin=1, n_boot=500, seed=3) == gap
    other_seed = recall_gap_interval(labels, first, second, margin=1, n_boot=500, seed=4)
    assert (other_seed["low"], other_seed["high"]) != (gap["low"], gap["high"])
    same = recall_gap_interval(labels, first, first, margin=1, n_boot=100, seed=0)
    assert same["diff"] == same["low"] == same["high"] == 0.0
    # Тревога вне окна допуска слом не находит.
    late = {f"s{i}": [13] for i in range(150)}
    assert recall_gap_interval(labels, late, second, margin=1, n_boot=50, seed=0)["recall_a"] == 0.0
    wider = recall_gap_interval(labels, first, second, margin=1, n_boot=500, seed=3, level=0.99)
    assert wider["low"] < gap["low"] and wider["high"] > gap["high"]


def test_recall_gap_interval_without_breaks_is_empty() -> None:
    labels = pd.DataFrame({"unique_id": ["a", "b"], "tau": [-1, -1], "n_obs": 24})
    gap = recall_gap_interval(labels, {"a": [3]}, {}, margin=1, n_boot=10, seed=0)
    assert gap["n"] == 0 and np.isnan(gap["diff"])


def test_nearest_rate_picks_the_schedule_as_noisy_as_the_detector() -> None:
    rates = {12: 0.42, 8: 0.87, 6: 1.28}
    assert nearest_rate(rates, 0.89) == 8
    assert nearest_rate(rates, 1.2) == 6
    assert nearest_rate(rates, 0.1) == 12
    # При равном расстоянии берётся более редкое расписание: оно не завышает полноту эталона.
    assert nearest_rate({8: 0.8, 6: 1.0}, 0.9) == 8
    with pytest.raises(ValueError):
        nearest_rate({}, 0.5)


def test_cusum_and_page_hinkley_detect_shift_but_not_noise() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(size=120)
    shifted = np.concatenate([rng.normal(size=60), rng.normal(loc=3.0, size=60)])
    assert cusum(noise, threshold=8.0) == []
    alarms = cusum(shifted, threshold=8.0)
    assert alarms and 60 <= alarms[0] <= 66
    assert page_hinkley(noise, threshold=20.0) == []
    ph = page_hinkley(shifted, threshold=20.0)
    assert ph and 60 <= ph[0] <= 70


def test_bocpd_flags_restart_near_shift() -> None:
    rng = np.random.default_rng(1)
    signal = np.concatenate([rng.normal(size=40), rng.normal(loc=5.0, size=40)])
    # Тревога датируется моментом подтверждения обрыва пробега: слом на 40 → тревога на 42.
    alarms = bocpd(signal, hazard=1 / 20, threshold=0.25, confirm=2)
    assert any(40 <= a <= 45 for a in alarms)
    quiet = bocpd(rng.normal(size=80), hazard=1 / 20, threshold=0.25, confirm=2)
    assert quiet == []


def test_bocpd_restart_probability_is_constant_for_fixed_hazard() -> None:
    """P(r_t = 0) при постоянном hazard тождественно равна hazard — поэтому сигнал другой."""
    rng = np.random.default_rng(3)
    signal = np.concatenate([rng.normal(size=20), rng.normal(loc=6.0, size=20)])
    probs = bocpd_run_length_probs(signal, hazard=0.05)
    np.testing.assert_allclose(probs[1:, 0], 0.05, atol=1e-9)


def test_detectors_use_only_past_observations() -> None:
    rng = np.random.default_rng(2)
    signal = np.concatenate([rng.normal(size=30), rng.normal(loc=4.0, size=30)])
    full = cusum(signal, threshold=6.0)
    prefix = cusum(signal[:35], threshold=6.0)
    assert [a for a in full if a < 35] == prefix
    full_ph = page_hinkley(signal, threshold=8.0)
    assert [a for a in full_ph if a < 35] == page_hinkley(signal[:35], threshold=8.0)


def test_conformal_alarm_needs_consecutive_exceedances() -> None:
    assert conformal_alarm([0, 1, 0, 1, 1, 0], consecutive=2) == [4]
    assert conformal_alarm([1, 1, 1], consecutive=3) == [2]
    y = np.array([1.0, 5.0, 2.0])
    assert list(interval_exceedance(y, np.array([0.0, 0.0, 0.0]), np.array([3.0, 3.0, 3.0]))) == [
        0,
        1,
        0,
    ]


def test_standardized_residuals_scale() -> None:
    y = np.array([10.0, 12.0, 14.0])
    y_hat = np.array([10.0, 10.0, 10.0])
    z = standardized_residuals(y, y_hat, scale=2.0)
    np.testing.assert_allclose(z, [0.0, 1.0, 2.0])


def test_hazard_labels_and_lead_time() -> None:
    labels = make_labels({"a": [10]}, {"a": 24}, horizons=(1, 2, 3))
    row = labels.set_index("t")
    assert row.loc[9, "z_h1"] == 1
    assert row.loc[8, "z_h1"] == 0 and row.loc[8, "z_h2"] == 1
    assert row.loc[10, "z_h3"] == 0, "в момент слома метка «слом впереди» уже нулевая"

    residuals = pd.DataFrame({"unique_id": "a", "t": range(24), "z": np.linspace(0, 3, 24)})
    table = build_hazard_table(residuals, labels, detector_alarms={"a": [9]})
    assert {"resid_z", "resid_mean_3", "detector_alarm"} <= set(table.columns)
    assert table.set_index("t").loc[9, "detector_alarm"] == 1

    probs = pd.DataFrame({"unique_id": "a", "t": range(24), "p": [0.1] * 8 + [0.9] + [0.1] * 15})
    assert alarms_from_probabilities(probs, 0.5) == {"a": [8]}
    lead = lead_time_table(probs, {"a": [10]}, 0.5)
    assert lead.iloc[0]["lead_time"] == 2 and bool(lead.iloc[0]["warned"])


def test_residual_trajectory_features_are_causal() -> None:
    z = np.array([0.0, 0.0, 5.0, 5.0])
    feats = residual_trajectory_features(z, windows=(2,))
    # признак на t=1 не должен зависеть от всплеска на t=2
    assert feats.loc[1, "resid_mean_2"] == pytest.approx(0.0)
    assert feats.loc[2, "resid_mean_2"] == pytest.approx(2.5)


def test_positions_for_dates_maps_events_to_grid() -> None:
    from sbx.core.changepoint.metrics import positions_for_dates

    monthly = pd.date_range("2020-01-01", periods=12, freq="MS")
    assert positions_for_dates(monthly, [pd.Timestamp("2020-04-01")]) == [3]
    # События вне сетки ряда отбрасываются, а не приписываются краю.
    assert positions_for_dates(monthly, [pd.Timestamp("2019-01-01")]) == []
    assert positions_for_dates(monthly, [pd.Timestamp("2021-06-01")]) == []
    # Недельная сетка: берётся первое наблюдение на дату события или после неё.
    weekly = pd.date_range("2024-09-29", periods=8, freq="W-SUN")
    assert positions_for_dates(weekly, [pd.Timestamp("2024-10-01")]) == [1]
    # Дубликаты схлопываются и порядок возрастающий.
    assert positions_for_dates(monthly, [pd.Timestamp("2020-03-01")] * 2) == [2]


def test_series_invariant_columns_flags_country_level_features() -> None:
    """Признак, одинаковый для всех рядов в каждом t, — это переобозначение позиции."""
    from sbx.core.changepoint.hazard import series_invariant_columns

    rows = []
    # Смещение задаётся явно, а не через hash(): хеш строк в Python рандомизируется от запуска
    # к запуску, и примерно в одном прогоне из девяти все три ряда получали одно смещение —
    # «региональный» признак становился одинаковым, и тест падал на ровном месте.
    for offset, uid in enumerate(("a", "b", "c")):
        for t in range(6):
            rows.append(
                {
                    "unique_id": uid,
                    "t": t,
                    "country_level": float(t),  # одинаков у всех рядов в момент t
                    "regional": float(t) + offset,  # различается между рядами
                    "constant": 1.0,
                }
            )
    table = pd.DataFrame(rows)
    invariant = series_invariant_columns(table, ["country_level", "regional", "constant"])
    assert "country_level" in invariant
    assert "constant" in invariant
    assert "regional" not in invariant


def test_series_invariant_ignores_missing_values() -> None:
    """Пропуск у части рядов не должен делать страновой признак «различающимся»."""
    from sbx.core.changepoint.hazard import series_invariant_columns

    rows = []
    for uid in ("a", "b"):
        for t in range(4):
            value = float(t) if uid == "a" else float("nan")
            rows.append({"unique_id": uid, "t": t, "country_level": value})
    table = pd.DataFrame(rows)
    assert series_invariant_columns(table, ["country_level"]) == ["country_level"]


def test_label_rate_depends_on_position_which_is_why_invariants_leak() -> None:
    """Обоснование отбраковки: доля положительных меток зависит от позиции в ряду."""
    breaks = {f"s{i}": [4 + (i % 12)] for i in range(120)}
    lengths = dict.fromkeys(breaks, 20)
    labels = make_labels(breaks, lengths, (3,))
    rate = labels.groupby("t")["z_h3"].mean()
    assert rate.loc[0] == 0.0, "у самого начала ряда слома впереди быть не может"
    assert rate.loc[19] == 0.0, "у конца ряда горизонт выходит за пределы"
    assert rate.max() > 0.1


def test_freshest_forecast_keeps_one_row_per_month_from_the_shortest_horizon() -> None:
    """Горизонты перекрываются: один месяц предсказан в фолдах h=1, 3 и 12.

    Поток для детектора и графиков — ряд без повторов, и на каждый месяц берётся самый
    короткий горизонт: это самый точный из доступных прогнозов.
    """
    ds = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"])
    oof = pd.DataFrame(
        {
            "unique_id": ["a"] * 5,
            "ds": ds,
            "model": ["M"] * 5,
            "horizon": [12, 1, 3, 12, 3],
            "cutoff": pd.to_datetime(
                ["2023-10-01", "2023-12-01", "2023-12-01", "2023-11-01", "2023-12-01"]
            ),
            "y_hat": [12.0, 1.0, 3.0, 12.0, 3.0],
        }
    )
    out = freshest_forecast(oof)
    assert list(out["ds"]) == list(pd.to_datetime(["2024-01-01", "2024-02-01"]))
    assert list(out["y_hat"]) == [1.0, 3.0]


def test_freshest_forecast_prefers_the_latest_cutoff_within_a_horizon() -> None:
    oof = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": pd.to_datetime(["2024-06-01", "2024-06-01"]),
            "model": ["M", "M"],
            "horizon": [6, 6],
            "cutoff": pd.to_datetime(["2023-12-01", "2024-03-01"]),
            "y_hat": [1.0, 2.0],
        }
    )
    assert list(freshest_forecast(oof)["y_hat"]) == [2.0]


def test_freshest_forecast_can_drop_long_horizons() -> None:
    """Живой детектор работает на коротких прогнозах: фолды h=6 и h=12 нужны для оценки
    точности на длинных горизонтах, а не для потока тревог."""
    oof = pd.DataFrame(
        {
            "unique_id": ["a", "a", "a"],
            "ds": pd.to_datetime(["2023-12-01", "2024-01-01", "2024-01-01"]),
            "model": ["M"] * 3,
            "horizon": [12, 12, 3],
            "y_hat": [12.0, 12.0, 3.0],
        }
    )
    out = freshest_forecast(oof, max_horizon=3)
    assert list(out["ds"]) == [pd.Timestamp("2024-01-01")]
    assert list(out["y_hat"]) == [3.0]
