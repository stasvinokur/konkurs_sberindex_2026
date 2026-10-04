"""Офлайн-детекторы и калибровка порогов на управляемом синтетическом наборе."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sbx.core.changepoint.metrics import score_detector
from sbx.core.changepoint.synthetic import inject, split_benchmark
from sbx.shell.cpd_offline import (
    OFFLINE_METHODS,
    adwin_alarms,
    detect_offline,
    detect_offline_panel,
)


def _series_with_shift(n: int = 36, tau: int = 18, delta: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = 1000.0 + rng.normal(0, 20, size=n)
    return inject(base, "level_shift", tau, delta, rng)


@pytest.mark.parametrize("method", OFFLINE_METHODS)
def test_offline_methods_find_injected_level_shift(method: str) -> None:
    """При подходящем штрафе каждый метод находит слом в пределах ±2 месяцев."""
    values = _series_with_shift()
    found = []
    for penalty in (1, 10, 100, 1000, 10000, 100000):
        found += [p for p in detect_offline(values, method, float(penalty)) if abs(p - 18) <= 2]
    assert found, f"{method} не нашёл слом ни при одном штрафе из сетки"


def test_offline_detects_nothing_on_flat_series_with_high_penalty() -> None:
    rng = np.random.default_rng(1)
    flat = 1000.0 + rng.normal(0, 20, size=36)
    assert detect_offline(flat, "pelt_l2", penalty=1e7) == []


# Штрафы из середины сетки `configs/cpd.yaml`, при которых метод работает: у квадратичной
# стоимости штраф — в дисперсиях шума, у ядерной стоимость сегмента ограничена его длиной. Это
# не откалиброванные значения: скользящему окну калибровка на бенчмарке отдаёт нижний край сетки
# (0,5), и при нём оно поднимает тревогу и на чистом шуме — см. reports/changepoints.md.
WORKING_PENALTY = {
    "pelt_l2": 30.0,
    "binseg_l2": 30.0,
    "window_l2": 30.0,
    "pelt_rbf": 3.0,
    "kernel_rbf": 3.0,
}


@pytest.mark.parametrize("method", OFFLINE_METHODS)
def test_offline_methods_locate_a_break_in_a_month_that_is_not_a_multiple_of_five(
    method: str,
) -> None:
    """Слом ставится там, где он есть. Умолчание ruptures (`jump=5`) оставляло методам только
    позиции, кратные пяти: на 24 точках это четыре месяца, и «найденный» слом был ближайшим
    узлом сетки, а не местом слома."""
    values = _series_with_shift(n=36, tau=17, delta=0.3, seed=0)
    assert detect_offline(values, method, WORKING_PENALTY[method]) == [17]


@pytest.mark.parametrize("method", OFFLINE_METHODS)
def test_offline_result_does_not_depend_on_the_units_of_the_series(method: str) -> None:
    """Один штраф означает одно и то же для ряда в сотни рублей и в сотни тысяч. Без этого
    квадратичная стоимость на крупных рядах окупает любой разрез, и метод режет ряд в каждой
    допустимой точке при любом штрафе из сетки."""
    values = _series_with_shift(n=36, tau=17, delta=0.3, seed=3)
    for penalty in (1.0, 10.0, 100.0, 1000.0):
        in_roubles = detect_offline(values, method, penalty)
        in_thousands = detect_offline(values / 1000.0, method, penalty)
        in_kopecks = detect_offline(values * 100.0, method, penalty)
        assert in_roubles == in_thousands == in_kopecks, f"штраф {penalty}"


@pytest.mark.parametrize("method", ["pelt_l2", "binseg_l2", "window_l2"])
def test_a_series_without_a_break_is_not_cut_on_a_grid(method: str) -> None:
    """Ряд уровня «Все категории» (десятки тысяч рублей) без слома: тревог быть не должно.
    До перевода входа в единицы шума метод ставил их в позициях 5, 10, 15 и 20 при любом
    штрафе из сетки."""
    rng = np.random.default_rng(1)
    flat = 25_000.0 + rng.normal(0, 400.0, size=24)
    assert detect_offline(flat, method, WORKING_PENALTY[method]) == []


def test_detect_offline_panel_returns_alarms_per_series() -> None:
    frames = []
    for i, tau in enumerate([10, 20]):
        values = _series_with_shift(tau=tau, seed=i)
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": f"s{i}",
                    "ds": pd.date_range("2023-01-01", periods=len(values), freq="MS"),
                    "y": values,
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)
    alarms = detect_offline_panel(panel, "pelt_l2", penalty=50.0)
    assert set(alarms) == {"s0", "s1"}
    assert alarms["s0"] and alarms["s1"]


def test_calibration_picks_penalty_with_best_f1() -> None:
    """Калибровка на одной части и оценка на другой — как в пайплайне."""
    frames, labels = [], []
    for i in range(12):
        tau = 12 + (i % 5)
        values = _series_with_shift(tau=tau, seed=i)
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": f"s{i}",
                    "ds": pd.date_range("2023-01-01", periods=len(values), freq="MS"),
                    "y": values,
                }
            )
        )
        labels.append(
            {
                "unique_id": f"s{i}",
                "tau": tau,
                "n_obs": len(values),
                "kind": "level_shift",
                "delta": 0.3,
            }
        )
    panel = pd.concat(frames, ignore_index=True)
    label_frame = pd.DataFrame(labels)

    scores = {}
    for penalty in (1.0, 100.0, 10000.0, 10_000_000.0):
        alarms = detect_offline_panel(panel, "pelt_l2", penalty)
        scores[penalty] = score_detector(label_frame, alarms)["f1"]
    best_penalty = max(scores, key=scores.get)
    assert scores[best_penalty] > 0.3
    # Слишком маленький штраф режет ряд на куски, слишком большой не находит ничего.
    assert scores[1.0] < scores[best_penalty]
    assert scores[10_000_000.0] == 0.0


SHORT_SERIES = "synth_00000"  # ряд из 13 месяцев; при `split_seed=1` попадает в тестовую часть


def _benchmark_with_breaks_on_a_schedule(n_series: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ряды из одного шума, а разметка стоит в месяцах 6, 12 и 18: расписание с шагом 6
    попадает в каждую метку, а методам, читающим ряд, искать нечего."""
    rng = np.random.default_rng(0)
    frames, labels = [], []
    for i in range(n_series):
        uid = f"synth_{i:05d}"
        n_obs = 13 if uid == SHORT_SERIES else 24
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": uid,
                    "ds": pd.date_range("2023-01-01", periods=n_obs, freq="MS"),
                    "y": 1000.0 + rng.normal(0, 20.0, size=n_obs),
                    "category": "Все категории",
                }
            )
        )
        labels.append(
            {
                "unique_id": uid,
                "tau": (6, 12, 18)[i % 3],
                "n_obs": n_obs,
                "kind": "level_shift",
                "delta": 0.05,
            }
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(labels)


def _compare_on_schedule_benchmark(tmp_path, monkeypatch, **settings) -> pd.DataFrame:
    from sbx.shell.pipelines import cpd

    seasonal = pd.DataFrame({"category": "Все категории", "month": range(1, 13), "index": 1.0})
    seasonal.to_parquet(tmp_path / "seasonal_index.parquet", index=False)
    monkeypatch.setattr(cpd, "PROCESSED", tmp_path)
    series, labels = _benchmark_with_breaks_on_a_schedule()
    cfg = {
        "offline_methods": ["pelt_l2"],
        "penalty_grid": [30.0],
        "online_detectors": ["cusum"],
        "threshold_grids": {"cusum": [3.0]},
        "split_seed": 1,
        "margin": 1,
        **settings,
    }
    return cpd.calibrate_and_evaluate(series, labels, cfg, out_dir=tmp_path / "cpd")


def test_comparison_includes_an_alarm_on_schedule_as_the_reference(tmp_path, monkeypatch) -> None:
    """Эталон сравнения — тревога по расписанию, не глядя на данные. Метод, который не лучше
    расписания, ничего не находит. Период подбирается на калибровочной части, как параметры
    методов."""
    results = _compare_on_schedule_benchmark(tmp_path, monkeypatch, baseline_periods=[4, 6])

    reference = results[results["kind"] == "baseline"]
    assert reference["detector"].tolist() == ["periodic"]
    row = reference.iloc[0]
    # Шаг 4 попадает только в метку 12, шаг 6 — в каждую: калибровка обязана выбрать 6.
    assert row["param"] == 6.0
    assert row["recall"] == 1.0
    _, labels = _benchmark_with_breaks_on_a_schedule()
    _, test_ids = split_benchmark(labels, seed=1)
    assert SHORT_SERIES in test_ids
    assert row["n_alarms"] == 3 * len(test_ids) - 1, "тревоги — только у рядов тестовой части"

    alarms = json.loads((tmp_path / "cpd" / "alarms.json").read_text(encoding="utf-8"))[
        "baseline::periodic"
    ]
    assert set(alarms) == set(test_ids)
    # Расписание идёт до конца своего ряда: у ряда из 13 месяцев тревог две, а не три.
    assert alarms.pop(SHORT_SERIES) == [6, 12]
    assert all(found == [6, 12, 18] for found in alarms.values())
    by_break = pd.read_csv(tmp_path / "cpd" / "recall_by_break.csv")
    assert "baseline::periodic" in set(by_break["detector"])


def test_the_period_of_the_schedule_is_chosen_on_the_calibration_half(
    tmp_path, monkeypatch
) -> None:
    """В калибровочной половине метки стоят в позициях, кратных шести, в тестовой — кратных
    четырём. Период обязан прийти из калибровочной половины, хотя на тестовой лучше другой."""
    from sbx.shell.pipelines import cpd

    seasonal = pd.DataFrame({"category": "Все категории", "month": range(1, 13), "index": 1.0})
    seasonal.to_parquet(tmp_path / "seasonal_index.parquet", index=False)
    monkeypatch.setattr(cpd, "PROCESSED", tmp_path)
    series, labels = _benchmark_with_breaks_on_a_schedule()
    calibration_ids, test_ids = split_benchmark(labels, seed=1)
    in_test = labels["unique_id"].isin(set(test_ids))
    labels.loc[in_test, "tau"] = [(4, 8, 12, 16, 20)[i % 5] for i in range(int(in_test.sum()))]
    cfg = {
        "offline_methods": ["pelt_l2"],
        "penalty_grid": [30.0],
        "online_detectors": ["cusum"],
        "threshold_grids": {"cusum": [3.0]},
        "baseline_periods": [4, 6],
        "split_seed": 1,
        "margin": 1,
    }
    results = cpd.calibrate_and_evaluate(series, labels, cfg, out_dir=tmp_path / "cpd")
    row = results[results["kind"] == "baseline"].iloc[0]
    assert row["param"] == 6.0
    assert row["recall"] < 1.0, "на тестовой половине шаг 6 попадает не во все метки"


def test_the_reference_is_never_reported_as_the_best_method(tmp_path, monkeypatch) -> None:
    """На этом наборе расписание точнее всех, но это не метод: разрез по типам сломов строится
    для лучшего метода, читающего ряд."""
    from sbx.shell.pipelines import cpd

    results = _compare_on_schedule_benchmark(tmp_path, monkeypatch, baseline_periods=[6])

    assert results.iloc[0]["kind"] == "baseline", "таблица отсортирована по F1"
    assert cpd.best_offline(results) == ("pelt_l2", 30.0)
    best = pd.read_csv(tmp_path / "cpd" / "best_detector_recall.csv")
    assert best["recall"].max() < 1.0, "полнота расписания здесь равна единице"


def test_comparison_has_no_reference_row_unless_periods_are_configured(
    tmp_path, monkeypatch
) -> None:
    results = _compare_on_schedule_benchmark(tmp_path, monkeypatch)
    assert set(results["kind"]) == {"offline", "online"}


def test_delay_curve_carries_the_schedule_at_every_period(tmp_path, monkeypatch) -> None:
    """На кривой «задержка — ложные тревоги» расписание даёт линию отсчёта: детектор лучше
    расписания, если при той же частоте ложных тревог срабатывает раньше."""
    from sbx.shell.pipelines import cpd

    seasonal = pd.DataFrame({"category": "Все категории", "month": range(1, 13), "index": 1.0})
    seasonal.to_parquet(tmp_path / "seasonal_index.parquet", index=False)
    monkeypatch.setattr(cpd, "PROCESSED", tmp_path)
    series, labels = _benchmark_with_breaks_on_a_schedule()
    cfg = {
        "online_detectors": ["cusum"],
        "threshold_grids": {"cusum": [3.0, 5.0]},
        "baseline_periods": [4, 6],
        "margin": 1,
    }
    curve = cpd.delay_vs_false_alarms(series, labels, cfg, out_dir=tmp_path)

    assert curve["detector"].tolist() == ["cusum", "cusum", "periodic", "periodic"]
    schedule = curve[curve["detector"] == "periodic"].set_index("threshold")
    assert schedule.index.tolist() == [4.0, 6.0]
    # Весь бенчмарк, как и у детекторов: 39 рядов по 24 месяца и один из 13.
    assert schedule.loc[6.0, "n_alarms"] == 39 * 3 + 2
    assert schedule.loc[4.0, "n_alarms"] == 39 * 5 + 3
    assert schedule.loc[6.0, "recall"] == 1.0 and schedule.loc[6.0, "mean_delay"] == 0.0
    written = pd.read_csv(tmp_path / "delay_vs_false_alarms.csv")
    assert written["detector"].tolist() == curve["detector"].tolist()
    without = cpd.delay_vs_false_alarms(
        series, labels, {k: v for k, v in cfg.items() if k != "baseline_periods"}, tmp_path
    )
    assert set(without["detector"]) == {"cusum"}


def test_schedule_match_puts_the_detector_against_an_equally_noisy_schedule(
    tmp_path, monkeypatch
) -> None:
    """Период расписания подбирается по частоте ложных тревог детектора на калибровочной
    половине, сравнение идёт на тестовой. В тестовой половине ряды неспокойные, и детектор
    тревожится там чаще: подбор по ней дал бы частое расписание."""
    from sbx.shell.pipelines import cpd

    seasonal = pd.DataFrame({"category": "Все категории", "month": range(1, 13), "index": 1.0})
    seasonal.to_parquet(tmp_path / "seasonal_index.parquet", index=False)
    monkeypatch.setattr(cpd, "PROCESSED", tmp_path)
    series, labels = _benchmark_with_breaks_on_a_schedule()
    calibration_ids, test_ids = split_benchmark(labels, seed=1)
    restless = series["unique_id"].isin(set(test_ids))
    month = series["ds"].dt.month
    series.loc[restless & month.isin([4, 5, 6, 10, 11, 12]), "y"] *= 3.0
    cfg = {
        "online_detectors": ["cusum"],
        "threshold_grids": {"cusum": [3.0]},
        "offline_methods": [],
        "baseline_periods": [4, 12],
        "split_seed": 1,
        "margin": 1,
        "seed": 11,
    }
    results = cpd.calibrate_and_evaluate(series, labels, cfg, out_dir=tmp_path)
    match = cpd.schedule_match(series, labels, cfg, results, out_dir=tmp_path)

    row = results[(results["kind"] == "online") & (results["input"] == "residual")].iloc[0]
    assert (match["detector"], match["threshold"]) == ("cusum", 3.0)
    assert match["detector_recall"] == pytest.approx(row["recall"])
    assert match["detector_false_alarms"] == pytest.approx(row["false_alarms_per_series_year"])
    assert match["detector_delay"] == pytest.approx(row["mean_delay"])
    assert match["period"] == 12, "на спокойной калибровочной половине детектор тревожится редко"
    test_labels = labels[labels["unique_id"].isin(set(test_ids))]
    assert match["n"] == len(test_labels)
    # Расписание раз в 12 месяцев попадает только в метку 12.
    assert match["schedule_recall"] == pytest.approx((test_labels["tau"] == 12).mean())
    assert match["recall_diff"] == pytest.approx(
        match["detector_recall"] - match["schedule_recall"]
    )
    assert match["recall_diff_low"] <= match["recall_diff"] <= match["recall_diff_high"]
    assert match["level"] == 0.95
    saved = json.loads((tmp_path / "schedule_match.json").read_text(encoding="utf-8"))
    assert saved == match
    assert cpd.schedule_match(series, labels, cfg, results, out_dir=tmp_path) == match
    # Число повторов и seed бутстрепа приходят из конфига: интервал воспроизводим.
    seen = {}
    interval = cpd.recall_gap_interval

    def recording(*args: object, **kwargs: object) -> dict[str, float]:
        seen.update(kwargs)
        return interval(*args, **kwargs)

    monkeypatch.setattr(cpd, "recall_gap_interval", recording)
    cpd.schedule_match(series, labels, {**cfg, "seed": 12, "bootstrap": 50}, results, tmp_path)
    assert (seen["seed"], seen["n_boot"], seen["level"], seen["margin"]) == (12, 50, 0.95, 1)


def test_schedule_match_needs_a_schedule_and_a_detector_on_residuals(tmp_path, monkeypatch) -> None:
    from sbx.shell.pipelines import cpd

    seasonal = pd.DataFrame({"category": "Все категории", "month": range(1, 13), "index": 1.0})
    seasonal.to_parquet(tmp_path / "seasonal_index.parquet", index=False)
    monkeypatch.setattr(cpd, "PROCESSED", tmp_path)
    series, labels = _benchmark_with_breaks_on_a_schedule()
    results = pd.DataFrame(
        [{"detector": "cusum", "kind": "online", "input": "residual", "param": 3.0, "f1": 0.2}]
    )
    out = tmp_path / "none"
    assert cpd.schedule_match(series, labels, {"split_seed": 1}, results, out_dir=out) == {}
    offline_only = results.assign(kind="offline", input="raw")
    cfg = {"baseline_periods": [6], "split_seed": 1}
    assert cpd.schedule_match(series, labels, cfg, offline_only, out_dir=out) == {}
    assert not (out / "schedule_match.json").exists()


def test_best_offline_is_the_best_method_that_segments_the_series() -> None:
    from sbx.shell.pipelines import cpd

    results = pd.DataFrame(
        [
            {"detector": "periodic", "kind": "baseline", "input": "none", "param": 5.0, "f1": 0.9},
            {"detector": "cusum", "kind": "online", "input": "residual", "param": 3.0, "f1": 0.5},
            {"detector": "pelt_l2", "kind": "offline", "input": "raw", "param": 30.0, "f1": 0.2},
            {"detector": "window_l2", "kind": "offline", "input": "raw", "param": 0.5, "f1": 0.3},
        ]
    )
    assert cpd.best_offline(results) == ("window_l2", 0.5)
    assert cpd.best_offline(results[results["kind"] != "offline"]) is None
    assert cpd.best_offline(pd.DataFrame()) is None


def test_adwin_detects_large_shift_on_long_series() -> None:
    rng = np.random.default_rng(2)
    values = np.concatenate([rng.normal(0, 1, size=200), rng.normal(6, 1, size=200)])
    alarms = adwin_alarms(values, delta=0.002)
    assert alarms and 200 <= alarms[0] <= 240


def test_all_offline_methods_are_callable() -> None:
    values = _series_with_shift()
    for method in OFFLINE_METHODS:
        result = detect_offline(values, method, penalty=100.0)
        assert isinstance(result, list)


def test_build_takes_the_frozen_benchmark_instead_of_sampling_anew(tmp_path, monkeypatch) -> None:
    """Бенчмарк опубликованного расчёта берётся из файлов, а не отбирается заново."""
    from sbx.shell.pipelines import cpd

    series = pd.DataFrame(
        {
            "unique_id": ["synth_00000"] * 2,
            "source_id": ["01-0001__total"] * 2,
            "ds": pd.to_datetime(["2023-01-01", "2023-02-01"]),
            "y": [1.0, 2.0],
            "category": "Все категории",
        }
    )
    labels = pd.DataFrame(
        {
            "unique_id": ["synth_00000"],
            "source_id": ["01-0001__total"],
            "kind": ["none"],
            "tau": [-1],
            "delta": [0.0],
            "n_obs": [2],
        }
    )
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    series.to_parquet(frozen / "bench_series.parquet", index=False)
    labels.to_parquet(frozen / "bench_labels.parquet", index=False)

    def no_panel() -> pd.DataFrame:
        raise AssertionError("замороженному бенчмарку панель не нужна")

    monkeypatch.setattr(cpd, "load_panel", no_panel)
    monkeypatch.setattr(cpd, "load_static", lambda: pd.DataFrame({"unique_id": ["01-0001__total"]}))
    cfg = {"frozen_benchmark": str(frozen / "bench"), "n_series": 5, "seed": 1}
    got_series, got_labels = cpd.build(cfg, out_dir=tmp_path / "out")

    pd.testing.assert_frame_equal(got_series, series, check_dtype=False)
    pd.testing.assert_frame_equal(got_labels, labels, check_dtype=False)
    written = pd.read_parquet(tmp_path / "out" / "benchmark_labels.parquet")
    assert written["source_id"].tolist() == ["01-0001__total"]
    assert (tmp_path / "out" / "benchmark_series.parquet").exists()

    monkeypatch.setattr(cpd, "load_static", lambda: pd.DataFrame({"unique_id": ["other"]}))
    with pytest.raises(ValueError, match="нет в панели"):
        cpd.build(cfg, out_dir=tmp_path / "out2")


def test_detector_inputs_on_a_prefix_equal_the_inputs_on_the_full_series() -> None:
    """Оба входных потока бенчмарка — остатки и сырой ряд — нормируются только по прошлому."""
    from sbx.shell.pipelines import cpd

    rng = np.random.default_rng(3)
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    values = 1000.0 + rng.normal(0, 15, size=24)
    values[15:] *= 1.4
    series = pd.DataFrame(
        {"unique_id": "synth_0", "ds": months, "y": values, "category": "Все категории"}
    )
    seasonal = pd.DataFrame({"category": "Все категории", "month": range(1, 13), "index": 1.0})

    full = cpd._streams(series, seasonal)
    prefix = cpd._streams(series[series["ds"] <= months[11]], seasonal)
    for name in ("residual", "raw"):
        np.testing.assert_allclose(
            full[name]["synth_0"][:12], prefix[name]["synth_0"], equal_nan=True, err_msg=name
        )
    assert np.nanmax(np.abs(full["raw"]["synth_0"][15:])) > 5, "сдвиг виден и в сыром потоке"


def test_live_alarms_use_ensemble_residuals_and_its_own_intervals(tmp_path, monkeypatch) -> None:
    """Эксплуатационный контур: остатки ансамбля плюс конформный детектор по его интервалам."""
    from sbx.shell.pipelines import cpd

    ds = pd.date_range("2024-01-01", periods=3, freq="MS")
    rows = []
    for uid, shift in [("a", 0.0), ("b", 900.0)]:
        for d in ds:
            rows.append(
                {
                    "unique_id": uid,
                    "ds": d,
                    "fold": "h3_2023-12",
                    "horizon": 3,
                    "step": 1,
                    "model": "Ensemble (по категориям)",
                    "y": 1000.0 + shift,
                    "y_hat": 1000.0,
                    "q_0.1": 900.0,
                    "q_0.9": 1100.0,
                    "mase_scale": 10.0,
                }
            )
    oof_dir = tmp_path / "oof"
    oof_dir.mkdir()
    pd.DataFrame(rows).to_parquet(oof_dir / "ensemble.parquet", index=False)
    monkeypatch.setattr(cpd, "OOF_DIR", oof_dir)

    results = pd.DataFrame(
        [{"detector": "cusum", "kind": "online", "input": "residual", "param": 2.0, "f1": 0.2}]
    )
    summary = cpd.live_alarms(results, out_dir=tmp_path)

    assert summary["model"] == "Ensemble (по категориям)"
    assert summary["series"] == 2
    # Смещение остатков и охват тревогами: без них частота тревог читается как частота сдвигов.
    assert summary["series_with_alarm"] == 1, "тревога есть только у ряда «b»"
    assert summary["above_forecast_share"] == 0.5, "факт выше прогноза в трёх месяцах из шести"
    payload = json.loads((tmp_path / "live_alarms.json").read_text(encoding="utf-8"))
    # Ряд «b» стабильно выше верхнего квантиля → конформная тревога; «a» внутри интервала.
    assert "b" in payload["conformal"] and "a" not in payload["conformal"]


def test_live_alarms_returns_empty_without_ensemble(tmp_path) -> None:
    from sbx.shell.pipelines import cpd

    assert cpd.live_alarms(pd.DataFrame(), out_dir=tmp_path) == {}


def test_threshold_for_alarm_rate_matches_requested_share() -> None:
    """Порог для сравнения при равном числе тревог: доля превышений равна заданной."""
    from sbx.shell.hazard_model import threshold_for_alarm_rate

    probabilities = pd.DataFrame({"p": np.linspace(0.0, 1.0, 1001)})
    for target in (0.05, 0.2, 0.5):
        threshold = threshold_for_alarm_rate(probabilities, target)
        share = float((probabilities["p"] >= threshold).mean())
        assert abs(share - target) < 0.01, f"доля {share} против цели {target}"
    # Вырожденные случаи не ломают расчёт.
    assert threshold_for_alarm_rate(pd.DataFrame({"p": []}), 0.1) == 1.0


def test_national_case_matches_events_and_writes_figure(tmp_path, monkeypatch) -> None:
    """Разбор примера: найденный слом сопоставляется событию, график сохраняется."""
    from sbx.shell.pipelines import cases

    ds = pd.date_range("2019-01-01", periods=48, freq="MS")
    values = np.full(48, 4000.0)
    shock = list(ds).index(pd.Timestamp("2020-04-01"))
    values[shock:] = 2500.0
    frame = pd.DataFrame({"series": "Всего", "ds": ds, "value": values})
    monkeypatch.setattr(
        cases, "load_national_series", lambda: {"monthly": frame, "weekly": frame.iloc[:0]}
    )
    monkeypatch.setattr(cases, "FIGURES", tmp_path)

    shocks = {
        "margin_months": 2,
        "events": [
            {"date": "2020-04-01", "name": "тест", "series": ["monthly"], "source": "http://x"},
            {"date": "2019-06-01", "name": "тишина", "series": ["monthly"], "source": "http://y"},
        ],
    }
    asked = []
    detect = cases.detect_offline

    def recording(values: np.ndarray, method: str, penalty: float) -> list[int]:
        asked.append((method, penalty))
        return detect(values, method, penalty)

    monkeypatch.setattr(cases, "detect_offline", recording)
    result = cases.national_case("Всего", shocks, method="binseg_l2", penalty=30.0)

    assert asked == [("binseg_l2", 30.0)], "сломы ищет тот метод, который передан"
    matched = {e["event"]: e for e in result["events"]}
    assert matched["тест"]["detected"], "явный провал 37% должен быть найден"
    assert matched["тест"]["offset_months"] == 0, "слом стоит в месяце события, а не рядом"
    assert not matched["тишина"]["detected"], "на ровном участке слома быть не должно"
    assert result["breaks"] == ["2020-04"]
    # Один слом с окном ±2 месяца накрывает 5 месяцев из 48: такова доля случайного попадания.
    assert result["coverage"] == pytest.approx(5 / 48) and result["margin"] == 2
    assert (tmp_path / result["figure"]).exists()
    # Отчёт и лендинг называют метод, которым разбор построен на самом деле.
    assert (result["method"], result["penalty"]) == ("binseg_l2", 30.0)


def test_national_cases_use_the_method_calibrated_on_the_benchmark(tmp_path, monkeypatch) -> None:
    """Разбор строится лучшим офлайн-методом бенчмарка с его штрафом. Зашитые «PELT, штраф
    1000» означали бы штраф, подобранный не на бенчмарке, а под картинку."""
    from sbx.shell.pipelines import cases

    cpd_dir = tmp_path / "cpd"
    cpd_dir.mkdir()
    pd.DataFrame(
        [
            {"detector": "periodic", "kind": "baseline", "input": "none", "param": 5.0, "f1": 0.9},
            {"detector": "pelt_l2", "kind": "offline", "input": "raw", "param": 30.0, "f1": 0.2},
            {"detector": "window_l2", "kind": "offline", "input": "raw", "param": 0.5, "f1": 0.3},
        ]
    ).to_csv(cpd_dir / "detector_comparison.csv", index=False)
    calls = []

    def record(name: str, shocks: dict, method: str, penalty: float) -> dict:
        calls.append((name, method, penalty))
        return {"series": name}

    monkeypatch.setattr(cases, "CPD_DIR", cpd_dir)
    monkeypatch.setattr(cases, "FIGURES", tmp_path / "figures")
    monkeypatch.setattr(cases, "load_known_shocks", lambda: {})
    monkeypatch.setattr(cases, "national_case", record)
    monkeypatch.setattr(cases, "municipal_case", lambda: None)
    monkeypatch.setattr(cases.stamps, "write", lambda *args, **kwargs: None)

    built = cases.run(out_dir=tmp_path / "cases")
    assert calls == [("Всего", "window_l2", 0.5), ("Услуги", "window_l2", 0.5)]
    assert [case["series"] for case in built["national"]] == ["Всего", "Услуги"]

    # Без сравнения детекторов метод взять неоткуда: разбора нет, а не разбор «по умолчанию».
    (cpd_dir / "detector_comparison.csv").unlink()
    calls.clear()
    assert cases.run(out_dir=tmp_path / "cases")["national"] == [] and calls == []


def test_live_alarms_deduplicate_overlapping_horizons(tmp_path, monkeypatch) -> None:
    """Один месяц предсказан в нескольких горизонтах — в поток он должен войти один раз.

    Онлайн-детекторы накопительные: повторённый месяц создаёт дрейф, которого в ряду нет.
    """
    from sbx.shell.pipelines import cpd

    ds = pd.date_range("2024-01-01", periods=4, freq="MS")
    rows = []
    for horizon, fold in [(3, "h3_2023-12"), (12, "h12_2023-12")]:
        for d in ds:
            rows.append(
                {
                    "unique_id": "a",
                    "ds": d,
                    "fold": fold,
                    "horizon": horizon,
                    "step": 1,
                    "model": "Ensemble (по категориям)",
                    "y": 1000.0,
                    # Длинный горизонт ошибается сильнее — по нему и видно, какой взят.
                    "y_hat": 990.0 if horizon == 3 else 500.0,
                    "q_0.1": 900.0,
                    "q_0.9": 1100.0,
                    "mase_scale": 10.0,
                }
            )
    oof_dir = tmp_path / "oof"
    oof_dir.mkdir()
    pd.DataFrame(rows).to_parquet(oof_dir / "ensemble.parquet", index=False)
    monkeypatch.setattr(cpd, "OOF_DIR", oof_dir)

    results = pd.DataFrame(
        [{"detector": "cusum", "kind": "online", "input": "residual", "param": 2.0, "f1": 0.2}]
    )
    summary = cpd.live_alarms(results, out_dir=tmp_path)

    # Четыре месяца, а не восемь: дубли схлопнуты.
    assert summary["months_per_series"] == 4.0
    assert summary["series"] == 1
