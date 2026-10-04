import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sbx.core.config import from_mapping
from sbx.core.folds import BacktestConfig, Fold, make_folds, split, step_of
from sbx.core.metrics import (
    coverage,
    leaderboard,
    mae,
    mase_scales,
    pinball,
    r2,
    rmse,
    summarize,
    wape,
)
from sbx.core.significance import (
    diebold_mariano,
    fold_consistency,
    friedman_nemenyi,
    nemenyi_critical_difference,
    panel_dm,
)

CFG = {
    "status": "aligned-with-criteria",
    "checked_at": "2026-09-29",
    "target": {"dataset": "d", "freq": "MS", "level": "x"},
    "folds": [
        {"horizon": 1, "origins": ["2023-12-01", "2024-03-01", "2024-06-01", "2024-09-01"]},
        {"horizon": 3, "origins": ["2023-12-01", "2024-03-01", "2024-06-01", "2024-09-01"]},
        {"horizon": 6, "origins": ["2023-12-01", "2024-03-01", "2024-06-01"]},
        {"horizon": 12, "origins": ["2023-10-01", "2023-11-01", "2023-12-01"]},
    ],
    "primary_metric": "mae",
    "mae_aggregation": "micro",
}


def _panel(n_series: int = 4, start: str = "2023-01-01", months: int = 24) -> pd.DataFrame:
    ds = pd.date_range(start, periods=months, freq="MS")
    return pd.DataFrame(
        [
            {
                "unique_id": f"s{i}",
                "ds": d,
                "y": float(100 * (i + 1) + t),
                "category": "Все категории",
            }
            for i in range(n_series)
            for t, d in enumerate(ds)
        ]
    )


def test_make_folds_covers_the_four_required_horizons() -> None:
    """Критерий конкурса требует горизонты 1, 3, 6 и 12 месяцев."""
    folds = make_folds(from_mapping(BacktestConfig, CFG))
    by_horizon: dict[int, list[str]] = {}
    for f in folds:
        by_horizon.setdefault(f.horizon, []).append(f.cutoff.strftime("%Y-%m"))

    assert sorted(by_horizon) == [1, 3, 6, 12]
    assert by_horizon[1] == ["2023-12", "2024-03", "2024-06", "2024-09"]
    assert by_horizon[3] == ["2023-12", "2024-03", "2024-06", "2024-09"]
    assert by_horizon[6] == ["2023-12", "2024-03", "2024-06"]
    # h=12 возможен только с ранних origin: панель кончается 2024-12.
    assert by_horizon[12] == ["2023-10", "2023-11", "2023-12"]
    assert len(folds) == 14


def test_fold_names_are_unique_and_carry_the_horizon() -> None:
    folds = make_folds(from_mapping(BacktestConfig, CFG))
    names = [f.name for f in folds]
    assert len(set(names)) == len(names)
    # Один и тот же origin встречается с разными горизонтами — имя обязано их различать.
    same_origin = sorted(f.name for f in folds if f.cutoff == pd.Timestamp("2023-12-01"))
    assert len(same_origin) == 4
    for f in folds:
        assert f"h{f.horizon}" in f.name


def test_test_window_never_runs_past_the_panel_for_h12() -> None:
    """h=12 от 2023-12 заканчивается ровно на последнем месяце панели."""
    folds = make_folds(from_mapping(BacktestConfig, CFG))
    latest = max((f for f in folds if f.horizon == 12), key=lambda f: f.cutoff)
    assert latest.cutoff == pd.Timestamp("2023-12-01")
    assert latest.test_end == pd.Timestamp("2024-12-01")


@settings(max_examples=60, deadline=None)
@given(
    cutoff_idx=st.integers(min_value=0, max_value=22),
    horizon=st.integers(min_value=1, max_value=6),
    min_obs=st.integers(min_value=1, max_value=12),
)
def test_split_properties(cutoff_idx: int, horizon: int, min_obs: int) -> None:
    panel = _panel()
    cutoff = pd.Timestamp("2023-01-01") + pd.DateOffset(months=cutoff_idx)
    fold = Fold("f", cutoff, horizon, "main")
    train, test = split(panel, fold, min_train_obs=min_obs)
    assert train["ds"].max() <= cutoff if len(train) else True
    if len(train):
        assert train["ds"].max() == cutoff
    assert ((test["ds"] > cutoff) & (test["ds"] <= fold.test_end)).all()
    assert set(test["unique_id"]) <= set(train["unique_id"])
    keys_train = set(zip(train["unique_id"], train["ds"], strict=True))
    keys_test = set(zip(test["unique_id"], test["ds"], strict=True))
    assert not keys_train & keys_test
    steps = step_of(test["ds"], cutoff)
    assert steps.between(1, horizon).all()


def test_split_never_leaks_future_rows() -> None:
    panel = _panel()
    poisoned = pd.concat(
        [
            panel,
            pd.DataFrame(
                [{"unique_id": "s0", "ds": pd.Timestamp("2030-01-01"), "y": 1e9, "category": "x"}]
            ),
        ]
    )
    fold = Fold("f", pd.Timestamp("2024-03-01"), 3, "main")
    train, test = split(poisoned, fold)
    assert (train["ds"] <= fold.cutoff).all() and 1e9 not in set(train["y"])
    assert 1e9 not in set(test["y"])


def test_point_metrics_known_values() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    yh = np.array([1.0, 3.0, 2.0, 6.0])
    assert mae(y, yh) == pytest.approx(1.0)
    assert rmse(y, yh) == pytest.approx(np.sqrt(6 / 4))
    assert wape(y, yh) == pytest.approx(4 / 10)
    assert r2(y, yh) == pytest.approx(1 - 6 / 5)
    assert pinball(np.array([10.0]), np.array([8.0]), 0.9) == pytest.approx(1.8)
    assert pinball(np.array([10.0]), np.array([12.0]), 0.9) == pytest.approx(0.2)
    assert coverage(y, y - 1, y + 0.5) == 1.0


def test_mase_scale_uses_seasonal_naive() -> None:
    ds = pd.date_range("2023-01-01", periods=4, freq="MS")
    train = pd.DataFrame(
        {"unique_id": "a", "ds": ds, "y": [100.0, 200.0, 200.0, 100.0], "category": "c"}
    )
    seasonal = pd.DataFrame({"category": "c", "month": [1, 2, 3, 4], "index": [1.0, 2.0, 2.0, 1.0]})
    assert mase_scales(train, seasonal)["a"] == pytest.approx(0.0)
    flat = seasonal.assign(index=1.0)
    assert mase_scales(train, flat)["a"] == pytest.approx((100 + 0 + 100) / 3)


def _oof() -> pd.DataFrame:
    rows = []
    for model, bias in [("good", 1.0), ("bad", 5.0)]:
        for uid, cat, size in [("a", "food", 1), ("b", "total", 2)]:
            for step, d in enumerate(pd.date_range("2024-01-01", periods=3, freq="MS"), start=1):
                y = 100.0 + step * (10 if uid == "a" else 20)
                rows.append(
                    {
                        "unique_id": uid,
                        "ds": d,
                        "fold": "f1",
                        "cutoff": pd.Timestamp("2023-12-01"),
                        "horizon": 3,
                        "step": step,
                        "model": model,
                        "y": y,
                        "y_hat": y + bias * (-1) ** step,
                        "q_0.1": y - 10,
                        "q_0.5": y,
                        "q_0.9": y + 10,
                        "category": cat,
                        "size_group": size,
                        "mase_scale": 2.0,
                    }
                )
    return pd.DataFrame(rows)


def test_summarize_slices_and_leaderboard() -> None:
    summary = summarize(_oof(), slices=("category", "size_group", "step", "fold"))
    board = leaderboard(summary)
    assert list(board["model"]) == ["good", "bad"]
    good = board.iloc[0]
    assert good["mae"] == pytest.approx(1.0) and good["mae_macro"] == pytest.approx(1.0)
    assert good["mase"] == pytest.approx(0.5)
    assert good["coverage_0.1_0.9"] == 1.0
    assert good["pinball_0.5"] == pytest.approx(0.0)
    slices = set(summary["slice"])
    assert slices == {"overall", "category", "size_group", "step", "fold"}
    by_step = summary[(summary["model"] == "bad") & (summary["slice"] == "step")]
    assert len(by_step) == 3 and np.allclose(by_step["mae"], 5.0)


def test_diebold_mariano_detects_difference_and_null() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=400) ** 2
    worse = base + 0.5 + rng.normal(scale=0.1, size=400)
    res = diebold_mariano(base, worse)
    assert res.statistic < -5 and res.p_value < 1e-6
    same = diebold_mariano(base, base + rng.normal(scale=0.01, size=400))
    assert same.p_value > 0.01
    # ручной пример: d = [1, -1, 1, -1] → среднее 0 → статистика 0
    zero = diebold_mariano(np.array([2.0, 0.0, 2.0, 0.0]), np.array([1.0, 1.0, 1.0, 1.0]))
    assert zero.statistic == pytest.approx(0.0) and zero.p_value == pytest.approx(1.0)


def test_panel_dm_prefers_better_model() -> None:
    oof = _oof()
    res = panel_dm(oof[oof["model"] == "good"], oof[oof["model"] == "bad"])
    assert res.mean_diff == pytest.approx(-4.0)


def test_friedman_nemenyi_ranks_and_cd() -> None:
    rng = np.random.default_rng(1)
    n = 200
    errors = pd.DataFrame(
        {"m1": rng.random(n), "m2": rng.random(n) + 0.5, "m3": rng.random(n) + 1.0}
    )
    res = friedman_nemenyi(errors)
    assert list(res.mean_ranks) == ["m1", "m2", "m3"]
    assert res.friedman_p_value < 1e-10
    # табличное значение q_0.05 для k=3 — 2.343 (Demšar, 2006)
    assert nemenyi_critical_difference(3, n) == pytest.approx(
        2.343 * np.sqrt(3 * 4 / (6 * n)), rel=1e-3
    )


def test_check_coverage_flags_folds_that_run_past_the_data() -> None:
    """Горизонт 12 от позднего origin молча обрезался бы до трёх месяцев."""
    from sbx.core.folds import check_coverage

    panel = _panel(months=24)  # 2023-01 … 2024-12
    good = Fold("h12_2023-12", pd.Timestamp("2023-12-01"), 12, "main")
    bad = Fold("h12_2024-09", pd.Timestamp("2024-09-01"), 12, "main")

    assert check_coverage(panel, [good]) == []
    problems = check_coverage(panel, [good, bad])
    assert len(problems) == 1 and "h12_2024-09" in problems[0]
    assert "2024-12" in problems[0]
    # Настоящий протокол не должен содержать таких фолдов.
    assert check_coverage(panel, make_folds(from_mapping(BacktestConfig, CFG))) == []


def _fold_pair(fold: str, left: float, right: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Прогнозы двух моделей в одном окне: ошибка левой и правой на двух рядах."""
    base = pd.DataFrame(
        {"unique_id": ["a", "b"], "ds": pd.Timestamp("2024-07-01"), "fold": fold, "y": 100.0}
    )
    return base.assign(y_hat=100.0 - left), base.assign(y_hat=100.0 - right)


def _windows(errors: dict[str, tuple[float, float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = [_fold_pair(fold, left, right) for fold, (left, right) in errors.items()]
    return pd.concat([p[0] for p in pairs]), pd.concat([p[1] for p in pairs])


def test_fold_consistency_counts_the_windows_where_the_left_model_is_better() -> None:
    """Все прогнозы окна делает одна модель, поэтому единица сравнения — окно, а не ряд."""
    left, right = _windows(
        {"f1": (5.0, 10.0), "f2": (12.0, 10.0), "f3": (3.0, 10.0), "f4": (9.0, 10.0)}
    )
    result = fold_consistency(left, right)
    assert (result.folds, result.better, result.worse) == (4, 3, 1)
    assert result.by_fold == {"f1": -5.0, "f2": 2.0, "f3": -7.0, "f4": -1.0}
    assert result.mean_of_folds == pytest.approx(-2.75)
    assert result.sign_p == pytest.approx(0.625), "три окна из четырёх — ещё не закономерность"
    assert result.as_dict() == {
        "folds": 4,
        "folds_better": 3,
        "folds_worse": 1,
        "sign_p": pytest.approx(0.625),
        "mean_of_folds": pytest.approx(-2.75),
        "by_fold": {"f1": -5.0, "f2": 2.0, "f3": -7.0, "f4": -1.0},
    }


def test_nine_windows_of_ten_pass_the_sign_test_and_eight_do_not() -> None:
    def sign_p(better: int) -> float:
        errors = {f"f{i}": ((5.0, 10.0) if i < better else (12.0, 10.0)) for i in range(10)}
        return fold_consistency(*_windows(errors)).sign_p

    assert sign_p(10) == pytest.approx(2 / 1024)
    assert sign_p(9) == pytest.approx(22 / 1024)
    assert sign_p(8) == pytest.approx(112 / 1024)
    assert sign_p(1) == pytest.approx(22 / 1024), "тест двусторонний: устойчиво хуже — тоже вывод"


def test_fold_consistency_uses_only_the_rows_both_models_have() -> None:
    left, right = _windows({"f1": (5.0, 10.0), "f2": (5.0, 10.0)})
    extra = left.iloc[:1].assign(unique_id="only-left", y_hat=-900.0)
    result = fold_consistency(pd.concat([left, extra]), right)
    assert result.by_fold == {"f1": -5.0, "f2": -5.0}
    tied = fold_consistency(left, left)
    assert (tied.folds, tied.better, tied.worse) == (2, 0, 0) and tied.sign_p == 1.0
