"""Прогон моделей через протокол бэктеста на маленькой фикстуре."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sbx.shell.models.statsforecast_  # noqa: F401  регистрация группы
from sbx.core.folds import BacktestConfig, Fold, make_folds
from sbx.core.forecasting import (
    ForecastContractError,
    build_oof,
    stratified_series_sample,
    validate_forecast,
)
from sbx.shell.models import base as model_base
from sbx.shell.pipelines import backtest

_folds_mod = __import__("sbx.core.folds", fromlist=["TargetSpec", "HorizonFolds"])

CFG = BacktestConfig(
    status="test",
    checked_at="2026-09-29",
    target=_folds_mod.TargetSpec("d", "MS", "l"),
    folds=(_folds_mod.HorizonFolds(horizon=3, origins=("2024-06-01",)),),
    primary_metric="mae",
    mae_aggregation="micro",
    min_train_obs=6,
)


@pytest.fixture
def fixture_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ds = pd.date_range("2023-01-01", periods=24, freq="MS")
    seasonal_shape = np.array([0.8, 0.85, 1.0, 1.0, 1.0, 1.05, 1.05, 1.05, 1.0, 1.0, 1.05, 1.35])
    rows = []
    for i in range(6):
        level = 1000.0 * (i + 1)
        for d in ds:
            rows.append(
                {
                    "unique_id": f"s{i}__total",
                    "territory_id": f"t{i}",
                    "category": "Все категории",
                    "ds": d,
                    "y": level * seasonal_shape[d.month - 1],
                }
            )
    panel = pd.DataFrame(rows)
    static = panel.drop_duplicates("unique_id")[["unique_id", "category"]].assign(size_group=1)
    seasonal = pd.DataFrame(
        {"category": "Все категории", "month": range(1, 13), "index": seasonal_shape}
    )
    return panel, static, seasonal


def test_seasonal_naive_is_exact_on_seasonal_fixture(fixture_data, tmp_path) -> None:
    panel, static, seasonal = fixture_data
    oof = backtest.run_group(
        "stats",
        panel,
        static,
        CFG,
        {"groups": {"stats": {"models": ["SeasonalNaive"], "n_jobs": 1}}},
        out_dir=tmp_path,
        seasonal=seasonal,
    )
    assert oof["y_hat"].notna().all()
    assert np.isfinite(oof["y_hat"]).all()
    # Ряд строго сезонный без тренда: SeasonalNaive повторяет прошлогодний месяц точно.
    np.testing.assert_allclose(oof["y_hat"], oof["y"], rtol=1e-9)


def test_run_group_is_cached_and_deterministic(fixture_data, tmp_path) -> None:
    panel, static, seasonal = fixture_data
    cfg = {"groups": {"stats": {"models": ["Naive", "AutoETS"], "n_jobs": 1}}}
    first = backtest.run_group(
        "stats", panel, static, CFG, cfg, out_dir=tmp_path, seasonal=seasonal
    )
    second = backtest.run_group(
        "stats", panel, static, CFG, cfg, out_dir=tmp_path, seasonal=seasonal
    )
    pd.testing.assert_frame_equal(first, second)
    recomputed = backtest.run_group(
        "stats", panel, static, CFG, cfg, out_dir=tmp_path, seasonal=seasonal, overwrite=True
    )
    pd.testing.assert_frame_equal(first, recomputed)
    assert set(first["model"]) == {"Naive", "AutoETS"}
    assert set(first["step"]) == {1, 2, 3}
    assert first["mase_scale"].notna().all()


@pytest.fixture
def counting_group(monkeypatch) -> list[str]:
    """Группа моделей, которая помнит, сколько раз её считали."""
    calls: list[str] = []

    def forecast(train, fold, group_cfg, context) -> pd.DataFrame:
        calls.append(fold.name)
        last = train.sort_values("ds").groupby("unique_id")["y"].last()
        months = pd.date_range(fold.cutoff, periods=fold.horizon + 1, freq="MS")[1:]
        rows = [
            {"unique_id": uid, "ds": month, "model": "Last", "y_hat": float(value)}
            for uid, value in last.items()
            for month in months
        ]
        return pd.DataFrame(rows)

    monkeypatch.setitem(model_base.REGISTRY, "counting", forecast)
    return calls


def test_run_group_recomputes_when_an_input_changes(
    fixture_data, tmp_path, counting_group, monkeypatch
) -> None:
    """Кэш прогнозов годится только для тех входов, на которых посчитан."""
    panel, static, seasonal = fixture_data
    inputs = {
        "panel": panel,
        "static": static,
        "cfg": CFG,
        "models_cfg": {"groups": {"counting": {"depth": 1}}},
        "seasonal": seasonal,
        "eval_ids": None,
        "context_extra": {"feature_blocks": {"macro": pd.DataFrame({"key_rate": [16.0]})}},
    }

    def run() -> pd.DataFrame:
        return backtest.run_group("counting", out_dir=tmp_path, **inputs)

    run()
    assert len(counting_group) == 1
    run()
    assert len(counting_group) == 1, "входы те же — кэш годится"

    changes = {
        "panel": panel.assign(y=panel["y"] * 2),
        "static": static.assign(size_group=2),
        "seasonal": seasonal.assign(index=1.0),
        "eval_ids": ["s0__total", "s1__total"],
        "models_cfg": {"groups": {"counting": {"depth": 2}}},
        # Тот же набор колонок, другое содержимое: так выглядит докачанный архив новостей.
        "context_extra": {"feature_blocks": {"macro": pd.DataFrame({"key_rate": [21.0]})}},
        "cfg": BacktestConfig(
            status="test",
            checked_at="2026-09-29",
            target=_folds_mod.TargetSpec("d", "MS", "l"),
            folds=(_folds_mod.HorizonFolds(horizon=3, origins=("2024-09-01",)),),
            primary_metric="mae",
            mae_aggregation="micro",
            min_train_obs=6,
        ),
    }
    for expected, (name, value) in enumerate(changes.items(), start=2):
        inputs[name] = value
        result = run()
        assert len(counting_group) == expected, (
            f"изменился вход «{name}» — прогнозы считаются заново"
        )
        run()
        assert len(counting_group) == expected, f"«{name}»: повторный запуск берёт кэш"
    assert set(result["unique_id"]) == {"s0__total", "s1__total"}

    count = len(counting_group)
    pinned = {"foundation_models": {"chronos2": {"revision": "abc"}}}
    inputs["models_cfg"] = {**inputs["models_cfg"], **pinned}
    run()
    assert len(counting_group) == count + 1, "сменилась ревизия весов foundation-модели"
    inputs["cfg"] = BacktestConfig(**{**inputs["cfg"].__dict__, "min_train_obs": 5})
    run()
    assert len(counting_group) == count + 2, "сменился порог длины обучающей части"
    monkeypatch.setattr(backtest, "CACHE_VERSION", backtest.CACHE_VERSION + 1)
    run()
    assert len(counting_group) == count + 3, "сменилась версия расчёта"
    run()
    assert len(counting_group) == count + 3
    assert result["y"].iloc[0] == pytest.approx(
        2
        * panel.set_index(["unique_id", "ds"])["y"].loc[
            (result["unique_id"].iloc[0], result["ds"].iloc[0])
        ]
    )


def _national(poison_after: pd.Timestamp | None = None) -> pd.DataFrame:
    """Короткий национальный ряд, в котором осень 2023 года выше обычного: окно с более
    поздним cutoff видит больше таких месяцев, и его сезонный индекс другой."""
    months = pd.date_range("2022-01-01", "2025-12-01", freq="MS")
    shape = np.array([0.8, 0.85, 1.0, 1.0, 1.0, 1.05, 1.05, 1.05, 1.0, 1.0, 1.05, 1.35])
    values = np.array(
        [
            1000.0 * shape[m.month - 1] * (1.2 if m.year == 2023 and m.month in (8, 9, 10) else 1)
            for m in months
        ]
    )
    if poison_after is not None:
        values = np.where(months > poison_after, values * np.linspace(1, 9, len(months)), values)
    return pd.DataFrame({"period": months, "type": "Всего", "value": values})


def test_each_fold_gets_a_seasonal_index_known_at_its_cutoff(
    fixture_data, tmp_path, monkeypatch
) -> None:
    """С национальным рядом сезонный индекс оценивается для каждого окна заново и только по
    месяцам, опубликованным к его cutoff; масштаб MASE считается по нему же."""
    from sbx.core.seasonality import seasonal_index_as_of

    panel, static, _ = fixture_data
    seen: dict[str, pd.DataFrame] = {}

    def forecast(train, fold, group_cfg, context) -> pd.DataFrame:
        seen[fold.name] = context["seasonal_index"].copy()
        last = train.sort_values("ds").groupby("unique_id")["y"].last()
        months = pd.date_range(fold.cutoff, periods=fold.horizon + 1, freq="MS")[1:]
        return pd.DataFrame(
            [
                {"unique_id": uid, "ds": month, "model": "Last", "y_hat": float(value)}
                for uid, value in last.items()
                for month in months
            ]
        )

    monkeypatch.setitem(model_base.REGISTRY, "spy", forecast)
    cfg = BacktestConfig(
        status="test",
        checked_at="2026-10-02",
        target=_folds_mod.TargetSpec("d", "MS", "l"),
        folds=(_folds_mod.HorizonFolds(horizon=3, origins=("2024-03-01", "2024-06-01")),),
        primary_metric="mae",
        mae_aggregation="micro",
        min_train_obs=6,
    )
    models_cfg = {"groups": {"spy": {}}}
    clean = _national()

    def run(national: pd.DataFrame, name: str) -> pd.DataFrame:
        return backtest.run_group(
            "spy",
            panel,
            static,
            cfg,
            models_cfg,
            out_dir=tmp_path / name,
            national=national,
            national_lag_days=35,
        )

    honest = run(clean, "clean")
    by_fold = dict(seen)
    assert set(by_fold) == {"h3_2024-03", "h3_2024-06"}
    for fold, cutoff in (("h3_2024-03", "2024-03-01"), ("h3_2024-06", "2024-06-01")):
        expected = seasonal_index_as_of(clean, pd.Timestamp(cutoff), 35)
        pd.testing.assert_frame_equal(by_fold[fold], expected)
    assert not by_fold["h3_2024-03"]["index"].equals(by_fold["h3_2024-06"]["index"]), (
        "у окон разные cutoff — и разные индексы; иначе тест не отличил бы один индекс на всех"
    )

    # Всё, что вышло после cutoff позднего окна (апрель 2024 — последний опубликованный
    # месяц), испорчено: ни индекс, ни масштаб MASE не должны этого заметить.
    seen.clear()
    poisoned = run(_national(poison_after=pd.Timestamp("2024-04-01")), "poisoned")
    pd.testing.assert_frame_equal(seen["h3_2024-06"], by_fold["h3_2024-06"])
    pd.testing.assert_frame_equal(seen["h3_2024-03"], by_fold["h3_2024-03"])
    pd.testing.assert_series_equal(poisoned["mase_scale"], honest["mase_scale"])


def test_run_group_recomputes_when_the_national_series_changes(
    fixture_data, tmp_path, counting_group
) -> None:
    panel, static, _ = fixture_data
    cfg = {"groups": {"counting": {}}}

    def run(national: pd.DataFrame, lag: int = 35) -> None:
        backtest.run_group(
            "counting",
            panel,
            static,
            CFG,
            cfg,
            out_dir=tmp_path,
            national=national,
            national_lag_days=lag,
        )

    run(_national())
    run(_national())
    assert len(counting_group) == 1
    run(_national(poison_after=pd.Timestamp("2023-01-01")))
    assert len(counting_group) == 2, "другой национальный ряд — другой сезонный индекс"
    run(_national(poison_after=pd.Timestamp("2023-01-01")), lag=60)
    assert len(counting_group) == 3, "другой лаг публикации — другой сезонный индекс"


def test_run_group_recomputes_a_cache_left_without_a_stamp(
    fixture_data, tmp_path, counting_group
) -> None:
    """Прогнозы без отметки входов посчитаны неизвестно на чём — доверять им нельзя."""
    panel, static, seasonal = fixture_data
    cfg = {"groups": {"counting": {}}}
    backtest.run_group("counting", panel, static, CFG, cfg, out_dir=tmp_path, seasonal=seasonal)
    (tmp_path / "counting_inputs.json").unlink()
    backtest.run_group("counting", panel, static, CFG, cfg, out_dir=tmp_path, seasonal=seasonal)
    assert len(counting_group) == 2
    assert (tmp_path / "counting_inputs.json").exists()


def _two_model_oof(cfg: BacktestConfig) -> pd.DataFrame:
    """Прогнозы двух моделей с ошибками разного знака на шести рядах двух категорий."""
    rows = []
    for fold in make_folds(cfg):
        month = fold.cutoff + pd.DateOffset(months=1)
        for i in range(6):
            truth = 1000.0 * (i + 1) + 10.0 * fold.cutoff.month
            for model, error in (("High", 0.10), ("Low", -0.04)):
                rows.append(
                    {
                        "unique_id": f"s{i}",
                        "ds": month,
                        "fold": fold.name,
                        "cutoff": fold.cutoff,
                        "horizon": fold.horizon,
                        "step": 1,
                        "model": model,
                        "y": truth,
                        "y_hat": truth * (1 + error),
                        "category": "Здоровье" if i % 2 else "Транспорт",
                    }
                )
    return pd.DataFrame(rows)


def test_ensemble_is_built_from_base_models_only(tmp_path) -> None:
    """Прежний ensemble.parquet в каталоге не должен влиять на новый ансамбль: иначе после
    смены панели или оценочной выборки ансамбль строился бы на пересечении со старым собой."""
    cfg = BacktestConfig(
        status="test",
        checked_at="2026-09-29",
        target=_folds_mod.TargetSpec("d", "MS", "l"),
        folds=(
            _folds_mod.HorizonFolds(horizon=1, origins=("2024-06-01", "2024-07-01", "2024-08-01")),
        ),
        primary_metric="mae",
        mae_aggregation="micro",
    )
    oof = _two_model_oof(cfg)
    for model, part in oof.groupby("model"):
        part.to_parquet(tmp_path / f"{model.lower()}.parquet", index=False)

    fresh = backtest.build_ensemble(cfg, out_dir=tmp_path)
    assert fresh.groupby("model").size().to_dict() == {
        "Ensemble": 18,
        "Ensemble (по категориям)": 18,
    }

    fresh.iloc[:1].to_parquet(tmp_path / "ensemble.parquet", index=False)
    again = backtest.build_ensemble(cfg, out_dir=tmp_path)
    pd.testing.assert_frame_equal(again, fresh)
    context = (tmp_path / "ensemble_context.json").read_text(encoding="utf-8")
    assert '"High"' in context and "Ensemble" not in context.split('"weights"')[0]

    # Отметка ансамбля — состояние прогнозов одиночных моделей: его собственный файл в неё
    # не входит, а смена любого из базовых делает ансамбль устаревшим.
    from sbx.shell import stamps

    stamp = tmp_path / "ensemble_inputs.json"
    assert len(stamps.read(stamp)) == 1 and stamps.stale(stamp) == []
    oof.assign(y_hat=1.0).query("model == 'High'").to_parquet(tmp_path / "high.parquet")
    assert len(stamps.stale(stamp)) == 1


def test_significance_step_stamps_its_result(tmp_path, monkeypatch) -> None:
    from sbx.shell import stamps

    cfg = BacktestConfig(
        status="test",
        checked_at="2026-10-02",
        target=_folds_mod.TargetSpec("d", "MS", "l"),
        folds=(_folds_mod.HorizonFolds(horizon=1, origins=("2024-06-01", "2024-07-01")),),
        primary_metric="mae",
        mae_aggregation="micro",
    )
    oof = _two_model_oof(cfg)
    oof = pd.concat([oof, oof[oof["model"] == "High"].assign(model="Prophet", y_hat=1.0)])
    oof_dir = tmp_path / "oof"
    oof_dir.mkdir()
    oof.to_parquet(oof_dir / "all.parquet", index=False)
    monkeypatch.setattr(backtest, "OOF_DIR", oof_dir)

    result = backtest.build_significance(oof, metrics_dir=tmp_path / "metrics")

    assert result["by_horizon"]["1"]["best_model"] == "Low"
    # Рядом с тестом по рядам — счёт окон: в скольких окнах лучшая модель точнее эталона.
    for row in result["by_horizon"]["1"]["dm"]:
        assert {"folds", "folds_better", "folds_worse", "sign_p", "by_fold"} <= set(row)
        assert row["folds"] == len(row["by_fold"]) >= 1
    stamp = tmp_path / "metrics" / "significance_inputs.json"
    assert len(stamps.read(stamp)) == 1 and stamps.stale(stamp) == []
    (oof_dir / "all.parquet").write_bytes(b"recomputed")
    assert len(stamps.stale(stamp)) == 1


def test_a_limited_run_leaves_the_full_cache_and_metrics_untouched(
    fixture_data, tmp_path, monkeypatch, counting_group
) -> None:
    """Пробный прогон на нескольких рядах не должен ни вернуть кэш полного прогона, ни затереть его."""
    panel, static, seasonal = fixture_data
    oof_dir, metrics_dir = tmp_path / "oof", tmp_path / "metrics"
    monkeypatch.setattr(backtest, "OOF_DIR", oof_dir)
    monkeypatch.setattr(backtest, "METRICS_DIR", metrics_dir)
    monkeypatch.setattr(backtest, "load_panel", lambda: panel)
    monkeypatch.setattr(backtest, "load_static", lambda: static)
    monkeypatch.setattr(backtest, "load_seasonal", lambda: seasonal)
    monkeypatch.setattr(backtest, "load_backtest_config", lambda path=None: CFG)
    monkeypatch.setattr(backtest, "load_models_config", lambda: {"groups": {"counting": {}}})

    full = backtest.run(["counting"])
    cache = (oof_dir / "counting.parquet").read_bytes()
    board = (metrics_dir / "leaderboard.csv").read_bytes()
    assert int(full["rows_common"].iloc[0]) == 6 * 3

    limited = backtest.run(["counting"], limit_series=2)
    assert int(limited["rows_common"].iloc[0]) == 2 * 3, "пробный прогон действительно посчитан"
    assert len(counting_group) == 2
    assert (oof_dir / "counting.parquet").read_bytes() == cache
    assert (metrics_dir / "leaderboard.csv").read_bytes() == board
    assert sorted(backtest.collect_oof(out_dir=oof_dir)["unique_id"].unique()) == sorted(
        static["unique_id"]
    )


def test_validate_forecast_rejects_broken_output(fixture_data) -> None:
    panel, _, _ = fixture_data
    fold = Fold("f", pd.Timestamp("2024-06-01"), 3, "main")
    good = pd.DataFrame(
        {
            "unique_id": ["s0__total"] * 3,
            "ds": pd.date_range("2024-07-01", periods=3, freq="MS"),
            "model": "m",
            "y_hat": [1.0, 2.0, 3.0],
        }
    )
    validate_forecast(good, ["s0__total"], fold)
    with pytest.raises(ForecastContractError, match="отрицательный"):
        validate_forecast(good.assign(y_hat=[-1.0, 2.0, 3.0]), ["s0__total"], fold)
    with pytest.raises(ForecastContractError, match="вне горизонта"):
        validate_forecast(
            good.assign(ds=pd.date_range("2024-11-01", periods=3, freq="MS")), ["s0__total"], fold
        )
    with pytest.raises(ForecastContractError, match="нет прогноза"):
        validate_forecast(good, ["s0__total", "s1__total"], fold)


def test_build_oof_matches_protocol_columns(fixture_data) -> None:
    panel, static, _ = fixture_data
    fold = make_folds(CFG)[0]
    test = panel[(panel["ds"] > fold.cutoff) & (panel["ds"] <= fold.test_end)]
    preds = test[["unique_id", "ds"]].assign(model="m", y_hat=1.0)
    oof = build_oof(preds, test, fold, static)
    assert list(oof.columns)[:9] == [
        "unique_id",
        "ds",
        "fold",
        "cutoff",
        "horizon",
        "step",
        "model",
        "y",
        "y_hat",
    ]
    # Горизонт фолда нужен, чтобы сравнивать модели внутри горизонта: при h=12 часть
    # моделей неопределима, и общее выравнивание выбросило бы весь горизонт.
    assert set(oof["horizon"]) == {fold.horizon}
    assert set(oof["step"]) == {1, 2, 3}
    assert (oof["cutoff"] == fold.cutoff).all()


def test_stratified_sample_is_deterministic_and_balanced() -> None:
    static = pd.DataFrame(
        {
            "unique_id": [f"u{i}" for i in range(400)],
            "category": ["a"] * 200 + ["b"] * 200,
            "size_group": ([1] * 100 + [2] * 100) * 2,
        }
    )
    sample = stratified_series_sample(static, 40, seed=1)
    again = stratified_series_sample(static, 40, seed=1)
    assert sample == again and len(sample) == 40
    picked = static[static["unique_id"].isin(sample)]
    assert picked.groupby(["category", "size_group"]).size().tolist() == [10, 10, 10, 10]


def _frozen_list(tmp_path, ids: list[str]):
    path = tmp_path / "evaluation.csv"
    pd.DataFrame({"unique_id": ids, "legacy_unique_id": [f"old-{i}" for i in ids]}).to_csv(
        path, index=False
    )
    return path


def test_evaluation_series_come_from_the_frozen_list(tmp_path) -> None:
    """Оценочные ряды не отбираются заново: это список опубликованного расчёта."""
    static = pd.DataFrame(
        {"unique_id": ["a", "b", "c"], "category": "Все категории", "size_group": 1}
    )
    frozen = _frozen_list(tmp_path, ["c", "a"])
    cfg = {"evaluation_sample": {"frozen": str(frozen), "n_series": 1, "seed": 7}}
    assert backtest.evaluation_series(static, cfg) == ["a", "c"]


def test_evaluation_series_refuse_a_frozen_list_made_for_another_panel(tmp_path) -> None:
    static = pd.DataFrame({"unique_id": ["a"], "category": "Все категории", "size_group": 1})
    cfg = {"evaluation_sample": {"frozen": str(_frozen_list(tmp_path, ["a", "zz"]))}}
    with pytest.raises(ValueError, match="нет в панели"):
        backtest.evaluation_series(static, cfg)


def test_collect_oof_ignores_nested_ablation_caches(tmp_path) -> None:
    """Абляционные варианты лежат во вложенной папке и не попадают в общий leaderboard.

    Иначе кандидаты итогового ансамбля и headline-метрика зависели бы от того, какие
    абляции сейчас закэшированы на диске.
    """
    main = pd.DataFrame(
        {
            "unique_id": ["a"],
            "ds": [pd.Timestamp("2024-01-01")],
            "fold": ["f1"],
            "cutoff": [pd.Timestamp("2023-12-01")],
            "step": [1],
            "model": ["LightGBM"],
            "y": [100.0],
            "y_hat": [101.0],
        }
    )
    main.to_parquet(tmp_path / "lgbm.parquet", index=False)
    nested = tmp_path / "ablations"
    nested.mkdir()
    main.assign(model="LightGBM (A: Сбер)").to_parquet(nested / "lgbm_A.parquet", index=False)

    assert sorted(backtest.collect_oof(out_dir=tmp_path)["model"]) == ["LightGBM"]
    assert sorted(backtest.collect_oof(out_dir=nested)["model"]) == ["LightGBM (A: Сбер)"]


def test_common_rows_aligns_inside_each_horizon() -> None:
    """При h=12 часть моделей неопределима — это не должно выбрасывать весь горизонт."""
    rows = []
    for horizon, fold, models in [
        (3, "h3_2024-09", ["stats", "lgbm", "chronos"]),
        (12, "h12_2023-12", ["stats", "chronos"]),  # lgbm на h=12 неопределим
    ]:
        for model in models:
            for month in range(1, 4):
                rows.append(
                    {
                        "unique_id": "a",
                        "ds": pd.Timestamp("2024-01-01") + pd.DateOffset(months=month),
                        "fold": fold,
                        "horizon": horizon,
                        "model": model,
                        "y": 100.0,
                        "y_hat": 99.0,
                    }
                )
    aligned = backtest.common_rows(pd.DataFrame(rows))

    kept = aligned.groupby("horizon")["model"].nunique().to_dict()
    assert kept == {3: 3, 12: 2}, "горизонт 12 не должен пропадать из-за отсутствия lgbm"
    assert set(aligned[aligned["horizon"] == 12]["model"]) == {"stats", "chronos"}
    # Без разреза по горизонту фолд h=12 исчез бы целиком: в нём не все модели дали прогноз,
    # а порог сравнения берётся по общему числу моделей во всём OOF.
    naive = backtest.common_rows(pd.DataFrame(rows).drop(columns=["horizon"]))
    assert set(naive["fold"]) == {"h3_2024-09"}, "наивное выравнивание теряет горизонт 12"


@pytest.fixture
def full_run(fixture_data, tmp_path, monkeypatch):
    """Полный прогон `backtest.run` на маленькой панели, каталоги — во временной папке."""
    panel, static, seasonal = fixture_data
    monkeypatch.setattr(backtest, "OOF_DIR", tmp_path / "oof")
    monkeypatch.setattr(backtest, "METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr(backtest, "load_panel", lambda: panel)
    monkeypatch.setattr(backtest, "load_static", lambda: static)
    monkeypatch.setattr(backtest, "load_seasonal", lambda: seasonal)
    monkeypatch.setattr(backtest, "load_backtest_config", lambda path=None: CFG)
    return tmp_path


def _calendar(working_days: int):
    from sbx.core.features.asof import KNOWN_FUTURE, ExogenousBlock

    months = pd.date_range("2023-01-01", periods=30, freq="MS")
    return ExogenousBlock(
        "calendar", pd.DataFrame({"ds": months, "working_days": working_days}), KNOWN_FUTURE
    )


def test_a_main_group_gets_the_feature_blocks_named_in_its_settings(
    full_run, monkeypatch, counting_group
) -> None:
    """Основная модель берёт блоки внешних признаков из своих настроек, и кэш её прогнозов
    следит за содержимым блоков — как у вариантов абляции."""
    received: list[dict] = []
    forecast = model_base.REGISTRY["counting"]

    def remembering(train, fold, group_cfg, context):
        received.append(dict(context.get("feature_blocks", {})))
        return forecast(train, fold, group_cfg, context)

    monkeypatch.setitem(model_base.REGISTRY, "counting", remembering)
    settings = {"groups": {"counting": {"feature_blocks": ["calendar"]}}}
    monkeypatch.setattr(backtest, "load_models_config", lambda: settings)
    blocks = {"calendar": _calendar(21), "macro": _calendar(0)}
    monkeypatch.setattr(backtest, "load_feature_blocks", lambda: blocks)

    backtest.run(["counting"])
    assert received and all(list(context) == ["calendar"] for context in received), (
        "модель получает только названные блоки"
    )
    computed = len(counting_group)
    backtest.run(["counting"])
    assert len(counting_group) == computed, "те же блоки — прогнозы из кэша"

    from sbx.shell import stamps

    recorded = stamps.read(full_run / "oof" / "counting_inputs.json")
    assert backtest.expected_inputs(["counting"])["counting"] == recorded, (
        "отчёт пересчитывает тот же отпечаток, что записан рядом с прогнозами"
    )

    blocks["calendar"] = _calendar(22)
    backtest.run(["counting"])
    assert len(counting_group) == 2 * computed, "содержимое блока изменилось — пересчёт"
    assert backtest.expected_inputs(["counting"])["counting"] != recorded


def test_a_main_group_refuses_to_run_without_a_block_it_names(
    full_run, monkeypatch, counting_group
) -> None:
    """Вариант абляции без блока пропускается; основная модель без своего блока — ошибка:
    иначе она молча стала бы другой моделью."""
    settings = {"groups": {"counting": {"feature_blocks": ["calendar"]}}}
    monkeypatch.setattr(backtest, "load_models_config", lambda: settings)
    monkeypatch.setattr(backtest, "load_feature_blocks", lambda: {})
    with pytest.raises(FileNotFoundError, match="make features"):
        backtest.run(["counting"])
    assert counting_group == []


def test_a_group_without_blocks_does_not_read_the_feature_tables(
    full_run, monkeypatch, counting_group
) -> None:
    monkeypatch.setattr(backtest, "load_models_config", lambda: {"groups": {"counting": {}}})

    def forbidden():
        raise AssertionError("группе без внешних признаков таблицы признаков не нужны")

    monkeypatch.setattr(backtest, "load_feature_blocks", forbidden)
    backtest.run(["counting"])
    assert counting_group


def test_the_main_lightgbm_takes_the_calendar_and_keeps_default_parameters() -> None:
    """Решение от 2 октября 2026 года: основной LightGBM получает производственный календарь на
    целевой месяц и не подбирает гиперпараметры."""
    settings = backtest.load_models_config()["groups"]["lgbm"]
    assert settings["feature_blocks"] == ["calendar"]
    assert settings["tune"] is False
