import json

import numpy as np
import pandas as pd
import pytest

from sbx.core.ablation import (
    DEFAULT_MATRIX,
    AblationResult,
    ablation_table,
    contribution,
    feature_sets,
    on_common_rows,
    source_contribution,
)
from sbx.core.ensemble import (
    attach_meta,
    combine,
    combine_quantiles,
    fit_weights,
    fit_weights_by_group,
    pivot_predictions,
    recenter_quantiles,
    rolling_origin_ensemble,
)
from sbx.core.features.calendar import calendar_features, parse_calendar
from sbx.core.features.lags import feature_columns, make_features, predictions_from_ratio
from sbx.core.metrics import OOF_COLUMNS
from sbx.core.news import add_novelty, event_calendar_features, map_regions, monthly_news_features


def _oof(bias_a: float = 1.0, bias_b: float = -3.0) -> pd.DataFrame:
    rows = []
    for fold, start in [("f1", "2024-01-01"), ("f2", "2024-04-01")]:
        for uid, cat in [("a", "food"), ("b", "total")]:
            for d in pd.date_range(start, periods=3, freq="MS"):
                y = 100.0 + d.month
                for model, bias in [("m1", bias_a), ("m2", bias_b)]:
                    rows.append(
                        {
                            "unique_id": uid,
                            "ds": d,
                            "fold": fold,
                            "model": model,
                            "y": y,
                            "y_hat": y + bias,
                            "category": cat,
                        }
                    )
    return pd.DataFrame(rows)


def test_weights_are_simplex_and_beat_single_model() -> None:
    oof = _oof(bias_a=2.0, bias_b=-2.0)
    preds, y = pivot_predictions(oof)
    weights = fit_weights(preds, y)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()
    # Ошибки противоположны по знаку → равные веса дают точный прогноз.
    np.testing.assert_allclose(weights.to_numpy(), [0.5, 0.5], atol=1e-6)
    combined = combine(preds, weights)
    assert np.abs(combined - y).mean() < 1e-4


def test_weights_collapse_to_best_model_when_one_dominates() -> None:
    oof = _oof(bias_a=0.5, bias_b=40.0)
    preds, y = pivot_predictions(oof)
    weights = fit_weights(preds, y)
    assert weights["m1"] > 0.9


def test_weights_by_group_differ() -> None:
    oof = _oof()
    oof.loc[(oof["category"] == "food") & (oof["model"] == "m1"), "y_hat"] += 50
    weights = fit_weights_by_group(oof, ["m1", "m2"], "category")
    assert weights["food"]["m2"] > weights["total"]["m2"]


def test_rolling_origin_ensemble_uses_only_previous_folds() -> None:
    oof = _oof(bias_a=1.0, bias_b=-1.0)
    combined, weights = rolling_origin_ensemble(oof, ["m1", "m2"], ["f1", "f2"])
    assert set(combined["fold"]) == {"f1", "f2"}
    np.testing.assert_allclose(weights["f1"].to_numpy(), [0.5, 0.5])
    assert weights["f2"].sum() == pytest.approx(1.0)
    assert combined["model"].unique().tolist() == ["Ensemble"]


def test_rolling_origin_ensemble_keeps_oof_contract_columns() -> None:
    """Выход ансамбля — такой же OOF, как у одиночных моделей, иначе метрики не считаются."""
    oof = _oof()
    oof["cutoff"] = oof["fold"].map(
        {"f1": pd.Timestamp("2023-12-01"), "f2": pd.Timestamp("2024-03-01")}
    )
    oof["step"] = [
        (d.year - c.year) * 12 + d.month - c.month
        for d, c in zip(oof["ds"], oof["cutoff"], strict=True)
    ]
    oof["horizon"] = 3
    combined, _ = rolling_origin_ensemble(oof, ["m1", "m2"], ["f1", "f2"])
    assert set(OOF_COLUMNS) <= set(combined.columns)
    assert "category" in combined.columns  # срезы отчёта тоже переносятся
    row = combined[(combined["unique_id"] == "a") & (combined["ds"] == pd.Timestamp("2024-02-01"))]
    assert row["cutoff"].iloc[0] == pd.Timestamp("2023-12-01")
    assert row["step"].iloc[0] == 2
    assert row["category"].iloc[0] == "food"
    assert len(combined) == len(oof) // 2  # ни одна строка не размножилась при переносе


def test_ensemble_intervals_are_its_own_not_borrowed() -> None:
    """Квантили ансамбля — взвешенные квантили участников, а не скопированные у одной модели."""
    oof = _oof(bias_a=1.0, bias_b=-1.0)
    # Интервалы есть только у m2 — так же, как в решении их дают лишь foundation-модели.
    oof["q_0.1"] = np.where(oof["model"] == "m2", oof["y_hat"] - 10.0, np.nan)
    oof["q_0.9"] = np.where(oof["model"] == "m2", oof["y_hat"] + 10.0, np.nan)
    combined, weights = rolling_origin_ensemble(oof, ["m1", "m2"], ["f1", "f2"])

    first = combined[combined["fold"] == "f1"]
    m2 = oof[(oof["model"] == "m2") & (oof["fold"] == "f1")].reset_index(drop=True)
    # Веса пересчитаны на тех, у кого квантили есть: остаётся только m2.
    np.testing.assert_allclose(first["q_0.1"].to_numpy(), m2["q_0.1"].to_numpy())
    # Точечный прогноз при этом усреднён по обеим моделям и не равен прогнозу m2.
    assert not np.allclose(first["y_hat"].to_numpy(), m2["y_hat"].to_numpy())
    assert (combined["q_0.1"] < combined["q_0.9"]).all()
    assert set(weights["f1"].index) == {"m1", "m2"}


def _quantile_oof() -> tuple[pd.DataFrame, pd.MultiIndex]:
    """Три модели на двух наблюдениях; квантили дают только q1 и q2."""
    rows = []
    for uid, shift in (("a", 0.0), ("b", 10.0)):
        for model, low, high in (("q1", 80.0, 120.0), ("q2", 90.0, 130.0), ("plain", None, None)):
            rows.append(
                {
                    "unique_id": uid,
                    "ds": pd.Timestamp("2024-07-01"),
                    "fold": "f1",
                    "model": model,
                    "y": 100.0,
                    "y_hat": 100.0,
                    "q_0.1": None if low is None else low + shift,
                    "q_0.9": None if high is None else high + shift,
                }
            )
    oof = pd.DataFrame(rows)
    index = pd.MultiIndex.from_frame(oof[["unique_id", "ds", "fold"]].drop_duplicates())
    return oof, index


def test_ensemble_interval_uses_the_weights_of_the_models_that_have_quantiles() -> None:
    oof, index = _quantile_oof()
    weights = pd.Series({"q1": 0.3, "q2": 0.1, "plain": 0.6})
    out = combine_quantiles(oof, ["q1", "q2", "plain"], weights, index)
    assert out["q_0.1"].tolist() == pytest.approx([82.5, 92.5]), "веса 0,75 и 0,25"
    assert out["q_0.9"].tolist() == pytest.approx([122.5, 132.5])


def test_ensemble_keeps_an_interval_when_all_weight_is_on_models_without_quantiles() -> None:
    """Вес целиком у модели без квантилей — интервал всё равно нужен: берётся равновзвешенный
    интервал моделей, которые его дают (так же ансамбль поступает без истории для весов)."""
    oof, index = _quantile_oof()
    weights = pd.Series({"q1": 0.0, "q2": 0.0, "plain": 1.0})
    out = combine_quantiles(oof, ["q1", "q2", "plain"], weights, index)
    assert out["q_0.1"].tolist() == pytest.approx([85.0, 95.0])
    assert out["q_0.9"].tolist() == pytest.approx([125.0, 135.0])


def test_recenter_quantiles_keeps_width_and_matches_point_forecast() -> None:
    quantiles = pd.DataFrame(
        {"q_0.1": [90.0, 80.0], "q_0.5": [100.0, 95.0], "q_0.9": [115.0, 99.0]}
    )
    point = pd.Series([102.0, 90.0])
    out = recenter_quantiles(quantiles, point)
    np.testing.assert_allclose(out["q_0.5"].to_numpy(), point.to_numpy())
    # Ширина интервала сохраняется, сдвигается только центр.
    np.testing.assert_allclose(
        (out["q_0.9"] - out["q_0.1"]).to_numpy(),
        (quantiles["q_0.9"] - quantiles["q_0.1"]).to_numpy(),
    )
    assert recenter_quantiles(pd.DataFrame(), point).empty


def test_attach_meta_never_carries_prediction_columns() -> None:
    frame = pd.DataFrame(
        {
            "unique_id": ["a"],
            "ds": [pd.Timestamp("2024-01-01")],
            "fold": ["f1"],
            "model": ["Ensemble"],
            "y": [100.0],
            "y_hat": [99.0],
        }
    )
    oof = pd.DataFrame(
        {
            "unique_id": ["a"],
            "ds": [pd.Timestamp("2024-01-01")],
            "fold": ["f1"],
            "model": ["m1"],
            "y": [100.0],
            "y_hat": [90.0],
            "q_0.9": [120.0],
            "category": ["food"],
            "step": [1],
        }
    )
    out = attach_meta(frame, oof)
    assert out["y_hat"].iloc[0] == 99.0  # прогноз ансамбля не подменён
    assert out["model"].iloc[0] == "Ensemble"
    assert "q_0.9" not in out.columns  # чужой интервал не переносится
    assert out["category"].iloc[0] == "food" and out["step"].iloc[0] == 1


def test_calendar_features_from_real_reference() -> None:
    raw = json.loads(
        pd.io.common.get_handle(
            "data/reference/ru_calendar.json", "r", encoding="utf-8"
        ).handle.read()
    )
    calendar = parse_calendar(raw)
    assert calendar["ds"].min().year == 2018
    jan = calendar[calendar["ds"] == pd.Timestamp("2024-01-01")].iloc[0]
    assert jan["working_days"] == 17  # январь 2024: 17 рабочих дней
    feats = calendar_features(pd.date_range("2024-01-01", periods=12, freq="MS"), calendar)
    assert feats["working_day_share"].between(0.5, 0.8).all()
    assert feats.loc[feats["month"] == 12, "is_december"].iloc[0] == 1


def _panel_for_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ds = pd.date_range("2023-01-01", periods=24, freq="MS")
    rows = [
        {"unique_id": "a", "ds": d, "y": 100.0 + i, "category": "Все категории"}
        for i, d in enumerate(ds)
    ]
    panel = pd.DataFrame(rows)
    static = pd.DataFrame(
        {
            "unique_id": ["a"],
            "category": ["Все категории"],
            "category_slug": ["total"],
            "region_code": ["46"],
            "federal_district": ["Центральный"],
            "mo_kind": ["urban_okrug"],
            "size_group": [3],
            "size_level": [5.0],
        }
    )
    seasonal = pd.DataFrame(
        {"category": "Все категории", "month": range(1, 13), "index": np.linspace(0.9, 1.1, 12)}
    )
    return panel, static, seasonal


def test_features_are_built_at_origin_without_future_leak() -> None:
    panel, static, seasonal = _panel_for_features()
    origin = pd.Timestamp("2023-12-01")
    features = make_features(panel, static, seasonal, [origin], horizon=3)
    assert set(features["step"]) == {1, 2, 3}
    assert (features["ds"] > origin).all()
    row = features[features["step"] == 1].iloc[0]
    assert row["y_last"] == 111.0  # значение на origin
    assert row["lag_1"] == pytest.approx(110.0 / 111.0)
    # Цель — логарифм отношения факта к последнему известному значению.
    assert row["target"] == pytest.approx(np.log(113.0 / 112.0), rel=1e-6)

    truncated = panel[panel["ds"] <= origin]
    same = make_features(truncated, static, seasonal, [origin], horizon=3, require_target=False)
    shared = [c for c in feature_columns(same) if c in features.columns]
    pd.testing.assert_frame_equal(
        features[features["step"] == 1][shared].reset_index(drop=True),
        same[same["step"] == 1][shared].reset_index(drop=True),
        check_dtype=False,
    )


def test_predictions_from_ratio_inverts_target() -> None:
    features = pd.DataFrame(
        {"unique_id": ["a"], "ds": [pd.Timestamp("2024-01-01")], "y_last": [100.0]}
    )
    out = predictions_from_ratio(features, np.array([np.log(110.0 / 101.0)]))
    assert out["y_hat"].iloc[0] == pytest.approx(109.0, rel=1e-6)
    negative = predictions_from_ratio(features, np.array([-50.0]))
    assert negative["y_hat"].iloc[0] == 0.0


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "adm1": ["RS48", "RS48", "RS16", "XX99"],
            "event_root_code": ["14", "19", "06", "14"],
            "quad_class": [3, 4, 1, 3],
            "goldstein": [-5.0, -9.0, 4.0, -2.0],
            "num_articles": [10, 20, 5, 1],
            "avg_tone": [-3.0, -8.0, 2.0, -1.0],
            "date_added": pd.to_datetime(["2024-01-05", "2024-01-20", "2024-02-03", "2024-01-07"]),
        }
    )


def test_news_features_respect_cutoff_and_region_mapping() -> None:
    reference = pd.DataFrame({"fips": ["RS48", "RS16"], "region_code": ["45", "27"]})
    events = map_regions(_events(), reference)
    assert events["region_code"].isna().sum() == 1  # неизвестный ADM1 сохраняется как None

    full = monthly_news_features(events)
    january = full[(full["region_code"] == "45") & (full["ds"] == pd.Timestamp("2024-01-01"))].iloc[
        0
    ]
    assert january["news_events"] == 2
    assert january["news_material_conflict_share"] == pytest.approx(0.5)
    assert january["news_share_protest"] == pytest.approx(0.5)

    early = monthly_news_features(events, cutoff=pd.Timestamp("2024-01-10"))
    assert early["news_events"].sum() == 1, "события, добавленные позже cutoff, не учитываются"
    assert pd.Timestamp("2024-02-01") not in set(early["ds"])


def test_novelty_uses_only_previous_months() -> None:
    frame = pd.DataFrame(
        {
            "region_code": ["45"] * 6,
            "ds": pd.date_range("2024-01-01", periods=6, freq="MS"),
            "news_events": [10.0, 10.0, 10.0, 10.0, 10.0, 50.0],
        }
    )
    out = add_novelty(frame, window=3)
    assert np.isnan(out["news_events_z"].iloc[0])
    # История постоянна (разброс 0), поэтому масштабом служит 10% среднего: всплеск виден.
    assert out["news_events_z"].iloc[-1] > 3, "всплеск последнего месяца выделяется"


def test_event_calendar_respects_publication_and_scope() -> None:
    events = [
        {"date": "2024-04-01", "published_at": "2024-04-05", "region_code": "53", "kind": "flood"},
        {
            "date": "2024-07-26",
            "published_at": "2024-07-26",
            "region_code": "all",
            "kind": "key_rate",
        },
    ]
    months = pd.date_range("2024-04-01", periods=5, freq="MS")
    out = event_calendar_features(events, months, ["53", "46"], cutoff=pd.Timestamp("2024-12-31"))
    indexed = out.set_index(["region_code", "ds"])
    assert indexed.loc[("53", pd.Timestamp("2024-04-01")), "event_flood"] == 1.0
    assert indexed.loc[("46", pd.Timestamp("2024-04-01")), "event_flood"] == 0.0
    assert indexed.loc[("46", pd.Timestamp("2024-07-01")), "event_key_rate"] == 1.0
    assert indexed.loc[("53", pd.Timestamp("2024-06-01")), "event_flood_decay"] == pytest.approx(
        0.25
    )

    before = event_calendar_features(
        events, months, ["53", "46"], cutoff=pd.Timestamp("2024-04-02")
    )
    assert before["event_flood"].sum() == 0.0, "событие ещё не опубликовано на момент cutoff"


def test_ablation_matrix_and_contributions() -> None:
    blocks = {
        "sber": ["lag_1"],
        "macro": ["key_rate"],
        "news": ["news_events"],
        "foundation": ["chronos"],
    }
    assert feature_sets(DEFAULT_MATRIX[0], blocks) == ["lag_1"]
    assert feature_sets(DEFAULT_MATRIX[2], blocks) == ["key_rate", "lag_1", "news_events"]
    results = [
        AblationResult("A", {"mae": 100.0, "f1": 0.3}),
        AblationResult("B", {"mae": 95.0, "f1": 0.3}),
        AblationResult("C", {"mae": 90.0, "f1": 0.4}),
    ]
    table = ablation_table(results, baseline="A", metrics=("mae", "f1"))
    assert table.set_index("name").loc["C", "delta_mae"] == pytest.approx(-10.0)
    assert contribution(table, "C", "B", "mae") == pytest.approx(-5.0)


def test_national_features_grouped_by_source_publication_lag() -> None:
    """Недельный индикатор доступен раньше месячного — лаги не должны схлопываться в один."""
    from sbx.core.features.national import build_national_features

    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    monthly = pd.DataFrame({"period": months, "value": range(24), "type": "Всего"})
    weekly = pd.DataFrame(
        {"period": pd.date_range("2023-01-01", periods=100, freq="W-SUN"), "value": 1.0}
    )
    frames = {
        "consumer-spending": monthly,
        "median-wages": monthly.drop(columns=["type"]),
        "nedelnaa-inflazia-v-razreze-analiticeskih-komponentov": weekly.assign(indicator="всё"),
    }
    groups = build_national_features(
        frames,
        {
            "median-wages": 60,
            "consumer-spending": 35,
            "nedelnaa-inflazia-v-razreze-analiticeskih-komponentov": 7,
        },
    )

    assert set(groups) == {35, 60, 7}, "источники с разными лагами не должны сливаться"
    assert any(c.startswith("wages") for c in groups[60].columns)
    assert any(c.startswith("infl") for c in groups[7].columns)
    # Дата публикации каждой группы — конец отчётного месяца плюс её лаг.
    for lag, frame in groups.items():
        month_end = frame["ds"] + pd.offsets.MonthEnd(0)
        assert ((frame["published_at"] - month_end).dt.days == lag).all()


def test_publication_lags_config_covers_every_national_source() -> None:
    """Каждый подключённый источник должен иметь обоснованный лаг, а не молчаливый дефолт."""
    from sbx.shell.pipelines.features import NATIONAL_SLUGS, load_publication_lags

    lags = load_publication_lags()
    missing = [slug for slug in NATIONAL_SLUGS if slug not in lags]
    assert not missing, f"нет лага публикации для источников: {missing}"
    assert all(0 <= v <= 120 for v in lags.values())


def test_ensemble_weights_never_see_the_current_folds_test_window() -> None:
    """Веса фолда подбираются только на наблюдениях, известных к его cutoff.

    При перекрывающихся горизонтах «предыдущий фолд» может содержать те самые месяцы, на
    которых текущий фолд оценивается: фолд h=12 от 2023-12 покрывает весь 2024 год, то есть
    надмножество тестовых окон всех остальных фолдов. Порядка фолдов недостаточно — нужен
    отбор по дате.
    """
    rows = []
    # Длинный фолд идёт первым и покрывает весь 2024 год.
    for fold, cutoff, months in [
        ("h12_2023-12", "2023-12-01", pd.date_range("2024-01-01", periods=12, freq="MS")),
        ("h3_2024-09", "2024-09-01", pd.date_range("2024-10-01", periods=3, freq="MS")),
    ]:
        for uid in ("a", "b"):
            for d in months:
                y = 100.0 + d.month
                for model, bias in [("m1", 1.0), ("m2", -1.0)]:
                    rows.append(
                        {
                            "unique_id": uid,
                            "ds": d,
                            "fold": fold,
                            "cutoff": pd.Timestamp(cutoff),
                            "horizon": 12 if fold.startswith("h12") else 3,
                            "step": 1,
                            "model": model,
                            "y": y,
                            "y_hat": y + bias,
                            "category": "total",
                        }
                    )
    oof = pd.DataFrame(rows)

    _, weights = rolling_origin_ensemble(
        oof, ["m1", "m2"], ["h12_2023-12", "h3_2024-09"], history_log=(log := {})
    )
    used = log["h3_2024-09"]
    # Октябрь–декабрь 2024 — тестовое окно фолда h3_2024-09; в истории его быть не должно.
    assert used, "история второго фолда не должна быть пустой: сентябрь и раньше доступны"
    assert max(used) <= pd.Timestamp("2024-09-01"), f"в историю попало будущее: {max(used)}"
    assert set(weights) == {"h12_2023-12", "h3_2024-09"}


def test_configurations_are_compared_on_the_observations_they_share() -> None:
    """У ансамбля строк меньше, чем у одиночной модели: ему нужен прогноз каждой своей модели.
    Разность MAE двух конфигураций имеет смысл только на общих наблюдениях."""
    wide = pd.DataFrame(
        {
            "unique_id": ["a", "b", "c"],
            "ds": pd.Timestamp("2024-07-01"),
            "fold": "f1",
            "y": 100.0,
            "y_hat": [90.0, 95.0, 50.0],
        }
    )
    narrow = wide.iloc[[1, 0]].assign(y_hat=[99.0, 98.0])
    shared = on_common_rows({"A": wide, "D": narrow})
    assert list(shared) == ["A", "D"]
    assert sorted(shared["A"]["unique_id"]) == ["a", "b"] and len(shared["D"]) == 2
    assert sorted(shared["A"]["y_hat"]) == [90.0, 95.0], "значения конфигурации не меняются"
    assert on_common_rows({}) == {}


def _errors(errors: dict[str, float]) -> pd.DataFrame:
    """Прогнозы с заданной ошибкой: ряды a и b — горизонт 1, остальные — горизонт 3."""
    return pd.DataFrame(
        {
            "unique_id": list(errors),
            "ds": pd.Timestamp("2024-07-01"),
            "fold": "f1",
            "horizon": [1 if uid in ("a", "b") else 3 for uid in errors],
            "y": 100.0,
            "y_hat": [100.0 - e for e in errors.values()],
        }
    )


def test_source_contribution_compares_the_model_with_the_block_and_without_it() -> None:
    base = _errors({"a": 10.0, "b": 10.0, "c": 10.0, "d": 100.0})
    with_block = _errors({"a": 4.0, "b": 6.0, "c": 14.0})
    out = source_contribution(with_block, base)
    assert out["rows"] == 3, "ряд d есть только у базовой модели и в сравнение не входит"
    assert out["mae_with"] == pytest.approx(8.0) and out["mae_base"] == pytest.approx(10.0)
    assert out["delta"] == pytest.approx(-2.0)
    assert out["mean_diff"] == pytest.approx(-2.0), "средняя по рядам разность ошибок"
    assert out["delta_h1"] == pytest.approx(-5.0) and out["delta_h3"] == pytest.approx(4.0)
    assert 0.0 <= out["p_value"] <= 1.0 and np.isfinite(out["statistic"])
    assert (out["folds"], out["folds_better"], out["folds_worse"]) == (1, 1, 0)
    assert out["sign_p"] == 1.0
