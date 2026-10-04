"""Сравнение двух прогонов: что изменилось и на каких наблюдениях это измерено."""

from __future__ import annotations

import pandas as pd
import pytest

from sbx.core.comparison import (
    align_oof,
    compare_models,
    compare_tables,
    comparison_markdown,
    horizon_leaders,
    self_check,
    split_series,
    tuning_changes,
)

MAPPING = pd.DataFrame(
    {
        "unique_id": ["01-0001__total", "61-1847__total", "61-3101__total"],
        "legacy_unique_id": ["01-a__total", "61-s__total", "61-s__total"],
    }
)


def _rows(uid: str, months: list[str], model: str, y_hat: float, fold: str = "h1_2024-06"):
    return pd.DataFrame(
        {
            "unique_id": uid,
            "ds": pd.to_datetime(months),
            "fold": fold,
            "horizon": 1,
            "model": model,
            "y": 100.0,
            "y_hat": y_hat,
        }
    )


def test_align_matches_the_same_observations_under_different_ids() -> None:
    old = pd.concat(
        [
            _rows("01-a__total", ["2024-07-01"], "Naive", 90.0),
            # Ряд-склейка прежней панели: его месяцы принадлежат двум официальным рядам.
            _rows("61-s__total", ["2023-12-01", "2024-07-01"], "Naive", 80.0),
            _rows("09-gone__total", ["2024-07-01"], "Naive", 70.0),
        ]
    )
    new = pd.concat(
        [
            _rows("01-0001__total", ["2024-07-01"], "Naive", 95.0),
            _rows("61-1847__total", ["2023-12-01"], "Naive", 85.0),
            _rows("61-3101__total", ["2024-07-01"], "Naive", 86.0),
            _rows("02-0002__total", ["2024-07-01"], "Naive", 60.0),
        ]
    )
    aligned = align_oof(old, new, MAPPING).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    assert aligned["unique_id"].tolist() == ["01-0001__total", "61-1847__total", "61-3101__total"]
    assert aligned["y_hat_old"].tolist() == [90.0, 80.0, 80.0]
    assert aligned["y_hat_new"].tolist() == [95.0, 85.0, 86.0]
    assert aligned["ds"].tolist() == list(
        pd.to_datetime(["2024-07-01", "2023-12-01", "2024-07-01"])
    )


def test_align_refuses_rows_whose_actuals_differ() -> None:
    """Общие строки — это одни и те же наблюдения: факт обязан совпадать, иначе сопоставление
    рядов неверно и сравнивать нечего."""
    old = _rows("01-a__total", ["2024-07-01"], "Naive", 90.0)
    new = _rows("01-0001__total", ["2024-07-01"], "Naive", 95.0).assign(y=101.0)
    with pytest.raises(ValueError, match="факт"):
        align_oof(old, new, MAPPING)


def test_compare_models_reports_mae_on_common_rows_and_exact_reproduction() -> None:
    months = ["2024-07-01"]
    old = pd.concat(
        [
            _rows("01-a__total", months, "Naive", 90.0),
            _rows("01-a__total", months, "LightGBM", 96.0),
            _rows("09-gone__total", months, "Naive", 10.0),
        ]
    )
    new = pd.concat(
        [
            _rows("01-0001__total", months, "Naive", 90.0),
            _rows("01-0001__total", months, "LightGBM", 99.0),
        ]
    )
    table = compare_models(align_oof(old, new, MAPPING), old, new).set_index("model")
    assert table.loc["Naive", ["rows_old", "rows_new", "rows_common"]].tolist() == [2, 1, 1]
    assert table.loc["Naive", "mae_old"] == 10.0 and table.loc["Naive", "mae_new"] == 10.0
    assert bool(table.loc["Naive", "identical"]) is True
    assert table.loc["LightGBM", "mae_old"] == 4.0 and table.loc["LightGBM", "mae_new"] == 1.0
    assert table.loc["LightGBM", "delta"] == -3.0
    assert bool(table.loc["LightGBM", "identical"]) is False
    assert table.loc["LightGBM", "max_abs_diff"] == 3.0


def test_compare_tables_puts_old_and_new_side_by_side() -> None:
    old = pd.DataFrame({"name": ["A", "B"], "mae": [662.5, 774.4]})
    new = pd.DataFrame({"name": ["A", "B", "G"], "mae": [660.0, 601.7, 590.0]})
    table = compare_tables(old, new, keys=["name"], columns=["mae"]).set_index("name")
    assert table.loc["B", ["mae_old", "mae_new", "mae_delta"]].round(1).tolist() == [
        774.4,
        601.7,
        -172.7,
    ]
    assert pd.isna(table.loc["G", "mae_old"]) and table.loc["G", "mae_new"] == 590.0


HEADLINE_OLD = pd.DataFrame(
    {
        "model": ["Chronos-2 (covariates)", "Prophet", "Ensemble", "Prophet"],
        "horizon": [1, 1, 6, 6],
        "mae": [580.8, 678.9, 488.2, 1354.9],
    }
)
HEADLINE_NEW = pd.DataFrame(
    {
        "model": ["Chronos-2 (covariates)", "Prophet", "LightGBM", "Prophet"],
        "horizon": [1, 1, 6, 6],
        "mae": [581.7, 679.0, 470.0, 1350.0],
    }
)


def test_horizon_leaders_name_the_best_model_and_the_reference_in_both_runs() -> None:
    leaders = horizon_leaders(HEADLINE_OLD, HEADLINE_NEW).set_index("horizon")
    assert leaders.loc[1, ["leader_old", "leader_new"]].tolist() == ["Chronos-2 (covariates)"] * 2
    assert leaders.loc[6, ["leader_old", "leader_new"]].tolist() == ["Ensemble", "LightGBM"]
    assert leaders.loc[6, "mae_new"] == 470.0 and leaders.loc[6, "reference_new"] == 1350.0
    assert bool(leaders.loc[6, "leader_changed"]) and not bool(leaders.loc[1, "leader_changed"])


def test_tuning_changes_list_only_the_folds_where_the_choice_moved() -> None:
    old = {"h1_2023-12": {"growth": "flat", "order": 2}, "h3_2023-12": {"growth": "linear"}}
    new = {"h1_2023-12": {"growth": "flat", "order": 2}, "h3_2023-12": {"growth": "flat"}}
    assert tuning_changes(old, new) == [
        {"fold": "h3_2023-12", "old": {"growth": "linear"}, "new": {"growth": "flat"}}
    ]
    assert tuning_changes(old, {}) == [], "окна, которого нет в одном из прогонов, не сравнить"


def _aligned(model: str, uid: str, old: float, new: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": [uid],
            "ds": pd.to_datetime(["2024-10-01"]),
            "fold": ["h1_2024-09"],
            "horizon": [1],
            "model": [model],
            "y": [100.0],
            "y_hat_old": [old],
            "y_hat_new": [new],
        }
    )


UNTOUCHED = ["Naive", "AutoETS", "TimesFM-2.5", "AutoTheta", "Chronos-2"]
ALIGNED = pd.concat(
    [
        _aligned("Naive", "01-0001__total", 90.0, 90.0),
        _aligned("Naive", "61-3101__total", 80.0, 80.0),
        _aligned("AutoETS", "01-0001__total", 90.0, 90.0),
        _aligned("AutoETS", "61-3101__total", 80.0, 95.0),  # расклеенный ряд: история другая
        _aligned("TimesFM-2.5", "01-0001__total", 90.0, 90.02),  # погрешность вычислений
        _aligned("TimesFM-2.5", "61-3101__total", 80.0, 83.0),
        _aligned("AutoTheta", "01-0001__total", 90.0, 95.0),  # настоящее изменение
        _aligned("LightGBM", "01-0001__total", 90.0, 70.0),  # исправления её касаются
    ],
    ignore_index=True,
)


def test_split_series_are_the_official_halves_of_a_glued_legacy_series() -> None:
    assert split_series(MAPPING) == ["61-1847__total", "61-3101__total"]


def test_self_check_says_where_the_untouched_models_differ() -> None:
    """Модель, которой исправления не касаются, обязана дать те же прогнозы. Исключение одно:
    ряды, которые прежняя панель склеивала, — у них изменилась сама история."""
    check = self_check(ALIGNED, UNTOUCHED, split_series(MAPPING)).set_index("model")
    assert list(check.index) == UNTOUCHED, "порядок — как задан; модели вне списка не входят"
    assert check["verdict"].to_dict() == {
        "Naive": "identical",
        "AutoETS": "split_only",
        "TimesFM-2.5": "numeric",
        "AutoTheta": "changed",
        "Chronos-2": "absent",
    }
    assert check.loc["AutoETS", ["rows", "differing", "differing_split"]].tolist() == [2, 1, 1]
    assert check.loc["TimesFM-2.5", ["differing", "differing_split"]].tolist() == [2, 1]
    assert check.loc["TimesFM-2.5", "max_diff_elsewhere"] == pytest.approx(0.02)
    assert check.loc["AutoTheta", "max_diff_elsewhere"] == 5.0
    assert check.loc["Chronos-2", "rows"] == 0


MODELS = pd.DataFrame(
    {
        "model": ["Naive", "LightGBM"],
        "horizon": [1, 1],
        "rows_old": [10, 10],
        "rows_new": [10, 10],
        "rows_common": [10, 10],
        "mae_old": [800.0, 700.0],
        "mae_new": [800.0, 650.0],
        "delta": [0.0, -50.0],
        "max_abs_diff": [0.0, 120.0],
        "identical": [True, False],
    }
)


def test_markdown_states_the_self_check_and_every_section() -> None:
    split = split_series(MAPPING)
    text = comparison_markdown(
        {
            "models": MODELS,
            "self_check": self_check(ALIGNED, ["Naive", "AutoETS", "TimesFM-2.5"], split),
            "split_series": split,
            "leaders": horizon_leaders(HEADLINE_OLD, HEADLINE_NEW),
            "ablations": compare_tables(
                pd.DataFrame({"name": ["A", "B"], "mae": [662.5, 774.4]}),
                pd.DataFrame({"name": ["A", "B"], "mae": [660.0, 601.7]}),
                keys=["name"],
                columns=["mae"],
            ),
        }
    )
    assert "## 1. Самопроверка перехода" in text
    assert "**Самопроверка пройдена.**" in text
    assert "| Naive | 2 | 0 | 0 | 0,00 | совпали до копейки |" in text
    assert "| AutoETS | 2 | 1 | 1 | 0,00 | совпали всюду, кроме расклеенных рядов |" in text
    assert "| TimesFM-2.5 | 2 | 2 | 1 | 0,02 | в пределах погрешности вычислений |" in text
    assert "61-1847__total, 61-3101__total" in text
    assert "LightGBM (наибольшее расхождение 120,0 руб.)" in text
    assert "Naive (наибольшее" not in text, "в списке затронутых — только затронутые модели"
    assert "| LightGBM | 700,0 → 650,0 (−50,0) |" in text
    assert "| Naive | 800,0 → 800,0 (0,0) |" in text
    assert "| 6 мес. | Ensemble (488,2) | LightGBM (470,0) |" in text
    assert "| B | 774,4 | 601,7 | −172,7 |" in text
    assert "нет данных" in text, "раздел без данных так и подписан, а не пропущен молча"


def test_markdown_says_plainly_when_the_self_check_fails() -> None:
    """Изменилась модель, которая меняться не должна, или её нет в одном из прогонов —
    переход не подтверждён, и документ говорит это первым делом."""
    check = self_check(ALIGNED, UNTOUCHED, split_series(MAPPING))
    text = comparison_markdown({"models": MODELS, "self_check": check, "split_series": []})
    assert "**Самопроверка не пройдена: AutoTheta, Chronos-2.**" in text
    assert "| AutoTheta | 1 | 1 | 0 | 5,00 | изменились |" in text
    assert "| Chronos-2 | 0 | 0 | 0 | — | нет общих строк |" in text
    assert "**Самопроверка пройдена.**" not in text
