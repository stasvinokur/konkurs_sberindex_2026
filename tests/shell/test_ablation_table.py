"""Матрица абляций: конфигурации сравниваются на общих наблюдениях."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sbx.core import folds
from sbx.shell.pipelines import ablations

BACKTEST = folds.BacktestConfig(
    status="test",
    checked_at="2026-10-02",
    target=folds.TargetSpec("d", "MS", "l"),
    folds=(folds.HorizonFolds(horizon=1, origins=("2024-06-01",)),),
    primary_metric="mae",
    mae_aggregation="micro",
)
CONFIG = {
    "baseline": "A",
    "configurations": {
        "A": {"models": ["LightGBM (A)"]},
        "B": {"models": ["LightGBM (B)"]},
        "D": {"models": ["LightGBM (B)", "Chronos-2"]},
    },
    "dm_pairs": [["B", "A"], ["D", "B"]],
}


def _rows(model: str, forecasts: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": list(forecasts),
            "ds": pd.Timestamp("2024-07-01"),
            "fold": "h1_2024-06",
            "cutoff": pd.Timestamp("2024-06-01"),
            "horizon": 1,
            "step": 1,
            "model": model,
            "y": 100.0,
            "y_hat": list(forecasts.values()),
        }
    )


@pytest.fixture
def built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[pd.DataFrame, Path]:
    # Chronos-2 пропускает ряд «c» — как короткие ряды в настоящем прогоне; на нём обе
    # LightGBM ошибаются сильнее всего.
    base = _rows("Chronos-2", {"a": 100.0, "b": 100.0})
    variants = pd.concat(
        [
            _rows("LightGBM (A)", {"a": 90.0, "b": 90.0, "c": 0.0}),
            _rows("LightGBM (B)", {"a": 95.0, "b": 95.0, "c": 0.0}),
        ]
    )
    monkeypatch.setattr(
        ablations, "collect_oof", lambda out_dir=None: base if out_dir is None else variants
    )
    monkeypatch.setattr(ablations, "load_backtest_config", lambda: BACKTEST)
    monkeypatch.setattr(ablations, "_cpd_metrics", lambda cfg: {})
    monkeypatch.setattr(ablations, "METRICS_DIR", tmp_path / "metrics")
    out = tmp_path / "ablations"
    return ablations.build_table(CONFIG, out), out


def test_every_configuration_is_measured_on_the_rows_all_of_them_have(
    built: tuple[pd.DataFrame, Path],
) -> None:
    """У ансамбля строк меньше, чем у одиночной модели. На своих строках A дала бы 40 руб., а
    разница с ансамблем включала бы ряд, которого у ансамбля нет."""
    table = built[0].set_index("name")
    assert table["rows"].tolist() == [2, 2, 2]
    assert table["mae"].tolist() == [10.0, 5.0, 2.5]
    assert table["delta_mae"].tolist() == [0.0, -5.0, -7.5]


def test_significance_tests_use_the_same_rows_as_the_table(
    built: tuple[pd.DataFrame, Path],
) -> None:
    saved = json.loads((built[1] / "contributions.json").read_text(encoding="utf-8"))
    assert saved["rows_common"] == 2
    assert saved["diebold_mariano"]["B_vs_A"]["mean_diff"] == -5.0
    b_vs_a = saved["diebold_mariano"]["B_vs_A"]
    assert (b_vs_a["folds"], b_vs_a["folds_better"], b_vs_a["folds_worse"]) == (1, 1, 0)
    assert b_vs_a["sign_p"] == 1.0, "по одному окну устойчивость не оценить"
    assert b_vs_a["by_fold"] == {"h1_2024-06": -5.0}
    assert saved["diebold_mariano"]["D_vs_B"]["mean_diff"] == -2.5
    assert pd.read_csv(built[1] / "ablations.csv")["rows"].tolist() == [2, 2, 2]


def test_sources_table_measures_each_block_against_the_variant_it_was_added_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {
        "feature_variants": {
            "A": {"blocks": [], "model_name": "LightGBM (A)"},
            "B": {"blocks": ["macro"], "model_name": "LightGBM (B)"},
        },
        "sources": {
            "base": "A",
            "on_top_of": "B",
            "blocks": {"macro": "Национальные ряды", "news": "Новости", "weather": "Погода"},
        },
    }
    variants = pd.concat(
        [
            _rows("LightGBM (A)", {"a": 90.0, "b": 90.0}),
            _rows("LightGBM (B)", {"a": 95.0, "b": 95.0}),
            _rows("LightGBM (A + macro)", {"a": 95.0, "b": 95.0}),
            _rows("LightGBM (A + news)", {"a": 88.0, "b": 90.0}),
            _rows("LightGBM (B + news)", {"a": 95.0, "b": 93.0}),
        ]
    )
    monkeypatch.setattr(ablations, "collect_oof", lambda out_dir=None: variants)
    table = ablations.build_sources(config, tmp_path, blocks=["macro", "news"])
    assert table[["block", "against"]].to_records(index=False).tolist() == [
        ("macro", "A"),
        ("news", "A"),
        ("news", "B"),
    ], "погоды в прогоне не было — строки о ней нет"
    row = table.set_index(["block", "against"])
    assert row.loc[("macro", "A"), "mean_diff"] == -5.0 and row.loc[("macro", "A"), "rows"] == 2
    assert row.loc[("news", "A"), "mean_diff"] == 1.0, "с новостями ошибка выше"
    assert row.loc[("news", "B"), "mean_diff"] == 1.0
    assert row.loc[("news", "B"), "title"] == "Новости"
    assert row.loc[("macro", "A"), "delta_h1"] == -5.0
    assert row.loc[("macro", "A"), ["folds", "folds_better", "folds_worse"]].tolist() == [1, 1, 0]
    assert row.loc[("news", "A"), "folds_worse"] == 1 and row.loc[("news", "A"), "sign_p"] == 1.0
    saved = pd.read_csv(tmp_path / "sources.csv")
    assert saved["block"].tolist() == ["macro", "news", "news"]


def test_sources_table_skips_a_source_whose_data_is_absent_now(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Прогнозы варианта с новостями остались на диске от прогона с архивом, а архива сейчас
    нет: строка о новостях была бы числом из другого прогона, выданным за нынешнее."""
    config = {
        "feature_variants": {"A": {"blocks": [], "model_name": "LightGBM (A)"}},
        "sources": {"base": "A", "blocks": {"macro": "Национальные ряды", "gdelt": "Новости"}},
    }
    variants = pd.concat(
        [
            _rows("LightGBM (A)", {"a": 90.0, "b": 90.0}),
            _rows("LightGBM (A + macro)", {"a": 95.0, "b": 95.0}),
            _rows("LightGBM (A + gdelt)", {"a": 88.0, "b": 90.0}),
        ]
    )
    monkeypatch.setattr(ablations, "collect_oof", lambda out_dir=None: variants)
    table = ablations.build_sources(config, tmp_path, blocks=["macro"])
    assert table["block"].tolist() == ["macro"]
    monkeypatch.setattr(ablations, "load_feature_blocks", lambda: {"macro": None, "gdelt": None})
    assert ablations.build_sources(config, tmp_path)["block"].tolist() == ["macro", "gdelt"]


def test_sources_table_is_empty_but_written_when_nothing_was_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ablations, "collect_oof", lambda out_dir=None: _rows("LightGBM (A)", {"a": 90.0})
    )
    table = ablations.build_sources({"feature_variants": {}}, tmp_path, blocks=[])
    assert table.empty and (tmp_path / "sources.csv").exists()
    saved = pd.read_csv(tmp_path / "sources.csv")
    assert {
        "block",
        "against",
        "mean_diff",
        "folds",
        "folds_better",
        "folds_worse",
        "sign_p",
    } <= set(saved.columns), "пустая таблица читается с теми же колонками, что и заполненная"
