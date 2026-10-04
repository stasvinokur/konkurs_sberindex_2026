"""Команда сравнения текущих артефактов с сохранённой копией прежнего прогона."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sbx.shell.pipelines import compare

MAPPING = pd.DataFrame({"unique_id": ["01-0001__total"], "legacy_unique_id": ["01-a__total"]})


def _oof(uid: str, model: str, y_hat: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": [uid],
            "ds": pd.to_datetime(["2024-07-01"]),
            "fold": ["h1_2024-06"],
            "horizon": [1],
            "model": [model],
            "y": [100.0],
            "y_hat": [y_hat],
        }
    )


def _run_dir(root: Path, uid: str, lightgbm: float, leader_mae: float) -> Path:
    (root / "oof").mkdir(parents=True)
    (root / "metrics").mkdir()
    _oof(uid, "Naive", 90.0).to_parquet(root / "oof" / "stats.parquet", index=False)
    _oof(uid, "LightGBM", lightgbm).to_parquet(root / "oof" / "lgbm.parquet", index=False)
    pd.DataFrame(
        {"model": ["LightGBM", "Prophet"], "horizon": [1, 1], "mae": [leader_mae, 20.0]}
    ).to_csv(root / "metrics" / "leaderboard_by_horizon.csv", index=False)
    tuning = {"lgbm_tuning": {"h1_2024-06": {"params": {"num_leaves": int(leader_mae)}}}}
    (root / "oof" / "lgbm_context.json").write_text(json.dumps(tuning), encoding="utf-8")
    return root


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    baseline = _run_dir(tmp_path / "baseline", "01-a__total", lightgbm=96.0, leader_mae=4.0)
    current = _run_dir(tmp_path / "current", "01-0001__total", lightgbm=99.0, leader_mae=1.0)
    monkeypatch.setattr(compare, "ARTIFACTS_DIR", current)
    monkeypatch.setattr(compare.samples, "load_evaluation", lambda: MAPPING)
    return baseline


def test_compare_writes_the_document_and_the_model_table(runs: Path, tmp_path: Path) -> None:
    summary = compare.run(runs)
    document = Path(summary["document"])
    assert document == tmp_path / "current" / "comparison" / "comparison.md"
    text = document.read_text(encoding="utf-8")
    assert "| Naive | 1 | 0 | 0 | 0,00 | совпали до копейки |" in text
    assert "**Самопроверка не пройдена:" in text, "остальных моделей списка в прогоне нет"
    assert "| LightGBM | 4,0 → 1,0 (−3,0) |" in text
    assert "| 1 мес. | LightGBM (4,0) | LightGBM (1,0) | 20,0 → 20,0 |" in text
    assert "**LightGBM:** выбор изменился в 1 окнах." in text
    assert summary["identical_models"] == ["Naive"] and summary["changed_models"] == ["LightGBM"]
    assert summary["self_check"]["Naive"] == "identical"
    assert set(summary["self_check"]) == set(compare.UNTOUCHED)
    assert summary["self_check_passed"] is False
    table = pd.read_csv(document.with_name("models.csv"))
    assert set(table["model"]) == {"Naive", "LightGBM"}


def test_compare_excuses_the_series_the_old_panel_glued(
    runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Прежний ряд-склейка разошёлся на два официальных: у преемника теперь своя история, и
    расхождение его прогнозов — не провал самопроверки."""
    glued = pd.DataFrame(
        {
            "unique_id": ["01-0001__total", "61-1847__total", "61-3101__total"],
            "legacy_unique_id": ["01-a__total", "61-s__total", "61-s__total"],
        }
    )
    monkeypatch.setattr(compare.samples, "load_evaluation", lambda: glued)
    _oof("61-s__total", "Naive", 80.0).to_parquet(runs / "oof" / "glued.parquet", index=False)
    _oof("61-3101__total", "Naive", 86.0).to_parquet(
        tmp_path / "current" / "oof" / "glued.parquet", index=False
    )
    summary = compare.run(runs)
    assert summary["self_check"]["Naive"] == "split_only"
    text = Path(summary["document"]).read_text(encoding="utf-8")
    assert "| Naive | 2 | 1 | 1 | 0,00 | совпали всюду, кроме расклеенных рядов |" in text
    assert "Расклеенные ряды: 61-1847__total, 61-3101__total." in text


def test_cli_compare_prints_where_the_document_is(runs: Path) -> None:
    from typer.testing import CliRunner

    from sbx.shell import cli

    result = CliRunner().invoke(cli.app, ["backtest", "compare", str(runs)])
    assert result.exit_code == 0, result.output
    assert "comparison.md" in result.output and "Naive" in result.output
    assert "самопроверка не пройдена" in result.output

    missing = CliRunner().invoke(cli.app, ["backtest", "compare", str(runs / "nope")])
    assert missing.exit_code == 1 and "нет каталога" in missing.output


def test_compare_shows_the_windows_where_the_new_run_counted_them(
    runs: Path, tmp_path: Path
) -> None:
    """Прежний прогон знал только p по рядам; в новом рядом стоит счёт окон проверки."""
    pair = {"pair": "LightGBM vs Prophet", "mean_diff": -97.5, "p_value": 2.0e-14}
    windows = {"folds": 4, "folds_better": 1, "folds_worse": 3, "sign_p": 0.625}
    for root, row in ((runs, pair), (tmp_path / "current", {**pair, **windows})):
        significance = {"by_horizon": {"1": {"best_model": "LightGBM", "dm": [row]}}}
        (root / "metrics" / "significance.json").write_text(
            json.dumps(significance), encoding="utf-8"
        )
        (root / "ablations").mkdir()
        pd.DataFrame({"name": ["A", "B"], "mae": [640.8, 611.3]}).to_csv(
            root / "ablations" / "ablations.csv", index=False
        )
        contributions = {"diebold_mariano": {"B_vs_A": {**row, "mean_diff": -29.6}}}
        (root / "ablations" / "contributions.json").write_text(
            json.dumps(contributions), encoding="utf-8"
        )
    text = Path(compare.run(runs)["document"]).read_text(encoding="utf-8")
    assert (
        "| 1 мес. | LightGBM: −97,5 руб., p < 0,001 | LightGBM: −97,5 руб., p < 0,001, "
        "точнее в 1 окне из 4 |" in text
    )
    assert (
        "| B_vs_A | −29,6 руб., p < 0,001 | −29,6 руб., p < 0,001, точнее в 1 окне из 4 |" in text
    )
    assert "p — тест по рядам" in text, "что значит p и почему рядом счёт окон"
