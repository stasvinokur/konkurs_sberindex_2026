"""Кэш прогнозов абляций пересчитывается, когда меняются внешние признаки варианта.

Без архива новостей блок `news` — один календарь событий. Если после такого прогона архив
скачать, прежний кэш варианта «+новости» вернул бы числа без новостей молча. Кэш сверяет
отпечаток входов, включая содержимое блоков признаков, которыми пользуется вариант.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sbx.core import folds
from sbx.core.features.asof import PAST_ONLY, ExogenousBlock
from sbx.shell.models import base as model_base
from sbx.shell.pipelines import ablations, backtest

MONTH = pd.to_datetime(["2024-01-01"])
MACRO = ExogenousBlock("macro", pd.DataFrame({"origin": MONTH, "key_rate": [16.0]}), PAST_ONLY)
EVENTS = pd.DataFrame({"origin": MONTH, "region_code": ["77"], "event_flood": [0]})
NEWS_AND_EVENTS = EVENTS.assign(news_events=[120], news_tone_mean=[-1.5])
CONFIG = {
    "feature_variants": {
        "B": {"blocks": ["macro"], "model_name": "LightGBM (B)"},
        "C": {"blocks": ["macro", "news"], "model_name": "LightGBM (C)"},
    }
}
BACKTEST = folds.BacktestConfig(
    status="test",
    checked_at="2026-10-02",
    target=folds.TargetSpec("d", "MS", "l"),
    folds=(folds.HorizonFolds(horizon=1, origins=("2024-06-01",)),),
    primary_metric="mae",
    mae_aggregation="micro",
)


class Runs:
    """Настоящий `run_group` с подменённой моделью: видно, какие варианты считались заново."""

    def __init__(self) -> None:
        self.computed: list[str] = []

    def forecast(self, train, fold, cfg, context) -> pd.DataFrame:
        self.computed.append(str(cfg["model_name"]))
        month = fold.cutoff + pd.DateOffset(months=1)
        ids = sorted(train["unique_id"].unique())
        return pd.DataFrame(
            {"unique_id": ids, "ds": month, "model": cfg["model_name"], "y_hat": 1.0}
        )

    def rerun(self, **kwargs: bool) -> list[str]:
        self.computed.clear()
        ablations.run_feature_ablations(CONFIG, **kwargs)
        return list(self.computed)


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Runs:
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    panel = pd.DataFrame(
        {"unique_id": "a__total", "category": "Все категории", "ds": months, "y": 100.0}
    )
    static = pd.DataFrame(
        {"unique_id": ["a__total"], "category": ["Все категории"], "size_group": [1]}
    )
    seasonal = pd.DataFrame({"category": "Все категории", "month": range(1, 13), "index": 1.0})
    fake = Runs()
    monkeypatch.setitem(model_base.REGISTRY, "lgbm", fake.forecast)
    monkeypatch.setattr(ablations, "ABLATION_OOF_DIR", tmp_path)
    monkeypatch.setattr(ablations, "load_models_config", lambda: {"groups": {"lgbm": {}}})
    monkeypatch.setattr(ablations, "load_backtest_config", lambda: BACKTEST)
    monkeypatch.setattr(ablations, "load_panel", lambda: panel)
    monkeypatch.setattr(ablations, "load_static", lambda: static)
    monkeypatch.setattr(backtest, "load_seasonal", lambda: seasonal)
    return fake


def _blocks(monkeypatch: pytest.MonkeyPatch, news: pd.DataFrame) -> None:
    blocks = {"macro": MACRO, "news": ExogenousBlock("news", news, PAST_ONLY)}
    monkeypatch.setattr(ablations, "load_feature_blocks", lambda: blocks)


def test_a_variant_is_recomputed_when_the_news_archive_appears_or_disappears(
    runs: Runs, monkeypatch: pytest.MonkeyPatch
) -> None:
    _blocks(monkeypatch, EVENTS)
    assert runs.rerun() == ["LightGBM (B)", "LightGBM (C)"], "первый прогон: архива новостей нет"
    assert runs.rerun() == [], "признаки те же — кэш годится"

    _blocks(monkeypatch, NEWS_AND_EVENTS)
    assert runs.rerun() == ["LightGBM (C)"], "архив скачан: вариант с новостями считается заново"
    assert runs.rerun() == []

    _blocks(monkeypatch, EVENTS)
    assert runs.rerun() == ["LightGBM (C)"], "архива снова нет"


def test_a_variant_is_recomputed_when_feature_values_change_under_the_same_columns(
    runs: Runs, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Набор колонок тот же, значения другие — например, архив новостей докачан до конца."""
    _blocks(monkeypatch, NEWS_AND_EVENTS)
    runs.rerun()
    _blocks(monkeypatch, NEWS_AND_EVENTS.assign(news_events=[150]))
    assert runs.rerun() == ["LightGBM (C)"]


def test_a_cache_without_a_stamp_is_recomputed(
    runs: Runs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Прогнозы без отметки входов посчитаны неизвестно на каких признаках."""
    _blocks(monkeypatch, NEWS_AND_EVENTS)
    runs.rerun()
    (tmp_path / "lgbm_C_inputs.json").unlink()
    assert runs.rerun() == ["LightGBM (C)"]
    assert not list(tmp_path.glob("*_exogenous.json")), "прежние отметки набора колонок не нужны"


def test_overwrite_still_recomputes_everything(runs: Runs, monkeypatch: pytest.MonkeyPatch) -> None:
    _blocks(monkeypatch, EVENTS)
    runs.rerun()
    assert runs.rerun(overwrite=True) == ["LightGBM (B)", "LightGBM (C)"]


SOURCES = {
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


def test_source_pairs_add_each_block_to_the_base_and_to_the_best_set() -> None:
    pairs = ablations.source_pairs(SOURCES)
    assert [(p["block"], p["against"]) for p in pairs] == [
        ("macro", "A"),
        ("news", "A"),
        ("news", "B"),
        ("weather", "A"),
        ("weather", "B"),
    ], "блок, уже входящий в набор, поверх него не считается"
    assert pairs[0] == {
        "block": "macro",
        "title": "Национальные ряды",
        "against": "A",
        "cache": "lgbm_src_A_macro",
        "model": "LightGBM (A + macro)",
        "base_model": "LightGBM (A)",
        "blocks": ["macro"],
    }
    assert pairs[2]["blocks"] == ["macro", "news"] and pairs[2]["base_model"] == "LightGBM (B)"
    assert ablations.source_pairs({"feature_variants": {}}) == []


def test_source_variants_run_next_to_the_matrix_and_skip_blocks_that_are_absent(
    runs: Runs, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Источник без данных (архива нет, таблица не построена) не считается и не выдумывается."""
    _blocks(monkeypatch, EVENTS)
    monkeypatch.setattr(runs, "rerun", lambda: None)
    runs.computed.clear()
    ablations.run_feature_ablations(SOURCES)
    assert runs.computed == [
        "LightGBM (A)",
        "LightGBM (B)",
        "LightGBM (A + macro)",
        "LightGBM (A + news)",
        "LightGBM (B + news)",
    ]
    runs.computed.clear()
    ablations.run_feature_ablations(SOURCES)
    assert runs.computed == [], "кэш вариантов источников сверяется так же, как у матрицы"
    monkeypatch.setattr(ablations, "evaluation_series", lambda static, cfg: None)
    assert sorted(ablations.expected_inputs(SOURCES)) == [
        "lgbm_A",
        "lgbm_B",
        "lgbm_src_A_macro",
        "lgbm_src_A_news",
        "lgbm_src_B_news",
    ]


def test_a_source_may_consist_of_several_blocks() -> None:
    """Погода — это отклонения на момент прогноза и норма целевого месяца: два блока с разной
    привязкой ко времени, но источник один и строка о нём одна."""
    config = {
        "feature_variants": {"A": {"blocks": [], "model_name": "LightGBM (A)"}},
        "sources": {
            "base": "A",
            "blocks": {"weather": {"title": "Погода", "blocks": ["weather", "climate"]}},
        },
    }
    pairs = ablations.source_pairs(config)
    assert pairs == [
        {
            "block": "weather",
            "title": "Погода",
            "against": "A",
            "cache": "lgbm_src_A_weather",
            "model": "LightGBM (A + weather)",
            "base_model": "LightGBM (A)",
            "blocks": ["weather", "climate"],
        }
    ]
