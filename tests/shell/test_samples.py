"""Замороженные выборки: оценочный список рядов и бенчмарк сдвигов в официальных кодах."""

from __future__ import annotations

import pandas as pd
import pytest

from sbx.core.panel import build_official_panel
from sbx.shell.download import official
from sbx.shell.pipelines import entity, samples


@pytest.fixture(scope="module")
def rebuilt() -> samples.Samples:
    return samples.build()


def test_frozen_evaluation_list_is_the_published_sample_in_official_ids(
    rebuilt: samples.Samples,
) -> None:
    frozen = samples.load_evaluation()
    assert list(frozen.columns) == ["unique_id", "legacy_unique_id"]
    assert len(frozen) == 1201 and frozen["unique_id"].is_unique
    assert frozen["legacy_unique_id"].nunique() == 1200, "один ряд-склейка стал двумя"
    pd.testing.assert_frame_equal(frozen, rebuilt.evaluation)


def test_frozen_benchmark_is_the_published_benchmark_with_official_sources(
    rebuilt: samples.Samples,
) -> None:
    series, labels = samples.load_benchmark()
    assert len(labels) == 1500 and len(series) == 36000
    assert list(series.columns) == ["unique_id", "source_id", "ds", "y", "category"]
    pd.testing.assert_frame_equal(series, rebuilt.benchmark_series, check_dtype=False)
    pd.testing.assert_frame_equal(labels, rebuilt.benchmark_labels, check_dtype=False)


def test_control_series_of_the_benchmark_are_untouched_official_series() -> None:
    """Перевод в официальные коды не перепутал ряды: без инъекции ряд равен ряду панели."""
    series, labels = samples.load_benchmark()
    consumption = official.load_consumption()
    panel = build_official_panel(consumption, entity.official_territories(consumption))
    control = labels.loc[labels["kind"] == "none", "unique_id"]
    assert len(control) > 300
    got = series[series["unique_id"].isin(set(control))]
    merged = got.merge(
        panel[["unique_id", "ds", "y"]].rename(columns={"unique_id": "source_id", "y": "real"}),
        on=["source_id", "ds"],
        how="left",
    )
    assert merged["real"].notna().all() and (merged["y"] == merged["real"]).all()


def test_samples_on_disk_match_what_the_published_run_used(
    rebuilt: samples.Samples, monkeypatch
) -> None:
    monkeypatch.setattr(samples, "build", lambda: rebuilt)
    assert samples.problems() == []


def test_configs_point_the_pipelines_at_the_frozen_files() -> None:
    from sbx.shell.io import CONFIG_DIR, ROOT, read_yaml
    from sbx.shell.pipelines import backtest

    models = read_yaml(CONFIG_DIR / "models.yaml")
    assert ROOT / models["evaluation_sample"]["frozen"] == samples.EVALUATION_SERIES
    cpd = read_yaml(CONFIG_DIR / "cpd.yaml")
    assert ROOT / cpd["frozen_benchmark"] == samples.BENCHMARK_PREFIX
    ids = samples.load_evaluation()["unique_id"]
    static = pd.DataFrame({"unique_id": ids, "category": "x", "size_group": 1})
    assert backtest.evaluation_series(static, models) == sorted(ids)


def test_cli_freeze_samples_checks_without_rewriting(rebuilt: samples.Samples, monkeypatch) -> None:
    from typer.testing import CliRunner

    from sbx.shell import cli

    monkeypatch.setattr(samples, "build", lambda: rebuilt)

    def no_write(_: samples.Samples) -> list:
        raise AssertionError("--check не должен переписывать файлы")

    monkeypatch.setattr(samples, "write", no_write)
    result = CliRunner().invoke(cli.app, ["data", "freeze-samples", "--check"])
    assert result.exit_code == 0 and "1201" in result.output

    monkeypatch.setattr(
        samples, "problems", lambda: ["configs/evaluation_series.csv: не совпадает"]
    )
    result = CliRunner().invoke(cli.app, ["data", "freeze-samples", "--check"])
    assert result.exit_code == 1 and "evaluation_series.csv" in result.output


def test_problems_name_a_frozen_file_that_drifted(
    rebuilt: samples.Samples, tmp_path, monkeypatch
) -> None:
    changed = rebuilt.evaluation.iloc[1:]
    path = tmp_path / "evaluation_series.csv"
    changed.to_csv(path, index=False)
    monkeypatch.setattr(samples, "EVALUATION_SERIES", path)
    monkeypatch.setattr(samples, "build", lambda: rebuilt)
    found = samples.problems()
    assert len(found) == 1 and "evaluation_series.csv" in found[0]


def test_the_name_heuristic_misplaced_31_series_of_the_evaluation_sample() -> None:
    """Число для разбора идентификации: сколько рядов оценки стояли в чужом регионе."""
    consumption = official.load_consumption()
    table, _ = entity.heuristic_crosswalk(consumption, entity.official_territories(consumption))
    frozen = samples.load_evaluation()
    rows = table[table["unique_id"].isin(set(frozen["unique_id"]))]
    wrong = rows[rows["heuristic_region"] != rows["official_region"]]
    assert wrong["legacy_unique_id"].nunique() == 31


def test_problems_name_a_drifted_benchmark_file(
    rebuilt: samples.Samples, tmp_path, monkeypatch
) -> None:
    prefix = tmp_path / "bench"
    rebuilt.benchmark_series.assign(y=0.0).to_parquet(f"{prefix}_series.parquet", index=False)
    rebuilt.benchmark_labels.iloc[1:].to_parquet(f"{prefix}_labels.parquet", index=False)
    monkeypatch.setattr(samples, "BENCHMARK_PREFIX", prefix)
    monkeypatch.setattr(samples, "build", lambda: rebuilt)
    found = samples.problems()
    assert [p.split(":")[0].rsplit("/", 1)[-1] for p in found] == [
        "bench_series.parquet",
        "bench_labels.parquet",
    ]


FROZEN_SHA256 = {
    "configs/evaluation_series.csv": (
        "71d78d15a6f109690ec500e8d4d927db65c07c20406b58cb27581664a5aa9561"
    ),
    "data/benchmark/cpd_benchmark_v1_series.parquet": (
        "1ea9586ca78167afd5bcb9ef8f0aa66fce13587432eecb515150beb8a4dbfa6b"
    ),
    "data/benchmark/cpd_benchmark_v1_labels.parquet": (
        "2d9ac358a45ec5ee303f977f88fbe6bc1b3b69fc5f871b061e72b44dbe11d31d"
    ),
}


def test_frozen_files_are_the_ones_fixed_on_2026_10_02() -> None:
    """Сверка «файлы равны повтору отбора» держится на нынешнем коде отбора. Контрольные суммы
    привязывают файлы к тем, что были сверены с опубликованным расчётом: перегенерация
    выборок не пройдёт незамеченной."""
    from sbx.shell.io import ROOT, sha256_file

    assert {path: sha256_file(ROOT / path) for path in FROZEN_SHA256} == FROZEN_SHA256
