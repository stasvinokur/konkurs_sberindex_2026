"""Замороженные выборки: оценочный список рядов и бенчмарк сдвигов.

Обе выборки получены случайным отбором на панели, построенной идентификацией по названиям.
На панели в официальных кодах тот же отбор дал бы другие ряды, и изменение метрик нельзя было
бы отделить от смены выборки. Поэтому выборки зафиксированы файлами: это те же ряды, что и в
опубликованном расчёте, переведённые в официальные коды точным сопоставлением по значениям.

`build` повторяет прежний путь целиком (выгрузка → идентификация по названиям → отбор), так
что происхождение файлов проверяемо: `sbx data freeze-samples --check`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sbx.core.changepoint.synthetic import build_benchmark
from sbx.core.crosswalk import crosswalk_table, match_blocks, rekey, to_official_ids
from sbx.core.forecasting import stratified_series_sample
from sbx.core.panel import build_panel, static_features
from sbx.shell.download import official
from sbx.shell.io import CONFIG_DIR, DATA_DIR, ROOT, read_yaml
from sbx.shell.pipelines import entity

EVALUATION_SERIES = CONFIG_DIR / "evaluation_series.csv"
BENCHMARK_PREFIX = DATA_DIR / "benchmark" / "cpd_benchmark_v1"


@dataclass(frozen=True)
class Samples:
    evaluation: pd.DataFrame
    benchmark_series: pd.DataFrame
    benchmark_labels: pd.DataFrame


def load_evaluation(path: Path | None = None) -> pd.DataFrame:
    """Оценочные ряды: официальный `unique_id` и прежний `legacy_unique_id`."""
    return pd.read_csv(path or EVALUATION_SERIES, dtype=str)


def benchmark_paths(prefix: Path | None = None) -> tuple[Path, Path]:
    prefix = Path(prefix or BENCHMARK_PREFIX)
    return (
        prefix.with_name(prefix.name + "_series.parquet"),
        prefix.with_name(prefix.name + "_labels.parquet"),
    )


def load_benchmark(prefix: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    series_path, labels_path = benchmark_paths(prefix)
    return pd.read_parquet(series_path), pd.read_parquet(labels_path)


def build() -> Samples:
    """Повторяет отбор опубликованного расчёта и переводит ряды в официальные коды."""
    raw = entity.load_raw_spending()
    mapping, heuristic = entity.heuristic_mapping(raw)
    panel = build_panel(raw, mapping)
    # Отбор опубликованного расчёта стратифицировал ряды по размеру, который у территорий без
    # данных за первый год оценивался по всему ряду. Повтор отбора берёт то же правило; в
    # признаки модели этот размер не идёт (см. `sbx.core.panel.static_features`).
    static = static_features(panel, heuristic, fallback_to_whole_series=True)
    consumption = official.load_consumption()
    table = crosswalk_table(
        match_blocks(raw, consumption), mapping, heuristic, entity.official_territories(consumption)
    )

    sample = read_yaml(CONFIG_DIR / "models.yaml")["evaluation_sample"]
    picked = stratified_series_sample(static, int(sample["n_series"]), int(sample["seed"]))
    evaluation = to_official_ids(picked, table)

    cpd = read_yaml(CONFIG_DIR / "cpd.yaml")
    series, labels = build_benchmark(
        panel,
        static,
        n_series=int(cpd["n_series"]),
        deltas=tuple(cpd["deltas"]),
        control_share=float(cpd["control_share"]),
        seed=int(cpd["seed"]),
        exclude_ids=static.loc[static["admin_break_at"].notna(), "unique_id"].tolist(),
    )
    return Samples(evaluation, rekey(series, "source_id", table), rekey(labels, "source_id", table))


def write(samples: Samples) -> list[Path]:
    series_path, labels_path = benchmark_paths()
    series_path.parent.mkdir(parents=True, exist_ok=True)
    samples.evaluation.to_csv(EVALUATION_SERIES, index=False, lineterminator="\n")
    samples.benchmark_series.to_parquet(series_path, index=False)
    samples.benchmark_labels.to_parquet(labels_path, index=False)
    return [EVALUATION_SERIES, series_path, labels_path]


def _differs(path: Path, expected: pd.DataFrame, read) -> bool:
    if not path.exists():
        return True
    try:
        pd.testing.assert_frame_equal(read(path), expected, check_dtype=False)
    except AssertionError:
        return True
    return False


def problems() -> list[str]:
    """Расхождения файлов на диске с тем, что даёт повтор отбора; пусто — всё совпадает."""
    samples = build()
    series_path, labels_path = benchmark_paths()
    checks = [
        (EVALUATION_SERIES, samples.evaluation, load_evaluation),
        (series_path, samples.benchmark_series, pd.read_parquet),
        (labels_path, samples.benchmark_labels, pd.read_parquet),
    ]
    return [
        f"{_relative(path)}: не совпадает с отбором опубликованного расчёта"
        for path, expected, read in checks
        if _differs(path, expected, read)
    ]


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name
