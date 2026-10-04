"""Матрица абляций: какие наборы признаков и компонентов входят в эксперимент."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sbx.core.significance import fold_consistency, panel_dm

FEATURE_BLOCKS = ("sber", "macro", "news", "foundation", "residual_cpd")


@dataclass(frozen=True)
class AblationSpec:
    name: str
    blocks: tuple[str, ...]
    description: str = ""

    def has(self, block: str) -> bool:
        return block in self.blocks


DEFAULT_MATRIX: tuple[AblationSpec, ...] = (
    AblationSpec("A", ("sber",), "только данные Сбера"),
    AblationSpec("B", ("sber", "macro"), "+ макро и календарь"),
    AblationSpec("C", ("sber", "macro", "news"), "+ новости и календарь событий"),
    AblationSpec("D", ("sber", "macro", "foundation"), "+ foundation-модели"),
    AblationSpec("E", ("sber", "macro", "news", "foundation"), "всё вместе"),
    AblationSpec(
        "F",
        ("sber", "macro", "news", "foundation", "residual_cpd"),
        "всё + детектор по остаткам и модель вероятности шока",
    ),
)


@dataclass
class AblationResult:
    name: str
    metrics: dict[str, float] = field(default_factory=dict)


def on_common_rows(
    predictions: Mapping[str, pd.DataFrame], keys: Sequence[str] = ("unique_id", "ds", "fold")
) -> dict[str, pd.DataFrame]:
    """Прогнозы конфигураций на наблюдениях, которые есть у каждой из них.

    Ансамблю нужен прогноз каждой его модели, поэтому строк у него меньше, чем у одиночной
    модели: часть моделей пропускает короткие ряды. Разность MAE двух конфигураций имеет смысл
    только на одних и тех же наблюдениях.
    """
    keys = list(keys)
    shared: pd.DataFrame | None = None
    for frame in predictions.values():
        rows = frame[keys].drop_duplicates()
        shared = rows if shared is None else shared.merge(rows, on=keys)
    return {name: frame.merge(shared, on=keys) for name, frame in predictions.items()}


def source_contribution(with_block: pd.DataFrame, base: pd.DataFrame) -> dict[str, float]:
    """Вклад блока признаков: модель с блоком против той же модели без него.

    Сравнение идёт на общих наблюдениях. `delta` — разность общих MAE (и `delta_h…` по
    горизонтам), `mean_diff` — средняя по рядам разность ошибок, по ней считается тест
    Диболда–Мариано. Отрицательные значения — блок помогает. `folds…` и `sign_p` — счёт окон
    проверки, в которых с блоком точнее, и знаковый тест по окнам (`fold_consistency`).
    """
    shared = on_common_rows({"with": with_block, "base": base})
    left, right = shared["with"], shared["base"]
    test = panel_dm(left, right)

    def error(frame: pd.DataFrame) -> pd.Series:
        return (frame["y"] - frame["y_hat"]).abs()

    out = {
        "rows": int(len(left)),
        "mae_with": float(error(left).mean()),
        "mae_base": float(error(right).mean()),
        "mean_diff": float(test.mean_diff),
        "statistic": float(test.statistic),
        "p_value": float(test.p_value),
    }
    out["delta"] = out["mae_with"] - out["mae_base"]
    windows = fold_consistency(left, right)
    out.update(
        folds=windows.folds,
        folds_better=windows.better,
        folds_worse=windows.worse,
        sign_p=windows.sign_p,
    )
    by_horizon = (
        error(left).groupby(left["horizon"]).mean() - error(right).groupby(right["horizon"]).mean()
    )
    out.update({f"delta_h{int(h)}": float(v) for h, v in by_horizon.items()})
    return out


def feature_sets(spec: AblationSpec, blocks: Mapping[str, Sequence[str]]) -> list[str]:
    """Список колонок-признаков для конфигурации абляции."""
    columns: list[str] = []
    for block in spec.blocks:
        columns += list(blocks.get(block, []))
    return sorted(dict.fromkeys(columns))


def ablation_table(
    results: Sequence[AblationResult],
    baseline: str = "A",
    metrics: Sequence[str] = (
        "mae",
        "r2_per_series_median",
        "f1",
        "mean_lead_time",
        "false_alarms_per_series_year",
    ),
) -> pd.DataFrame:
    """Таблица абляций с разницей относительно базовой конфигурации."""
    frame = pd.DataFrame([{"name": r.name, **r.metrics} for r in results]).set_index("name")
    if baseline not in frame.index:
        raise KeyError(f"нет базовой конфигурации {baseline}")
    base = frame.loc[baseline]
    for metric in metrics:
        if metric in frame.columns:
            frame[f"delta_{metric}"] = frame[metric] - base[metric]
    return frame.reset_index()


def contribution(table: pd.DataFrame, better: str, worse: str, metric: str = "mae") -> float:
    """Вклад блока как разница метрики между двумя конфигурациями (например, C против B)."""
    indexed = table.set_index("name")
    if better not in indexed.index or worse not in indexed.index:
        return float("nan")
    return float(indexed.loc[better, metric] - indexed.loc[worse, metric])


def significance_note(p_value: float, alpha: float = 0.05) -> str:
    if not np.isfinite(p_value):
        return "нет оценки"
    return "значимо" if p_value < alpha else "незначимо"
