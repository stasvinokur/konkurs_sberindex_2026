"""Реестр прогнозных моделей оболочки.

Каждая функция получает обучающую часть фолда и возвращает long-прогноз
`unique_id, ds, model, y_hat` (опционально квантили `q_*`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from sbx.core.folds import Fold

ForecastFn = Callable[[pd.DataFrame, Fold, Mapping[str, Any], Mapping[str, Any]], pd.DataFrame]
REGISTRY: dict[str, ForecastFn] = {}


@dataclass(frozen=True)
class ModelGroup:
    name: str
    models: tuple[str, ...]


def register(name: str) -> Callable[[ForecastFn], ForecastFn]:
    def decorator(fn: ForecastFn) -> ForecastFn:
        REGISTRY[name] = fn
        return fn

    return decorator


def get(name: str) -> ForecastFn:
    if name not in REGISTRY:
        raise KeyError(f"неизвестная группа моделей: {name}; есть {sorted(REGISTRY)}")
    return REGISTRY[name]
