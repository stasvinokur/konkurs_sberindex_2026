"""Протокол rolling-origin бэктеста: конфигурация, фолды и разбиение панели."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TargetSpec:
    dataset: str
    freq: str
    level: str


@dataclass(frozen=True)
class HorizonFolds:
    """Набор origin-ов для одного горизонта прогноза.

    Горизонт задаётся явно, а не выводится из origin: для direct-моделей (глобальный LightGBM)
    горизонт меняет саму обучающую выборку, поэтому фолд «origin 2023-12, h=3» и фолд
    «origin 2023-12, h=12» — это два разных эксперимента, а не срез одного.
    """

    horizon: int
    origins: tuple[str, ...]


@dataclass(frozen=True)
class BacktestConfig:
    status: str
    checked_at: str
    target: TargetSpec
    folds: tuple[HorizonFolds, ...]
    primary_metric: str
    mae_aggregation: str
    reference_baselines: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    min_train_obs: int = 6
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    metric_slices: tuple[str, ...] = ("category", "size_group", "step", "fold", "horizon")

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(sorted({spec.horizon for spec in self.folds}))


@dataclass(frozen=True)
class Fold:
    name: str
    cutoff: pd.Timestamp
    horizon: int
    kind: str  # main (понятие demo-фолда убрано: h=6 и h=12 полноценные)

    @property
    def test_end(self) -> pd.Timestamp:
        return self.cutoff + pd.DateOffset(months=self.horizon)


def make_folds(cfg: BacktestConfig) -> tuple[Fold, ...]:
    """Фолды по одному на пару (горизонт, origin).

    Имя фолда несёт и origin, и горизонт: один и тот же origin участвует в нескольких
    горизонтах, и без горизонта в имени фолды схлопнулись бы.
    """
    folds = [
        Fold(f"h{spec.horizon}_{pd.Timestamp(o):%Y-%m}", pd.Timestamp(o), spec.horizon, "main")
        for spec in cfg.folds
        for o in spec.origins
    ]
    if len({f.name for f in folds}) != len(folds):
        raise ValueError("fold names must be unique")
    return tuple(sorted(folds, key=lambda f: (f.horizon, f.cutoff)))


def split(
    panel: pd.DataFrame, fold: Fold, min_train_obs: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Разбиение панели по фолду.

    train — все наблюдения с ds ≤ cutoff; test — наблюдения в (cutoff, cutoff + h] только для
    рядов, у которых в train не меньше `min_train_obs` точек (иначе прогноз не определён).
    """
    train = panel[panel["ds"] <= fold.cutoff]
    counts = train.groupby("unique_id")["ds"].size()
    eligible = counts.index[counts >= min_train_obs]
    train = train[train["unique_id"].isin(eligible)]
    test = panel[(panel["ds"] > fold.cutoff) & (panel["ds"] <= fold.test_end)]
    test = test[test["unique_id"].isin(eligible)]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def step_of(ds: pd.Series, cutoff: pd.Timestamp) -> pd.Series:
    """Номер шага прогноза: 1 для первого месяца после cutoff."""
    return (ds.dt.year - cutoff.year) * 12 + ds.dt.month - cutoff.month


def check_coverage(panel: pd.DataFrame, folds: Sequence[Fold]) -> list[str]:
    """Фолды, у которых тестовое окно выходит за пределы данных.

    `split` просто обрезает окно по доступным месяцам и ничего не сообщает. Поэтому фолд с
    именем `h12_2024-09` молча дал бы три месяца вместо двенадцати и попал бы в отчёт как
    «горизонт 12». Проверка возвращает список расхождений, чтобы прогон падал явно.
    """
    if panel.empty:
        return []
    last = pd.Timestamp(panel["ds"].max())
    return [
        f"{f.name}: тестовое окно до {f.test_end:%Y-%m}, данные кончаются {last:%Y-%m}"
        for f in folds
        if f.test_end > last
    ]
