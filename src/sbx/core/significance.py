"""Статистическая значимость различий прогнозов: Diebold–Mariano, Friedman–Nemenyi."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class DMResult:
    statistic: float
    p_value: float
    mean_diff: float
    n: int


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int = 1) -> DMResult:
    """DM-тест для последовательности потерь (H0: равная точность), HAC и поправка HLN.

    Отрицательная статистика — модель A точнее. p-value двусторонний, распределение Стьюдента
    с n−1 степенями свободы (Harvey, Leybourne, Newbold, 1997).
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return DMResult(float("nan"), float("nan"), float(np.mean(d)) if n else float("nan"), n)
    mean = d.mean()
    dc = d - mean
    gamma0 = np.dot(dc, dc) / n
    var = gamma0
    for lag in range(1, horizon):
        var += 2 * np.dot(dc[lag:], dc[:-lag]) / n
    if var <= 0:
        return DMResult(float("nan"), float("nan"), float(mean), n)
    dm = mean / np.sqrt(var / n)
    hln = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat = float(dm * hln)
    p = float(2 * stats.t.sf(abs(stat), df=n - 1))
    return DMResult(stat, p, float(mean), n)


def panel_dm(oof_a: pd.DataFrame, oof_b: pd.DataFrame, loss: str = "abs") -> DMResult:
    """DM-тип теста на панели: разности потерь усредняются по ряду, ряды — независимые наблюдения.

    На 24 месяцах временной ряд разностей слишком короток для HAC-оценки; усреднение по ряду
    устраняет автокорреляцию внутри ряда, t-тест идёт по поперечному срезу рядов.
    """
    keys = ["unique_id", "ds", "fold"]
    merged = oof_a[keys + ["y", "y_hat"]].merge(
        oof_b[keys + ["y_hat"]], on=keys, suffixes=("_a", "_b")
    )
    if loss == "abs":
        la, lb = (merged["y"] - merged["y_hat_a"]).abs(), (merged["y"] - merged["y_hat_b"]).abs()
    elif loss == "sq":
        la, lb = (merged["y"] - merged["y_hat_a"]) ** 2, (merged["y"] - merged["y_hat_b"]) ** 2
    else:
        raise ValueError(loss)
    d = (la - lb).groupby(merged["unique_id"]).mean()
    return diebold_mariano(d.to_numpy(), np.zeros(len(d)), horizon=1)


@dataclass(frozen=True)
class FoldConsistency:
    """Согласованность сравнения двух моделей по окнам проверки."""

    folds: int
    better: int  # окна, в которых левая модель точнее
    worse: int
    sign_p: float
    mean_of_folds: float
    by_fold: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "folds": self.folds,
            "folds_better": self.better,
            "folds_worse": self.worse,
            "sign_p": self.sign_p,
            "mean_of_folds": self.mean_of_folds,
            "by_fold": dict(self.by_fold),
        }


def fold_consistency(oof_a: pd.DataFrame, oof_b: pd.DataFrame) -> FoldConsistency:
    """В скольких окнах проверки модель A точнее модели B и держится ли знак разности.

    Тест по рядам (`panel_dm`) считает ряды независимыми. Но все прогнозы окна строит одна
    обученная модель, а шоки месяца у рядов общие, поэтому разности ошибок внутри окна
    сдвинуты в одну сторону, и p-value по рядам завышает уверенность: оно отвечает на вопрос
    «различаются ли модели в этих окнах», а не «повторится ли это в другом окне». Единица
    повторения — окно. Здесь по каждому окну считается средняя разность абсолютных ошибок на
    общих наблюдениях и точный двусторонний знаковый тест: насколько вероятно такое число
    окон одного знака, если знак случаен. При десяти окнах он проходит уровень 0,05 только
    при девяти и десяти согласных окнах; при трёх-четырёх окнах пройти его нельзя вовсе.
    """
    keys = ["unique_id", "ds", "fold"]
    merged = oof_a[[*keys, "y", "y_hat"]].merge(
        oof_b[[*keys, "y_hat"]], on=keys, suffixes=("_a", "_b")
    )
    diff = (merged["y"] - merged["y_hat_a"]).abs() - (merged["y"] - merged["y_hat_b"]).abs()
    by_fold = diff.groupby(merged["fold"]).mean()
    better, worse = int((by_fold < 0).sum()), int((by_fold > 0).sum())
    decided = better + worse
    sign_p = float(stats.binomtest(better, decided, 0.5).pvalue) if decided else 1.0
    return FoldConsistency(
        folds=int(len(by_fold)),
        better=better,
        worse=worse,
        sign_p=sign_p,
        mean_of_folds=float(by_fold.mean()) if len(by_fold) else float("nan"),
        by_fold={str(fold): float(value) for fold, value in by_fold.items()},
    )


@dataclass(frozen=True)
class RankTestResult:
    friedman_statistic: float
    friedman_p_value: float
    mean_ranks: dict[str, float]
    critical_difference: float
    n_blocks: int


def nemenyi_critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    """CD Немени: q_α · sqrt(k(k+1)/(6N)), q_α = studentized range / √2."""
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


def friedman_nemenyi(errors: pd.DataFrame, alpha: float = 0.05) -> RankTestResult:
    """Ранговое сравнение моделей: строки — блоки (ряды), столбцы — модели, значения — ошибка."""
    clean = errors.dropna()
    k, n = clean.shape[1], clean.shape[0]
    ranks = clean.rank(axis=1, method="average")
    mean_ranks = ranks.mean().sort_values()
    if k < 3:
        stat, p = float("nan"), float("nan")
    else:
        stat, p = stats.friedmanchisquare(*[clean[c].to_numpy() for c in clean.columns])
    return RankTestResult(
        float(stat), float(p), mean_ranks.to_dict(), nemenyi_critical_difference(k, n, alpha), n
    )
