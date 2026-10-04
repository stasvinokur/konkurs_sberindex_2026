"""Онлайн-детекторы структурных сдвигов.

Все функции — чистые scan-процедуры: тревога в момент t зависит только от наблюдений до t
включительно. Это обязательное свойство для задачи «раннего выявления»: детектор не должен
видеть будущее.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import gammaln


def cusum_path(
    x: Sequence[float], threshold: float = 5.0, drift: float = 0.5, reset: bool = True
) -> tuple[np.ndarray, list[int]]:
    """Двусторонний CUSUM: статистика на каждом шаге и индексы тревог.

    Статистика — большая из двух накопленных сумм; тревога — её выход за порог. На пропуске
    статистика не определена (NaN), состояние не меняется.
    """
    values = np.asarray(x, dtype=float)
    statistic = np.full(len(values), np.nan)
    pos = neg = 0.0
    alarms = []
    for t, value in enumerate(values):
        if not np.isfinite(value):
            continue
        pos = max(0.0, pos + value - drift)
        neg = max(0.0, neg - value - drift)
        statistic[t] = max(pos, neg)
        if statistic[t] > threshold:
            alarms.append(t)
            if reset:
                pos = neg = 0.0
    return statistic, alarms


def cusum(
    x: Sequence[float], threshold: float = 5.0, drift: float = 0.5, reset: bool = True
) -> list[int]:
    """Двусторонний CUSUM по стандартизованному сигналу; возвращает индексы тревог."""
    return cusum_path(x, threshold, drift, reset)[1]


def page_hinkley_path(
    x: Sequence[float], threshold: float = 5.0, delta: float = 0.05, reset: bool = True
) -> tuple[np.ndarray, list[int]]:
    """Тест Пейджа–Хинкли: статистика на каждом шаге и индексы тревог.

    Статистика — наибольшее из отклонений накопленной суммы от её минимума и максимума;
    тревога — её выход за порог.
    """
    values = np.asarray(x, dtype=float)
    statistic = np.full(len(values), np.nan)
    mean = 0.0
    cumulative = 0.0
    minimum = 0.0
    maximum = 0.0
    alarms = []
    n = 0
    for t, value in enumerate(values):
        if not np.isfinite(value):
            continue
        n += 1
        mean += (value - mean) / n
        cumulative += value - mean - delta
        minimum = min(minimum, cumulative)
        maximum = max(maximum, cumulative)
        statistic[t] = max(cumulative - minimum, maximum - cumulative)
        if statistic[t] > threshold:
            alarms.append(t)
            if reset:
                mean, cumulative, minimum, maximum, n = 0.0, 0.0, 0.0, 0.0, 0
    return statistic, alarms


def page_hinkley(
    x: Sequence[float], threshold: float = 5.0, delta: float = 0.05, reset: bool = True
) -> list[int]:
    """Тест Пейджа–Хинкли на сдвиг среднего."""
    return page_hinkley_path(x, threshold, delta, reset)[1]


def bocpd_run_length_probs(
    x: Sequence[float],
    hazard: float = 1 / 12,
    mu0: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> np.ndarray:
    """Bayesian Online Changepoint Detection (Adams & MacKay, 2007), модель Normal-Gamma.

    Возвращает матрицу апостериорных вероятностей длины пробега: строки — моменты времени,
    столбцы — длина пробега r.
    """
    values = np.asarray(x, dtype=float)
    n = len(values)
    probs = np.zeros((n + 1, n + 1))
    probs[0, 0] = 1.0
    mu = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta = np.array([beta0])
    for t, value in enumerate(values):
        if not np.isfinite(value):
            probs[t + 1, : t + 2] = probs[t, : t + 2]
            continue
        scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
        df = 2.0 * alpha
        z = (value - mu) / scale
        # плотность t-распределения Стьюдента
        log_pred = (
            gammaln((df + 1) / 2)
            - gammaln(df / 2)
            - 0.5 * np.log(df * np.pi)
            - np.log(scale)
            - (df + 1) / 2 * np.log1p(z**2 / df)
        )
        pred = np.exp(log_pred)
        growth = probs[t, : t + 1] * pred * (1 - hazard)
        change = np.sum(probs[t, : t + 1] * pred * hazard)
        probs[t + 1, 1 : t + 2] = growth
        probs[t + 1, 0] = change
        total = probs[t + 1, : t + 2].sum()
        if total > 0:
            probs[t + 1, : t + 2] /= total
        mu_new = (kappa * mu + value) / (kappa + 1.0)
        kappa_new = kappa + 1.0
        alpha_new = alpha + 0.5
        beta_new = beta + kappa * (value - mu) ** 2 / (2.0 * (kappa + 1.0))
        mu = np.concatenate([[mu0], mu_new])
        kappa = np.concatenate([[kappa0], kappa_new])
        alpha = np.concatenate([[alpha0], alpha_new])
        beta = np.concatenate([[beta0], beta_new])
    return probs


def bocpd(
    x: Sequence[float],
    hazard: float = 1 / 12,
    threshold: float = 0.25,
    confirm: int = 2,
    **kwargs,
) -> list[int]:
    """Тревога по «обрыву пробега» в апостериорном распределении длины пробега.

    При постоянном hazard величина `P(r_t = 0)` тождественно равна hazard и сигналом быть
    не может. Информативен обрыв: MAP-оценка длины пробега падает, а масса коротких пробегов
    растёт. Чтобы не реагировать на разовые выбросы, обрыв подтверждается `confirm` шагами,
    на которых пробег последовательно растёт от нового начала; тревога датируется моментом
    подтверждения (это и есть честная задержка обнаружения).
    """
    probs = bocpd_run_length_probs(x, hazard=hazard, **kwargs)
    posterior = probs[1:]
    if len(posterior) == 0:
        return []
    short_mass = posterior[:, :3].sum(axis=1)
    map_run = posterior.argmax(axis=1)
    alarms = []
    for t in range(1, len(posterior) - confirm):
        if map_run[t] >= map_run[t - 1] or short_mass[t] < threshold:
            continue
        confirmed = all(map_run[t + k] == map_run[t] + k for k in range(1, confirm + 1))
        if confirmed:
            alarms.append(t + confirm)
    return alarms


def conformal_alarm(exceedance: Sequence[int], consecutive: int = 2) -> list[int]:
    """Тревога после `consecutive` подряд выходов факта за квантильный интервал."""
    streak = 0
    alarms = []
    for t, flag in enumerate(np.asarray(exceedance, dtype=float)):
        streak = streak + 1 if flag > 0 else 0
        if streak >= consecutive:
            alarms.append(t)
            streak = 0
    return alarms


DETECTORS = {
    "cusum": cusum,
    "page_hinkley": page_hinkley,
    "bocpd": bocpd,
}
