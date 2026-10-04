"""Метрики обнаружения точек структурных изменений.

Для ретроспективных методов — протокол TCPDBench (van den Burg & Williams, arXiv:2003.06222):
F1 с окном допуска и covering. Для «раннего выявления» дополнительно нужны задержка обнаружения
и частота ложных тревог.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DetectionScore:
    precision: float
    recall: float
    f1: float
    n_true: int
    n_pred: int
    matched: int


def positions_for_dates(index: Sequence[pd.Timestamp], dates: Sequence[pd.Timestamp]) -> list[int]:
    """Позиции событий в сетке ряда; события вне сетки отбрасываются.

    Нужна для разметки реальных шоков: в конфиге они заданы датами, а метрики работают с
    индексами. Для недельного ряда точной даты события в сетке нет, поэтому берётся первое
    наблюдение на дату события или после неё.
    """
    grid = pd.DatetimeIndex(index)
    out = []
    for date in pd.DatetimeIndex(dates):
        if date > grid[-1] or date < grid[0]:
            continue
        out.append(int(grid.searchsorted(date, side="left")))
    return sorted(set(out))


def match_changepoints(
    true_taus: Sequence[int], pred_taus: Sequence[int], margin: int = 1
) -> list[tuple[int, int]]:
    """Жадное сопоставление прогнозных точек истинным в пределах окна ±margin."""
    available = sorted(pred_taus)
    pairs = []
    for t in sorted(true_taus):
        best, best_dist = None, None
        for p in available:
            dist = abs(p - t)
            if dist <= margin and (best_dist is None or dist < best_dist):
                best, best_dist = p, dist
        if best is not None:
            pairs.append((t, best))
            available.remove(best)
    return pairs


def f1_margin(
    true_by_series: Mapping[str, Sequence[int]],
    pred_by_series: Mapping[str, Sequence[int]],
    margin: int = 1,
) -> DetectionScore:
    """F1 по всем рядам: точка засчитана, если найдена в пределах ±margin месяцев."""
    matched = n_true = n_pred = 0
    for uid in set(true_by_series) | set(pred_by_series):
        true_taus = [t for t in true_by_series.get(uid, []) if t >= 0]
        pred_taus = list(pred_by_series.get(uid, []))
        n_true += len(true_taus)
        n_pred += len(pred_taus)
        matched += len(match_changepoints(true_taus, pred_taus, margin))
    precision = matched / n_pred if n_pred else float("nan")
    recall = matched / n_true if n_true else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if matched else 0.0
    return DetectionScore(precision, recall, f1, n_true, n_pred, matched)


def segments(taus: Sequence[int], n: int) -> list[set[int]]:
    bounds = [0, *sorted(t for t in taus if 0 < t < n), n]
    return [set(range(bounds[i], bounds[i + 1])) for i in range(len(bounds) - 1)]


def covering(true_taus: Sequence[int], pred_taus: Sequence[int], n: int) -> float:
    """Covering metric TCPDBench: взвешенное по длине лучшее перекрытие сегментов."""
    true_segments = segments(true_taus, n)
    pred_segments = segments(pred_taus, n)
    total = 0.0
    for seg in true_segments:
        best = max((len(seg & other) / len(seg | other) for other in pred_segments), default=0.0)
        total += len(seg) * best
    return total / n


def mean_covering(
    true_by_series: Mapping[str, Sequence[int]],
    pred_by_series: Mapping[str, Sequence[int]],
    lengths: Mapping[str, int],
) -> float:
    values = [
        covering(true_by_series.get(uid, []), pred_by_series.get(uid, []), lengths[uid])
        for uid in lengths
    ]
    return float(np.mean(values)) if values else float("nan")


def detection_delays(
    true_by_series: Mapping[str, int],
    alarms_by_series: Mapping[str, Sequence[int]],
    horizon: int | None = None,
) -> pd.DataFrame:
    """Задержка первой тревоги после слома (в месяцах); NaN — слом не обнаружен.

    Отрицательная задержка означает предупреждение до слома (lead time).
    """
    rows = []
    for uid, tau in true_by_series.items():
        if tau is None or tau < 0:
            continue
        alarms = sorted(alarms_by_series.get(uid, []))
        after = [a for a in alarms if a >= tau and (horizon is None or a - tau <= horizon)]
        before = [a for a in alarms if a < tau]
        rows.append(
            {
                "unique_id": uid,
                "tau": tau,
                "delay": (after[0] - tau) if after else np.nan,
                "detected": bool(after),
                "first_alarm": alarms[0] if alarms else np.nan,
                "lead_time": (tau - before[-1]) if before else np.nan,
            }
        )
    return pd.DataFrame(rows)


def false_alarm_rate(
    alarms_by_series: Mapping[str, Sequence[int]],
    true_by_series: Mapping[str, int],
    lengths: Mapping[str, int],
    margin: int = 1,
    freq_per_year: int = 12,
) -> float:
    """Ложные тревоги на ряд-год: тревоги вне окна ±margin вокруг истинного слома."""
    false_alarms = 0
    series_years = 0.0
    for uid, length in lengths.items():
        tau = true_by_series.get(uid, -1)
        alarms = alarms_by_series.get(uid, [])
        for alarm in alarms:
            if tau is None or tau < 0 or abs(alarm - tau) > margin:
                false_alarms += 1
        series_years += length / freq_per_year
    return false_alarms / series_years if series_years else float("nan")


def score_detector(
    labels: pd.DataFrame,
    alarms_by_series: Mapping[str, Sequence[int]],
    margin: int = 1,
) -> dict[str, float]:
    """Сводка метрик детектора на бенчмарке с разметкой `labels`."""
    true_by_series = {r.unique_id: int(r.tau) for r in labels.itertuples()}
    lengths = {r.unique_id: int(r.n_obs) for r in labels.itertuples()}
    true_lists = {uid: ([tau] if tau >= 0 else []) for uid, tau in true_by_series.items()}
    score = f1_margin(true_lists, alarms_by_series, margin)
    delays = detection_delays(true_by_series, alarms_by_series)
    detected = delays[delays["detected"]]

    def alarmed(with_break: bool) -> float:
        """Доля рядов хотя бы с одной тревогой — среди рядов со сломом или без него."""
        flags = [
            bool(alarms_by_series.get(uid))
            for uid, tau in true_by_series.items()
            if (tau >= 0) == with_break
        ]
        return float(np.mean(flags)) if flags else float("nan")

    return {
        "precision": score.precision,
        "recall": score.recall,
        "f1": score.f1,
        "covering": mean_covering(true_lists, alarms_by_series, lengths),
        "mean_delay": float(detected["delay"].mean()) if len(detected) else float("nan"),
        "median_delay": float(detected["delay"].median()) if len(detected) else float("nan"),
        "false_alarms_per_series_year": false_alarm_rate(
            alarms_by_series, true_by_series, lengths, margin
        ),
        "n_true": score.n_true,
        "n_alarms": score.n_pred,
        # Метод, который тревожится на рядах без слома так же часто, как на рядах со сломом,
        # сломов не находит, какой бы ни была его полнота в окне допуска.
        "alarmed_break_share": alarmed(True),
        "alarmed_control_share": alarmed(False),
    }


def window_coverage(n: int, positions: Sequence[int], margin: int) -> float:
    """Доля точек ряда длины `n`, попадающих в окно ±margin хотя бы одной тревоги.

    С такой вероятностью «найденной» оказалась бы случайная дата: без этого числа счёт
    совпадений тревог с событиями ни о чём не говорит.
    """
    if n <= 0:
        return 0.0
    covered = np.zeros(n, dtype=bool)
    for position in positions:
        covered[max(0, int(position) - margin) : int(position) + margin + 1] = True
    return float(covered.mean())


def recall_gap_interval(
    labels: pd.DataFrame,
    alarms_a: Mapping[str, Sequence[int]],
    alarms_b: Mapping[str, Sequence[int]],
    margin: int = 1,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Разность полноты двух наборов тревог и её бутстреп-интервал по рядам.

    Пересэмплируются ряды со сломом, и в каждой выборке полнота обоих наборов считается на
    одних и тех же рядах: разность парная. Слом найден, если тревога есть в окне ±margin.
    """
    taus = {r.unique_id: int(r.tau) for r in labels.itertuples() if int(r.tau) >= 0}
    if not taus:
        nan = float("nan")
        return {"recall_a": nan, "recall_b": nan, "diff": nan, "low": nan, "high": nan, "n": 0}

    def hits(alarms: Mapping[str, Sequence[int]]) -> np.ndarray:
        return np.array(
            [
                any(abs(int(a) - tau) <= margin for a in alarms.get(uid, []))
                for uid, tau in taus.items()
            ],
            dtype=float,
        )

    gap = hits(alarms_a) - hits(alarms_b)
    rng = np.random.default_rng(seed)
    draws = gap[rng.integers(0, len(gap), size=(n_boot, len(gap)))].mean(axis=1)
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(draws, [tail, 100 - tail])
    return {
        "recall_a": float(hits(alarms_a).mean()),
        "recall_b": float(hits(alarms_b).mean()),
        "diff": float(gap.mean()),
        "low": float(low),
        "high": float(high),
        "n": len(gap),
    }


def nearest_rate(rates: Mapping[int, float], rate: float) -> int:
    """Период расписания, у которого частота ложных тревог ближе всего к заданной.

    При равном расстоянии берётся более редкое расписание (больший период).
    """
    if not rates:
        raise ValueError("нет расписаний для сравнения")
    return min(rates, key=lambda period: (abs(rates[period] - rate), -period))


def recall_by(
    labels: pd.DataFrame, alarms_by_series: Mapping[str, Sequence[int]], by: str, margin: int = 1
) -> pd.DataFrame:
    """Доля обнаруженных сломов в разрезе колонки `by` (тип слома, величина)."""
    rows = []
    for value, g in labels[labels["tau"] >= 0].groupby(by):
        true_lists = {r.unique_id: [int(r.tau)] for r in g.itertuples()}
        preds = {uid: alarms_by_series.get(uid, []) for uid in true_lists}
        score = f1_margin(true_lists, preds, margin)
        rows.append({by: value, "recall": score.recall, "n": score.n_true})
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True)
