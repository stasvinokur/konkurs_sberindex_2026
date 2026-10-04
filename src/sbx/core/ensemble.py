"""Ансамбль прогнозов: неотрицательные веса с суммой 1, подобранные по OOF-прогнозам."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

KEYS = ["unique_id", "ds", "fold"]


def pivot_predictions(
    oof: pd.DataFrame, models: Sequence[str] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Широкая таблица прогнозов (строки — наблюдения, столбцы — модели) и вектор факта."""
    wide = oof.pivot_table(index=KEYS, columns="model", values="y_hat")
    truth = oof.pivot_table(index=KEYS, columns="model", values="y").mean(axis=1)
    if models is not None:
        missing = set(models) - set(wide.columns)
        if missing:
            raise KeyError(f"в OOF нет моделей: {sorted(missing)}")
        wide = wide[list(models)]
    wide = wide.dropna()
    return wide, truth.reindex(wide.index)


def fit_weights(preds: pd.DataFrame, y: pd.Series, loss: str = "abs") -> pd.Series:
    """Веса, минимизирующие ошибку комбинации; ограничения: w ≥ 0, Σw = 1."""
    matrix = preds.to_numpy(dtype=float)
    target = y.to_numpy(dtype=float)
    k = matrix.shape[1]
    if k == 1:
        return pd.Series([1.0], index=preds.columns)

    def objective(w: np.ndarray) -> float:
        residual = target - matrix @ w
        return float(np.mean(np.abs(residual)) if loss == "abs" else np.mean(residual**2))

    start = np.full(k, 1.0 / k)
    constraints = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    bounds = [(0.0, 1.0)] * k
    result = minimize(objective, start, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = np.clip(result.x, 0.0, None)
    total = weights.sum()
    weights = weights / total if total > 0 else start
    return pd.Series(weights, index=preds.columns)


def combine(preds: pd.DataFrame, weights: pd.Series) -> pd.Series:
    aligned = weights.reindex(preds.columns).fillna(0.0)
    return pd.Series(preds.to_numpy(dtype=float) @ aligned.to_numpy(), index=preds.index)


def is_prediction_column(name: str) -> bool:
    """Колонка зависит от модели, а не от наблюдения (прогноз и его квантили)."""
    return name in {"y_hat", "model"} or name.startswith("q_")


def attach_meta(frame: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """Переносит на прогноз ансамбля колонки наблюдения из исходного OOF.

    Ансамбль обязан удовлетворять тому же контракту, что и одиночные модели (`cutoff`, `step`),
    иначе метрики по срезам на нём не считаются. Переносятся только колонки, одинаковые для всех
    моделей в этом наблюдении: `cutoff`, `step`, срезы (категория, размер, регион), `mase_scale`.
    Квантили переносить нельзя: они принадлежат конкретной модели, и ансамбль с чужими
    интервалами получил бы чужое покрытие. Свои интервалы ансамбль считает в `combine_quantiles`.
    """
    carried = [c for c in oof.columns if c not in {*KEYS, "y"} and not is_prediction_column(c)]
    if not carried:
        return frame
    meta = oof[[*KEYS, *carried]].drop_duplicates(subset=KEYS)
    merged = frame.merge(meta, on=KEYS, how="left", suffixes=("", "__dup"))
    return merged.drop(columns=[c for c in merged.columns if c.endswith("__dup")])


def combine_quantiles(
    oof: pd.DataFrame, models: Sequence[str], weights: pd.Series, index: pd.MultiIndex
) -> pd.DataFrame:
    """Интервалы ансамбля: взвешенное среднее квантилей моделей теми же весами.

    Усреднение квантилей (vincentization) сохраняет монотонность уровней, поэтому интервал
    ансамбля остаётся корректным. Квантили есть не у всех моделей (их дают Chronos-2 и
    TimesFM), поэтому веса пересчитываются на подмножество тех, у кого они есть. Если весь вес
    достался моделям без квантилей, берётся равновзвешенный интервал моделей с квантилями —
    так же ансамбль поступает, когда истории для подбора весов нет; без этого у части прогнозов
    не было бы интервала вовсе.
    """
    columns = sorted({c for c in oof.columns if c.startswith("q_")})
    out = pd.DataFrame(index=index)
    for column in columns:
        wide = oof.pivot_table(index=KEYS, columns="model", values=column).reindex(index)
        providers = [m for m in models if m in wide.columns and wide[m].notna().any()]
        if not providers:
            continue
        sub = weights.reindex(providers).fillna(0.0)
        if float(sub.sum()) <= 0:
            sub = pd.Series(1.0, index=providers)
        out[column] = wide[providers].to_numpy(dtype=float) @ (sub.to_numpy() / float(sub.sum()))
    return out


def recenter_quantiles(quantiles: pd.DataFrame, point: pd.Series) -> pd.DataFrame:
    """Сдвигает интервалы так, чтобы медиана совпала с точечным прогнозом ансамбля.

    Квантили дают лишь часть моделей, а точечный прогноз считается по всем, поэтому без сдвига
    интервал был бы центрирован не на прогнозе решения: медиана расходилась бы с `y_hat`, и
    pinball на уровне 0,5 не сходился бы с MAE. Ширина интервала при сдвиге сохраняется.
    """
    if "q_0.5" not in quantiles.columns or quantiles.empty:
        return quantiles
    offset = point.to_numpy(dtype=float) - quantiles["q_0.5"].to_numpy(dtype=float)
    return quantiles.apply(lambda column: column.to_numpy(dtype=float) + offset)


def fit_weights_by_group(
    oof: pd.DataFrame, models: Sequence[str], group_col: str, loss: str = "abs"
) -> dict[str, pd.Series]:
    """Отдельные веса для каждой группы (например, категории расходов)."""
    out = {}
    groups = oof[[*KEYS, group_col]].drop_duplicates().set_index(KEYS)[group_col]
    preds, y = pivot_predictions(oof, models)
    labels = groups.reindex(preds.index)
    for value, idx in labels.groupby(labels).groups.items():
        out[str(value)] = fit_weights(preds.loc[idx], y.loc[idx], loss)
    return out


def rolling_origin_ensemble(
    oof: pd.DataFrame,
    models: Sequence[str],
    fold_order: Sequence[str],
    name: str = "Ensemble",
    loss: str = "abs",
    history_log: dict[str, list[pd.Timestamp]] | None = None,
    history_pool: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Честная оценка ансамбля: веса фолда подобраны только на данных, известных к его cutoff.

    Порядка фолдов для этого недостаточно. Горизонты перекрываются: фолд h=12 от 2023-12
    покрывает весь 2024 год, то есть надмножество тестовых окон всех более поздних фолдов.
    Если брать «все предыдущие фолды», веса позднего фолда подбирались бы на тех самых месяцах,
    на которых он потом оценивается. Поэтому история отбирается по дате: только наблюдения с
    `ds <= cutoff` текущего фолда.

    Фолд, для которого истории не нашлось, считается с равными весами.
    """
    preds, y = pivot_predictions(oof, models)
    frames, weights_by_fold = [], {}
    folds = preds.index.get_level_values("fold")

    # История для подбора весов берётся из общего пула, а не только из фолдов своего горизонта.
    # На коротком горизонте «предыдущих фолдов» почти нет: к cutoff второго фолда h=1 известен
    # ровно один месяц, и пятнадцать весов, подогнанные под него, вырождаются (в прогоне
    # SeasonalNaive получал 62% за удачно угаданный январь). Длинные горизонты к тому же
    # моменту уже дали несколько месяцев OOF — это законная, доступная к cutoff информация.
    pool = history_pool if history_pool is not None else oof
    pool_preds, pool_y = pivot_predictions(pool, None)
    pool_folds = pool_preds.index.get_level_values("fold")
    pool_months = pool_preds.index.get_level_values("ds")
    if "cutoff" in oof.columns:
        cutoffs = oof.drop_duplicates("fold").set_index("fold")["cutoff"]
    else:
        # Cutoff по определению — месяц перед первым прогнозом фолда (шаг 1).
        cutoffs = oof.groupby("fold")["ds"].min() - pd.DateOffset(months=1)
    equal = pd.Series(np.full(len(models), 1.0 / len(models)), index=list(models))

    for fold in fold_order:
        mask = folds == fold
        if not mask.any():
            continue
        cutoff = pd.Timestamp(cutoffs.get(fold, pd.Timestamp.min))
        history = (pool_folds != fold) & (pool_months <= cutoff)
        if history_log is not None:
            history_log[fold] = sorted(set(pool_months[history]))

        # Веса подбираются только по тем моделям, у которых история к этому cutoff есть.
        # Модель без истории получает ноль: это факт доступности данных, а не отбор по качеству.
        past = pool_preds[history]
        available = [m for m in models if m in past.columns and past[m].notna().all()]
        if not history.any() or len(available) < 2:
            weights = equal.copy()
        else:
            weights = fit_weights(past[available], pool_y[history], loss)
            weights = weights.reindex(models).fillna(0.0)
        weights_by_fold[fold] = weights
        combined = combine(preds[mask], weights)
        index = preds.index[mask]
        quantiles = recenter_quantiles(combine_quantiles(oof, models, weights, index), combined)
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": index.get_level_values("unique_id"),
                    "ds": index.get_level_values("ds"),
                    "fold": fold,
                    "model": name,
                    "y": y[mask].to_numpy(),
                    "y_hat": combined.to_numpy(),
                    **{c: quantiles[c].to_numpy() for c in quantiles.columns},
                }
            )
        )
    return attach_meta(pd.concat(frames, ignore_index=True), oof), weights_by_fold
