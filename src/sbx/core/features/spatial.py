"""Признаки из таблиц организаторов: доступность рынков и соседние МО по автодорогам.

Все прежние внешние данные решения — страновые или региональные, то есть одинаковые для всех
МО региона. Эти признаки различаются между муниципалитетами.

- Постоянные: индекс доступности рынков и среднее расстояние до ближайших МО. Организаторы
  рассчитали их на конец панели (индекс — «в 2024 году», расстояния — «на 31 декабря 2024
  года»); это медленно меняющаяся география, и берётся она как постоянная характеристика.
- На момент прогноза: динамика и уровень той же категории у ближайших соседей. Строка
  origin считается только по значениям не позже origin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NEIGHBOURS = 5
FEATURES = ("nb_growth_1", "nb_growth_3", "nb_level_gap")


def nearest_neighbours(
    connection: pd.DataFrame, territories: pd.DataFrame, k: int = NEIGHBOURS, kind: str = "highway"
) -> pd.DataFrame:
    """Ближайшие по дороге МО панели для каждой территории: `territory_id`, `neighbour`,
    `rank`, `distance` (км).

    В таблице организаторов расстояние записано в одну сторону (x → y) в предположении, что
    обратный путь такой же, поэтому пары берутся в обоих направлениях. Соседом считается только
    территория, которая есть в панели: у остальных нет рядов расходов.
    """
    ids = territories.set_index("official_id")["territory_id"]
    roads = connection[connection["type"] == kind]
    roads = roads[roads["territory_id_x"].isin(ids.index) & roads["territory_id_y"].isin(ids.index)]
    forward = roads.rename(columns={"territory_id_x": "a", "territory_id_y": "b"})
    backward = roads.rename(columns={"territory_id_y": "a", "territory_id_x": "b"})
    pairs = pd.concat([forward[["a", "b", "distance"]], backward[["a", "b", "distance"]]])
    pairs = pairs[pairs["a"] != pairs["b"]]
    pairs = pairs.groupby(["a", "b"], as_index=False)["distance"].min()
    pairs = pairs.assign(territory_id=pairs["a"].map(ids), neighbour=pairs["b"].map(ids))
    pairs = pairs.sort_values(["territory_id", "distance", "neighbour"])
    pairs["rank"] = pairs.groupby("territory_id").cumcount() + 1
    near = pairs[pairs["rank"] <= k]
    return near[["territory_id", "neighbour", "rank", "distance"]].reset_index(drop=True)


def access_features(
    market_access: pd.DataFrame, neighbours: pd.DataFrame, territories: pd.DataFrame
) -> pd.DataFrame:
    """Постоянные характеристики территории: логарифм индекса доступности рынков и логарифм
    среднего расстояния до ближайших соседей. Нет значения у организаторов — пропуск."""
    index = market_access.set_index("territory_id")["market_access"]
    distance = neighbours.groupby("territory_id")["distance"].mean()
    out = pd.DataFrame({"territory_id": territories["territory_id"].to_numpy()})
    access = territories["official_id"].map(index).to_numpy(dtype=float)
    out["market_access_log"] = np.log(access)
    out["neighbour_distance_log"] = np.log1p(
        out["territory_id"].map(distance).to_numpy(dtype=float)
    )
    return out.sort_values("territory_id").reset_index(drop=True)


def _mean(values: np.ndarray) -> np.ndarray:
    """Среднее по соседям (ось 1) без учёта пропусков; нет ни одного значения — пропуск."""
    known = np.isfinite(values)
    count = known.sum(axis=1)
    total = np.where(known, values, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def neighbour_features(
    panel: pd.DataFrame, static: pd.DataFrame, neighbours: pd.DataFrame
) -> pd.DataFrame:
    """Динамика и уровень той же категории у ближайших МО — по состоянию на каждый месяц.

    Для ряда и месяца origin: средний по соседям прирост логарифма за 1 и 3 месяца и разница
    логарифма собственного уровня со средним уровнем соседей. Используются значения не позже
    origin, поэтому таблицу можно построить один раз по всей панели: строка origin от будущих
    месяцев не зависит.
    """
    meta = static.set_index("unique_id")[["territory_id", "category"]]
    months = pd.DatetimeIndex(sorted(panel["ds"].unique()))
    values = panel.pivot(index="unique_id", columns="ds", values="y").reindex(columns=months)
    logs = np.log(values.where(values > 0))  # нулевой месяц — пропуск, а не минус бесконечность
    near = {
        territory: list(group.sort_values("rank")["neighbour"])
        for territory, group in neighbours.groupby("territory_id")
    }
    width = max((len(found) for found in near.values()), default=0)
    frames = []
    for _category, part in meta.loc[logs.index].groupby("category", sort=True):
        ids = part.index.to_numpy()
        row_of = {territory: i for i, territory in enumerate(part["territory_id"])}
        level = logs.loc[ids].to_numpy(dtype=float)
        # Последняя строка — пустая: на неё указывают отсутствующие соседи.
        padded = np.vstack([level, np.full((1, level.shape[1]), np.nan)])
        index = np.full((len(ids), max(width, 1)), len(ids), dtype=int)
        for i, territory in enumerate(part["territory_id"]):
            found = [row_of[n] for n in near.get(territory, []) if n in row_of]
            index[i, : len(found)] = found
        columns = {"nb_level_gap": level - _mean(padded[index])}
        for step in (1, 3):
            change = np.full_like(padded, np.nan)
            change[:, step:] = padded[:, step:] - padded[:, :-step]
            columns[f"nb_growth_{step}"] = _mean(change[index])
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": np.repeat(ids, len(months)),
                    "origin": np.tile(months, len(ids)),
                    **{name: columns[name].ravel() for name in FEATURES},
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    # Строка есть только там, где ряд наблюдался: иначе у ряда, начавшегося позже, появились бы
    # строки за прежние месяцы, и набор строк origin зависел бы от будущего.
    observed = panel[["unique_id", "ds"]].rename(columns={"ds": "origin"})
    out = out.merge(observed, on=["unique_id", "origin"])
    return out.sort_values(["unique_id", "origin"]).reset_index(drop=True)
