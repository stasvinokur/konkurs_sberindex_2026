"""Согласование внешних данных по времени: что известно к моменту прогноза.

Правило одно для всех источников. Момент прогноза — конец последнего наблюдаемого месяца
(origin). Значение попадает в прогноз, если оно опубликовано не позже этого момента; дата
публикации — конец отчётного периода плюс лаг источника в днях. Исключение — то, что известно
заранее (производственный календарь): оно берётся на целевой месяц.

Правило консервативно: сами расходы МО выходят примерно через пять недель после конца месяца,
так что на практике к моменту прогноза известно больше.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

KNOWN_FUTURE = "known_future"
PAST_ONLY = "past_only"
STATIC = "static"
TIME_KEYS = {KNOWN_FUTURE: "ds", PAST_ONLY: "origin", STATIC: None}
# Ключи, по которым блок привязан к ряду, помимо времени.
ENTITY_KEYS = ("region_code", "territory_id", "unique_id")


def published_at(period, lag_days: int):
    """Дата публикации значения за месяц `period`: последний день месяца плюс лаг в днях."""
    return period + pd.offsets.MonthEnd(0) + pd.Timedelta(days=int(lag_days))


def latest_published_month(origin: pd.Timestamp, lag_days: int) -> pd.Timestamp:
    """Последний месяц, значение за который опубликовано к концу месяца `origin`."""
    origin = pd.Timestamp(origin)
    moment = origin + pd.offsets.MonthEnd(0)
    month = origin
    while published_at(month, lag_days) > moment:
        month = month - pd.DateOffset(months=1)
    return month


def as_of_origin(
    frame: pd.DataFrame, origins: Sequence[pd.Timestamp], lag_days: int, period_col: str = "ds"
) -> pd.DataFrame:
    """Таблица «период → значения» → таблица «origin → значения, известные к его концу».

    Каждому origin соответствует последний период, опубликованный к концу его месяца. Если к
    этому моменту ничего не опубликовано, строка origin остаётся с пропусками.
    """
    known = pd.DataFrame(
        {
            "origin": [pd.Timestamp(o) for o in origins],
            period_col: [latest_published_month(o, lag_days) for o in origins],
        }
    ).sort_values("origin")
    out = known.merge(frame, on=period_col, how="left").drop(columns=[period_col])
    return out.reset_index(drop=True)


@dataclass(frozen=True)
class ExogenousBlock:
    """Блок внешних признаков и способ его привязки ко времени.

    - `known_future` — известно заранее; ключ времени `ds`, значение берётся на целевой месяц;
    - `past_only` — известно только после публикации; ключ времени `origin`, строка содержит
      то, что опубликовано к концу этого месяца;
    - `static` — постоянная характеристика ряда или территории; ключа времени нет, значение
      одно на все строки ряда.

    Тип и ключ обязаны соответствовать друг другу: прошлое, присоединённое по целевому месяцу,
    — это заглядывание в будущее, и блок с таким ключом не создаётся.
    """

    name: str
    frame: pd.DataFrame
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in TIME_KEYS:
            raise ValueError(f"блок «{self.name}»: неизвестный вид {self.kind!r}")
        columns = set(self.frame.columns)
        if {"ds", "origin"} <= columns:
            raise ValueError(f"блок «{self.name}»: в таблице и ds, и origin — ключ неоднозначен")
        if self.kind == STATIC:
            if columns & {"ds", "origin"}:
                raise ValueError(
                    f"блок «{self.name}» вида {self.kind} не зависит от времени: ключ времени лишний"
                )
            if not self.keys:
                raise ValueError(f"блок «{self.name}» вида {self.kind}: нужен ключ ряда")
        elif self.time_key not in columns:
            raise ValueError(
                f"блок «{self.name}» вида {self.kind} должен иметь ключ времени {self.time_key}"
            )
        if self.frame.duplicated(self.keys).any():
            raise ValueError(f"блок «{self.name}»: повтор ключа {self.keys}")

    @property
    def time_key(self) -> str | None:
        return TIME_KEYS[self.kind]

    @property
    def keys(self) -> list[str]:
        entity = [k for k in ENTITY_KEYS if k in self.frame.columns]
        return [self.time_key, *entity] if self.time_key else entity

    @property
    def columns(self) -> list[str]:
        """Колонки признаков блока (без ключей)."""
        return [c for c in self.frame.columns if c not in set(self.keys)]
