"""Национальные и макро-признаки из длинных рядов СберИндекса.

Все ряды приводятся к месячной сетке и снабжаются датой публикации: конец отчётного месяца
плюс лаг источника. Месячные показатели СберИндекса выходят примерно через пять недель после
конца месяца, недельные — через неделю. В прогноз значение попадает только после публикации
(см. `sbx.core.features.asof`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from sbx.core.features.asof import as_of_origin, published_at

MONTHLY_LAG_DAYS = 35

# Категории недельного ряда «Изменение потребительских расходов», близкие к категориям МО.
WEEKLY_CATEGORY_MAP = {
    "Все категории ": "total",
    "Продовольственные товары": "food",
    "Маркетплейсы": "marketplaces",
    "Общественное питание": "catering",
    "Лекарства и медицинские товары": "health",
    "Локальный транспорт": "transport",
}


def _slug(value: str) -> str:
    table = str.maketrans("абвгдеёжзийклмнопрстуфхцчшщъыьэюя", "abvgdeejzijklmnoprstufhccss_y_eua")
    text = str(value).lower().translate(table)
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")[:40]


def to_monthly(frame: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """Приводит ряд к месячной сетке (для недельных данных — агрегация внутри месяца)."""
    out = frame.assign(ds=pd.to_datetime(frame["period"]).dt.to_period("M").dt.to_timestamp())
    return out.groupby("ds", as_index=False).agg({"value": how})


def pivot_indicator(
    frame: pd.DataFrame, prefix: str, dims: Sequence[str] = (), how: str = "mean"
) -> pd.DataFrame:
    """Из long-ряда делает широкую месячную таблицу с префиксом в именах колонок."""
    df = frame.assign(ds=pd.to_datetime(frame["period"]).dt.to_period("M").dt.to_timestamp())
    if not dims:
        wide = (
            df.groupby("ds", as_index=False).agg({"value": how}).rename(columns={"value": prefix})
        )
        return wide
    df["_key"] = df[list(dims)].astype(str).agg("|".join, axis=1).map(_slug)
    wide = df.pivot_table(index="ds", columns="_key", values="value", aggfunc=how)
    wide.columns = [f"{prefix}_{c}" for c in wide.columns]
    return wide.reset_index()


def growth_features(levels: pd.DataFrame, prefix: str, dims: Sequence[str] = ()) -> pd.DataFrame:
    """Приросты к предыдущему месяцу и к тому же месяцу прошлого года."""
    wide = pivot_indicator(levels, prefix, dims).set_index("ds").sort_index()
    out = pd.DataFrame(index=wide.index)
    for col in wide.columns:
        out[f"{col}_mom"] = wide[col].pct_change()
        out[f"{col}_yoy"] = wide[col].pct_change(12)
    return out.reset_index()


def national_parts(frames: Mapping[str, pd.DataFrame]) -> list[tuple[str, pd.DataFrame]]:
    """Признаки каждого источника отдельно, вместе со slug источника.

    Slug нужен, чтобы применить к признакам именно тот лаг публикации, с которым выходит
    их источник: недельная инфляция известна через неделю, зарплаты — через два месяца.
    """
    parts: list[tuple[str, pd.DataFrame]] = []

    if "consumer-spending" in frames:
        parts.append(
            (
                "consumer-spending",
                growth_features(frames["consumer-spending"], "nat_spending", ("type",)),
            )
        )
    if "consumer-spending-growth" in frames:
        parts.append(
            (
                "consumer-spending-growth",
                pivot_indicator(
                    frames["consumer-spending-growth"], "nat_growth", ("type", "value_type")
                ),
            )
        )
    if "consumper-spending-index-sa" in frames:
        parts.append(
            (
                "consumper-spending-index-sa",
                pivot_indicator(frames["consumper-spending-index-sa"], "nat_index_sa", ("type",)),
            )
        )
    if "median-wages" in frames:
        parts.append(("median-wages", growth_features(frames["median-wages"], "wages", ())))
    if "oboroty-biznesa" in frames:
        parts.append(
            ("oboroty-biznesa", growth_features(frames["oboroty-biznesa"], "business", ()))
        )
    if "izmenenie-obema-fot" in frames:
        parts.append(
            ("izmenenie-obema-fot", growth_features(frames["izmenenie-obema-fot"], "payroll", ()))
        )
    if "real-key-interest-rate" in frames:
        parts.append(
            (
                "real-key-interest-rate",
                pivot_indicator(
                    frames["real-key-interest-rate"], "key_rate", ("key_rate_categories",)
                ),
            )
        )
    if "nedelnaa-inflazia-v-razreze-analiticeskih-komponentov" in frames:
        weekly = frames["nedelnaa-inflazia-v-razreze-analiticeskih-komponentov"]
        parts.append(
            (
                "nedelnaa-inflazia-v-razreze-analiticeskih-komponentov",
                pivot_indicator(weekly, "infl", ("indicator",)),
            )
        )
    if "ver-izmenenie-trat-po-kategoriyam" in frames:
        weekly = frames["ver-izmenenie-trat-po-kategoriyam"]
        subset = weekly[weekly["category"].isin(WEEKLY_CATEGORY_MAP)].copy()
        subset["category"] = subset["category"].map(WEEKLY_CATEGORY_MAP)
        monthly = pivot_indicator(subset, "weekly_spend_yoy", ("category",))
        accel = monthly.set_index("ds").diff().add_suffix("_accel").reset_index()
        slug = "ver-izmenenie-trat-po-kategoriyam"
        parts += [(slug, monthly), (slug, accel)]
    if "potrebitelskaya-aktivnost-po-kategoriyam-tovarov-v-razreze-vozrastov" in frames:
        slug = "potrebitelskaya-aktivnost-po-kategoriyam-tovarov-v-razreze-vozrastov"
        parts.append((slug, pivot_indicator(frames[slug], "activity_index", ("age",))))
    return parts


def _merge_on_ds(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    merged = frames[0]
    for part in frames[1:]:
        merged = merged.merge(part, on="ds", how="outer")
    merged = merged.sort_values("ds").reset_index(drop=True)
    return merged.replace([np.inf, -np.inf], np.nan)


def build_national_features(
    frames: Mapping[str, pd.DataFrame], lags: Mapping[str, int] | None = None
) -> dict[int, pd.DataFrame]:
    """Национальные признаки, сгруппированные по лагу публикации их источника.

    Возвращается словарь «лаг в днях → месячная таблица признаков». Разные лаги нельзя
    свести в одну таблицу: к каждой группе применяется свой сдвиг point-in-time.
    """
    lags = dict(lags or {})
    groups: dict[int, list[pd.DataFrame]] = {}
    for slug, part in national_parts(frames):
        lag = int(lags.get(slug, MONTHLY_LAG_DAYS))
        groups.setdefault(lag, []).append(part)
    out = {}
    for lag, parts in groups.items():
        features = _merge_on_ds(parts)
        features["published_at"] = published_at(features["ds"], lag)
        out[lag] = features
    return out


def national_as_of(
    groups: Mapping[int, pd.DataFrame], origins: Sequence[pd.Timestamp]
) -> pd.DataFrame:
    """Национальные признаки, известные к концу каждого месяца origin.

    Для каждой группы источников берётся последний месяц, опубликованный к этому моменту:
    недельные индикаторы свежее месячных на месяц, зарплаты отстают ещё сильнее. Колонки
    получают префикс группы (`nat7__`, `nat35__`, …), чтобы возраст значения был виден в имени.
    """
    out = pd.DataFrame({"origin": sorted(pd.Timestamp(o) for o in origins)})
    for lag, frame in sorted(groups.items()):
        values = frame.drop(columns=["published_at"])
        known = as_of_origin(values, out["origin"], lag)
        known = known.rename(columns={c: f"nat{lag}__{c}" for c in known.columns if c != "origin"})
        out = out.merge(known, on="origin", how="left")
    return out
