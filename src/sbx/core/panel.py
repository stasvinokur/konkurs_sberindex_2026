"""Каноническая панель расходов «территория × категория × месяц» и статические признаки."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sbx.core.entity import normalize_name, series_blocks, squash_spaces
from sbx.core.regions import region_by_code

TOTAL_CATEGORY = "Все категории"
CATEGORY_SLUGS = {
    "Все категории": "total",
    "Продовольствие": "food",
    "Здоровье": "health",
    "Маркетплейсы": "marketplaces",
    "Общественное питание": "catering",
    "Транспорт": "transport",
}
PANEL_COLUMNS = ["unique_id", "territory_id", "category", "ds", "y"]


class PanelContractError(ValueError):
    """Нарушение контракта панели."""


def build_panel(raw: pd.DataFrame, block_territory: pd.DataFrame) -> pd.DataFrame:
    """Сырые строки + соответствие блоков территориям → long-панель с флагами рядов."""
    df = raw.assign(block_id=series_blocks(raw).to_numpy(), mo=raw["mo"].map(squash_spaces))
    df = df.merge(block_territory[["block_id", "territory_id", "admin_break_at"]], on="block_id")
    unknown = set(df["category_15"]) - set(CATEGORY_SLUGS)
    if unknown:
        raise PanelContractError(f"неизвестные категории: {sorted(unknown)}")
    panel = pd.DataFrame(
        {
            "unique_id": df["territory_id"] + "__" + df["category_15"].map(CATEGORY_SLUGS),
            "territory_id": df["territory_id"],
            "category": df["category_15"],
            "ds": pd.to_datetime(df["period"]),
            "y": df["value"].astype(float),
            "admin_break_at": pd.to_datetime(df["admin_break_at"]),
        }
    )
    return _with_series_flags(panel)


def build_official_panel(consumption: pd.DataFrame, territories: pd.DataFrame) -> pd.DataFrame:
    """Набор расходов организаторов + таблица территорий → та же long-панель с флагами рядов.

    В наборе территория задана официальным кодом (`territory_id` организаторов), в панели —
    нашим `territory_id` из таблицы территорий. Разрывов рядов нет: официальный код означает
    МО в постоянных границах.
    """
    unknown = set(consumption["category"]) - set(CATEGORY_SLUGS)
    if unknown:
        raise PanelContractError(f"неизвестные категории: {sorted(unknown)}")
    ids = territories.set_index("official_id")["territory_id"]
    absent = sorted(set(consumption["territory_id"]) - set(ids.index))
    if absent:
        raise PanelContractError(f"кодов нет в таблице территорий: {absent[:10]}")
    territory = consumption["territory_id"].map(ids)
    panel = pd.DataFrame(
        {
            "unique_id": territory + "__" + consumption["category"].map(CATEGORY_SLUGS),
            "territory_id": territory,
            "category": consumption["category"],
            "ds": pd.to_datetime(consumption["date"] + "-01"),
            "y": consumption["value"].astype(float),
            "admin_break_at": pd.Series(pd.NaT, index=consumption.index, dtype="datetime64[ns]"),
        }
    )
    return _with_series_flags(panel)


def _with_series_flags(panel: pd.DataFrame) -> pd.DataFrame:
    """Сортировка, флаги полноты рядов и проверка контракта — общий хвост обоих путей сборки."""
    panel = panel.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    stats = panel.groupby("unique_id")["ds"].agg(n_obs="size", first_ds="min", last_ds="max")
    all_months = pd.date_range(panel["ds"].min(), panel["ds"].max(), freq="MS")
    stats["is_complete"] = stats["n_obs"] == len(all_months)
    span = (
        (stats["last_ds"].dt.year - stats["first_ds"].dt.year) * 12
        + stats["last_ds"].dt.month
        - stats["first_ds"].dt.month
        + 1
    )
    stats["has_gaps"] = stats["n_obs"] < span
    panel = panel.merge(stats, left_on="unique_id", right_index=True)
    # Разрыв задан на уровне территории; для continuation в одной территории два исходных названия.
    breaks = panel.groupby("unique_id")["admin_break_at"].transform("max")
    panel["admin_break_at"] = breaks
    validate_panel(panel)
    return panel


def validate_panel(panel: pd.DataFrame) -> None:
    """Контракт: колонки и типы, уникальный ключ, y конечный и неотрицательный."""
    missing = set(PANEL_COLUMNS) - set(panel.columns)
    if missing:
        raise PanelContractError(f"нет колонок {sorted(missing)}")
    if not pd.api.types.is_datetime64_any_dtype(panel["ds"]):
        raise PanelContractError("ds должен быть datetime")
    if not pd.api.types.is_float_dtype(panel["y"]):
        raise PanelContractError("y должен быть float")
    if panel[PANEL_COLUMNS].isna().any().any():
        raise PanelContractError("пропуски в ключевых колонках")
    if panel.duplicated(["unique_id", "ds"]).any():
        raise PanelContractError("дубли ключа (unique_id, ds)")
    if not np.isfinite(panel["y"]).all() or (panel["y"] < 0).any():
        raise PanelContractError("y должен быть конечным и неотрицательным")
    if (panel["ds"].dt.day != 1).any():
        raise PanelContractError("ds должен быть началом месяца")


def static_features(
    panel: pd.DataFrame,
    territories: pd.DataFrame,
    n_size_groups: int = 5,
    level_until: pd.Timestamp | None = None,
    fallback_to_whole_series: bool = False,
) -> pd.DataFrame:
    """Статические признаки ряда: регион, ФО, тип МО, категория, размер.

    Размер территории — логарифм медианы «Все категории» по месяцам не позже `level_until`
    (по умолчанию — конец первого календарного года панели) и квантильная группа этой медианы.
    Признак один на все окна проверки, поэтому `level_until` — самый ранний момент прогноза.
    Территория без наблюдений по эти месяцы размера не получает: оценка по её более поздним
    месяцам была бы значением из будущего.

    `fallback_to_whole_series` возвращает прежнее правило — размер такой территории по всему
    её ряду. Оно нужно только чтобы повторить отбор замороженных выборок, сделанный по нему;
    в признаки модели такое значение идти не должно.
    """
    terr = territories.drop_duplicates("territory_id", keep="last").set_index("territory_id")
    series = panel.drop_duplicates("unique_id")[
        [
            "unique_id",
            "territory_id",
            "category",
            "n_obs",
            "first_ds",
            "last_ds",
            "is_complete",
            "admin_break_at",
        ]
    ].copy()
    series["region_code"] = series["territory_id"].map(terr["region_code"])
    series["region_name"] = series["region_code"].map(lambda c: region_by_code(c).name)
    series["federal_district"] = series["region_code"].map(
        lambda c: region_by_code(c).federal_district
    )
    series["oktmo"] = series["territory_id"].map(terr["oktmo"])
    if "mo_kind" in terr.columns:
        series["mo_kind"] = series["territory_id"].map(terr["mo_kind"])
    else:
        # Выгрузка без кодов: тип МО известен только из названия.
        series["mo_kind"] = (
            series["territory_id"].map(terr["raw_mo"]).map(lambda n: normalize_name(n)[1])
        )
    series["category_slug"] = series["category"].map(CATEGORY_SLUGS)

    total = panel[panel["category"] == TOTAL_CATEGORY]
    if level_until is None:
        known = total[total["ds"].dt.year == total["ds"].dt.year.min()]
    else:
        known = total[total["ds"] <= pd.Timestamp(level_until)]
    level = known.groupby("territory_id")["y"].median()
    if fallback_to_whole_series:
        whole = total.groupby("territory_id")["y"].median()
        level = level.reindex(whole.index).fillna(whole)
    series["size_level"] = np.log(series["territory_id"].map(level).astype(float))
    ranks = level.rank(method="first")
    groups = pd.qcut(ranks, n_size_groups, labels=False) + 1
    series["size_group"] = series["territory_id"].map(groups).astype("Int64")
    # Название и официальный код — для подписей и стыковки с таблицами организаторов;
    # в признаки модели они не входят (см. STATIC_COLUMNS в features/lags.py).
    for column in ("mo_name", "official_id"):
        if column in terr.columns:
            series[column] = series["territory_id"].map(terr[column])
    return series.reset_index(drop=True)


def additivity(panel: pd.DataFrame) -> dict[str, float]:
    """Сравнение суммы пяти категорий с «Все категории» по (территория, месяц)."""
    wide = panel.pivot_table(index=["territory_id", "ds"], columns="category", values="y")
    parts = [c for c in CATEGORY_SLUGS if c != TOTAL_CATEGORY and c in wide.columns]
    complete = wide.dropna(subset=parts + [TOTAL_CATEGORY])
    ratio = complete[parts].sum(axis=1) / complete[TOTAL_CATEGORY]
    return {
        "n": int(len(ratio)),
        "ratio_median": float(ratio.median()),
        "ratio_p05": float(ratio.quantile(0.05)),
        "ratio_p95": float(ratio.quantile(0.95)),
        "share_sum_exceeds_total": float((ratio > 1.0).mean()),
        "share_each_part_le_total": float(
            (complete[parts].le(complete[TOTAL_CATEGORY], axis=0)).all(axis=1).mean()
        ),
    }
