"""Новостные признаки из событий GDELT и курируемого календаря событий.

Правило point-in-time: событие учитывается, только если оно попало в базу (`date_added`) не
позже даты отсечения прогноза. Полные тексты статей не хранятся — только агрегаты и ссылки.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Корневые коды CAMEO, за которыми осмысленно следить для потребления.
CAMEO_ROOT_GROUPS = {
    "protest": ("14",),
    "violence": ("18", "19", "20"),
    "coerce": ("17",),
    "cooperate": ("05", "06", "07"),
    "statements": ("01", "02", "03", "04"),
}
NEWS_LAG_DAYS = 0  # GDELT публикует событие в тот же день, дата публикации — date_added


@dataclass(frozen=True)
class EventRecord:
    date: str
    published_at: str
    region_code: str | None
    kind: str
    description: str
    source: str


def map_regions(events: pd.DataFrame, fips_reference: pd.DataFrame) -> pd.DataFrame:
    """Добавляет код региона по FIPS ADM1; строки без сопоставления сохраняются с None."""
    mapping = fips_reference.set_index("fips")["region_code"].astype(str).to_dict()
    out = events.copy()
    out["region_code"] = out["adm1"].map(mapping)
    return out


def unmapped_share(events: pd.DataFrame) -> pd.DataFrame:
    """Доля событий по несопоставленным кодам ADM1 (для отчёта о покрытии)."""
    missing = events[events["region_code"].isna()]
    if missing.empty:
        return pd.DataFrame(columns=["adm1", "events", "share"])
    counts = missing["adm1"].value_counts()
    return pd.DataFrame(
        {
            "adm1": counts.index,
            "events": counts.to_numpy(),
            "share": counts.to_numpy() / len(events),
        }
    )


def monthly_news_features(
    events: pd.DataFrame, cutoff: pd.Timestamp | None = None, level: str = "region"
) -> pd.DataFrame:
    """Агрегирует события в признаки «регион × месяц» (или «страна × месяц»).

    `cutoff` отбрасывает события, добавленные в базу позже даты отсечения.
    """
    df = events.copy()
    df["date_added"] = pd.to_datetime(df["date_added"])
    if cutoff is not None:
        df = df[df["date_added"] <= cutoff]
    if df.empty:
        return pd.DataFrame()
    df["ds"] = df["date_added"].dt.to_period("M").dt.to_timestamp()
    keys = ["ds"] if level == "country" else ["region_code", "ds"]
    if level != "country":
        df = df[df["region_code"].notna()]

    df["root"] = df["event_root_code"].astype(str).str.zfill(2)
    grouped = df.groupby(keys)
    out = grouped.agg(
        news_events=("event_root_code", "size"),
        news_articles=("num_articles", "sum"),
        news_tone_mean=("avg_tone", "mean"),
        news_tone_min=("avg_tone", "min"),
        news_goldstein_mean=("goldstein", "mean"),
        news_material_conflict_share=("quad_class", lambda s: float((s == 4).mean())),
    )
    for name, roots in CAMEO_ROOT_GROUPS.items():
        share = grouped["root"].apply(lambda s, roots=roots: float(s.isin(roots).mean()))
        out[f"news_share_{name}"] = share
    out = out.reset_index()
    prefix = "country_" if level == "country" else ""
    if prefix:
        out = out.rename(columns={c: prefix + c for c in out.columns if c.startswith("news_")})
    return out.sort_values(keys).reset_index(drop=True)


def add_novelty(
    features: pd.DataFrame, window: int = 6, group_col: str = "region_code"
) -> pd.DataFrame:
    """Новизна: z-оценка относительно скользящего среднего предыдущих месяцев."""
    out = (
        features.sort_values([group_col, "ds"]).copy()
        if group_col in features
        else features.sort_values("ds").copy()
    )
    value_cols = [c for c in out.columns if c.startswith(("news_", "country_news_"))]
    grouper = out.groupby(group_col) if group_col in out else None
    for col in value_cols:
        series = out[col]
        if grouper is not None:
            mean = grouper[col].transform(
                lambda s: s.shift(1).rolling(window, min_periods=2).mean()
            )
            std = grouper[col].transform(lambda s: s.shift(1).rolling(window, min_periods=2).std())
        else:
            mean = series.shift(1).rolling(window, min_periods=2).mean()
            std = series.shift(1).rolling(window, min_periods=2).std()
        # У постоянной истории разброс нулевой: масштабом берём 10% среднего, иначе всплеск
        # после «тишины» дал бы NaN вместо большого z.
        denom = std.where(std > 0, (mean.abs() * 0.1).clip(lower=1e-9))
        out[f"{col}_z"] = (series - mean) / denom
    return out.replace([np.inf, -np.inf], np.nan)


def event_calendar_features(
    events: Sequence[Mapping[str, str]],
    months: Sequence[pd.Timestamp],
    regions: Sequence[str],
    decay: float = 0.5,
    cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Бинарные и затухающие признаки из курируемого календаря событий.

    Для каждого типа события: флаг «событие в этом месяце» и затухающий след
    `decay**(месяцев с события)`. Учитываются только события, опубликованные до `cutoff`.
    """
    months = pd.DatetimeIndex(months)
    kinds = sorted({str(e["kind"]) for e in events})
    index = pd.MultiIndex.from_product([sorted(regions), months], names=["region_code", "ds"])
    out = pd.DataFrame(0.0, index=index, columns=[f"event_{k}" for k in kinds])
    for kind in kinds:
        out[f"event_{kind}_decay"] = 0.0

    for event in events:
        published = pd.Timestamp(event.get("published_at") or event["date"])
        if cutoff is not None and published > cutoff:
            continue
        start = pd.Timestamp(event["date"]).to_period("M").to_timestamp()
        kind = str(event["kind"])
        scope = str(event.get("region_code") or "")
        targets = sorted(regions) if scope in {"", "all"} else [scope]
        for region in targets:
            if region not in out.index.get_level_values(0):
                continue
            for ds in months:
                if ds == start:
                    out.loc[(region, ds), f"event_{kind}"] = 1.0
                if ds >= start:
                    gap = (ds.year - start.year) * 12 + ds.month - start.month
                    value = decay**gap
                    col = f"event_{kind}_decay"
                    out.loc[(region, ds), col] = max(out.loc[(region, ds), col], value)
    return out.reset_index()


def event_features_as_of(
    events: Sequence[Mapping[str, str]],
    origins: Sequence[pd.Timestamp],
    regions: Sequence[str],
    decay: float = 0.5,
) -> pd.DataFrame:
    """Признаки календаря событий, известные к концу каждого месяца `origin`.

    Строка origin учитывает только события, опубликованные не позже конца этого месяца:
    событие, о котором стало известно позже, в неё не попадает, даже если случилось раньше.
    """
    frames = []
    for origin in pd.DatetimeIndex(origins):
        cutoff = origin + pd.offsets.MonthEnd(0)
        frames.append(event_calendar_features(events, [origin], regions, decay, cutoff=cutoff))
    out = pd.concat(frames, ignore_index=True).rename(columns={"ds": "origin"})
    return out.sort_values(["region_code", "origin"]).reset_index(drop=True)
