"""Календарные признаки месяца: рабочие дни, праздники, распродажи.

Это заранее известные признаки: их значение на будущий месяц известно в момент прогноза,
поэтому они передаются моделям как known-future ковариаты.
"""

from __future__ import annotations

import calendar as pycalendar
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

# Крупные распродажи, влияющие на месячные расходы (маркетплейсы).
SALES_MONTHS = {11: "black_friday_11_11", 3: "spring_sales"}


def parse_calendar(raw: Mapping[str, Mapping]) -> pd.DataFrame:
    """Разбор ответа xmlcalendar.ru: нерабочие и сокращённые дни по месяцам."""
    rows = []
    for year_str, payload in raw.items():
        year = int(year_str)
        for month_entry in payload["months"]:
            month = int(month_entry["month"])
            holidays, short_days = [], []
            for token in str(month_entry["days"]).split(","):
                token = token.strip()
                if not token:
                    continue
                if token.endswith("*"):
                    short_days.append(int(token[:-1]))
                elif token.endswith("+"):
                    holidays.append(int(token[:-1]))
                else:
                    holidays.append(int(token))
            days_in_month = pycalendar.monthrange(year, month)[1]
            rows.append(
                {
                    "ds": pd.Timestamp(year=year, month=month, day=1),
                    "days_in_month": days_in_month,
                    "holidays": len(holidays),
                    "short_days": len(short_days),
                    "working_days": days_in_month - len(holidays),
                }
            )
    return pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)


def calendar_features(months: Sequence[pd.Timestamp], calendar: pd.DataFrame) -> pd.DataFrame:
    """Признаки календаря для заданных месяцев (известны заранее)."""
    index = calendar.set_index("ds")
    rows = []
    for ds in pd.DatetimeIndex(months):
        entry = index.loc[ds] if ds in index.index else None
        days = pycalendar.monthrange(ds.year, ds.month)[1]
        working = float(entry["working_days"]) if entry is not None else np.nan
        rows.append(
            {
                "ds": ds,
                "month": ds.month,
                "month_sin": np.sin(2 * np.pi * ds.month / 12),
                "month_cos": np.cos(2 * np.pi * ds.month / 12),
                "days_in_month": days,
                "working_days": working,
                "holidays": float(entry["holidays"]) if entry is not None else np.nan,
                "short_days": float(entry["short_days"]) if entry is not None else np.nan,
                "working_day_share": working / days if entry is not None else np.nan,
                "is_sales_month": int(ds.month in SALES_MONTHS),
                "is_december": int(ds.month == 12),
                "is_january": int(ds.month == 1),
            }
        )
    return pd.DataFrame(rows)
