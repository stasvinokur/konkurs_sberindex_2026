"""Порядок рядов в совместном режиме Chronos-2."""

from __future__ import annotations

import pandas as pd

from sbx.shell.models import chronos2_


def test_grouped_mode_feeds_series_region_by_region() -> None:
    """В совместном режиме модель делит информацию между рядами одного пакета, а пакет — это
    соседние ряды входа. Соседями должны быть МО одного региона при любом виде идентификатора."""
    months = pd.date_range("2024-01-01", periods=2, freq="MS")
    ids = ["z-volga__total", "a-moscow__total", "m-volga__food"]
    history = pd.DataFrame(
        [{"unique_id": uid, "ds": month, "y": 1.0} for uid in ids for month in reversed(months)]
    )
    static = pd.DataFrame({"unique_id": ids, "region_code": ["36", "45", "36"]})

    ordered = chronos2_.region_ordered(history, static)

    assert list(pd.unique(ordered["unique_id"])) == [
        "m-volga__food",
        "z-volga__total",
        "a-moscow__total",
    ]
    assert ordered.groupby("unique_id", sort=False)["ds"].apply(list).tolist() == [list(months)] * 3
    assert list(ordered.columns) == ["unique_id", "ds", "y"]
