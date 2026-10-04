"""Согласование внешних данных по времени: что известно к моменту прогноза."""

from __future__ import annotations

import pandas as pd
import pytest

from sbx.core.features.asof import (
    KNOWN_FUTURE,
    PAST_ONLY,
    STATIC,
    ExogenousBlock,
    as_of_origin,
    latest_published_month,
    published_at,
)

JUNE = pd.Timestamp("2024-06-01")


def test_publication_date_is_counted_from_the_end_of_the_period() -> None:
    """Лаг источника — дни от конца отчётного месяца, а не от его начала."""
    assert published_at(pd.Timestamp("2024-04-01"), 35) == pd.Timestamp("2024-06-04")
    assert published_at(pd.Timestamp("2024-04-01"), 0) == pd.Timestamp("2024-04-30")
    months = pd.Series(pd.to_datetime(["2024-01-01", "2024-02-01"]))
    assert published_at(months, 7).tolist() == [
        pd.Timestamp("2024-02-07"),
        pd.Timestamp("2024-03-07"),
    ]


def test_latest_published_month_is_the_last_one_out_by_the_end_of_the_origin_month() -> None:
    assert latest_published_month(JUNE, 0) == JUNE, "месяц закончился — его новости известны"
    assert latest_published_month(JUNE, 7) == pd.Timestamp("2024-05-01"), (
        "последняя неделя июня выйдет только в июле"
    )
    assert latest_published_month(JUNE, 35) == pd.Timestamp("2024-04-01"), (
        "апрель вышел 4 июня, май выйдет 5 июля"
    )
    assert latest_published_month(JUNE, 45) == pd.Timestamp("2024-04-01")
    assert latest_published_month(JUNE, 60) == pd.Timestamp("2024-04-01"), (
        "30 апреля + 60 = 29 июня"
    )
    # Граница считается по дням: 31 января + 60 дней — это 31 марта в високосном году
    # и 1 апреля в обычном.
    assert latest_published_month(pd.Timestamp("2024-03-01"), 60) == pd.Timestamp("2024-01-01")
    assert latest_published_month(pd.Timestamp("2023-03-01"), 60) == pd.Timestamp("2022-12-01")


def test_as_of_origin_gives_each_origin_the_latest_published_period() -> None:
    frame = pd.DataFrame(
        {"ds": pd.date_range("2024-01-01", periods=6, freq="MS"), "rate": [1.0, 2, 3, 4, 5, 6]}
    )
    out = as_of_origin(frame, [JUNE, pd.Timestamp("2024-03-01")], lag_days=35)
    assert list(out.columns) == ["origin", "rate"]
    assert out.to_dict("records") == [
        {"origin": pd.Timestamp("2024-03-01"), "rate": 1.0},
        {"origin": JUNE, "rate": 4.0},
    ]
    early = as_of_origin(frame, [pd.Timestamp("2024-02-01")], lag_days=35)
    assert len(early) == 1 and early["rate"].isna().all(), "к концу февраля ещё ничего не вышло"


def test_as_of_origin_keeps_one_row_per_region() -> None:
    frame = pd.DataFrame(
        {
            "region_code": ["45", "46", "45"],
            "ds": pd.to_datetime(["2024-06-01", "2024-06-01", "2024-07-01"]),
            "news_events": [10, 20, 99],
        }
    )
    out = as_of_origin(frame, [JUNE], lag_days=0)
    assert out.sort_values("region_code").to_dict("records") == [
        {"origin": JUNE, "region_code": "45", "news_events": 10},
        {"origin": JUNE, "region_code": "46", "news_events": 20},
    ]


def test_a_block_declares_how_it_is_aligned_in_time() -> None:
    months = pd.date_range("2024-01-01", periods=3, freq="MS")
    calendar = ExogenousBlock(
        "calendar", pd.DataFrame({"ds": months, "working_days": [17, 20, 20]}), KNOWN_FUTURE
    )
    news = ExogenousBlock(
        "news",
        pd.DataFrame({"origin": months, "region_code": "45", "news_events": [1, 2, 3]}),
        PAST_ONLY,
    )
    assert calendar.time_key == "ds" and calendar.keys == ["ds"]
    assert news.time_key == "origin" and news.keys == ["origin", "region_code"]
    assert news.columns == ["news_events"]


def test_a_block_with_the_wrong_time_key_is_refused() -> None:
    """Прошлое, присоединённое по целевому месяцу, — это заглядывание в будущее. Тип блока и
    его ключ обязаны соответствовать друг другу."""
    months = pd.date_range("2024-01-01", periods=3, freq="MS")
    by_month = pd.DataFrame({"ds": months, "news_events": [1, 2, 3]})
    by_origin = pd.DataFrame({"origin": months, "working_days": [17, 20, 20]})
    with pytest.raises(ValueError, match="origin"):
        ExogenousBlock("news", by_month, PAST_ONLY)
    with pytest.raises(ValueError, match="ds"):
        ExogenousBlock("calendar", by_origin, KNOWN_FUTURE)
    with pytest.raises(ValueError, match="вид"):
        ExogenousBlock("news", by_origin, "whenever")
    with pytest.raises(ValueError, match="и ds, и origin"):
        ExogenousBlock("news", by_origin.assign(ds=months), PAST_ONLY)
    with pytest.raises(ValueError, match="повтор"):
        ExogenousBlock("calendar", pd.concat([by_month, by_month]), KNOWN_FUTURE)


def test_a_static_block_is_keyed_by_the_series_alone() -> None:
    """Постоянная характеристика МО (доступность рынков) от времени не зависит: ключа времени у
    блока нет, зато обязателен ключ ряда — иначе его не к чему присоединить."""
    frame = pd.DataFrame({"territory_id": ["t0", "t1"], "market_access_log": [1.0, 2.0]})
    access = ExogenousBlock("access", frame, STATIC)
    assert access.time_key is None and access.keys == ["territory_id"]
    assert access.columns == ["market_access_log"]
    months = pd.date_range("2024-01-01", periods=2, freq="MS")
    with pytest.raises(ValueError, match="не зависит от времени"):
        ExogenousBlock("access", frame.assign(origin=months), STATIC)
    with pytest.raises(ValueError, match="ключ ряда"):
        ExogenousBlock("access", pd.DataFrame({"market_access_log": [1.0]}), STATIC)
    with pytest.raises(ValueError, match="повтор"):
        ExogenousBlock("access", pd.concat([frame, frame]), STATIC)
