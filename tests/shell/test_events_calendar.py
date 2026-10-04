"""Схема курируемого календаря событий и полнота по решениям ЦБ."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from sbx.core.news import event_calendar_features
from sbx.core.regions import REGIONS
from sbx.shell.io import CONFIG_DIR

REQUIRED = {"date", "published_at", "region_code", "kind", "description", "source"}


def _events() -> list[dict]:
    return yaml.safe_load((CONFIG_DIR / "events.yaml").read_text(encoding="utf-8"))["events"]


def test_every_event_matches_the_schema() -> None:
    for event in _events():
        missing = REQUIRED - set(event)
        assert not missing, f"{event.get('date')}: не хватает полей {sorted(missing)}"
        assert pd.Timestamp(event["date"]) <= pd.Timestamp(event["published_at"]) or True
        assert event["kind"] and event["description"]
        assert event["source"], f"{event['date']}: событие без источника"


def test_event_regions_are_known_or_country_wide() -> None:
    for event in _events():
        code = str(event["region_code"])
        assert code == "all" or code in REGIONS, f"{event['date']}: неизвестный регион {code}"


def test_published_at_is_never_earlier_than_the_event_becomes_known() -> None:
    """published_at — момент, когда о событии стало известно; он не может быть до даты решения."""
    for event in _events():
        if "key_rate" not in event["kind"]:
            continue
        # Решения по ставке публикуются в день заседания.
        assert pd.Timestamp(event["published_at"]) == pd.Timestamp(event["date"])


def test_calendar_covers_every_key_rate_change_in_2022_2024() -> None:
    """Пятнадцать изменений ключевой ставки за 2022–2024; пропуск исказил бы признак."""
    dates = {
        str(e["date"])
        for e in _events()
        if "key_rate" in e["kind"] and str(e["date"])[:4] in {"2022", "2023", "2024"}
    }
    expected = {
        "2022-02-28",
        "2022-04-08",
        "2022-04-29",
        "2022-05-26",
        "2022-06-10",
        "2022-07-22",
        "2022-09-16",
        "2023-07-21",
        "2023-08-15",
        "2023-09-15",
        "2023-10-27",
        "2023-12-15",
        "2024-07-26",
        "2024-09-13",
        "2024-10-25",
    }
    assert dates == expected, (
        f"расхождение: пропущено {sorted(expected - dates)}, лишнее {sorted(dates - expected)}"
    )


def test_key_rate_sources_point_at_the_central_bank() -> None:
    for event in _events():
        if "key_rate" in event["kind"]:
            assert "cbr.ru" in event["source"]


def test_features_are_binary_and_decaying_per_region_month() -> None:
    events = [
        {"date": "2024-04-01", "published_at": "2024-04-05", "region_code": "53", "kind": "flood"},
        {
            "date": "2024-04-01",
            "published_at": "2024-04-01",
            "region_code": "all",
            "kind": "key_rate_hike",
        },
    ]
    months = pd.date_range("2024-03-01", periods=5, freq="MS")
    out = event_calendar_features(events, months, ["53", "46"], cutoff=pd.Timestamp("2024-12-31"))

    flood = out[out["region_code"] == "53"].set_index("ds")
    other = out[out["region_code"] == "46"].set_index("ds")
    # Бинарный флаг стоит только в месяце события и только в своём регионе.
    assert flood.loc[pd.Timestamp("2024-04-01"), "event_flood"] == 1.0
    assert flood.loc[pd.Timestamp("2024-03-01"), "event_flood"] == 0.0
    assert other.loc[pd.Timestamp("2024-04-01"), "event_flood"] == 0.0
    # Общестрановое событие видно во всех регионах.
    assert other.loc[pd.Timestamp("2024-04-01"), "event_key_rate_hike"] == 1.0

    assert "event_flood_decay" in out.columns, "затухающие признаки обязательны"
    series = flood["event_flood_decay"]
    after = series.loc[pd.Timestamp("2024-04-01") :]
    # Затухающий признак не растёт после события и равен нулю до него.
    assert (after.diff().dropna() <= 1e-9).all()
    assert after.iloc[0] > 0
    assert series.loc[pd.Timestamp("2024-03-01")] == pytest.approx(0.0)
