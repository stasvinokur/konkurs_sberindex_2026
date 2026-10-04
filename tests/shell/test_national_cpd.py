"""Проверка методов CPD на реальных национальных рядах и разметке известных шоков."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sbx.shell.pipelines import national_cpd


def test_known_shocks_config_is_complete_and_sourced() -> None:
    """Каждое событие проверяемо: есть дата, описание, ряды и ссылка на официальный источник."""
    shocks = national_cpd.load_known_shocks()
    assert shocks["events"], "календарь шоков пуст"
    for event in shocks["events"]:
        assert set(event) >= {"date", "name", "expect", "series", "description", "source"}
        assert pd.Timestamp(event["date"])
        assert event["source"].startswith("http")
        assert set(event["series"]) <= {"monthly", "weekly"}
        assert event["expect"] in {"drop", "spike", "regime"}


def test_national_series_cover_known_shock_dates() -> None:
    """Ряды должны покрывать период размеченных событий, иначе проверка беспредметна."""
    data = national_cpd.load_national_series()
    monthly = data["monthly"]
    assert monthly["ds"].min() <= pd.Timestamp("2020-03-01")
    assert monthly["series"].nunique() >= 5
    weekly = data["weekly"]
    assert weekly["ds"].min() <= pd.Timestamp("2024-10-01")


def test_evaluate_scores_methods_on_injected_shock(tmp_path, monkeypatch) -> None:
    """На ряду с явным провалом в дату события метод обязан его найти."""
    ds = pd.date_range("2019-01-01", periods=48, freq="MS")
    values = np.full(48, 1000.0)
    shock = list(ds).index(pd.Timestamp("2020-04-01"))
    values[shock:] = 600.0  # провал на 40% начиная с апреля 2020
    frame = pd.DataFrame({"series": "Всего", "ds": ds, "value": values})
    monkeypatch.setattr(
        national_cpd, "load_national_series", lambda: {"monthly": frame, "weekly": frame.iloc[:0]}
    )
    # Штраф — в дисперсиях шума ряда (середина сетки `configs/cpd.yaml`), а не в рублях.
    monkeypatch.setattr(national_cpd, "_penalties_from_benchmark", lambda: {"pelt_l2": 30.0})

    shocks = {
        "margin_months": 2,
        "events": [{"date": "2020-04-01", "name": "тест", "series": ["monthly"]}],
    }
    comparison, per_event = national_cpd.evaluate(shocks, methods=["pelt_l2"], out_dir=tmp_path)
    assert comparison.iloc[0]["recall"] == 1.0
    assert bool(per_event.iloc[0]["detected"])
    assert (tmp_path / "national_comparison.csv").exists()


def test_penalties_come_from_the_methods_of_the_benchmark_not_from_its_reference(tmp_path) -> None:
    """Тревога по расписанию — эталон сравнения, а не метод: на национальных рядах её нет."""
    path = tmp_path / "detector_comparison.csv"
    pd.DataFrame(
        [
            {"detector": "periodic", "kind": "baseline", "input": "none", "param": 5.0, "f1": 0.9},
            {"detector": "pelt_l2", "kind": "offline", "input": "raw", "param": 30.0, "f1": 0.2},
            {"detector": "cusum", "kind": "online", "input": "residual", "param": 3.0, "f1": 0.5},
        ]
    ).to_csv(path, index=False)
    assert national_cpd._penalties_from_benchmark(path) == {"pelt_l2": 30.0}


def test_evaluate_ignores_events_outside_series_range(tmp_path, monkeypatch) -> None:
    ds = pd.date_range("2023-01-01", periods=24, freq="MS")
    frame = pd.DataFrame({"series": "Всего", "ds": ds, "value": np.full(24, 1000.0)})
    monkeypatch.setattr(
        national_cpd, "load_national_series", lambda: {"monthly": frame, "weekly": frame.iloc[:0]}
    )
    monkeypatch.setattr(national_cpd, "_penalties_from_benchmark", lambda: {"pelt_l2": 1000.0})
    shocks = {"events": [{"date": "2020-04-01", "name": "до начала ряда", "series": ["monthly"]}]}
    comparison, _ = national_cpd.evaluate(shocks, methods=["pelt_l2"], out_dir=tmp_path)
    assert comparison.empty


def test_fips_reference_covers_every_adm1_code_seen_in_gdelt() -> None:
    """Все встречающиеся коды ADM1 сопоставлены; остаток — только события уровня страны."""
    from pathlib import Path

    unmapped_path = Path("artifacts/news/unmapped_adm1.csv")
    if not unmapped_path.exists():
        import pytest

        pytest.skip("нет выгрузки GDELT")
    unmapped = pd.read_csv(unmapped_path, dtype={"adm1": str})
    # RS — событие привязано только к стране, RS00 — пустой код: региона нет в принципе.
    structural = {"RS", "RS00"}
    leftovers = sorted(set(unmapped["adm1"]) - structural)
    assert not leftovers, f"коды ADM1 без сопоставления: {leftovers}"


def test_fips_reference_has_no_duplicate_codes() -> None:
    reference = pd.read_csv("data/reference/fips_adm1_ru.csv", dtype=str)
    assert not reference["fips"].duplicated().any()
    assert reference["region_code"].notna().all()
