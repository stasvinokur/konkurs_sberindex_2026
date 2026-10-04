"""Отпечатки входов расчёта: по ним кэш узнаёт, что входы сменились."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sbx.core.features.asof import KNOWN_FUTURE, PAST_ONLY, ExogenousBlock
from sbx.core.fingerprint import changed, digest, fingerprint, frame_digest


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["a", "a", "b"],
            "ds": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01"]),
            "y": [1.0, 2.0, 3.0],
        }
    )


def test_frame_digest_follows_the_content() -> None:
    frame = _frame()
    same = frame_digest(frame)
    assert same == frame_digest(frame.copy())
    assert same == frame_digest(frame.set_index(pd.Index([7, 8, 9]))), (
        "индекс в отпечаток не входит"
    )
    assert same != frame_digest(frame.assign(y=[1.0, 2.0, 3.5]))
    assert same != frame_digest(frame.iloc[::-1]), "порядок строк значим"
    assert same != frame_digest(frame.rename(columns={"y": "value"}))
    assert same != frame_digest(frame.iloc[:2])
    assert same != frame_digest(frame[["ds", "unique_id", "y"]]), "порядок колонок значим"


def test_digest_of_nested_inputs_ignores_key_order_but_not_content() -> None:
    frame = _frame()
    base = digest({"blocks": {"macro": frame}, "n": 3, "names": ["x", "y"]})
    assert base == digest({"names": ["x", "y"], "n": 3, "blocks": {"macro": frame.copy()}})
    assert base != digest({"blocks": {"macro": frame.assign(y=0.0)}, "n": 3, "names": ["x", "y"]})
    assert base != digest({"blocks": {"macro": frame}, "n": 4, "names": ["x", "y"]})
    assert base != digest({"blocks": {"macro": frame}, "n": 3, "names": ["y", "x"]})
    assert digest(None) == digest(None) and digest(None) != digest({})


def test_fingerprint_names_the_parts_that_changed() -> None:
    old = fingerprint({"panel": _frame(), "config": {"a": 1}})
    new = fingerprint({"panel": _frame().assign(y=0.0), "config": {"a": 1}, "extra": 1})
    assert set(old) == {"panel", "config"} and all(isinstance(v, str) for v in old.values())
    assert changed(old, old) == []
    assert changed(old, new) == ["extra", "panel"]
    assert changed(None, new) == ["config", "extra", "panel"], "отметки нет — изменилось всё"
    assert changed(new, old) == ["extra", "panel"], "исчезнувший вход — тоже изменение"


def test_digest_looks_inside_a_feature_block() -> None:
    """Блок признаков — объект с таблицей внутри: отпечаток обязан следовать её содержимому,
    а не строковому представлению объекта (оно обрезает длинные таблицы)."""
    months = pd.date_range("2023-01-01", periods=200, freq="MS")
    frame = pd.DataFrame({"origin": months, "key_rate": range(200)})
    block = ExogenousBlock("macro", frame, PAST_ONLY)
    assert digest(block) == digest(ExogenousBlock("macro", frame.copy(), PAST_ONLY))
    middle = frame.copy()
    middle.loc[100, "key_rate"] = -1
    assert digest(block) != digest(ExogenousBlock("macro", middle, PAST_ONLY))
    assert digest(block) != digest(ExogenousBlock("other", frame, PAST_ONLY))
    by_month = frame.rename(columns={"origin": "ds"})
    assert digest(ExogenousBlock("macro", by_month, KNOWN_FUTURE)) != digest(block)


def test_digest_follows_arrays_and_refuses_what_it_cannot_read() -> None:
    """Запасной путь через строковое представление обрезал бы длинный массив: изменение в его
    середине осталось бы незамеченным. Массив читается по содержимому, неизвестный тип — ошибка."""
    values = np.arange(5000, dtype=float)
    changed_middle = values.copy()
    changed_middle[2500] = -1.0
    assert digest(values) == digest(values.copy())
    assert digest(values) != digest(changed_middle)
    assert digest(pd.Index(values)) != digest(pd.Index(changed_middle))
    assert digest(pd.Timestamp("2024-06-01")) == digest(pd.Timestamp("2024-06-01"))
    assert digest(pd.Timestamp("2024-06-01")) != digest(pd.Timestamp("2024-07-01"))

    class Opaque:
        pass

    with pytest.raises(TypeError, match="отпечаток"):
        digest(Opaque())
    with pytest.raises(TypeError, match="отпечаток"):
        digest({"nested": [Opaque()]})
