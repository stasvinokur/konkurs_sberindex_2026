"""Отметки входов рядом с артефактами: шаг записывает, на чём посчитан, отчёт это сверяет."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sbx.shell import stamps


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Копия раскладки проекта во временном каталоге."""
    monkeypatch.setattr(stamps, "ROOT", tmp_path)
    monkeypatch.setattr(stamps, "PROCESSED", tmp_path / "data" / "processed")
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "artifacts" / "oof").mkdir(parents=True)
    pd.DataFrame({"unique_id": ["a"], "y": [1.0]}).to_parquet(
        tmp_path / "data" / "processed" / "panel.parquet", index=False
    )
    return tmp_path


def test_keys_are_relative_to_the_project_root(tree: Path) -> None:
    assert stamps.file_key(tree / "artifacts" / "cpd" / "alarms.json") == (
        "file:artifacts/cpd/alarms.json"
    )
    assert stamps.glob_key(
        tree / "artifacts" / "oof", "*.parquet", exclude=["ensemble.parquet"]
    ) == ("glob:artifacts/oof/*.parquet;except=ensemble.parquet")
    assert stamps.file_key(Path("artifacts/news/x.parquet")) == "file:artifacts/news/x.parquet"
    outside = Path("/somewhere/else/x.json")
    assert stamps.file_key(outside) == "file:/somewhere/else/x.json"


def test_a_stamp_goes_stale_when_a_recorded_input_changes(tree: Path) -> None:
    oof = tree / "artifacts" / "oof"
    (oof / "stats.parquet").write_bytes(b"one")
    config = tree / "configs.yaml"
    config.write_text("a: 1", encoding="utf-8")
    stamp = tree / "artifacts" / "metrics" / "inputs.json"
    keys = [stamps.PANEL, stamps.file_key(config), stamps.glob_key(oof, "*.parquet")]

    assert stamps.stale(stamp) is None, "отметки ещё нет"
    stamps.write(stamp, keys)
    assert stamps.stale(stamp) == []

    config.write_text("a: 2", encoding="utf-8")
    assert stamps.stale(stamp) == ["file:configs.yaml"]
    config.write_text("a: 1", encoding="utf-8")
    assert stamps.stale(stamp) == [], "то же содержимое — та же отметка"

    (oof / "lgbm.parquet").write_bytes(b"two")
    assert stamps.stale(stamp) == ["glob:artifacts/oof/*.parquet"], "появился новый файл"
    (oof / "lgbm.parquet").unlink()

    pd.DataFrame({"unique_id": ["a"], "y": [2.0]}).to_parquet(
        tree / "data" / "processed" / "panel.parquet", index=False
    )
    assert stamps.stale(stamp) == [stamps.PANEL]


def test_an_excluded_file_does_not_change_the_state(tree: Path) -> None:
    oof = tree / "artifacts" / "oof"
    (oof / "stats.parquet").write_bytes(b"one")
    stamp = oof / "ensemble_inputs.json"
    stamps.write(stamp, [stamps.glob_key(oof, "*.parquet", exclude=["ensemble.parquet"])])
    (oof / "ensemble.parquet").write_bytes(b"result")
    assert stamps.stale(stamp) == []
    (oof / "stats.parquet").write_bytes(b"changed")
    assert stamps.stale(stamp) == ["glob:artifacts/oof/*.parquet;except=ensemble.parquet"]


def test_a_missing_input_is_recorded_and_noticed_when_it_appears(tree: Path) -> None:
    stamp = tree / "artifacts" / "hazard" / "inputs.json"
    news = tree / "artifacts" / "news" / "features.parquet"
    stamps.write(stamp, [stamps.file_key(news)])
    assert stamps.stale(stamp) == []
    news.parent.mkdir(parents=True)
    news.write_bytes(b"archive downloaded")
    assert stamps.stale(stamp) == ["file:artifacts/news/features.parquet"]


def test_content_digests_of_a_run_are_compared_with_the_data_on_disk(tree: Path) -> None:
    """Отметка прогона моделей хранит отпечатки таблиц в памяти; известные из них сверяются
    с тем, что лежит в data/processed, остальные (конфиг, фолды) отчёту недоступны."""
    from sbx.core.fingerprint import frame_digest

    panel = pd.read_parquet(tree / "data" / "processed" / "panel.parquet")
    stamp = tree / "artifacts" / "oof" / "stats_inputs.json"
    stamps.save(stamp, {stamps.PANEL: frame_digest(panel), "config": "abc"})
    assert stamps.stale(stamp) == []
    stamps.save(stamp, {stamps.PANEL: frame_digest(panel.assign(y=5.0)), "config": "abc"})
    assert stamps.stale(stamp) == [stamps.PANEL]


def test_a_stamp_without_checkable_inputs_vouches_for_nothing(tree: Path) -> None:
    """Пустая отметка оставалась бы «свежей» навсегда; такая отметка ничего не подтверждает."""
    stamp = tree / "artifacts" / "cpd" / "inputs.json"
    stamps.save(stamp, {})
    assert stamps.stale(stamp) is None
    stamps.save(stamp, {"config": "abc"})
    assert stamps.stale(stamp) is None, "ни один вход нельзя сверить по имени"
