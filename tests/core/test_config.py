from dataclasses import FrozenInstanceError, dataclass, field
from typing import Literal

import pytest

from sbx.core.config import ConfigError, from_mapping
from sbx.core.data_manifest import DataManifest, FileSpec, download_url, verify_file


@dataclass(frozen=True)
class Inner:
    name: str
    weight: float = 1.0


@dataclass(frozen=True)
class Outer:
    horizon: int
    mode: Literal["a", "b"]
    items: tuple[Inner, ...]
    tags: tuple[str, ...] = field(default_factory=tuple)
    note: str | None = None


def test_builds_nested_frozen_objects() -> None:
    cfg = from_mapping(
        Outer, {"horizon": 3, "mode": "a", "items": [{"name": "x"}, {"name": "y", "weight": 2}]}
    )
    assert cfg.horizon == 3
    assert cfg.items[1] == Inner("y", 2.0)
    assert cfg.tags == ()
    with pytest.raises(FrozenInstanceError):
        cfg.horizon = 4  # type: ignore[misc]


def test_unknown_field_is_rejected_with_path() -> None:
    with pytest.raises(ConfigError, match=r"\$\.items\[0\].*неизвестные поля \['wieght'\]"):
        from_mapping(Outer, {"horizon": 1, "mode": "a", "items": [{"name": "x", "wieght": 2}]})


def test_missing_field_is_rejected() -> None:
    with pytest.raises(ConfigError, match="отсутствует обязательное поле 'mode'"):
        from_mapping(Outer, {"horizon": 1, "items": []})


@pytest.mark.parametrize(
    "patch",
    [{"horizon": "3"}, {"horizon": True}, {"mode": "c"}, {"items": "x"}],
)
def test_wrong_types_are_rejected(patch: dict) -> None:
    data = {"horizon": 1, "mode": "a", "items": []} | patch
    with pytest.raises(ConfigError):
        from_mapping(Outer, data)


def test_manifest_schema_roundtrip() -> None:
    manifest = from_mapping(
        DataManifest,
        {
            "source": "s",
            "download_url_template": "https://x/{slug}/{fmt}",
            "datasets": [
                {
                    "slug": "d",
                    "title": "t",
                    "files": [{"path": "raw/d/d.parquet", "url": "u", "sha256": "ab", "rows": 5}],
                }
            ],
        },
    )
    assert manifest.all_files() == (FileSpec("raw/d/d.parquet", "u", "ab", 5),)
    assert download_url(manifest.download_url_template, "d", "csv") == "https://x/d/csv"


def test_verify_file_reports_each_problem() -> None:
    spec = FileSpec("raw/a.parquet", "u", "a" * 64, rows=10)
    assert verify_file(spec, "a" * 64, 10) == []
    assert verify_file(spec, None, None) == ["raw/a.parquet: файл отсутствует"]
    problems = verify_file(spec, "b" * 64, 9)
    assert len(problems) == 2
    assert "sha256" in problems[0] and "строк 9" in problems[1]
