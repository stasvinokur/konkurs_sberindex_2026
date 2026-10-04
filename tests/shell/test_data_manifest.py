import hashlib
from pathlib import Path

import pandas as pd
import pytest

from sbx.core.config import ConfigError
from sbx.core.data_manifest import DataManifest, DatasetSpec, FileSpec
from sbx.shell.download import sberindex
from sbx.shell.io import load_config, sha256_file


def _manifest(spec: FileSpec) -> DataManifest:
    return DataManifest("s", "https://x/{slug}/{fmt}", (DatasetSpec("d", "t", (spec,)),))


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"sberindex")
    assert sha256_file(path) == hashlib.sha256(b"sberindex").hexdigest()
    assert sha256_file(tmp_path / "missing") is None


def test_validate_detects_match_and_mismatch(tmp_path: Path) -> None:
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(tmp_path / "d.parquet")
    good = FileSpec("d.parquet", "u", sha256_file(tmp_path / "d.parquet"), rows=3)
    assert sberindex.validate(_manifest(good), tmp_path) == []

    bad_sha = FileSpec("d.parquet", "u", "0" * 64, rows=3)
    assert "sha256" in sberindex.validate(_manifest(bad_sha), tmp_path)[0]

    bad_rows = FileSpec("d.parquet", "u", good.sha256, rows=4)
    assert "строк 3" in sberindex.validate(_manifest(bad_rows), tmp_path)[0]


def test_sync_downloads_missing_and_never_overwrites(tmp_path: Path) -> None:
    payload = b"new-content"
    spec = FileSpec("raw/d/d.zip", "https://x/d/csv", hashlib.sha256(payload).hexdigest())
    calls: list[str] = []

    def fake_fetch(url: str, dest: Path) -> None:
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)

    report = sberindex.sync(_manifest(spec), tmp_path, fetcher=fake_fetch)
    assert report.downloaded == ["raw/d/d.zip"] and not report.problems
    assert (tmp_path / "raw/d/d.zip").read_bytes() == payload

    # Файл уже есть: повторная синхронизация ничего не качает.
    report = sberindex.sync(_manifest(spec), tmp_path, fetcher=fake_fetch)
    assert calls == ["https://x/d/csv"] and report.ok == ["raw/d/d.zip"]


def test_sync_keeps_mismatched_download_aside(tmp_path: Path) -> None:
    spec = FileSpec("raw/d/d.zip", "u", "0" * 64)

    def fake_fetch(url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"changed upstream")

    report = sberindex.sync(_manifest(spec), tmp_path, fetcher=fake_fetch)
    assert not (tmp_path / "raw/d/d.zip").exists()
    assert (tmp_path / "raw/d/d.zip.downloaded").exists()
    assert "sha256" in report.problems[0]


def test_repository_manifest_matches_files() -> None:
    manifest = sberindex.load_manifest()
    slugs = {d.slug for d in manifest.datasets}
    assert "potrebitelskie-beznalicnye-rashody-na-urovne-munizipalnyh-obrazovanij" in slugs
    assert len(manifest.all_files()) == 2 * len(manifest.datasets)
    assert sberindex.validate(manifest) == []


def test_load_config_reports_bad_yaml_structure(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text("source: s\ndatasets: []\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="download_url_template"):
        load_config(DataManifest, path)
