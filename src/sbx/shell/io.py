"""Файловый ввод-вывод: YAML, контрольные суммы, parquet."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, TypeVar

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from sbx.core.config import from_mapping

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "configs"
ARTIFACTS_DIR = ROOT / "artifacts"

T = TypeVar("T")


def read_yaml(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def write_yaml(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)


def load_config(cls: type[T], path: Path) -> T:
    """Читает YAML и собирает из него неизменяемый объект конфигурации ядра."""
    return from_mapping(cls, read_yaml(path), path=str(path))


def publication_lags(path: Path | None = None) -> dict[str, int]:
    """Лаги публикации источников в днях от конца отчётного периода (`configs/features.yaml`).

    Лаги заданы в конфиге с обоснованием, а не константами в коде.
    """
    cfg = read_yaml(path or CONFIG_DIR / "features.yaml")
    return {str(k): int(v) for k, v in (cfg.get("publication_lag_days") or {}).items()}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def parquet_rows(path: Path) -> int | None:
    """Число строк parquet-файла; None, если это не parquet или файл не читается.

    Оборванный файл не должен ронять сверку: его выдаст контрольная сумма.
    """
    path = Path(path)
    if path.suffix != ".parquet" or not path.exists():
        return None
    try:
        return pq.ParquetFile(path).metadata.num_rows
    except pa.ArrowInvalid:
        return None
