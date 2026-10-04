"""Отметки входов рядом с артефактами.

Шаг расчёта записывает рядом со своим результатом, на чём он посчитан: панель, прогнозы,
результаты предыдущих шагов, конфиг. Отметка — словарь «вход → отпечаток»; вход назван так,
чтобы его нынешний отпечаток можно было посчитать заново:

- `panel`, `static`, `seasonal` — таблицы в `data/processed`, `national` — национальный ряд
  расходов в `data/raw`; отпечаток по содержимому;
- `file:<путь>` — файл, sha256;
- `glob:<каталог>/<маска>[;except=<имя>,…]` — набор файлов каталога.

По отметкам отчёт видит, что все артефакты посчитаны на текущих входах, и не собирается из
смеси прогнозов разных панелей.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pandas as pd

from sbx.core.fingerprint import changed, digest, frame_digest
from sbx.shell.io import DATA_DIR, ROOT, sha256_file

PROCESSED = DATA_DIR / "processed"
RAW = DATA_DIR / "raw"
PANEL, STATIC, SEASONAL, NATIONAL = "panel", "static", "seasonal", "national"
_MISSING = "нет"
# Отпечаток таблицы пересчитывается только когда файл сменился.
_table_cache: dict[tuple[str, int, int], str] = {}


def _relative(path: Path) -> str:
    """Путь от корня проекта: так отметка годится и на хосте, и в контейнере."""
    path = Path(path)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def file_key(path: Path) -> str:
    return f"file:{_relative(path)}"


def glob_key(directory: Path, pattern: str, exclude: Sequence[str] = ()) -> str:
    key = f"glob:{_relative(directory)}/{pattern}"
    return key + (f";except={','.join(sorted(exclude))}" if exclude else "")


def _table_path(key: str) -> Path | None:
    """Таблица, отпечаток содержимого которой записан под этим именем входа."""
    return {
        PANEL: PROCESSED / "panel.parquet",
        STATIC: PROCESSED / "static.parquet",
        SEASONAL: PROCESSED / "seasonal_index.parquet",
        # Национальный ряд расходов: по нему для каждого окна оценивается сезонный индекс.
        NATIONAL: RAW / "consumer-spending" / "consumer-spending.parquet",
    }.get(key)


def _table_digest(path: Path) -> str:
    if not path.exists():
        return _MISSING
    stat = path.stat()
    cache_key = (str(path), stat.st_mtime_ns, stat.st_size)
    if cache_key not in _table_cache:
        _table_cache[cache_key] = frame_digest(pd.read_parquet(path))
    return _table_cache[cache_key]


def current(key: str) -> str | None:
    """Нынешний отпечаток входа; None, если по имени входа его не посчитать."""
    table = _table_path(key)
    if table is not None:
        return _table_digest(table)
    if key.startswith("file:"):
        return sha256_file(ROOT / key.removeprefix("file:")) or _MISSING
    if key.startswith("glob:"):
        spec, _, excluded = key.removeprefix("glob:").partition(";except=")
        directory, _, pattern = spec.rpartition("/")
        skip = set(excluded.split(",")) if excluded else set()
        files = sorted(p for p in (ROOT / directory).glob(pattern) if p.name not in skip)
        return digest({p.name: sha256_file(p) for p in files})
    return None


def save(path: Path, parts: Mapping[str, str]) -> None:
    """Записывает готовую отметку «вход → отпечаток»."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(parts), ensure_ascii=False, indent=1), encoding="utf-8")


def write(path: Path, keys: Iterable[str]) -> None:
    """Записывает отметку с нынешними отпечатками перечисленных входов."""
    save(path, {key: current(key) or _MISSING for key in keys})


def read(path: Path) -> dict[str, str] | None:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def stale(path: Path) -> list[str] | None:
    """Входы, изменившиеся с момента записи отметки; None — отметки нет.

    Сверяются входы, чей отпечаток можно посчитать по имени. Отметка, в которой таких входов
    нет вовсе, ничего не подтверждает и приравнивается к отсутствующей: иначе она оставалась
    бы «свежей» навсегда. Отметки прогнозов (в них отпечатки таблиц в памяти, конфига и
    фолдов) сверяются иначе — пересчётом ожидаемого отпечатка, см. `report.stale_artifacts`.
    """
    recorded = read(path)
    if recorded is None:
        return None
    now = {key: value for key in recorded if (value := current(key)) is not None}
    if not now:
        return None
    return changed({key: recorded[key] for key in now}, now)
