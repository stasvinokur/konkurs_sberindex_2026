"""Отпечатки входов расчёта.

Кэш прогнозов и отметки рядом с артефактами должны замечать, что входы сменились: другая
панель, другой конфиг, другой набор внешних признаков. Отпечаток считается по содержимому, а
не по имени файла или времени записи.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def frame_digest(frame: pd.DataFrame) -> str:
    """Отпечаток таблицы: имена и порядок колонок, значения и порядок строк; индекс не входит."""
    hasher = hashlib.sha256()
    hasher.update(json.dumps([str(c) for c in frame.columns], ensure_ascii=False).encode())
    hasher.update(str(len(frame)).encode())
    try:
        rows = pd.util.hash_pandas_object(frame, index=False)
    except TypeError:
        # В колонке лежат списки или словари: сводим значения к строкам.
        rows = pd.util.hash_pandas_object(frame.astype(str), index=False)
    hasher.update(rows.to_numpy().tobytes())
    return hasher.hexdigest()


def digest(value: Any) -> str:
    """Отпечаток значения: таблица и массив — по содержимому, словарь и список — по элементам.

    Тип, содержимое которого здесь прочитать нельзя, — ошибка. Запасной путь через строковое
    представление обрезал бы длинные объекты, и изменение в их середине осталось бы незамеченным.
    """
    if isinstance(value, pd.DataFrame):
        return frame_digest(value)
    if isinstance(value, pd.Series | pd.Index):
        return frame_digest(pd.DataFrame({"values": value}))
    if isinstance(value, np.ndarray):
        header = json.dumps({"shape": value.shape, "dtype": str(value.dtype)}).encode()
        return _sha(header + np.ascontiguousarray(value).tobytes())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        # Объект с таблицей внутри (блок признаков): отпечаток берётся по полям.
        fields = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
        return digest({"type": type(value).__name__, "fields": fields})
    if isinstance(value, Mapping):
        parts = {str(key): digest(item) for key, item in value.items()}
        return _sha(json.dumps({"mapping": parts}, sort_keys=True).encode())
    if isinstance(value, list | tuple):
        return _sha(json.dumps({"sequence": [digest(item) for item in value]}).encode())
    if isinstance(value, pd.Timestamp | Path):
        value = str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, str | int | float | bool):
        return _sha(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())
    raise TypeError(f"отпечаток не умеет читать значение типа {type(value).__name__}")


def fingerprint(parts: Mapping[str, Any]) -> dict[str, str]:
    """Отпечаток каждого именованного входа."""
    return {name: digest(value) for name, value in parts.items()}


def changed(old: Mapping[str, str] | None, new: Mapping[str, str]) -> list[str]:
    """Имена входов, чей отпечаток изменился, появился или исчез; без прежней отметки — все."""
    old = dict(old or {})
    return sorted(name for name in {*old, *new} if old.get(name) != new.get(name))
