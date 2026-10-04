"""Манифест файлов организаторов конкурса и чистая сверка с ним.

В отличие от выгрузок СберИндекса (`sbx.core.data_manifest`) у этих файлов есть лицензия с
обязательной цитатой, а справочник территорий опубликован внутри архива, поэтому у записи
больше полей.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from sbx.core.data_manifest import FileSpec, verify_file


@dataclass(frozen=True)
class OfficialFile:
    """Файл в каталоге данных: откуда он, чем должен быть и как на него ссылаться."""

    path: str
    url: str
    sha256: str
    title: str
    license: str
    # Подписи источника в том виде, в каком их требует правообладатель; у архива из
    # нескольких таблиц — по одной на таблицу.
    citations: tuple[str, ...]
    retrieved: str = ""
    # Файл лежит внутри опубликованного архива: имя внутри него и контрольная сумма архива.
    archive_member: str | None = None
    archive_sha256: str | None = None
    # Файл получен преобразованием другого файла манифеста, а не скачан.
    derived_from: str | None = None


@dataclass(frozen=True)
class OfficialManifest:
    source: str
    files: tuple[OfficialFile, ...]


def file_problems(spec: OfficialFile, actual_sha256: str | None) -> list[str]:
    return verify_file(FileSpec(spec.path, spec.url, spec.sha256), actual_sha256, None)


def manifest_problems(manifest: OfficialManifest, actual: Mapping[str, str | None]) -> list[str]:
    """Расхождения файлов с манифестом; `actual` — sha256 по путям (None — файла нет)."""
    return [
        problem for spec in manifest.files for problem in file_problems(spec, actual.get(spec.path))
    ]
