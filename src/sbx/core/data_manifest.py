"""Манифест сырых данных и чистая сверка файлов с ним."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FileSpec:
    """Один файл датасета: путь относительно каталога данных, URL, контрольная сумма."""

    path: str
    url: str
    sha256: str
    rows: int | None = None


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    title: str
    files: tuple[FileSpec, ...]


@dataclass(frozen=True)
class DataManifest:
    source: str
    download_url_template: str
    datasets: tuple[DatasetSpec, ...]

    def all_files(self) -> tuple[FileSpec, ...]:
        return tuple(f for d in self.datasets for f in d.files)


def download_url(template: str, slug: str, fmt: str) -> str:
    """URL публичной выгрузки датасета в формате `csv` (zip) или `parquet`."""
    if fmt not in {"csv", "parquet"}:
        raise ValueError(f"unsupported format: {fmt}")
    return template.format(slug=slug, fmt=fmt)


def verify_file(spec: FileSpec, actual_sha256: str | None, actual_rows: int | None) -> list[str]:
    """Сравнивает фактические свойства файла с манифестом; пустой список — файл в порядке."""
    if actual_sha256 is None:
        return [f"{spec.path}: файл отсутствует"]
    problems = []
    if actual_sha256 != spec.sha256:
        problems.append(
            f"{spec.path}: sha256 не совпадает (ожидалось {spec.sha256[:12]}…, "
            f"получено {actual_sha256[:12]}…)"
        )
    if spec.rows is not None and actual_rows is not None and actual_rows != spec.rows:
        problems.append(f"{spec.path}: строк {actual_rows}, ожидалось {spec.rows}")
    return problems
