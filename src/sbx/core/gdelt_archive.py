"""Описание архива событий GDELT и чистая сверка его состояния.

Архив — помесячные файлы событий по одной стране. Месяц собирается из суточных файлов источника,
поэтому у него нет одного URL: манифест хранит шаблон суточного адреса, период и дни, за которые
у источника файла нет.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from sbx.core.data_manifest import FileSpec, verify_file


@dataclass(frozen=True)
class MonthSpec:
    """Один месяц архива: путь относительно каталога данных, контрольная сумма, число событий."""

    month: str
    path: str
    sha256: str
    rows: int


@dataclass(frozen=True)
class GdeltManifest:
    source: str
    url_template: str
    country: str
    start: str
    end: str
    missing_days: tuple[str, ...]
    months: tuple[MonthSpec, ...]


def months_between(start: str, end: str) -> list[str]:
    """Месяцы периода в виде `YYYY-MM`, оба конца включены."""
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def month_file(month: str, country: str) -> str:
    """Путь файла месяца относительно каталога данных."""
    return f"external/gdelt/gdelt_{country.lower()}_{month.replace('-', '')}.parquet"


def month_days(month: str, missing_days: Sequence[str] = ()) -> list[pd.Timestamp]:
    """Дни месяца, за которые у источника есть суточный файл."""
    period = pd.Period(month, freq="M")
    missing = {pd.Timestamp(d) for d in missing_days}
    days = pd.date_range(period.start_time, period.end_time.normalize(), freq="D")
    return [day for day in days if day not in missing]


def coverage_problems(
    month: str, days_present: Iterable[pd.Timestamp], missing_days: Sequence[str] = ()
) -> list[str]:
    """Сверяет дни добавления событий в файле месяца с днями, за которые у источника есть файл."""
    expected = set(month_days(month, missing_days))
    present = {pd.Timestamp(day).normalize() for day in days_present}
    problems: list[str] = []
    if lost := sorted(expected - present):
        problems.append(f"{month}: нет событий за {_days(lost)}")
    if stray := sorted(present - expected):
        problems.append(f"{month}: есть события за {_days(stray)}, которых в месяце быть не должно")
    return problems


def _days(days: Iterable[pd.Timestamp]) -> str:
    return ", ".join(f"{day:%Y-%m-%d}" for day in days)


def month_problems(spec: MonthSpec, sha256: str | None, rows: int | None) -> list[str]:
    """Сверка одного файла месяца — та же, что у сырых данных СберИндекса."""
    # У месяца нет одного адреса: он собирается из суточных файлов.
    expected = FileSpec(spec.path, url="", sha256=spec.sha256, rows=spec.rows)
    return verify_file(expected, sha256, rows)


def archive_problems(
    manifest: GdeltManifest, actual: Mapping[str, tuple[str | None, int | None]]
) -> list[str]:
    """Расхождения архива с манифестом; `actual` — sha256 и число строк по путям файлов.

    Пустой список — архив полон и совпадает с тем, на котором получены опубликованные результаты.
    """
    return [
        problem
        for spec in manifest.months
        for problem in month_problems(spec, *actual.get(spec.path, (None, None)))
    ]


def manifest_problems(manifest: GdeltManifest) -> list[str]:
    """Внутренние противоречия манифеста: месяцы не покрывают период, чужие пути, лишние дни."""
    problems: list[str] = []
    expected = months_between(manifest.start, manifest.end)
    listed = [spec.month for spec in manifest.months]
    if listed != expected:
        absent = [m for m in expected if m not in listed]
        extra = [m for m in listed if m not in expected]
        problems.append(
            f"месяцы манифеста не совпадают с периодом {manifest.start}…{manifest.end}: "
            f"нет {absent}, лишние {extra}"
        )
    for spec in manifest.months:
        if spec.path != month_file(spec.month, manifest.country):
            problems.append(f"{spec.month}: путь {spec.path} не соответствует месяцу и стране")
    covered = {day for month in expected for day in month_days(month)}
    for day in manifest.missing_days:
        if pd.Timestamp(day) not in covered:
            problems.append(f"отсутствующий день {day} лежит вне периода манифеста")
    return problems
