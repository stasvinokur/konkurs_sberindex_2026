"""Архив событий GDELT 1.0 по России: загрузка суточных файлов и сверка с манифестом.

DOC API GDELT отдаёт только последние три месяца, поэтому для 2023–2024 нужен архив суточных
файлов `https://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip` (около 8 МБ в день).
Файл скачивается в память, из него сразу отбираются строки с `ActionGeo_CountryCode == 'RS'`
и сохраняются помесячными parquet-файлами; полные выгрузки на диск не кладутся.

Какие месяцы составляют архив и какими они должны получиться, записано в
`data/gdelt_manifest.yaml`. Месяц пишется на диск, только если скачан каждый его день и файл
совпал с манифестом: неполный месяц дал бы другие новостные признаки незаметно.
"""

from __future__ import annotations

import http.client
import io
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from sbx.core.config import ConfigError
from sbx.core.gdelt_archive import (
    GdeltManifest,
    archive_problems,
    coverage_problems,
    manifest_problems,
    month_days,
    month_file,
    month_problems,
    months_between,
)
from sbx.shell.download.sberindex import SyncReport, ssl_context
from sbx.shell.io import DATA_DIR, load_config, parquet_rows, sha256_file

MANIFEST_PATH = DATA_DIR / "gdelt_manifest.yaml"
HEADER_FIELDS = ("source", "url_template", "country", "start", "end", "missing_days")
URL = "https://data.gdeltproject.org/events/{date:%Y%m%d}.export.CSV.zip"
COUNTRY_COLUMN = 51
COLUMNS = {
    "sqldate": 1,
    "event_code": 26,
    "event_root_code": 28,
    "quad_class": 29,
    "goldstein": 30,
    "num_mentions": 31,
    "num_articles": 33,
    "avg_tone": 34,
    "country": 51,
    "adm1": 52,
    "lat": 53,
    "lon": 54,
    "date_added": 56,
    "source_url": 57,
}
NUMERIC = ("quad_class", "goldstein", "num_mentions", "num_articles", "avg_tone", "lat", "lon")
# Попыток на суточный файл и пауза перед повтором в секундах (удваивается с каждой попыткой):
# сбой источника короче 75 секунд загрузку не прерывает.
RETRIES = 5
RETRY_PAUSE = 5.0

DayFetcher = Callable[[pd.Timestamp], pd.DataFrame]


class DayMissing(Exception):
    """У источника нет суточного файла за этот день (HTTP 404); повтор не поможет."""


class SourceUnavailable(Exception):
    """Источник не ответил или отдал повреждённый файл; возможно, временно."""


class ArchiveAbsent(FileNotFoundError):
    """Архив не скачан вовсе: нет ни одного файла из манифеста."""


class ArchiveMismatch(RuntimeError):
    """Архив скачан не полностью или не совпадает с манифестом."""


def load_manifest(path: Path = MANIFEST_PATH) -> GdeltManifest:
    manifest = load_config(GdeltManifest, path)
    if problems := manifest_problems(manifest):
        raise ConfigError(f"{path}: " + "; ".join(problems))
    return manifest


def parse_day(payload: bytes, country: str) -> pd.DataFrame:
    """События выбранной страны из суточного zip-файла."""
    rows = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if not names:
            raise zipfile.BadZipFile("в архиве нет файлов")
        for line in archive.read(names[0]).decode("utf-8", "ignore").splitlines():
            parts = line.split("\t")
            if len(parts) <= COLUMNS["source_url"] or parts[COUNTRY_COLUMN] != country:
                continue
            rows.append({key: parts[idx] for key, idx in COLUMNS.items()})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for col in NUMERIC:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["date_added"] = pd.to_datetime(frame["date_added"], format="%Y%m%d", errors="coerce")
    frame["sqldate"] = pd.to_datetime(frame["sqldate"], format="%Y%m%d", errors="coerce")
    return frame


def fetch_day(
    date: pd.Timestamp, country: str = "RS", url_template: str = URL, timeout: int = 60
) -> pd.DataFrame:
    """Скачивает суточный файл и возвращает только события выбранной страны."""
    request = urllib.request.Request(
        url_template.format(date=date), headers={"User-Agent": "sbx-research/0.1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=ssl_context()) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise DayMissing("у источника нет суточного файла (HTTP 404)") from exc
        raise SourceUnavailable(f"HTTP {exc.code}") from exc
    except (OSError, http.client.HTTPException) as exc:
        raise SourceUnavailable(str(exc) or type(exc).__name__) from exc
    try:
        return parse_day(payload, country)
    except (zipfile.BadZipFile, zlib.error) as exc:
        raise SourceUnavailable("суточный файл повреждён") from exc


def _fetch_with_retries(
    fetcher: DayFetcher,
    day: pd.Timestamp,
    retries: int,
    pause: float,
    sleep: Callable[[float], None],
) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            return fetcher(day)
        except SourceUnavailable:
            if attempt == retries - 1:
                raise
            sleep(pause * 2**attempt)
    raise ValueError("retries должно быть не меньше 1")


def download_month(
    month: str,
    missing_days: Sequence[str],
    fetcher: DayFetcher,
    retries: int = RETRIES,
    pause: float = RETRY_PAUSE,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Собирает месяц из суточных файлов; неудача любого дня отменяет месяц целиком."""
    frames = []
    for day in month_days(month, missing_days):
        try:
            frame = _fetch_with_retries(fetcher, day, retries, pause, sleep)
        except DayMissing as exc:
            raise DayMissing(f"{day:%Y-%m-%d}: {exc}") from exc
        except SourceUnavailable as exc:
            raise SourceUnavailable(f"{day:%Y-%m-%d}: {exc}") from exc
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def pending_months(manifest: GdeltManifest, data_dir: Path = DATA_DIR) -> list[str]:
    """Месяцы манифеста, файлов которых на диске ещё нет."""
    return [spec.month for spec in manifest.months if not (data_dir / spec.path).exists()]


def validate(manifest: GdeltManifest, data_dir: Path = DATA_DIR) -> list[str]:
    """Расхождения архива на диске с манифестом; пустой список — архив полон и совпадает."""
    actual = {
        spec.path: (sha256_file(data_dir / spec.path), parquet_rows(data_dir / spec.path))
        for spec in manifest.months
    }
    return archive_problems(manifest, actual)


def sync(
    manifest: GdeltManifest,
    data_dir: Path = DATA_DIR,
    fetcher: DayFetcher | None = None,
    retries: int = RETRIES,
    pause: float = RETRY_PAUSE,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[str, int, float], None] | None = None,
) -> SyncReport:
    """Докачивает отсутствующие месяцы. Существующие файлы не перезаписываются.

    Месяц, не совпавший с манифестом, сохраняется рядом с суффиксом `.downloaded`. Если источник
    не отвечает, загрузка останавливается: повторный запуск продолжит с этого же месяца.
    """
    fetch = fetcher or (lambda day: fetch_day(day, manifest.country, manifest.url_template))
    report = SyncReport()
    for spec in manifest.months:
        path = data_dir / spec.path
        if path.exists():
            problems = month_problems(spec, sha256_file(path), parquet_rows(path))
            (report.problems.extend(problems) if problems else report.ok.append(spec.path))
            continue
        started = time.monotonic()
        try:
            frame = download_month(spec.month, manifest.missing_days, fetch, retries, pause, sleep)
        except DayMissing as exc:
            report.problems.append(f"{spec.path}: месяц не записан — {exc}")
            continue
        except SourceUnavailable as exc:
            report.problems.append(f"{spec.path}: месяц не записан, загрузка остановлена — {exc}")
            break
        tmp_target = path.with_name(path.name + ".downloaded")
        tmp_target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(tmp_target, index=False)
        problems = month_problems(spec, sha256_file(tmp_target), len(frame))
        if problems:
            report.problems.extend(
                p + f" (скачанный файл сохранён как {tmp_target.name})" for p in problems
            )
            continue
        tmp_target.replace(path)
        report.downloaded.append(spec.path)
        if progress is not None:
            progress(spec.month, len(frame), time.monotonic() - started)
    return report


def build_manifest(header: Mapping[str, Any], data_dir: Path = DATA_DIR) -> dict[str, Any]:
    """Строит манифест по архиву на диске; период, страна и отсутствующие дни — из `header`.

    Архив, в котором при загрузке пропал день, эталоном сверки стать не может: дни добавления
    событий в каждом файле обязаны покрывать месяц.
    """
    months = []
    for month in months_between(header["start"], header["end"]):
        rel = month_file(month, header["country"])
        path = data_dir / rel
        if not path.exists():
            raise FileNotFoundError(f"нельзя построить манифест: нет файла {rel}")
        days = pd.read_parquet(path, columns=["date_added"])["date_added"].dropna().unique()
        if problems := coverage_problems(month, days, header["missing_days"]):
            raise ValueError("нельзя построить манифест: " + "; ".join(problems))
        months.append(
            {"month": month, "path": rel, "sha256": sha256_file(path), "rows": parquet_rows(path)}
        )
    return {**{key: header[key] for key in HEADER_FIELDS}, "months": months}


def load_archive(manifest: GdeltManifest, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """События архива, сверенного с манифестом; читаются только месяцы манифеста."""
    if len(pending_months(manifest, data_dir)) == len(manifest.months):
        raise ArchiveAbsent(f"архива GDELT нет в {(data_dir / manifest.months[0].path).parent}")
    if problems := validate(manifest, data_dir):
        raise ArchiveMismatch(
            "архив GDELT не совпадает с манифестом:\n  "
            + "\n  ".join(problems)
            + "\nДокачайте его командой `make gdelt` или уберите каталог архива, "
            "чтобы собрать признаки без новостей."
        )
    frames = [pd.read_parquet(data_dir / spec.path) for spec in manifest.months]
    return pd.concat(frames, ignore_index=True)
