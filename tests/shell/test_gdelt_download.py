"""Архив GDELT: докачка по манифесту, отказ от неполных месяцев, состояния архива. Без сети."""

from __future__ import annotations

import http.client
import io
import urllib.error
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from sbx.core.config import ConfigError, from_mapping
from sbx.core.gdelt_archive import GdeltManifest, manifest_problems, month_days, month_file
from sbx.shell.download import gdelt
from sbx.shell.io import write_yaml

HEADER = {
    "source": "тест",
    "url_template": "https://x/{date:%Y%m%d}.zip",
    "country": "RS",
    "start": "2024-01",
    "end": "2024-02",
    "missing_days": ["2024-01-15"],
}
JANUARY = month_file("2024-01", "RS")
FEBRUARY = month_file("2024-02", "RS")


def _day_frame(day: pd.Timestamp) -> pd.DataFrame:
    """Суточный кадр-заглушка: два события, по которым виден день."""
    return pd.DataFrame({"date_added": [day, day], "num_articles": [day.day, 1]})


def _write_month(data_dir: Path, month: str, days: list[pd.Timestamp]) -> None:
    path = data_dir / month_file(month, "RS")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([_day_frame(d) for d in days], ignore_index=True).to_parquet(path, index=False)


def _reference_archive(data_dir: Path) -> GdeltManifest:
    """Архив «автора» и манифест, построенный по его файлам."""
    for month in ("2024-01", "2024-02"):
        _write_month(data_dir, month, month_days(month, HEADER["missing_days"]))
    return from_mapping(GdeltManifest, gdelt.build_manifest(HEADER, data_dir))


class FakeSource:
    """Источник суточных файлов без сети: запоминает запросы и отказывает по сценарию."""

    def __init__(self, failures: dict[str, list[Exception]] | None = None) -> None:
        self.calls: list[str] = []
        self.failures = failures or {}

    def __call__(self, day: pd.Timestamp) -> pd.DataFrame:
        key = f"{day:%Y-%m-%d}"
        self.calls.append(key)
        if self.failures.get(key):
            raise self.failures[key].pop(0)
        return _day_frame(day)


def _no_sleep(_: float) -> None:
    return None


def test_build_manifest_describes_the_archive_on_disk(tmp_path: Path) -> None:
    manifest = _reference_archive(tmp_path)
    assert manifest_problems(manifest) == []
    assert [m.path for m in manifest.months] == [JANUARY, FEBRUARY]
    assert manifest.months[0].rows == 2 * 30, "31 день января без дня, которого нет у источника"
    assert gdelt.validate(manifest, tmp_path) == []


def test_build_manifest_refuses_an_archive_with_a_silently_skipped_day(tmp_path: Path) -> None:
    """Эталоном сверки не может стать архив, в котором день пропал при загрузке."""
    _reference_archive(tmp_path)
    days = [d for d in month_days("2024-02") if d != pd.Timestamp("2024-02-07")]
    _write_month(tmp_path, "2024-02", days)
    with pytest.raises(ValueError, match="2024-02-07"):
        gdelt.build_manifest(HEADER, tmp_path)

    (tmp_path / FEBRUARY).unlink()
    with pytest.raises(FileNotFoundError, match="gdelt_rs_202402"):
        gdelt.build_manifest(HEADER, tmp_path)


def test_sync_downloads_only_missing_months(tmp_path: Path) -> None:
    manifest = _reference_archive(tmp_path)
    (tmp_path / JANUARY).unlink()
    source = FakeSource()

    report = gdelt.sync(manifest, tmp_path, fetcher=source)

    assert report.downloaded == [JANUARY] and report.ok == [FEBRUARY] and not report.problems
    assert gdelt.validate(manifest, tmp_path) == []
    # Запрошены только дни января, и среди них нет дня, за который у источника нет файла.
    assert len(source.calls) == 30 and all(c.startswith("2024-01") for c in source.calls)
    assert "2024-01-15" not in source.calls

    # Архив полон: повторный запуск к источнику не обращается.
    report = gdelt.sync(manifest, tmp_path, fetcher=source)
    assert len(source.calls) == 30
    assert report.ok == [JANUARY, FEBRUARY] and not report.downloaded


def test_sync_reports_each_downloaded_month(tmp_path: Path) -> None:
    manifest = _reference_archive(tmp_path)
    (tmp_path / FEBRUARY).unlink()
    seen: list[tuple[str, int]] = []

    gdelt.sync(
        manifest,
        tmp_path,
        fetcher=FakeSource(),
        progress=lambda month, rows, seconds: seen.append((month, rows)),
    )

    assert seen == [("2024-02", 2 * 29)]


def test_sync_retries_a_day_that_failed_temporarily(tmp_path: Path) -> None:
    manifest = _reference_archive(tmp_path)
    (tmp_path / FEBRUARY).unlink()
    source = FakeSource(
        {"2024-02-10": [gdelt.SourceUnavailable("HTTP 503"), gdelt.SourceUnavailable("таймаут")]}
    )
    pauses: list[float] = []

    report = gdelt.sync(manifest, tmp_path, fetcher=source, pause=1.0, sleep=pauses.append)

    assert report.downloaded == [FEBRUARY] and not report.problems
    assert source.calls.count("2024-02-10") == 3
    assert pauses == [1.0, 2.0], "пауза растёт с каждой попыткой"


def test_sync_writes_no_month_when_a_day_cannot_be_fetched(tmp_path: Path) -> None:
    """Прежний загрузчик пропускал такой день молча, и месяц выходил неполным незаметно."""
    manifest = _reference_archive(tmp_path)
    (tmp_path / JANUARY).unlink()
    (tmp_path / FEBRUARY).unlink()
    source = FakeSource({"2024-01-20": [gdelt.SourceUnavailable("HTTP 503")] * 10})

    report = gdelt.sync(manifest, tmp_path, fetcher=source, retries=3, pause=0, sleep=_no_sleep)

    assert list((tmp_path / "external" / "gdelt").iterdir()) == [], "ни месяца, ни отложенной копии"
    assert len(report.problems) == 1
    assert "2024-01-20" in report.problems[0] and "HTTP 503" in report.problems[0]
    assert source.calls.count("2024-01-20") == 3
    # Источник не отвечает: следующий месяц не запрашивается, повторный запуск продолжит отсюда.
    assert not any(c.startswith("2024-02") for c in source.calls)


def test_sync_goes_on_when_the_source_has_lost_one_day(tmp_path: Path) -> None:
    manifest = _reference_archive(tmp_path)
    (tmp_path / JANUARY).unlink()
    (tmp_path / FEBRUARY).unlink()
    source = FakeSource({"2024-01-20": [gdelt.DayMissing("HTTP 404")]})

    report = gdelt.sync(manifest, tmp_path, fetcher=source, pause=0, sleep=_no_sleep)

    assert source.calls.count("2024-01-20") == 1, "отсутствие файла повтором не лечится"
    assert report.downloaded == [FEBRUARY]
    assert len(report.problems) == 1 and "2024-01-20" in report.problems[0]
    assert not (tmp_path / JANUARY).exists()


def test_sync_keeps_a_month_that_differs_from_the_manifest_aside(tmp_path: Path) -> None:
    manifest = _reference_archive(tmp_path)
    january = tmp_path / JANUARY
    january.unlink()

    def changed_upstream(day: pd.Timestamp) -> pd.DataFrame:
        return _day_frame(day).assign(num_articles=99)

    report = gdelt.sync(manifest, tmp_path, fetcher=changed_upstream)

    assert not january.exists()
    assert january.with_name(january.name + ".downloaded").exists()
    assert "sha256" in report.problems[0] and ".downloaded" in report.problems[0]


def test_load_archive_tells_a_complete_archive_from_an_incomplete_and_an_absent_one(
    tmp_path: Path,
) -> None:
    manifest = _reference_archive(tmp_path)
    events = gdelt.load_archive(manifest, tmp_path)
    assert len(events) == sum(m.rows for m in manifest.months)
    assert events["date_added"].is_monotonic_increasing, "месяцы идут по порядку"

    # Недокачанный архив — ошибка, а не признаки по части данных.
    (tmp_path / FEBRUARY).unlink()
    with pytest.raises(gdelt.ArchiveMismatch, match="gdelt_rs_202402.parquet: файл отсутствует"):
        gdelt.load_archive(manifest, tmp_path)

    # Архива нет вовсе — отдельный случай: шаг новостей его пропускает.
    (tmp_path / JANUARY).unlink()
    with pytest.raises(gdelt.ArchiveAbsent):
        gdelt.load_archive(manifest, tmp_path)


def test_a_month_with_a_lost_day_is_rejected_everywhere(tmp_path: Path) -> None:
    """Ради этого случая задача и делалась: файл месяца на месте, но одного дня в нём нет."""
    manifest = _reference_archive(tmp_path)
    days = [d for d in month_days("2024-02") if d != pd.Timestamp("2024-02-07")]
    _write_month(tmp_path, "2024-02", days)
    damaged = (tmp_path / FEBRUARY).read_bytes()

    problems = gdelt.validate(manifest, tmp_path)
    assert len(problems) == 2
    assert "gdelt_rs_202402.parquet: sha256 не совпадает" in problems[0]
    assert "строк 56, ожидалось 58" in problems[1]
    with pytest.raises(gdelt.ArchiveMismatch, match="sha256 не совпадает"):
        gdelt.load_archive(manifest, tmp_path)

    # Докачка такой файл не чинит и не трогает: решать, удалять ли его, должен человек.
    source = FakeSource()
    report = gdelt.sync(manifest, tmp_path, fetcher=source)
    assert source.calls == []
    assert report.ok == [JANUARY] and len(report.problems) == 2
    assert (tmp_path / FEBRUARY).read_bytes() == damaged


def test_a_truncated_month_file_is_reported_not_crashed_on(tmp_path: Path) -> None:
    """Оборванная копия архива — не parquet вовсе; сверка должна назвать файл, а не упасть."""
    manifest = _reference_archive(tmp_path)
    path = tmp_path / FEBRUARY
    path.write_bytes(path.read_bytes()[:100])

    assert "gdelt_rs_202402.parquet: sha256 не совпадает" in gdelt.validate(manifest, tmp_path)[0]
    with pytest.raises(gdelt.ArchiveMismatch):
        gdelt.load_archive(manifest, tmp_path)
    assert gdelt.sync(manifest, tmp_path, fetcher=FakeSource()).ok == [JANUARY]

    path.write_bytes(b"")
    assert len(gdelt.validate(manifest, tmp_path)) == 1


def test_sync_fetches_from_the_source_named_in_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _reference_archive(tmp_path)
    (tmp_path / FEBRUARY).unlink()
    seen: set[tuple[str, str]] = set()

    def fake_fetch_day(day: pd.Timestamp, country: str, url_template: str) -> pd.DataFrame:
        seen.add((country, url_template))
        return _day_frame(day)

    monkeypatch.setattr(gdelt, "fetch_day", fake_fetch_day)
    report = gdelt.sync(manifest, tmp_path)
    assert report.downloaded == [FEBRUARY]
    assert seen == {("RS", "https://x/{date:%Y%m%d}.zip")}


def test_load_archive_reads_only_the_months_of_the_manifest(tmp_path: Path) -> None:
    manifest = _reference_archive(tmp_path)
    _write_month(tmp_path, "2024-03", month_days("2024-03"))
    assert len(gdelt.load_archive(manifest, tmp_path)) == sum(m.rows for m in manifest.months)


def test_load_manifest_rejects_a_manifest_with_a_gap(tmp_path: Path) -> None:
    _reference_archive(tmp_path)
    broken = gdelt.build_manifest(HEADER, tmp_path)
    broken["months"] = broken["months"][:1]
    write_yaml(tmp_path / "m.yaml", broken)
    with pytest.raises(ConfigError, match="2024-02"):
        gdelt.load_manifest(tmp_path / "m.yaml")


def test_repository_manifest_describes_the_published_period() -> None:
    manifest = gdelt.load_manifest()
    assert (manifest.start, manifest.end, manifest.country) == ("2022-10", "2024-12", "RS")
    assert len(manifest.months) == 27
    assert sum(m.rows for m in manifest.months) == 3_011_689
    assert manifest.missing_days == ("2022-11-10", "2023-03-23")


def test_local_archive_matches_the_repository_manifest() -> None:
    manifest = gdelt.load_manifest()
    # Идущая или прерванная загрузка — не дефект кода: полноту архива стережёт `make gdelt`.
    if gdelt.pending_months(manifest):
        pytest.skip("архив GDELT не скачан или скачан не полностью")
    assert gdelt.validate(manifest) == []


def _zip_payload(lines: list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("20240105.export.CSV", "\n".join(lines))
    return buffer.getvalue()


def _event_line(country: str, adm1: str, tone: str) -> str:
    parts = [""] * 58
    values = {
        "sqldate": "20240104",
        "event_code": "043",
        "event_root_code": "04",
        "quad_class": "1",
        "goldstein": "2.8",
        "num_mentions": "4",
        "num_articles": "3",
        "avg_tone": tone,
        "country": country,
        "adm1": adm1,
        "lat": "55.75",
        "lon": "37.61",
        "date_added": "20240105",
        "source_url": "https://example.org/a",
    }
    for key, value in values.items():
        parts[gdelt.COLUMNS[key]] = value
    return "\t".join(parts)


def test_parse_day_keeps_only_the_country_and_types_the_columns() -> None:
    payload = _zip_payload(
        [
            _event_line("RS", "RS48", "-1.5"),
            _event_line("US", "USCA", "0.3"),
            _event_line("RS", "RS", ""),
            "обрезанная\tстрока",
        ]
    )
    frame = gdelt.parse_day(payload, "RS")
    assert list(frame.columns) == list(gdelt.COLUMNS)
    assert frame["adm1"].tolist() == ["RS48", "RS"]
    assert frame["avg_tone"].iloc[0] == -1.5 and pd.isna(frame["avg_tone"].iloc[1])
    assert frame["date_added"].iloc[0] == pd.Timestamp("2024-01-05")
    assert frame["sqldate"].iloc[0] == pd.Timestamp("2024-01-04")
    assert gdelt.parse_day(_zip_payload([_event_line("US", "USCA", "0.3")]), "RS").empty


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (urllib.error.HTTPError("u", 404, "Not Found", None, None), gdelt.DayMissing),
        (urllib.error.HTTPError("u", 503, "Unavailable", None, None), gdelt.SourceUnavailable),
        (urllib.error.URLError("нет сети"), gdelt.SourceUnavailable),
        (TimeoutError("timed out"), gdelt.SourceUnavailable),
        (http.client.IncompleteRead(b"abc"), gdelt.SourceUnavailable),
    ],
)
def test_fetch_day_tells_a_missing_day_from_an_unavailable_source(
    monkeypatch: pytest.MonkeyPatch, error: Exception, expected: type[Exception]
) -> None:
    def failing_urlopen(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(gdelt.urllib.request, "urlopen", failing_urlopen)
    with pytest.raises(expected):
        gdelt.fetch_day(pd.Timestamp("2024-01-05"))


def _corrupt_deflate() -> bytes:
    """Zip с целым оглавлением и испорченным сжатым потоком."""
    lines = [_event_line("RS", "RS48", str(i)) for i in range(200)]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("20240105.export.CSV", "\n".join(lines))
    payload = bytearray(buffer.getvalue())
    start = 30 + len("20240105.export.CSV")
    payload[start : start + 40] = b"\xff" * 40
    return bytes(payload)


@pytest.mark.parametrize(
    "payload",
    [b"not a zip", _zip_payload([])[:-10], _corrupt_deflate()],
    ids=["не zip", "обрезанный zip", "испорченный поток"],
)
def test_fetch_day_treats_a_damaged_file_as_a_temporary_failure(
    monkeypatch: pytest.MonkeyPatch, payload: bytes
) -> None:
    monkeypatch.setattr(gdelt.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload))
    with pytest.raises(gdelt.SourceUnavailable, match="повреждён"):
        gdelt.fetch_day(pd.Timestamp("2024-01-05"))


def test_fetch_day_treats_an_empty_archive_as_a_temporary_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = io.BytesIO()
    zipfile.ZipFile(buffer, "w").close()
    empty = buffer.getvalue()
    monkeypatch.setattr(gdelt.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(empty))
    with pytest.raises(gdelt.SourceUnavailable, match="повреждён"):
        gdelt.fetch_day(pd.Timestamp("2024-01-05"))


def test_news_step_is_skipped_without_the_archive_and_fails_on_an_incomplete_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sbx.shell.pipelines import features

    manifest = _reference_archive(tmp_path)
    load_archive = gdelt.load_archive
    monkeypatch.setattr(gdelt, "load_manifest", lambda: manifest)
    monkeypatch.setattr(gdelt, "load_archive", lambda m: load_archive(m, tmp_path))
    months = pd.DatetimeIndex(["2024-01-01", "2024-02-01"])

    (tmp_path / FEBRUARY).unlink()
    with pytest.raises(gdelt.ArchiveMismatch, match="make gdelt"):
        features.news_step(months, out_dir=tmp_path / "news")

    (tmp_path / JANUARY).unlink()
    skipped = features.news_step(months, out_dir=tmp_path / "news")
    assert isinstance(skipped, str) and "пропущен" in skipped and "make gdelt" in skipped
    assert not (tmp_path / "news").exists(), "без архива новостные признаки не пишутся"


def test_cli_check_reports_the_archive_state_by_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from sbx.shell import cli

    manifest = _reference_archive(tmp_path)
    validate = gdelt.validate
    monkeypatch.setattr(gdelt, "load_manifest", lambda: manifest)
    monkeypatch.setattr(gdelt, "validate", lambda m: validate(m, tmp_path))

    result = CliRunner().invoke(cli.app, ["data", "gdelt", "--check"])
    assert result.exit_code == 0
    assert "совпадают с манифестом" in result.output

    (tmp_path / FEBRUARY).unlink()
    result = CliRunner().invoke(cli.app, ["data", "gdelt", "--check"])
    assert result.exit_code == 1
    assert "gdelt_rs_202402.parquet: файл отсутствует" in result.output


def test_cli_download_fails_while_the_archive_is_incomplete_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Код выхода 1 останавливает `make all` на этом шаге: дальше с частью архива идти нельзя."""
    from typer.testing import CliRunner

    from sbx.shell import cli

    manifest = _reference_archive(tmp_path)
    (tmp_path / JANUARY).unlink()
    source = FakeSource({"2024-01-03": [gdelt.SourceUnavailable("нет сети")] * 10})
    sync, pending = gdelt.sync, gdelt.pending_months
    monkeypatch.setattr(gdelt, "load_manifest", lambda: manifest)
    monkeypatch.setattr(gdelt, "pending_months", lambda m: pending(m, tmp_path))
    monkeypatch.setattr(
        gdelt,
        "sync",
        lambda m, progress: sync(
            m, tmp_path, fetcher=source, pause=0, sleep=_no_sleep, progress=progress
        ),
    )

    result = CliRunner().invoke(cli.app, ["data", "gdelt"])
    assert result.exit_code == 1
    assert "нужно скачать месяцев: 1 из 2" in result.output
    assert "2024-01-03" in result.output and "make gdelt" in result.output

    # Источник снова отвечает: повторный запуск докачивает недостающий месяц.
    source.failures.clear()
    result = CliRunner().invoke(cli.app, ["data", "gdelt"])
    assert result.exit_code == 0
    assert "скачан 2024-01" in result.output
    assert gdelt.validate(manifest, tmp_path) == []
