"""Численность населения Росстата: загрузка бюллетеней, таблица в репозитории и её сверка."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from sbx.shell.download import rosstat

MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG = "http://schemas.openxmlformats.org/package/2006/relationships"


def _letters(index: int) -> str:
    return "ABCDEFGHIJ"[index]


def _xlsx(rows: list[list[str]]) -> bytes:
    """Книга с одним листом «Численность_по_МО»; ячейки — строки, как их отдаёт читатель."""
    body = "".join(
        f'<row r="{r}">'
        + "".join(
            f'<c r="{_letters(c)}{r}" t="inlineStr"><is><t>{value}</t></is></c>'
            for c, value in enumerate(row)
            if value != ""
        )
        + "</row>"
        for r, row in enumerate(rows, start=1)
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as book:
        book.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{MAIN}" xmlns:r="{REL}"><sheets>'
            '<sheet name="Численность_по_МО" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        book.writestr(
            "xl/_rels/workbook.xml.rels",
            f'<Relationships xmlns="{PKG}"><Relationship Id="rId1" '
            'Target="worksheets/sheet1.xml"/></Relationships>',
        )
        book.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{MAIN}"><sheetData>{body}</sheetData></worksheet>',
        )
    return buffer.getvalue()


CODED = _xlsx(
    [
        ["7900000000", "Республика Адыгея", "500591", "243610", "256981"],
        ["7970100000", "Городской округ - Город Майкоп", "161898", "137965", "23933"],
        ["7960500000", "Гиагинский муниципальный район", "31872", "0", "31872"],
        ["0100000000", "Алтайский край", "2115308", "1237124", "878184"],
        ["01701000 0 0", "Городской округ г Барнаул", "687601", "638173", "49428"],
    ]
)
NAMED = _xlsx(
    [
        ["Республика Адыгея", "497985", "243944", "254041"],
        ["Городской округ - Город Майкоп", "163766", "139687", "24079"],
        ["Гиагинский муниципальный район", "31933", "0", "31933"],
        ["Алтайский край", "2130950", "1241693", "889257"],
        ["Городской округ город Барнаул", "695003", "644852", "50151"],
    ]
)
TERRITORIES = pd.DataFrame(
    {
        "official_id": [1, 2, 3],
        "territory_id": ["79-0001", "79-0002", "01-0003"],
        "oktmo": ["79701000", "79605000", "01701000"],
    }
)


class Site:
    """Сайт-заглушка: отдаёт бюллетень по году в адресе и помнит запросы."""

    def __init__(self) -> None:
        self.asked: list[str] = []
        self.values = NAMED

    def __call__(self, url: str) -> bytes:
        self.asked.append(url)
        return self.values if "BUL_MO_2023" in url else CODED


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Site:
    site = Site()
    monkeypatch.setattr(rosstat, "BULLETINS_DIR", tmp_path / "external")
    monkeypatch.setattr(rosstat, "TABLE_PATH", tmp_path / "population_2023.csv")
    monkeypatch.setattr(rosstat, "MANIFEST_PATH", tmp_path / "rosstat_manifest.yaml")
    monkeypatch.setattr(rosstat, "VALUES_SHA256", hashlib.sha256(NAMED).hexdigest())
    monkeypatch.setattr(rosstat, "fetch", site)
    return site


def test_sync_downloads_both_bulletins_and_writes_the_table_with_its_manifest(home: Site) -> None:
    record = rosstat.sync(TERRITORIES, downloaded="2026-10-02")
    assert home.asked == [
        "https://web.archive.org/web/20230904080244id_/"
        "https://rosstat.gov.ru/storage/mediabank/BUL_MO_2023.xlsx",
        "https://rosstat.gov.ru/storage/mediabank/BUL_MO_2024.xlsx",
    ], "значения — из редакции, сохранённой архивом интернета; коды — с сайта Росстата"
    table = pd.read_csv(rosstat.TABLE_PATH, dtype={"oktmo": str})
    assert table["official_id"].tolist() == [1, 2], "Барнаул в бюллетене 2023 назван иначе"
    assert table["total"].tolist() == [163766, 31933]
    assert record["territories"] == 3 and record["matched"] == 2
    assert record["share"] == pytest.approx(2 / 3)
    assert record["values_year"] == 2023 and record["codes_year"] == 2024
    assert record["values_published"] == "2023-09-04"
    assert [item["file"] for item in record["bulletins"]] == [
        "BUL_MO_2023_20230904.xlsx",
        "BUL_MO_2024.xlsx",
    ]
    assert [item["url"] for item in record["bulletins"]] == home.asked
    assert record["bulletins"][0]["sha256"] == rosstat.VALUES_SHA256
    assert all(len(item["sha256"]) == 64 for item in record["bulletins"])
    assert "Росстат" in record["citation"] and record["downloaded"] == "2026-10-02"
    assert rosstat.validate(rosstat.MANIFEST_PATH, rosstat.TABLE_PATH) == []
    pd.testing.assert_frame_equal(rosstat.load_table(rosstat.TABLE_PATH), table)


def test_sync_reuses_the_bulletins_already_on_disk(home: Site) -> None:
    rosstat.sync(TERRITORIES)
    home.asked.clear()
    rosstat.sync(TERRITORIES)
    assert home.asked == [], "бюллетени уже скачаны — сеть не нужна"


def test_sync_refuses_a_values_bulletin_that_is_not_the_pinned_edition(home: Site) -> None:
    """Росстат заменял файл бюллетеня. В расчёт идёт редакция, опубликованная до первого
    момента прогноза; файл с другой суммой — не она, и таблица из него не строится."""
    home.values = CODED
    with pytest.raises(ValueError, match="не та редакция"):
        rosstat.sync(TERRITORIES)
    assert not rosstat.TABLE_PATH.exists() and not rosstat.MANIFEST_PATH.exists()
    assert not list(rosstat.BULLETINS_DIR.glob("BUL_MO_2023*")), "чужой файл на диск не кладётся"

    home.values = NAMED
    rosstat.sync(TERRITORIES)
    rosstat.bulletin_path(rosstat.VALUES_YEAR).write_bytes(CODED)
    with pytest.raises(ValueError, match="не та редакция"):
        rosstat.sync(TERRITORIES)


def test_the_tracked_table_comes_from_the_edition_published_before_the_first_forecast() -> None:
    """Правило времени для постоянной характеристики: значение опубликовано до первого момента
    прогноза. Редакция закреплена по sha256, и манифест в репозитории называет именно её."""
    import pandas as pd

    from sbx.core.folds import make_folds
    from sbx.shell.io import read_yaml
    from sbx.shell.pipelines.backtest import load_backtest_config

    first_forecast = min(f.cutoff for f in make_folds(load_backtest_config()))
    first_forecast = first_forecast + pd.offsets.MonthEnd(0)
    assert pd.Timestamp(rosstat.VALUES_PUBLISHED) < first_forecast
    assert rosstat.VALUES_PUBLISHED.replace("-", "") in rosstat.VALUES_URL
    manifest = read_yaml(rosstat.MANIFEST_PATH)
    assert manifest["values_published"] == rosstat.VALUES_PUBLISHED
    assert manifest["bulletins"][0]["sha256"] == rosstat.VALUES_SHA256
    assert manifest["bulletins"][0]["url"] == rosstat.VALUES_URL


def test_validate_reports_a_changed_or_missing_table(home: Site) -> None:
    assert rosstat.validate(rosstat.MANIFEST_PATH, rosstat.TABLE_PATH) == [], (
        "источник не подключён"
    )
    assert rosstat.load_table(rosstat.TABLE_PATH) is None
    rosstat.sync(TERRITORIES)
    rosstat.TABLE_PATH.write_text("official_id,total\n1,1\n", encoding="utf-8")
    assert (
        "не совпадает с манифестом"
        in rosstat.validate(rosstat.MANIFEST_PATH, rosstat.TABLE_PATH)[0]
    )
    rosstat.TABLE_PATH.unlink()
    assert "нет файла" in rosstat.validate(rosstat.MANIFEST_PATH, rosstat.TABLE_PATH)[0]


def test_cli_population_builds_the_table_and_says_how_many_territories_matched(
    home: Site, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from sbx.shell import cli
    from sbx.shell.pipelines import features

    monkeypatch.setattr(features, "load_territories", lambda: TERRITORIES)
    absent = CliRunner().invoke(cli.app, ["data", "population", "--check"])
    assert absent.exit_code == 0 and "таблицы населения нет" in absent.output

    result = CliRunner().invoke(cli.app, ["data", "population"])
    assert result.exit_code == 0, result.output
    assert "привязано территорий: 2 из 3 (66,7%)" in result.output

    check = CliRunner().invoke(cli.app, ["data", "population", "--check"])
    assert check.exit_code == 0 and "совпадает с манифестом" in check.output
    rosstat.TABLE_PATH.write_text("broken", encoding="utf-8")
    broken = CliRunner().invoke(cli.app, ["data", "population", "--check"])
    assert broken.exit_code == 1 and "не совпадает" in broken.output
    overall = CliRunner().invoke(cli.app, ["data", "validate"])
    assert overall.exit_code == 1 and "население" in overall.output
