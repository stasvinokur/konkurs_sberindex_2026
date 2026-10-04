"""Погода NASA POWER: докачка ячеек, таблица в репозитории и её сверка с манифестом."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sbx.shell.download import power

CONFIG = {
    "api": "https://power.example/api",
    "community": "AG",
    "parameters": ["T2M", "PRECTOTCORR"],
    "start_year": 2019,
    "end_year": 2023,
    "norm_years": [2019, 2020],
    "keep_from": "2023-01-01",
    "city_centers": {"45": [55.7558, 37.6173]},
}
CELLS = pd.DataFrame({"lat": [56.0, 60.0], "lon": [37.5, 30.625]})


def _payload(shift: float) -> dict:
    """Ответ источника: январи и июли 2019–2023 и «тринадцатый месяц»."""
    t2m, rain = {}, {}
    for year in range(2019, 2024):
        t2m[f"{year}01"], t2m[f"{year}07"], t2m[f"{year}13"] = -8.0 + shift, 19.0 + shift, 5.0
        rain[f"{year}01"], rain[f"{year}07"], rain[f"{year}13"] = 1.0, 3.0, 2.0
    t2m["202301"] = -12.0 + shift
    return {
        "header": {
            "fill_value": -999.0,
            "api": {"version": "v2.10.0", "name": "POWER Monthly and Annual API"},
        },
        "properties": {"parameter": {"T2M": t2m, "PRECTOTCORR": rain}},
    }


class Source:
    """Источник-заглушка: помнит запросы и может отказать на заданной ячейке."""

    def __init__(self, failing: tuple[float, float] | None = None) -> None:
        self.asked: list[tuple[float, float]] = []
        self.failing = failing

    def __call__(self, lat: float, lon: float, cfg: dict) -> dict:
        self.asked.append((lat, lon))
        if (lat, lon) == self.failing:
            raise OSError("источник не отвечает")
        return _payload(0.0 if lat == 56.0 else -4.0)


def test_request_names_the_point_the_parameters_and_the_years() -> None:
    url = power.request_url(56.0, 37.5, CONFIG)
    assert url == (
        "https://power.example/api?parameters=T2M,PRECTOTCORR&community=AG"
        "&longitude=37.5&latitude=56.0&start=2019&end=2023&format=JSON"
    )


def test_download_fetches_only_the_cells_that_are_missing(tmp_path: Path) -> None:
    source = Source()
    first = power.download(CELLS, tmp_path, CONFIG, fetch=source, pause=0)
    assert first == {"cells": 2, "downloaded": 2, "cached": 0, "failed": []}
    assert len(list(tmp_path.glob("*.json"))) == 2
    again = Source()
    second = power.download(CELLS, tmp_path, CONFIG, fetch=again, pause=0)
    assert second["downloaded"] == 0 and second["cached"] == 2 and again.asked == []


def test_a_failed_cell_does_not_stop_the_rest_and_is_fetched_next_time(tmp_path: Path) -> None:
    source = Source(failing=(56.0, 37.5))
    first = power.download(CELLS, tmp_path, CONFIG, fetch=source, pause=0, attempts=2)
    assert first["downloaded"] == 1 and first["failed"] == [(56.0, 37.5)]
    assert source.asked.count((56.0, 37.5)) == 2, "две попытки на ячейку"
    assert not power.cell_path(tmp_path, 56.0, 37.5).exists(), "недокачанного файла не остаётся"
    retry = Source()
    second = power.download(CELLS, tmp_path, CONFIG, fetch=retry, pause=0)
    assert retry.asked == [(56.0, 37.5)] and second["failed"] == []


def test_table_holds_the_kept_months_with_the_norms_of_every_cell(tmp_path: Path) -> None:
    power.download(CELLS, tmp_path, CONFIG, fetch=Source(), pause=0)
    table = power.build_table(CELLS, tmp_path, CONFIG)
    assert list(table.columns) == ["lat", "lon", "ds", "t2m", "prectot", "t2m_norm", "prectot_norm"]
    assert table["ds"].min() == pd.Timestamp("2023-01-01"), "годы нормы в таблицу не идут"
    assert len(table) == 4, "две ячейки, январь и июль 2023"
    january = table[table["ds"] == pd.Timestamp("2023-01-01")].set_index("lat")
    assert january.loc[56.0, "t2m"] == -12.0 and january.loc[56.0, "t2m_norm"] == -8.0
    assert january.loc[60.0, "t2m_norm"] == -12.0, "норма у каждой ячейки своя"


def test_table_is_not_built_from_an_incomplete_download(tmp_path: Path) -> None:
    """Таблица по части ячеек выглядела бы как настоящая, но часть МО осталась бы без погоды."""
    power.download(CELLS.iloc[:1], tmp_path, CONFIG, fetch=Source(), pause=0)
    with pytest.raises(power.CellsMissing, match="1 из 2"):
        power.build_table(CELLS, tmp_path, CONFIG)


def test_written_table_is_checked_against_its_manifest(tmp_path: Path) -> None:
    power.download(CELLS, tmp_path / "cells", CONFIG, fetch=Source(), pause=0)
    table = power.build_table(CELLS, tmp_path / "cells", CONFIG)
    path, manifest = tmp_path / "power_monthly.parquet", tmp_path / "weather_manifest.yaml"
    version = power.service_version(tmp_path / "cells")
    assert version == "Monthly and Annual 2.10.0"
    record = power.write(table, CONFIG, path, manifest, downloaded="2026-10-02", version=version)
    assert record["cells"] == 2 and record["rows"] == 4
    assert record["first_month"] == "2023-01" and record["last_month"] == "2023-07"
    assert record["norm_years"] == [2019, 2020]
    # Подписи — в формулировке источника: о проекте и о версии сервиса с датой обращения.
    assert record["citation"] == power.CITATION and record["citation"].startswith(
        "The data was obtained from National Aeronautics and Space Administration (NASA) Langley"
    )
    assert record["data_reference"] == (
        "The data was obtained from the POWER Project's Monthly and Annual 2.10.0 version on "
        "2026/10/02."
    )
    assert power.validate(manifest, path) == []
    pd.testing.assert_frame_equal(power.load_table(path), table)

    path.write_bytes(path.read_bytes() + b"x")
    assert "не совпадает с манифестом" in power.validate(manifest, path)[0]
    path.unlink()
    assert "нет файла" in power.validate(manifest, path)[0]
    assert power.load_table(path) is None, "нет таблицы — нет и блока признаков"
    assert power.validate(tmp_path / "absent.yaml", path) == [], "источник не подключён"


def test_cell_files_are_named_by_their_coordinates(tmp_path: Path) -> None:
    assert power.cell_path(tmp_path, 56.0, 37.5).name == "+56.000_+037.500.json"
    assert power.cell_path(tmp_path, 66.5, -179.375).name == "+66.500_-179.375.json"
    payload = json.dumps(_payload(0.0))
    assert json.loads(payload)["header"]["fill_value"] == -999.0


TERRITORIES = pd.DataFrame(
    {
        "territory_id": ["45-0001", "40-0002", "45-0003"],
        "region_code": ["45", "40", "45"],
        "center_lat": [float("nan"), 59.9386, 55.9],
        "center_lon": [float("nan"), 30.3141, 37.4],
    }
)


@pytest.fixture
def weather_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Каталоги погоды во временной папке и источник-заглушка вместо сети."""
    from sbx.shell.pipelines import features

    monkeypatch.setattr(power, "CELLS_DIR", tmp_path / "cells")
    monkeypatch.setattr(power, "TABLE_PATH", tmp_path / "power_monthly.parquet")
    monkeypatch.setattr(power, "MANIFEST_PATH", tmp_path / "weather_manifest.yaml")
    monkeypatch.setattr(
        power, "load_config", lambda: CONFIG | {"city_centers": {"45": [55.7558, 37.6173]}}
    )
    monkeypatch.setattr(power, "fetch_cell", Source())
    monkeypatch.setattr(power, "PAUSE", 0.0)
    monkeypatch.setattr(features, "load_territories", lambda: TERRITORIES)
    return tmp_path


def test_sync_downloads_the_cells_of_the_territories_and_writes_the_table(
    weather_home: Path,
) -> None:
    result = power.sync(TERRITORIES)
    assert result["cells"] == 2, "две территории Москвы — одна ячейка, Санкт-Петербург — вторая"
    assert result["downloaded"] == 2 and result["failed"] == []
    assert result["manifest"]["rows"] == 4
    assert "Monthly and Annual 2.10.0 version on" in result["manifest"]["data_reference"]
    assert power.validate(power.MANIFEST_PATH, power.TABLE_PATH) == []


def test_sync_writes_no_table_while_cells_are_missing(
    weather_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(power, "fetch_cell", Source(failing=(56.0, 37.5)))
    result = power.sync(TERRITORIES)
    assert result["failed"] == [(56.0, 37.5)] and "manifest" not in result
    assert not power.TABLE_PATH.exists() and not power.MANIFEST_PATH.exists()


def test_cli_weather_downloads_then_checks_the_table(weather_home: Path) -> None:
    from typer.testing import CliRunner

    from sbx.shell import cli

    absent = CliRunner().invoke(cli.app, ["data", "weather", "--check"])
    assert absent.exit_code == 0 and "таблицы погоды нет" in absent.output

    result = CliRunner().invoke(cli.app, ["data", "weather"])
    assert result.exit_code == 0, result.output
    assert "ячеек: 2, скачано: 2" in result.output and "power_monthly.parquet" in result.output

    check = CliRunner().invoke(cli.app, ["data", "weather", "--check"])
    assert check.exit_code == 0 and "совпадает с манифестом" in check.output

    power.TABLE_PATH.write_bytes(b"broken")
    broken = CliRunner().invoke(cli.app, ["data", "weather", "--check"])
    assert broken.exit_code == 1 and "не совпадает" in broken.output
    validate = CliRunner().invoke(cli.app, ["data", "validate"])
    assert validate.exit_code == 1 and "погода" in validate.output, "общая сверка видит и погоду"


def test_cli_weather_fails_when_the_source_does_not_answer(
    weather_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from sbx.shell import cli

    monkeypatch.setattr(power, "fetch_cell", Source(failing=(56.0, 37.5)))
    result = CliRunner().invoke(cli.app, ["data", "weather"])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit), "отказ — сообщением, а не падением"
    assert "не скачано ячеек: 1" in result.output and "повторить" in result.output
