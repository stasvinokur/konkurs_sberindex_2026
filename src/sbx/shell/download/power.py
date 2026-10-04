"""Погода NASA POWER: докачка месячных значений по ячейкам сетки и таблица для репозитория.

Запрос — один на ячейку сетки MERRA-2, в которую попадает центр МО (около 1 300 ячеек на всю
панель, секунда на запрос). Ответы складываются по файлу на ячейку в `data/external/weather`
(вне git), поэтому прерванная докачка продолжается с того же места. Из ответов собирается
таблица «ячейка × месяц» с нормами; она лежит в репозитории и сверяется с манифестом, так что
расчёту сеть не нужна.

Условия источника (https://power.larc.nasa.gov/docs/referencing/): в работе приводятся две
подписи — о проекте (`CITATION`) и о версии сервиса с датой обращения (`data_reference` в
манифесте). Проект также просит сообщать ему о публикациях и о передаче данных другим
исследователям — это просьба об уведомлении, а не ограничение; см. `DATA_LICENSES.md`.
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from sbx.core.weather import CELL, parse_power, territory_cells, with_norms
from sbx.shell.download.sberindex import ssl_context
from sbx.shell.io import CONFIG_DIR, DATA_DIR, read_yaml, sha256_file, write_yaml

CONFIG_PATH = CONFIG_DIR / "weather.yaml"
CELLS_DIR = DATA_DIR / "external" / "weather" / "cells"
TABLE_PATH = DATA_DIR / "reference" / "weather" / "power_monthly.parquet"
MANIFEST_PATH = DATA_DIR / "weather_manifest.yaml"
CITATION = (
    "The data was obtained from National Aeronautics and Space Administration (NASA) Langley "
    "Research Center's Prediction Of Worldwide Energy Resources (POWER) project funded through "
    "the NASA Earth Science Division."
)
Fetch = Callable[[float, float, Mapping[str, Any]], Mapping[str, Any]]
# Пауза между запросами, секунды: источник открытый и без ключа, нагружать его незачем.
PAUSE = 0.2


class CellsMissing(RuntimeError):
    """Часть ячеек не скачана: таблица по неполному набору не строится."""


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return dict(read_yaml(path))


def cell_path(directory: Path, lat: float, lon: float) -> Path:
    return Path(directory) / f"{lat:+07.3f}_{lon:+08.3f}.json"


def request_url(lat: float, lon: float, cfg: Mapping[str, Any]) -> str:
    return (
        f"{cfg['api']}?parameters={','.join(cfg['parameters'])}&community={cfg['community']}"
        f"&longitude={lon}&latitude={lat}&start={cfg['start_year']}&end={cfg['end_year']}"
        "&format=JSON"
    )


def fetch_cell(lat: float, lon: float, cfg: Mapping[str, Any], timeout: int = 90) -> dict[str, Any]:
    request = urllib.request.Request(
        request_url(lat, lon, cfg), headers={"User-Agent": "sbx-weather"}
    )
    with urllib.request.urlopen(request, timeout=timeout, context=ssl_context()) as response:
        return json.loads(response.read().decode("utf-8"))


def download(
    cells: pd.DataFrame,
    directory: Path,
    cfg: Mapping[str, Any],
    fetch: Fetch | None = None,
    pause: float | None = None,
    attempts: int = 3,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Скачивает ячейки, которых ещё нет на диске; отказ на одной ячейке не останавливает остальные.

    Ответ пишется во временный файл и переименовывается: прерванная запись не оставляет
    половины файла, который следующий запуск принял бы за готовый.
    """
    fetch = fetch_cell if fetch is None else fetch
    pause = PAUSE if pause is None else pause
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    points = [(float(lat), float(lon)) for lat, lon in cells[CELL].drop_duplicates().to_numpy()]
    downloaded, cached, failed = 0, 0, []
    for number, (lat, lon) in enumerate(points, start=1):
        path = cell_path(directory, lat, lon)
        if path.exists():
            cached += 1
        else:
            payload = None
            for attempt in range(attempts):
                try:
                    payload = fetch(lat, lon, cfg)
                    break
                except (OSError, ValueError):
                    time.sleep(pause * (attempt + 1))
            if payload is None:
                failed.append((lat, lon))
            else:
                partial = path.with_suffix(".part")
                partial.write_text(json.dumps(payload), encoding="utf-8")
                partial.replace(path)
                downloaded += 1
                time.sleep(pause)
        if progress:
            progress(number, len(points))
    return {"cells": len(points), "downloaded": downloaded, "cached": cached, "failed": failed}


def build_table(cells: pd.DataFrame, directory: Path, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Таблица «ячейка × месяц» с нормами — только месяцы, нужные расчёту (`keep_from`).

    Нормы считаются по всем скачанным годам из `norm_years`, а в таблицу идут месяцы панели и
    запас перед ней: годы нормы в репозитории не нужны.
    """
    points = [(float(lat), float(lon)) for lat, lon in cells[CELL].drop_duplicates().to_numpy()]
    missing = [point for point in points if not cell_path(directory, *point).exists()]
    if missing:
        raise CellsMissing(
            f"не скачано ячеек: {len(missing)} из {len(points)} — докачать: sbx data weather"
        )
    frames = []
    for lat, lon in points:
        payload = json.loads(cell_path(directory, lat, lon).read_text(encoding="utf-8"))
        frames.append(parse_power(payload).assign(lat=lat, lon=lon))
    first, last = (int(year) for year in cfg["norm_years"])
    table = with_norms(pd.concat(frames, ignore_index=True), first, last)
    table = table[table["ds"] >= pd.Timestamp(cfg["keep_from"])]
    columns = [*CELL, "ds", "t2m", "prectot", "t2m_norm", "prectot_norm"]
    return table[columns].sort_values([*CELL, "ds"]).reset_index(drop=True)


def service_version(directory: Path) -> str:
    """Название и версия сервиса из ответов источника: «Monthly and Annual 2.10.0»."""
    first = sorted(Path(directory).glob("*.json"))[0]
    api = json.loads(first.read_text(encoding="utf-8"))["header"]["api"]
    name = str(api["name"]).removeprefix("POWER ").removesuffix(" API")
    return f"{name} {str(api['version']).removeprefix('v')}"


def write(
    table: pd.DataFrame,
    cfg: Mapping[str, Any],
    path: Path = TABLE_PATH,
    manifest_path: Path = MANIFEST_PATH,
    downloaded: str | None = None,
    version: str = "",
) -> dict[str, Any]:
    """Записывает таблицу и манифест с её sha256 — эталон сверки."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    downloaded = downloaded or f"{pd.Timestamp.now():%Y-%m-%d}"
    record = {
        "source": "NASA POWER, реанализ MERRA-2: месячные значения по ячейкам сетки",
        "api": str(cfg["api"]),
        "parameters": list(cfg["parameters"]),
        "community": str(cfg["community"]),
        "norm_years": [int(year) for year in cfg["norm_years"]],
        "cells": int(len(table[CELL].drop_duplicates())),
        "rows": int(len(table)),
        "first_month": f"{table['ds'].min():%Y-%m}",
        "last_month": f"{table['ds'].max():%Y-%m}",
        "downloaded": downloaded,
        "file": path.name,
        "sha256": sha256_file(path),
        "citation": CITATION,
        "data_reference": (
            f"The data was obtained from the POWER Project's {version} version on "
            f"{downloaded.replace('-', '/')}."
        ),
    }
    write_yaml(manifest_path, record)
    return record


def sync(
    territories: pd.DataFrame, progress: Callable[[int, int], None] | None = None
) -> dict[str, Any]:
    """Докачивает ячейки территорий и, когда скачаны все, пишет таблицу и манифест.

    Пока хоть одна ячейка не скачана, таблица не пишется: прежняя остаётся как была.
    """
    cfg = load_config()
    cells = territory_cells(territories, cfg["city_centers"])
    result = download(cells, CELLS_DIR, cfg, progress=progress)
    if result["failed"]:
        return result
    table = build_table(cells, CELLS_DIR, cfg)
    record = write(table, cfg, TABLE_PATH, MANIFEST_PATH, version=service_version(CELLS_DIR))
    return {**result, "manifest": record}


def validate(manifest_path: Path = MANIFEST_PATH, path: Path = TABLE_PATH) -> list[str]:
    """Сверка таблицы с манифестом. Манифеста нет — источник не подключён, сверять нечего."""
    if not Path(manifest_path).exists():
        return []
    actual = sha256_file(path)
    if actual is None:
        return [f"погода: нет файла {path}"]
    if actual != read_yaml(manifest_path)["sha256"]:
        return [f"погода: {path} не совпадает с манифестом {manifest_path}"]
    return []


def load_table(path: Path = TABLE_PATH) -> pd.DataFrame | None:
    return pd.read_parquet(path) if Path(path).exists() else None
