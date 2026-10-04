"""Численность населения МО: бюллетени Росстата и таблица для репозитория.

Значения — оценка на 1 января 2023 года; коды — из бюллетеня на 1 января 2024 года (в бюллетене
2023 года их нет). Бюллетени скачиваются в `data/external/rosstat` (вне git), в репозитории лежит
только таблица численности территорий панели и манифест с долей привязки, sha256 бюллетеней и
самой таблицы. Расчёту сеть не нужна.

Росстат несколько раз заменял файл бюллетеня 2023 года (нынешний датирован 29 февраля 2024 года).
Постоянная характеристика идёт в прогноз, только если была опубликована до первого момента
прогноза, поэтому значения берутся из редакции, которую архив интернета сохранил 4 сентября 2023
года; её sha256 закреплён здесь, и файл с другой суммой в расчёт не идёт.

Сайт rosstat.gov.ru подписан национальным удостоверяющим центром; загрузка идёт с проверкой
цепочки до закреплённого корневого сертификата (`sbx.shell.download.official.trusted_context`).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from sbx.core.rosstat import SHEET, bridge_names, parse_bulletin, population_table
from sbx.core.xlsx import read_sheet
from sbx.shell.download.official import fetch
from sbx.shell.io import DATA_DIR, read_yaml, sha256_file, write_yaml

PAGE = "https://rosstat.gov.ru/compendium/document/13282"
URL = "https://rosstat.gov.ru/storage/mediabank/BUL_MO_{year}.xlsx"
BULLETINS_DIR = DATA_DIR / "external" / "rosstat"
TABLE_PATH = DATA_DIR / "reference" / "rosstat" / "population_2023.csv"
MANIFEST_PATH = DATA_DIR / "rosstat_manifest.yaml"
VALUES_YEAR, CODES_YEAR = 2023, 2024
# Редакция бюллетеня 2023 года: не позже этого дня она была на сайте Росстата (снимок архива
# интернета; сам файл изменён 18 августа 2023 года).
VALUES_PUBLISHED = "2023-09-04"
VALUES_URL = "https://web.archive.org/web/20230904080244id_/" + URL.format(year=VALUES_YEAR)
VALUES_SHA256 = "82f4c08b9340618a7abfb6717e8f6ca5a3852b828586e61ff4013b4b98c23c2d"
CITATION = (
    "Росстат. Численность населения Российской Федерации по муниципальным образованиям на "
    f"1 января {VALUES_YEAR} года (статистический бюллетень). {PAGE}"
)


def bulletin_path(year: int) -> Path:
    stamp = f"_{VALUES_PUBLISHED.replace('-', '')}" if year == VALUES_YEAR else ""
    return BULLETINS_DIR / f"BUL_MO_{year}{stamp}.xlsx"


def bulletin_url(year: int) -> str:
    return VALUES_URL if year == VALUES_YEAR else URL.format(year=year)


def _bulletin(year: int) -> bytes:
    """Бюллетень с диска; если его там нет — из сети, с сохранением на диск.

    Бюллетень со значениями сверяется с закреплённой суммой: другая редакция не принимается и
    на диск не кладётся.
    """
    path = bulletin_path(year)
    payload = path.read_bytes() if path.exists() else fetch(bulletin_url(year))
    if year == VALUES_YEAR and hashlib.sha256(payload).hexdigest() != VALUES_SHA256:
        raise ValueError(
            f"{path.name}: не та редакция бюллетеня — sha256 не совпадает с закреплённым; "
            f"нужна редакция от {VALUES_PUBLISHED} ({VALUES_URL})"
        )
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        partial.write_bytes(payload)
        partial.replace(path)
    return payload


def sync(territories: pd.DataFrame, downloaded: str | None = None) -> dict[str, Any]:
    """Скачивает недостающие бюллетени, привязывает численность к территориям панели и
    записывает таблицу с манифестом (манифест — эталон сверки)."""
    payloads = {year: _bulletin(year) for year in (VALUES_YEAR, CODES_YEAR)}
    coded = parse_bulletin(read_sheet(payloads[CODES_YEAR], SHEET))
    named = parse_bulletin(read_sheet(payloads[VALUES_YEAR], SHEET))
    table = population_table(bridge_names(coded, named), territories)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_PATH, index=False)
    record = {
        "source": "Росстат: численность населения по муниципальным образованиям",
        "page": PAGE,
        "values_year": VALUES_YEAR,
        "values_published": VALUES_PUBLISHED,
        "codes_year": CODES_YEAR,
        "bulletins": [
            {
                "file": bulletin_path(year).name,
                "url": bulletin_url(year),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for year, payload in payloads.items()
        ],
        "territories": int(len(territories)),
        "matched": int(len(table)),
        "share": float(len(table) / len(territories)),
        "downloaded": downloaded or f"{pd.Timestamp.now():%Y-%m-%d}",
        "file": TABLE_PATH.name,
        "sha256": sha256_file(TABLE_PATH),
        "citation": CITATION,
    }
    write_yaml(MANIFEST_PATH, record)
    return record


def validate(manifest_path: Path = MANIFEST_PATH, path: Path = TABLE_PATH) -> list[str]:
    """Сверка таблицы с манифестом. Манифеста нет — источник не подключён, сверять нечего."""
    if not Path(manifest_path).exists():
        return []
    actual = sha256_file(path)
    if actual is None:
        return [f"население: нет файла {path}"]
    if actual != read_yaml(manifest_path)["sha256"]:
        return [f"население: {path} не совпадает с манифестом {manifest_path}"]
    return []


def load_table(path: Path = TABLE_PATH) -> pd.DataFrame | None:
    return pd.read_csv(path, dtype={"oktmo": str}) if Path(path).exists() else None
