"""Загрузка датасетов СберИндекса по публичным ссылкам выгрузки и сверка с манифестом."""

from __future__ import annotations

import shutil
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import certifi

from sbx.core.data_manifest import (
    DataManifest,
    DatasetSpec,
    FileSpec,
    download_url,
    verify_file,
)
from sbx.shell.io import DATA_DIR, load_config, parquet_rows, sha256_file

MANIFEST_PATH = DATA_DIR / "manifest.yaml"
USER_AGENT = "Mozilla/5.0 (sbx reproducibility downloader)"

DATASET_TITLES = {
    "potrebitelskie-beznalicnye-rashody-na-urovne-munizipalnyh-obrazovanij": "Потребительские безналичные расходы на уровне МО",
    "consumer-spending": "Потребительские расходы",
    "consumer-spending-growth": "Потребительские расходы, приросты",
    "consumper-spending-index-sa": "Индекс реальных потребительских расходов",
    "ver-izmenenie-trat-po-kategoriyam": "Изменение потребительских расходов (недельное)",
    "potrebitelskaya-aktivnost-po-kategoriyam-tovarov-v-razreze-vozrastov": "Потребительская активность",
    "indeks-mobilnosti": "Индекс мобильности по СЗФО",
    "median-wages": "Медианная заработная плата",
    "oboroty-biznesa": "Обороты бизнеса",
    "real-key-interest-rate": "Ключевая ставка в реальном выражении",
    "nedelnaa-inflazia-v-razreze-analiticeskih-komponentov": "Недельная инфляция по аналитическим компонентам",
    "izmenenie-obema-fot": "Объём фонда оплаты труда",
}
URL_TEMPLATE = "https://sberindex.ru/api/dataset/v1//download/{slug}/{fmt}"


@dataclass
class SyncReport:
    downloaded: list[str] = field(default_factory=list)
    ok: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def load_manifest(path: Path = MANIFEST_PATH) -> DataManifest:
    return load_config(DataManifest, path)


def fetch(url: str, dest: Path, timeout: int = 120) -> None:
    """Скачивает URL во временный файл и атомарно переносит в `dest`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
        with urllib.request.urlopen(request, timeout=timeout, context=ssl_context()) as response:
            shutil.copyfileobj(response, tmp)
    tmp_path.replace(dest)


def check_file(spec: FileSpec, data_dir: Path = DATA_DIR) -> list[str]:
    path = data_dir / spec.path
    return verify_file(spec, sha256_file(path), parquet_rows(path))


def validate(manifest: DataManifest, data_dir: Path = DATA_DIR) -> list[str]:
    return [p for spec in manifest.all_files() for p in check_file(spec, data_dir)]


def sync(manifest: DataManifest, data_dir: Path = DATA_DIR, fetcher=fetch) -> SyncReport:
    """Докачивает отсутствующие файлы. Существующие файлы не перезаписываются.

    Если скачанный файл не совпадает с манифестом, он сохраняется рядом с суффиксом
    `.downloaded`, а расхождение попадает в отчёт.
    """
    report = SyncReport()
    for spec in manifest.all_files():
        path = data_dir / spec.path
        if path.exists():
            problems = check_file(spec, data_dir)
            (report.problems.extend(problems) if problems else report.ok.append(spec.path))
            continue
        tmp_target = path.with_name(path.name + ".downloaded")
        fetcher(spec.url, tmp_target)
        problems = verify_file(spec, sha256_file(tmp_target), parquet_rows(tmp_target))
        if problems:
            report.problems.extend(
                p + f" (скачанный файл сохранён как {tmp_target.name})" for p in problems
            )
        else:
            tmp_target.replace(path)
            report.downloaded.append(spec.path)
    return report


def build_manifest(data_dir: Path = DATA_DIR) -> dict:
    """Строит манифест по файлам, лежащим в `data/raw/<slug>/`."""
    datasets = []
    for slug, title in DATASET_TITLES.items():
        files = []
        for fmt, suffix in (("csv", ".zip"), ("parquet", ".parquet")):
            rel = Path("raw") / slug / f"{slug}{suffix}"
            path = data_dir / rel
            if not path.exists():
                continue
            files.append(
                {
                    "path": rel.as_posix(),
                    "url": download_url(URL_TEMPLATE, slug, fmt),
                    "sha256": sha256_file(path),
                    "rows": parquet_rows(path),
                }
            )
        datasets.append({"slug": slug, "title": title, "files": files})
    return {
        "source": "Данные СберИндекса (https://sberindex.ru)",
        "download_url_template": URL_TEMPLATE,
        "datasets": datasets,
    }


__all__ = ["DatasetSpec", "SyncReport", "build_manifest", "load_manifest", "sync", "validate"]


def ssl_context():
    """CA-хранилище certifi: в изолированном окружении uv системные сертификаты недоступны."""
    import ssl

    return ssl.create_default_context(cafile=certifi.where())
