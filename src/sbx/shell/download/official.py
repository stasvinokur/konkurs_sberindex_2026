"""Файлы организаторов конкурса: загрузка, сверка с манифестом и чтение таблиц.

Основной набор данных конкурса — архив Лаборатории СберИндекс: расходы на уровне МО с
официальным `territory_id`, индекс доступности рынков и транспортные связи между МО. К нему
относится справочник территорий. Файлы лежат в репозитории в том виде, в каком опубликованы,
и сверяются по `data/official_manifest.yaml`; сеть нужна только чтобы получить их заново.

sberbank.com подписан национальным УЦ Минцифры, которого нет в хранилище certifi. Корневой
сертификат лежит в репозитории, а его отпечаток закреплён здесь: проверка цепочки не
отключается, и подменённый файл сертификата не станет доверенным.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import ssl
import subprocess
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

import certifi
import pandas as pd

from sbx.core.official_data import OfficialManifest, file_problems, manifest_problems
from sbx.core.territories import reference_frame
from sbx.core.xlsx import read_sheet
from sbx.shell.download.sberindex import USER_AGENT, SyncReport
from sbx.shell.io import DATA_DIR, load_config, sha256_file

MANIFEST_PATH = DATA_DIR / "official_manifest.yaml"
ARCHIVE = DATA_DIR / "raw" / "sberindex-municipal" / "hackathonlicence.zip"
REFERENCE_CSV = DATA_DIR / "reference" / "sberindex_municipal_districts.csv"
TRUSTED_ROOT_PATH = DATA_DIR / "reference" / "russian_trusted_root_ca.pem"
# SHA-256 сертификата «Russian Trusted Root CA» (Минцифры России), действует до 2032-02-27.
TRUSTED_ROOT_SHA256 = "D26D2D0231B7C39F92CC738512BA54103519E4405D68B5BD703E9788CA8ECF31"
# Промежуточный сертификат того же удостоверяющего центра (выпущен в июле 2024 года, серийный
# номер 1005). Часть сайтов — например rosstat.gov.ru — его не присылает, и без него цепочку
# до корня не построить.
TRUSTED_SUB_CA_PATH = DATA_DIR / "reference" / "russian_trusted_sub_ca_2024.pem"
TRUSTED_SUB_CA_SHA256 = "2155785036C900DBB5F1BB2A1569C80C55595BD6BF94867A29BBDDBC7D88A3F2"

Fetcher = Callable[[str], bytes]
Extractor = Callable[[bytes, str], bytes]


def load_manifest(path: Path = MANIFEST_PATH) -> OfficialManifest:
    return load_config(OfficialManifest, path)


def validate(manifest: OfficialManifest, data_dir: Path = DATA_DIR) -> list[str]:
    """Расхождения файлов на диске с манифестом; пустой список — всё на месте."""
    actual = {spec.path: sha256_file(data_dir / spec.path) for spec in manifest.files}
    return manifest_problems(manifest, actual)


def _pinned(path: Path, expected: str) -> str:
    """Читает сертификат и сверяет его отпечаток с закреплённым в коде."""
    pem = Path(path).read_text(encoding="ascii")
    digest = hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem.strip())).hexdigest().upper()
    if digest != expected:
        raise ValueError(
            f"{path}: отпечаток сертификата {digest[:16]}… не совпадает с закреплённым"
        )
    return pem


def trusted_context(
    pem_path: Path = TRUSTED_ROOT_PATH, sub_ca_path: Path = TRUSTED_SUB_CA_PATH
) -> ssl.SSLContext:
    """Обычные корневые сертификаты плюс корень Минцифры с закреплённым отпечатком.

    Промежуточный сертификат лежит рядом и тоже закреплён по отпечатку: сервер может его не
    прислать. Якорем доверия он не становится — неполная цепочка запрещена, поэтому проверка
    обязана дойти до самоподписанного корня, которым этот сертификат подписан.
    """
    root = _pinned(pem_path, TRUSTED_ROOT_SHA256)
    sub_ca = _pinned(sub_ca_path, TRUSTED_SUB_CA_SHA256)
    context = ssl.create_default_context(cafile=certifi.where())
    context.load_verify_locations(cadata=root)
    context.load_verify_locations(cadata=sub_ca)
    context.verify_flags &= ~ssl.VERIFY_X509_PARTIAL_CHAIN
    return context


def fetch(url: str, timeout: int = 300) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout, context=trusted_context()) as response:
        return response.read()


def extract_rar_member(payload: bytes, member: str) -> bytes:
    """Достаёт файл из RAR-архива в памяти; нужен `bsdtar` (libarchive)."""
    tool = shutil.which("bsdtar")
    if tool is None:
        raise RuntimeError(
            "для распаковки архива справочника нужен bsdtar (пакет libarchive-tools); "
            "сам справочник уже лежит в репозитории, распаковка нужна только для перезагрузки"
        )
    return subprocess.run(
        [tool, "-xOf", "-", member], input=payload, capture_output=True, check=True
    ).stdout


def reference_csv_bytes(xlsx_payload: bytes) -> bytes:
    """Справочник территорий из xlsx в CSV: все строки и колонки, без преобразований."""
    frame = reference_frame(read_sheet(xlsx_payload))
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def sync(
    manifest: OfficialManifest,
    data_dir: Path = DATA_DIR,
    fetcher: Fetcher = fetch,
    extractor: Extractor = extract_rar_member,
) -> SyncReport:
    """Получает отсутствующие файлы. Существующие не перезаписываются.

    Скачанный файл, не совпавший с манифестом, сохраняется рядом с суффиксом `.downloaded`.
    """
    report = SyncReport()
    for spec in manifest.files:
        path = data_dir / spec.path
        if path.exists():
            problems = file_problems(spec, sha256_file(path))
            (report.problems.extend(problems) if problems else report.ok.append(spec.path))
            continue
        if spec.derived_from:
            payload = reference_csv_bytes((data_dir / spec.derived_from).read_bytes())
        else:
            payload = fetcher(spec.url)
            if spec.archive_member:
                if hashlib.sha256(payload).hexdigest() != spec.archive_sha256:
                    report.problems.append(
                        f"{spec.path}: архив по адресу {spec.url} не совпадает с манифестом"
                    )
                    continue
                payload = extractor(payload, spec.archive_member)
        path.parent.mkdir(parents=True, exist_ok=True)
        problems = file_problems(spec, hashlib.sha256(payload).hexdigest())
        if problems:
            aside = path.with_name(path.name + ".downloaded")
            aside.write_bytes(payload)
            report.problems.extend(
                p + f" (полученный файл сохранён как {aside.name})" for p in problems
            )
            continue
        path.write_bytes(payload)
        report.downloaded.append(spec.path)
    return report


def load_reference(path: Path = REFERENCE_CSV) -> pd.DataFrame:
    """Справочник территорий: все версии всех территорий, ключи — целые числа."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in ("territory_id", "year_from", "year_to"):
        frame[column] = frame[column].astype(int)
    return frame


def _table(name: str, archive: Path = ARCHIVE) -> pd.DataFrame:
    """Таблица из архива организаторов; читается прямо из zip, без распаковки на диск."""
    with zipfile.ZipFile(archive) as book:
        member = next(m for m in book.namelist() if m.rsplit("/", 1)[-1] == f"{name}.parquet")
        return pd.read_parquet(io.BytesIO(book.read(member))).reset_index(drop=True)


def load_consumption(archive: Path = ARCHIVE) -> pd.DataFrame:
    """Расходы: `date` (ГГГГ-ММ), официальный `territory_id`, `category`, `value` (руб.)."""
    return _table("consumption", archive)


def load_market_access(archive: Path = ARCHIVE) -> pd.DataFrame:
    return _table("market_access", archive)


def load_connection(archive: Path = ARCHIVE) -> pd.DataFrame:
    return _table("connection", archive)
