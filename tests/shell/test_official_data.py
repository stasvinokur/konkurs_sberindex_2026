"""Файлы организаторов конкурса: манифест, сверка, загрузка с закреплённым сертификатом, чтение."""

from __future__ import annotations

import hashlib
import ssl
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from sbx.core.config import from_mapping
from sbx.core.official_data import OfficialManifest
from sbx.shell.download import official
from sbx.shell.io import DATA_DIR
from sbx.shell.pipelines.entity import load_raw_spending


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest(files: list[dict]) -> OfficialManifest:
    return from_mapping(OfficialManifest, {"source": "тест", "files": files})


def _entry(path: str, payload: bytes, **extra: str) -> dict:
    return {
        "path": path,
        "url": f"https://example.org/{Path(path).name}",
        "sha256": _sha(payload),
        "title": "набор",
        "license": "CC BY-SA 4.0",
        "citations": ["Набор. СберИндекс."],
        **extra,
    }


def test_repository_official_files_match_manifest() -> None:
    manifest = official.load_manifest()
    assert official.validate(manifest) == []
    paths = [spec.path for spec in manifest.files]
    assert "raw/sberindex-municipal/hackathonlicence.zip" in paths
    assert all(spec.license and spec.citations for spec in manifest.files), (
        "подпись источника обязательна"
    )


def test_manifest_carries_the_citations_required_by_the_licences() -> None:
    """Лицензия CC BY-SA 4.0 требует указания источника; текст цитат задан правообладателем:
    по одной на каждую таблицу архива и одна на справочник территорий."""
    files = {spec.path: spec for spec in official.load_manifest().files}
    archive = files["raw/sberindex-municipal/hackathonlicence.zip"]
    assert [c.split(". СберИндекс.")[0] for c in archive.citations] == [
        "Потребительские безналичные расходы на уровне муниципальных образований по категориям трат",
        "Индекс доступности рынков на уровне муниципальных образований",
        "Автодорожные и железнодорожные связи между муниципальными образованиями",
    ]
    reference = files["raw/sberindex-municipal/t_dict_municipal_districts.xlsx"]
    assert reference.citations == files["reference/sberindex_municipal_districts.csv"].citations
    assert reference.citations[0].startswith(
        "Данные о границах и преобразованиях муниципальных образований. Сбериндекс. "
        "Данные доступны по адресу https://sberindex.ru/ru/research/"
    )
    for spec in files.values():
        assert spec.license == "CC BY-SA 4.0"
        for citation in spec.citations:
            assert citation.endswith(f"(данные скачаны {spec.retrieved}).")
            assert "Данные доступны по адресу https://sberindex.ru/ru/research/" in citation


def test_validate_reports_a_missing_and_a_changed_file(tmp_path: Path) -> None:
    manifest = _manifest([_entry("raw/a.zip", b"published"), _entry("raw/b.zip", b"other")])
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "a.zip").write_bytes(b"tampered")
    problems = official.validate(manifest, tmp_path)
    assert len(problems) == 2
    assert "raw/a.zip: sha256 не совпадает" in problems[0]
    assert "raw/b.zip: файл отсутствует" in problems[1]


def test_sync_fetches_only_missing_files_and_keeps_a_mismatch_aside(tmp_path: Path) -> None:
    manifest = _manifest([_entry("raw/a.zip", b"published"), _entry("raw/b.zip", b"expected")])
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "a.zip").write_bytes(b"published")
    calls: list[str] = []

    def changed_upstream(url: str) -> bytes:
        calls.append(url)
        return b"something else"

    report = official.sync(manifest, tmp_path, fetcher=changed_upstream)

    assert calls == ["https://example.org/b.zip"], "файл, который уже на месте, не скачивается"
    assert report.ok == ["raw/a.zip"] and not report.downloaded
    assert not (tmp_path / "raw" / "b.zip").exists()
    assert (tmp_path / "raw" / "b.zip.downloaded").read_bytes() == b"something else"
    assert "sha256 не совпадает" in report.problems[0]

    report = official.sync(manifest, tmp_path, fetcher=lambda url: b"expected")
    assert report.downloaded == ["raw/b.zip"] and not report.problems
    assert (tmp_path / "raw" / "b.zip").read_bytes() == b"expected"


def test_sync_takes_a_file_out_of_the_published_archive(tmp_path: Path) -> None:
    """Справочник опубликован внутри архива: сверяются и архив, и извлечённый файл."""
    archive, member = b"RAR-ARCHIVE", b"xlsx-bytes"
    entry = _entry("raw/ref.xlsx", member, archive_member="ref.xlsx", archive_sha256=_sha(archive))
    manifest = _manifest([entry])
    seen: list[tuple[bytes, str]] = []

    def extract(payload: bytes, name: str) -> bytes:
        seen.append((payload, name))
        return member

    report = official.sync(manifest, tmp_path, fetcher=lambda url: archive, extractor=extract)
    assert report.downloaded == ["raw/ref.xlsx"] and seen == [(archive, "ref.xlsx")]

    (tmp_path / "raw" / "ref.xlsx").unlink()
    report = official.sync(manifest, tmp_path, fetcher=lambda url: b"OTHER", extractor=extract)
    assert "архив" in report.problems[0] and not (tmp_path / "raw" / "ref.xlsx").exists()


def test_pinned_root_certificate_has_the_documented_fingerprint() -> None:
    pem = official.TRUSTED_ROOT_PATH.read_text(encoding="ascii")
    digest = hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem)).hexdigest().upper()
    assert digest == official.TRUSTED_ROOT_SHA256
    assert isinstance(official.trusted_context(), ssl.SSLContext)


def test_trusted_context_refuses_a_certificate_with_another_fingerprint(tmp_path: Path) -> None:
    """Подменённый файл сертификата не должен молча стать доверенным корнем."""
    import certifi

    other = Path(certifi.where()).read_text(encoding="ascii")
    first = other[
        other.index("-----BEGIN CERTIFICATE-----") : other.index("-----END CERTIFICATE-----") + 25
    ]
    fake = tmp_path / "root.pem"
    fake.write_text(first + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="отпечаток"):
        official.trusted_context(fake)


def test_committed_reference_csv_is_the_converted_official_xlsx() -> None:
    xlsx = DATA_DIR / "raw" / "sberindex-municipal" / "t_dict_municipal_districts.xlsx"
    assert official.reference_csv_bytes(xlsx.read_bytes()) == official.REFERENCE_CSV.read_bytes()
    reference = official.load_reference()
    assert len(reference) == 3101 and reference["territory_id"].nunique() == 2660
    assert reference["territory_id"].dtype.kind == "i"


def test_official_tables_have_the_documented_shape() -> None:
    consumption = official.load_consumption()
    assert list(consumption.columns) == ["date", "territory_id", "category", "value"]
    assert len(consumption) == 303_126 and consumption["territory_id"].nunique() == 2_190
    market = official.load_market_access()
    assert {"territory_id", "market_access"} <= set(market.columns)
    connection = official.load_connection()
    assert set(connection["type"].unique()) == {"highway", "railway"}


def test_official_consumption_equals_the_dashboard_export_as_a_multiset() -> None:
    """Архив организаторов и публичная выгрузка — одни и те же наблюдения; отличие только в том,
    что в архиве территория задана кодом, а в выгрузке — названием."""
    official_rows = official.load_consumption()
    public = load_raw_spending()
    left = Counter(
        zip(
            official_rows["date"],
            official_rows["category"],
            official_rows["value"].astype(int),
            strict=True,
        )
    )
    right = Counter(
        zip(
            pd.to_datetime(public["period"]).dt.strftime("%Y-%m"),
            public["category_15"],
            public["value"].round().astype(int),
            strict=True,
        )
    )
    assert left == right


def test_cli_validate_covers_the_official_files_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """Полный прогон начинается с `make validate`: подменённый архив организаторов должен
    остановить его так же, как подменённая выгрузка СберИндекса."""
    from typer.testing import CliRunner

    from sbx.shell import cli

    result = CliRunner().invoke(cli.app, ["data", "validate"])
    assert result.exit_code == 0
    # Выгрузки СберИндекса, файлы организаторов, таблица погоды и таблица населения.
    total = 24 + len(official.load_manifest().files) + 2
    assert f"все {total} файлов" in result.output

    monkeypatch.setattr(official, "validate", lambda manifest: ["raw/x.zip: sha256 не совпадает"])
    result = CliRunner().invoke(cli.app, ["data", "validate"])
    assert result.exit_code == 1
    assert "raw/x.zip: sha256 не совпадает" in result.output


def test_cli_official_checks_without_network_and_fetches_what_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from sbx.shell import cli

    manifest = _manifest([_entry("raw/a.zip", b"published")])
    validate, sync = official.validate, official.sync
    fetched: list[str] = []

    def fetcher(url: str) -> bytes:
        fetched.append(url)
        return b"published"

    monkeypatch.setattr(official, "load_manifest", lambda: manifest)
    monkeypatch.setattr(official, "validate", lambda m: validate(m, tmp_path))
    monkeypatch.setattr(official, "sync", lambda m: sync(m, tmp_path, fetcher=fetcher))

    result = CliRunner().invoke(cli.app, ["data", "official", "--check"])
    assert result.exit_code == 1 and "raw/a.zip: файл отсутствует" in result.output
    assert fetched == [], "проверка не обращается к сети"

    result = CliRunner().invoke(cli.app, ["data", "official"])
    assert result.exit_code == 0 and "получен: raw/a.zip" in result.output
    assert fetched == ["https://example.org/a.zip"]

    result = CliRunner().invoke(cli.app, ["data", "official", "--check"])
    assert result.exit_code == 0


def test_the_intermediate_certificate_is_pinned_and_issued_by_the_pinned_root() -> None:
    """Сервер Росстата не присылает промежуточный сертификат, поэтому он лежит в репозитории.
    Доверие от этого не расширяется: его отпечаток закреплён, и он подписан закреплённым корнем."""
    import shutil
    import subprocess

    pem = official.TRUSTED_SUB_CA_PATH.read_text(encoding="ascii")
    digest = hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem.strip())).hexdigest().upper()
    assert digest == official.TRUSTED_SUB_CA_SHA256
    if shutil.which("openssl") is None:
        pytest.skip("нет утилиты openssl: подпись промежуточного сертификата не проверить")
    check = subprocess.run(
        [
            "openssl",
            "verify",
            "-CAfile",
            str(official.TRUSTED_ROOT_PATH),
            str(official.TRUSTED_SUB_CA_PATH),
        ],
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0 and check.stdout.strip().endswith(": OK"), check.stderr


def test_trusted_context_anchors_trust_in_the_root_only(tmp_path: Path) -> None:
    """Промежуточный сертификат помогает построить цепочку, но якорем доверия не служит:
    неполная цепочка запрещена, и она обязана дойти до самоподписанного корня."""
    context = official.trusted_context()
    assert not context.verify_flags & ssl.VERIFY_X509_PARTIAL_CHAIN
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    loaded = set(context.get_ca_certs(binary_form=True))
    for path in (official.TRUSTED_ROOT_PATH, official.TRUSTED_SUB_CA_PATH):
        der = ssl.PEM_cert_to_DER_cert(path.read_text(encoding="ascii").strip())
        assert der in loaded, f"{path.name} не загружен — цепочку Росстата не построить"

    other = tmp_path / "sub.pem"
    other.write_text(official.TRUSTED_ROOT_PATH.read_text(encoding="ascii"), encoding="ascii")
    with pytest.raises(ValueError, match="не совпадает с закреплённым"):
        official.trusted_context(sub_ca_path=other)
