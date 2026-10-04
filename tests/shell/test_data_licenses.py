"""DATA_LICENSES.md: условия использования данных и обязательные подписи источников."""

from __future__ import annotations

from pathlib import Path

from sbx.shell.download import official
from sbx.shell.io import DATA_DIR, ROOT

DOCUMENT = ROOT / "DATA_LICENSES.md"


def _text() -> str:
    return DOCUMENT.read_text(encoding="utf-8")


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_every_required_citation_is_printed_in_full() -> None:
    text = _flat(_text())
    for spec in official.load_manifest().files:
        for citation in spec.citations:
            assert _flat(citation) in text, f"нет цитаты: {citation[:60]}…"


def test_every_data_source_in_the_repository_is_covered() -> None:
    text = _text()
    for name in (
        "CC BY-SA 4.0",
        "Данные СберИндекса",
        "GDELT",
        "Wikidata",
        "CC0",
        "xmlcalendar.ru",
        "NASA POWER",
        "Росстат",
    ):
        assert name in text, f"в документе нет «{name}»"
    tracked = [
        "data/raw/sberindex-municipal/",
        "data/reference/sberindex_municipal_districts.csv",
        "data/reference/wikidata_oktmo_municipal.csv",
        "data/reference/ru_calendar.json",
        "data/reference/mo_filter_order.txt",
        "data/reference/fips_adm1_ru.csv",
        "data/reference/russian_trusted_root_ca.pem",
        "data/benchmark/",
        "data/reference/weather/power_monthly.parquet",
        "data/weather_manifest.yaml",
        "data/reference/rosstat/population_2023.csv",
        "data/rosstat_manifest.yaml",
        "data/reference/russian_trusted_sub_ca_2024.pem",
        "configs/evaluation_series.csv",
        "configs/events.yaml",
        "configs/known_shocks.yaml",
    ]
    for path in tracked:
        assert (ROOT / path).exists(), f"в репозитории нет {path}"
        assert path in text, f"документ не называет {path}"
    reference_files = {p.name for p in (DATA_DIR / "reference").iterdir() if p.is_file()}
    unnamed = sorted(name for name in reference_files if name not in text)
    assert not unnamed, f"файлы data/reference без записи об условиях: {unnamed}"


def test_share_alike_terms_name_the_derived_files_and_the_changes() -> None:
    """CC BY-SA 4.0: производные материалы распространяются на тех же условиях, изменения
    должны быть названы."""
    text = _flat(_text())
    assert "на тех же условиях" in text and "Что изменено" in text
    assert "MIT" in text, "код под MIT, данные — нет: документ обязан это развести"


def test_readme_points_to_the_document() -> None:
    readme = Path(ROOT / "README.md").read_text(encoding="utf-8")
    assert "DATA_LICENSES.md" in readme
