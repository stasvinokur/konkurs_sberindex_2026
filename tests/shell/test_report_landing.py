"""Что шаг отчёта передаёт лендингу сверх чисел: шапка, ограничения, ссылки, копия PDF."""

from __future__ import annotations

from pathlib import Path

from sbx.shell.pipelines import report
from tests.core.test_report import DM, WINS_SIGNIFICANCE


def test_headline_keeps_the_report_subtitle_and_splits_it_for_the_page() -> None:
    head = report.headline({"series": 13140, "territories": 2190})
    assert head["subtitle"] == (
        "Онлайн-конкурс СберИндекса 2026, направление «Прогнозирование». Панель: 13 140 рядов "
        "«территория × категория», 2 190 территорий, январь 2023 — декабрь 2024."
    ), "строка отчёта не меняется"
    assert head["contest"] == "Онлайн-конкурс СберИндекса 2026 · направление «Прогнозирование»"
    assert head["panel_line"].startswith("Панель: 13 140 рядов")
    assert head["subtitle"].endswith(head["panel_line"])


def test_page_limitations_carry_titles_and_follow_the_measured_numbers() -> None:
    items = report.page_limitations(
        r2_note="в сводке по всем горизонтам R² по отдельным рядам отрицателен у 14 из 15 моделей",
        dm=DM,
        significance=WINS_SIGNIFICANCE,
        model_note="Основной LightGBM получает календарь.",
        caveat="",
        schedule="Тревога по расписанию даёт F1 0,227.",
    )
    titles = [item["title"] for item in items]
    assert titles[:3] == [
        "Короткая история рядов",
        "На уровне МО шоки почти не размечены",
        "Публичным лидербордам не доверяем",
    ]
    assert "Календарь и национальные ряды" in titles and "Мало окон проверки" in titles
    assert "Часть признаков — из более поздних редакций" not in titles, "оговорки нет — нет пункта"
    assert items[0]["text"].startswith("История каждого ряда — 24 месяца: в сводке по всем")
    by_title = {item["title"]: item for item in items}
    assert by_title["Календарь и национальные ряды"]["text"].startswith(
        "Календарь и национальные ряды: MAE ниже на 60,8 руб."
    )
    assert by_title["F1 не отличает метод от расписания"]["area"] == "сдвиги"


def test_pdf_is_copied_next_to_the_page(tmp_path: Path) -> None:
    pdf = tmp_path / "reports" / "methodology.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"%PDF-1.4 test")
    out = tmp_path / "landing"
    assert report.publish_pdf(pdf, out) == "methodology.pdf"
    assert (out / "methodology.pdf").read_bytes() == b"%PDF-1.4 test"
    assert report.publish_pdf(tmp_path / "missing.pdf", out) == "", "нет файла — нет и ссылки"


def test_repository_link_is_read_from_the_project_metadata(tmp_path: Path) -> None:
    project = tmp_path / "pyproject.toml"
    project.write_text(
        '[project]\nname = "x"\n\n[project.urls]\nRepository = "https://example.org/repo"\n',
        encoding="utf-8",
    )
    assert report.repository_url(project) == "https://example.org/repo"
    project.write_text('[project]\nname = "x"\n', encoding="utf-8")
    assert report.repository_url(project) == "", "адрес не указан — ссылки на странице нет"
    assert report.repository_url(tmp_path / "none.toml") == ""
