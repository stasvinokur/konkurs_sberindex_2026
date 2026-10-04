"""Экспорт методологического отчёта в PDF.

Без pandoc и системных зависимостей: fpdf2 плюс шрифты DejaVu, которые уже поставляются с
matplotlib. Так сборка PDF работает на чистом клоне после `uv sync`, без установки чего-либо
через пакетный менеджер операционной системы, и кириллица не ломается.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
from fpdf import FPDF

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


class _Report(FPDF):
    def header(self) -> None:  # pragma: no cover - оформление
        pass

    def footer(self) -> None:  # pragma: no cover - оформление
        self.set_y(-15)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, str(self.page_no()), align="C")
        self.set_text_color(0, 0, 0)


def _strip_markup(text: str) -> str:
    """Убирает разметку, которую нечем отрисовать: ссылки, выделение, инлайн-код."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = text.replace("**", "").replace("`", "")
    return text.strip()


def _table_rows(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        if set(line.replace("|", "").replace(" ", "")) <= {"-", ":"}:
            continue
        cells = [_strip_markup(c) for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def _draw_table(pdf: FPDF, rows: list[list[str]]) -> None:
    if not rows:
        return
    columns = max(len(row) for row in rows)
    # Ширина колонки пропорциональна её содержимому: при равных долях первая колонка с
    # названием модели обрезается, а числовые остаются полупустыми.
    longest = [
        max((len(row[i]) for row in rows if i < len(row)), default=1) for i in range(columns)
    ]
    capped = [min(value, 34) for value in longest]
    total = sum(capped) or 1
    widths = [max(pdf.epw * value / total, 8.0) for value in capped]
    scale = pdf.epw / sum(widths)
    widths = [w * scale for w in widths]

    pdf.set_font("DejaVuMono", "", 6.5)
    for index, row in enumerate(rows):
        if pdf.get_y() > 258:
            pdf.add_page()
        height = 5
        pdf.set_x(pdf.l_margin)
        padded = list(row) + [""] * (columns - len(row))
        for cell, width in zip(padded[:columns], widths, strict=True):
            limit = max(int(width / 1.35), 3)
            text = cell if len(cell) <= limit else cell[: limit - 1] + "…"
            pdf.cell(width, height, text, border=1, align="L")
        pdf.ln(height)
        if index == 0:
            pdf.set_font("DejaVuMono", "", 6.5)
    pdf.ln(2)


def markdown_to_pdf(markdown_path: Path, pdf_path: Path) -> Path:
    """Простой конвертер: заголовки, абзацы, списки, таблицы и блоки кода."""
    pdf = _Report(format="A4")
    pdf.add_font("DejaVu", "", FONT_DIR / "DejaVuSans.ttf")
    pdf.add_font("DejaVu", "B", FONT_DIR / "DejaVuSans-Bold.ttf")
    pdf.add_font("DejaVuMono", "", FONT_DIR / "DejaVuSansMono.ttf")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    lines = markdown_path.read_text(encoding="utf-8").split("\n")
    buffer: list[str] = []
    in_code = False

    def flush_table() -> None:
        if buffer:
            _draw_table(pdf, _table_rows(buffer))
            buffer.clear()

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("```"):
            flush_table()
            in_code = not in_code
            continue
        if in_code:
            pdf.set_font("DejaVuMono", "", 8)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.epw, 4.5, _strip_markup(line) or " ")
            continue
        if line.startswith("|"):
            buffer.append(line)
            continue
        flush_table()
        if not line:
            pdf.ln(2)
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            pdf.ln(3)
            pdf.set_font("DejaVu", "B", {1: 16, 2: 13, 3: 11}.get(level, 10))
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.epw, 7, _strip_markup(line.lstrip("# ")))
            pdf.ln(1)
            continue
        if re.match(r"^[-*]\s+|^\d+\.\s+", line):
            pdf.set_font("DejaVu", "", 9.5)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(
                pdf.epw, 5, "• " + _strip_markup(re.sub(r"^[-*]\s+|^\d+\.\s+", "", line))
            )
            continue
        pdf.set_font("DejaVu", "", 9.5)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(pdf.epw, 5, _strip_markup(line))
    flush_table()

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(pdf_path))
    return pdf_path
