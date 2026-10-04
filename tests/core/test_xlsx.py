"""Чтение листа xlsx стандартной библиотекой: справочник организаторов и бюллетени Росстата."""

from __future__ import annotations

import io
import zipfile

import pytest

from sbx.core.xlsx import read_sheet, sheet_names

MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG = "http://schemas.openxmlformats.org/package/2006/relationships"


def _workbook(sheets: dict[str, str], shared: list[str]) -> bytes:
    """Минимальная книга: имя листа → XML строк; общие строки — отдельной частью, как в Excel."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as book:
        entries = "".join(
            f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>'
            for i, name in enumerate(sheets, start=1)
        )
        book.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{MAIN}" xmlns:r="{REL}"><sheets>{entries}</sheets></workbook>',
        )
        # Файлы листов названы не по порядку: имя листа связано с файлом только через rels.
        targets = {name: f"worksheets/part{len(sheets) - i}.xml" for i, name in enumerate(sheets)}
        relations = "".join(
            f'<Relationship Id="rId{i}" Target="{targets[name]}"/>'
            for i, name in enumerate(sheets, start=1)
        )
        book.writestr(
            "xl/_rels/workbook.xml.rels",
            f'<Relationships xmlns="{PKG}">{relations}</Relationships>',
        )
        items = "".join(f"<si><t>{text}</t></si>" for text in shared)
        book.writestr("xl/sharedStrings.xml", f'<sst xmlns="{MAIN}">{items}</sst>')
        for name, rows in sheets.items():
            book.writestr(
                f"xl/{targets[name]}",
                f'<worksheet xmlns="{MAIN}"><sheetData>{rows}</sheetData></worksheet>',
            )
    return buffer.getvalue()


def test_read_sheet_returns_shared_inline_and_numeric_cells_as_text() -> None:
    rows = (
        '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
        '<row r="2"><c r="A2" t="inlineStr"><is><t>Майкоп</t></is></c>'
        '<c r="B2"><v>44.606207900000001</v></c></row>'
    )
    payload = _workbook({"Лист1": rows}, shared=["name", "lat"])
    assert read_sheet(payload) == [["name", "lat"], ["Майкоп", "44.606207900000001"]]


def test_read_sheet_keeps_column_positions_when_cells_are_missing() -> None:
    """Excel не пишет пустые ячейки: без учёта адреса значения съехали бы влево."""
    rows = (
        '<row r="1"><c r="A1"><v>1</v></c><c r="C1"><v>3</v></c></row>'
        '<row r="3"><c r="B3"><v>5</v></c><c r="AA3"><v>27</v></c></row>'
    )
    table = read_sheet(_workbook({"Лист1": rows}, shared=[]))
    assert table[0] == ["1", "", "3"]
    assert table[1][:2] == ["", "5"]
    assert len(table[1]) == 27 and table[1][26] == "27", "AA — двадцать седьмая колонка"


def test_read_sheet_joins_rich_text_runs_of_a_shared_string() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as book:
        book.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{MAIN}" xmlns:r="{REL}"><sheets>'
            '<sheet name="s" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        book.writestr(
            "xl/_rels/workbook.xml.rels",
            f'<Relationships xmlns="{PKG}"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/>'
            "</Relationships>",
        )
        book.writestr(
            "xl/sharedStrings.xml",
            f'<sst xmlns="{MAIN}"><si><r><t>Городской </t></r><r><t>округ</t></r></si></sst>',
        )
        book.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{MAIN}"><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c></row>'
            "</sheetData></worksheet>",
        )
    assert read_sheet(buffer.getvalue()) == [["Городской округ"]]


def test_read_sheet_selects_a_sheet_by_name_not_by_file_order() -> None:
    first = '<row r="1"><c r="A1"><v>1</v></c></row>'
    second = '<row r="1"><c r="A1"><v>2</v></c></row>'
    payload = _workbook({"Титул": first, "Численность_по_МО": second}, shared=[])
    assert sheet_names(payload) == ["Титул", "Численность_по_МО"]
    assert read_sheet(payload) == [["1"]], "по умолчанию — первый лист книги"
    assert read_sheet(payload, "Численность_по_МО") == [["2"]]
    with pytest.raises(KeyError, match="Титул"):
        read_sheet(payload, "нет такого")
