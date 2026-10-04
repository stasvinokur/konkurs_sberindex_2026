"""Чтение листа xlsx из байтов средствами стандартной библиотеки.

Нужно для двух файлов-источников: справочника территорий организаторов и бюллетеней Росстата.
Ради них не добавляется зависимость в окружение, от версий которого зависят опубликованные числа.
Читаются только значения ячеек как текст; формулы, даты и стили не разбираются.
"""

from __future__ import annotations

import io
import posixpath
import zipfile
from xml.etree import ElementTree

MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RELATIONSHIP = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PACKAGE = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def _sheets(book: zipfile.ZipFile) -> dict[str, str]:
    """Имя листа → путь его файла в архиве; порядок — как в книге."""
    relations = ElementTree.fromstring(book.read("xl/_rels/workbook.xml.rels"))
    targets = {rel.get("Id"): rel.get("Target") for rel in relations.iter(f"{PACKAGE}Relationship")}
    workbook = ElementTree.fromstring(book.read("xl/workbook.xml"))
    out = {}
    for sheet in workbook.iter(f"{MAIN}sheet"):
        target = str(targets[sheet.get(f"{RELATIONSHIP}id")])
        out[str(sheet.get("name"))] = posixpath.normpath(posixpath.join("xl", target.lstrip("/")))
    return out


def _shared_strings(book: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in book.namelist():
        return []
    root = ElementTree.fromstring(book.read("xl/sharedStrings.xml"))
    # Строка может состоять из нескольких фрагментов с разным оформлением.
    return ["".join(t.text or "" for t in item.iter(f"{MAIN}t")) for item in root.iter(f"{MAIN}si")]


def _column(reference: str) -> int:
    """Номер колонки по адресу ячейки: A1 → 0, AA3 → 26."""
    index = 0
    for char in reference:
        if not char.isalpha():
            break
        index = index * 26 + (ord(char.upper()) - 64)
    return index - 1


def sheet_names(payload: bytes) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(payload)) as book:
        return list(_sheets(book))


def read_sheet(payload: bytes, sheet: str | None = None) -> list[list[str]]:
    """Строки листа; ячейки — текст, пропущенные внутри строки ячейки — пустые строки.

    `sheet` — имя листа, по умолчанию первый лист книги. Строки без ячеек пропускаются.
    """
    with zipfile.ZipFile(io.BytesIO(payload)) as book:
        sheets = _sheets(book)
        if sheet is None:
            sheet = next(iter(sheets))
        if sheet not in sheets:
            raise KeyError(f"нет листа {sheet!r}; есть: {list(sheets)}")
        shared = _shared_strings(book)
        root = ElementTree.fromstring(book.read(sheets[sheet]))

    rows = []
    for row in root.iter(f"{MAIN}row"):
        cells: dict[int, str] = {}
        for cell in row.iter(f"{MAIN}c"):
            kind = cell.get("t")
            if kind == "inlineStr":
                text = "".join(t.text or "" for t in cell.iter(f"{MAIN}t"))
            else:
                value = cell.find(f"{MAIN}v")
                text = value.text or "" if value is not None else ""
                if kind == "s" and text:
                    text = shared[int(text)]
            cells[_column(str(cell.get("r")))] = text
        if cells:
            rows.append([cells.get(i, "") for i in range(max(cells) + 1)])
    return rows
