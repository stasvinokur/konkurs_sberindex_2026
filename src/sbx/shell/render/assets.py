"""Ресурсы страницы: таблица стилей, сценарий и шрифты, встроенные в HTML.

Лендинг — один файл без сетевых запросов: его открывают с диска, с GitHub Pages и из контейнера
без сети. Поэтому шрифты IBM Plex вшиты как `data:`-адреса, а не подключены с чужого сервера.
Файлы шрифтов — официальные неизменённые подмножества из пакетов `@ibm/plex-*` (лицензия
SIL OFL 1.1, текст — `assets/fonts/LICENSE.txt`); диапазоны символов взяты из самих файлов.
"""

from __future__ import annotations

import base64
from functools import cache
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"
FONTS = ASSETS / "fonts"

CYRILLIC = (
    "U+0400-045F, U+0462-0463, U+046A-046B, U+0472-0475, U+0490-04C2, U+04CF-04D9, U+04DC-04E9, "
    "U+04EE-04F9, U+0524-0525"
)
LATIN = (
    "U+0000, U+000D, U+0020-007E, U+00A0-00FF, U+0131, U+0152-0153, U+02C6, U+02DA, U+02DC, "
    "U+2013-2014, U+2018-201A, U+201C-201E, U+2020-2022, U+2026, U+2030, U+2039-203A, U+2044, "
    "U+20AC, U+2122, U+2212, U+FB01-FB02"
)
MONO_LATIN = LATIN.removeprefix("U+0000, U+000D, ")
# Из второго латинского подмножества странице нужен один знак — рубль.
RUBLE = "U+20BD"
# Стрелки и математические знаки: →, ←, ∆, ∑, ≈, ≤, ≥, №.
SYMBOLS = (
    "U+2032-2033, U+2116, U+2190-2199, U+2202, U+2206, U+220F, U+2211, U+221A, U+221E, U+2248, "
    "U+2260, U+2264-2265"
)

# Шрифты встроены в страницу, поэтому их лицензия названа в самой странице.
FONT_NOTICE = (
    "/* Шрифты IBM Plex Sans, IBM Plex Sans Condensed и IBM Plex Mono: © IBM Corp., лицензия "
    "SIL Open Font License 1.1; файлы не изменены, текст лицензии — в репозитории решения, "
    "src/sbx/shell/render/assets/fonts/LICENSE.txt */"
)

# (семейство, насыщенность, файл, диапазон символов)
FACES = (
    ("IBM Plex Sans", 400, "IBMPlexSans-Regular-Cyrillic.woff2", CYRILLIC),
    ("IBM Plex Sans", 400, "IBMPlexSans-Regular-Latin1.woff2", LATIN),
    ("IBM Plex Sans", 400, "IBMPlexSans-Regular-Latin2.woff2", RUBLE),
    ("IBM Plex Sans", 400, "IBMPlexSans-Regular-Pi.woff2", SYMBOLS),
    ("IBM Plex Sans", 500, "IBMPlexSans-Medium-Cyrillic.woff2", CYRILLIC),
    ("IBM Plex Sans", 500, "IBMPlexSans-Medium-Latin1.woff2", LATIN),
    ("IBM Plex Sans", 600, "IBMPlexSans-SemiBold-Cyrillic.woff2", CYRILLIC),
    ("IBM Plex Sans", 600, "IBMPlexSans-SemiBold-Latin1.woff2", LATIN),
    ("IBM Plex Sans", 600, "IBMPlexSans-SemiBold-Latin2.woff2", RUBLE),
    ("IBM Plex Sans Condensed", 600, "IBMPlexSansCondensed-SemiBold-Cyrillic.woff2", CYRILLIC),
    ("IBM Plex Sans Condensed", 600, "IBMPlexSansCondensed-SemiBold-Latin1.woff2", LATIN),
    ("IBM Plex Sans Condensed", 600, "IBMPlexSansCondensed-SemiBold-Latin2.woff2", RUBLE),
    ("IBM Plex Mono", 400, "IBMPlexMono-Regular-Cyrillic.woff2", CYRILLIC),
    ("IBM Plex Mono", 400, "IBMPlexMono-Regular-Latin1.woff2", MONO_LATIN),
)


@cache
def font_faces() -> str:
    """Правила `@font-face` со шрифтами внутри: браузер разбирает только нужные подмножества."""
    rules = []
    for family, weight, name, unicode_range in FACES:
        data = base64.b64encode((FONTS / name).read_bytes()).decode("ascii")
        rules.append(
            f'@font-face{{font-family:"{family}";font-style:normal;font-weight:{weight};'
            f"font-display:swap;src:url(data:font/woff2;base64,{data}) format('woff2');"
            f"unicode-range:{unicode_range}}}"
        )
    return FONT_NOTICE + "\n" + "\n".join(rules)


@cache
def stylesheet() -> str:
    """Таблица стилей страницы целиком: шрифты и оформление."""
    return font_faces() + "\n" + (ASSETS / "landing.css").read_text(encoding="utf-8")


@cache
def script() -> str:
    """Сценарий страницы: оглавление, выбор ряда, пример детектора, кнопки копирования."""
    return (ASSETS / "landing.js").read_text(encoding="utf-8")
