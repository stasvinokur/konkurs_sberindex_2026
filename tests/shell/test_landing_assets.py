"""Ресурсы лендинга: шрифты, таблица стилей и сценарий лежат в пакете и встраиваются в страницу."""

from __future__ import annotations

import re

import pytest

from sbx.shell.render import assets, charts


def test_every_declared_font_file_is_a_woff2_and_no_file_is_dead_weight() -> None:
    declared = {name for _, _, name, _ in assets.FACES}
    on_disk = {path.name for path in assets.FONTS.glob("*.woff2")}
    assert declared == on_disk, "объявлены ровно те файлы, что лежат в пакете"
    for name in declared:
        assert (assets.FONTS / name).read_bytes()[:4] == b"wOF2", name
    licence = (assets.FONTS / "LICENSE.txt").read_text(encoding="utf-8")
    assert "SIL OPEN FONT LICENSE" in licence and "IBM" in licence


def test_font_faces_are_embedded_with_their_character_ranges() -> None:
    css = assets.font_faces()
    rules = re.findall(r"@font-face\{([^}]*)\}", css)
    assert len(rules) == len(assets.FACES)
    for rule in rules:
        assert "src:url(data:font/woff2;base64," in rule and "unicode-range:U+" in rule
        assert "font-display:swap" in rule
    families = set(re.findall(r'font-family:"([^"]+)"', css))
    assert families == {"IBM Plex Sans", "IBM Plex Sans Condensed", "IBM Plex Mono"}
    assert css.startswith("/*") and "SIL Open Font License 1.1" in css.split("*/")[0], (
        "страница со встроенными шрифтами называет их лицензию"
    )
    ruble = [rule for rule in rules if "unicode-range:U+20BD" in rule]
    assert len(ruble) == 3, "знак рубля — для обычного, полужирного и узкого начертаний"


def _tokens(block: str) -> dict[str, str]:
    return dict(re.findall(r"(--[a-z0-9-]+):\s*([^;]+);", block))


def test_dark_theme_redefines_tokens_and_introduces_none_of_its_own() -> None:
    """Цвет, заданный только в тёмной теме, в светлой остался бы пустым."""
    css = (assets.ASSETS / "landing.css").read_text(encoding="utf-8")
    light = _tokens(css[css.index(":root {") :].split("\n}\n", 1)[0])
    dark_block = css[css.index("@media (prefers-color-scheme: dark)") :].split("\n}\n", 1)[0]
    dark = _tokens(dark_block)
    assert dark and set(dark) <= set(light), sorted(set(dark) - set(light))
    for name in ("--bg", "--ink", "--acc", "--line", "--alarm", "--found", "--grid"):
        assert dark[name] != light[name], name
    body = css[css.index("\n}\n", css.index("@media (prefers-color-scheme: dark)")) :]
    literals = [
        line
        for line in body.splitlines()
        if re.search(r"#[0-9A-Fa-f]{3,8}\b|rgba?\(", line) and "/*" not in line
    ]
    assert not literals, "цвета компонентов — только через переменные: " + "; ".join(literals[:3])


def _luminance(colour: str) -> float:
    channels = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(first: str, second: str) -> float:
    high, low = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_text_colours_are_readable_on_their_backgrounds_in_both_themes() -> None:
    """Контраст текста с фоном — не ниже 4,5:1 (WCAG AA) для каждой пары «цвет текста, фон»."""
    css = (assets.ASSETS / "landing.css").read_text(encoding="utf-8")
    light = _tokens(css[css.index(":root {") :].split("\n}\n", 1)[0])
    dark = {
        **light,
        **_tokens(css[css.index("@media (prefers-color-scheme: dark)") :].split("\n}\n", 1)[0]),
    }
    pairs = [
        ("ink", "bg"), ("ink2", "bg"), ("ink3", "bg"), ("ink", "sub"), ("ink2", "sub"),
        ("ink3", "sub"), ("ink3", "sub2"), ("acc", "bg"), ("acc", "acc-soft"),
        ("acc-strong", "acc-soft"), ("ref", "ref-soft"), ("ref", "bg"),
        ("alarm-ink", "alarm-soft"), ("alarm-ink", "alarm-pale"), ("alarm-ink", "bg"),
        ("harm", "harm-soft"), ("event", "bg"), ("found", "bg"), ("amber-ink", "bg"),
        ("stat", "bg"), ("neutral-strong", "bg"), ("on-acc", "acc"), ("ink", "surf"),
    ]  # fmt: skip
    for theme, tokens in (("светлая", light), ("тёмная", dark)):
        for text, ground in pairs:
            ratio = _contrast(tokens[f"--{text}"], tokens[f"--{ground}"])
            assert ratio >= 4.5, f"{theme} тема: {text} на {ground} — {ratio:.2f}"
    assert "text.c-alarm { fill: var(--alarm-ink); }" in css, "подпись тревоги — текстовым цветом"


def test_page_script_makes_no_network_requests() -> None:
    script = assets.script()
    for call in ("fetch(", "XMLHttpRequest", "import(", "WebSocket", "sendBeacon"):
        assert call not in script, call
    assert "http://" not in script and "https://" not in script


def test_stylesheet_has_no_external_references() -> None:
    css = assets.stylesheet()
    assert set(re.findall(r"url\(([a-z]+):", css)) == {"data"}
    assert "@import" not in css
    assert "@media (max-width: 900px)" in css and "@media print" in css


def _blocks(css: str) -> list[tuple[str, str]]:
    """Правила верхнего уровня: заголовок (селектор или @-правило) и тело."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    found, depth, start, head = [], 0, 0, ""
    for index, char in enumerate(css):
        if char == "{":
            if depth == 0:
                head, start = css[start:index].strip(), index + 1
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                found.append((head, css[start:index]))
                start = index + 1
    return found


def _style(selector: str, media: str | None = None) -> dict[str, str]:
    """Объявления правила: вне медиазапросов, а если запрос назван — с его поправками."""
    blocks = _blocks((assets.ASSETS / "landing.css").read_text(encoding="utf-8"))
    rules = [block for block in blocks if not block[0].startswith("@")]
    if media:
        rules += [rule for head, body in blocks if head == media for rule in _blocks(body)]
    style: dict[str, str] = {}
    for head, body in rules:
        if selector in [part.strip() for part in head.split(",")]:
            style.update(dict(re.findall(r"([a-z-]+):\s*([^;]+);", body)))
    assert style, selector
    return style


def _px(value: str) -> float:
    assert value.endswith("px"), value
    return float(value[:-2])


PHONE = "@media (max-width: 900px)"
CASE_CHART = ".case figure .scroll-x > svg"
# Блок, стоящий в ряд с соседом, и график внутри него.
HOLDERS = [
    (".two > .chart-col", ".chart-col .scroll-x > svg"),
    (".split > figure", ".split > figure > .scroll-x > svg"),
    (".curve figure", ".curve figure .scroll-x > svg"),
]


@pytest.mark.parametrize(("holder", "chart"), HOLDERS)
def test_chart_is_not_wider_than_the_block_it_sits_in(holder: str, chart: str) -> None:
    """Блок уходит на новую строку раньше, чем график перестаёт в нём помещаться."""
    basis = _px(_style(holder)["flex"].split()[-1])
    assert _px(_style(chart)["min-width"]) <= basis


def test_case_chart_fits_the_text_column_of_a_1280_window() -> None:
    """Колонка текста — окно без полей, оглавления и зазора; график разбора занимает её всю."""
    shell, side = _style(".shell"), _style(".side")
    column = 1280 - 2 * _px(shell["padding"].split()[-1]) - _px(shell["gap"].split()[-1])
    column -= _px(side["flex"].split()[-1])
    assert _px(_style(CASE_CHART)["min-width"]) <= column, column


@pytest.mark.parametrize(("media", "floor"), [(None, 8.5), (PHONE, 10)])
def test_chart_text_stays_readable_at_the_narrowest_width(media: str | None, floor: float) -> None:
    """Текст сжимается вместе с графиком: на телефоне блок прокручивается, а не мельчит."""
    explorer = re.search(r"const \[W, H, LEFT, TOP, INSET\] = \[(\d+),", assets.script())
    assert explorer, "ширина графика выбора ряда задана в сценарии страницы"
    drawn = {
        ".chart-col .scroll-x > svg": [charts.HORIZON_SIZE[0]],
        ".split > figure > .scroll-x > svg": [charts.DEMO_SIZE[0], int(explorer.group(1))],
        ".curve figure .scroll-x > svg": [charts.DELAY_SIZE[0]],
        CASE_CHART: [charts.CASE_SIZE[0]],
    }
    assert _style(".ch text.s11")["font-size"] == "11px", "наименьший кегль на графиках"
    for chart, widths in drawn.items():
        narrowest = _px(_style(chart, media)["min-width"])
        for width in widths:
            assert 11 * narrowest / width >= floor, (chart, narrowest, width)


def test_horizon_chart_stays_within_one_screen_when_it_takes_a_row_of_its_own() -> None:
    """Колонка графика, ушедшая под таблицу, не растягивает его на всю ширину текста."""
    column = _style(".two > .chart-col")
    assert "max-width" in column, "у колонки графика есть предел ширины"
    width, height = charts.HORIZON_SIZE
    assert _px(column["flex"].split()[-1]) <= _px(column["max-width"])
    assert _px(column["max-width"]) * height / width <= 720, (
        "высота графика — не больше окна 1280×720"
    )
