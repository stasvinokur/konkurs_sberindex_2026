"""Графики лендинга — встроенный SVG.

Рисунок собирается здесь из тех же данных, что и текст страницы. Цвета заданы классами из
таблицы стилей (`assets/landing.css`), поэтому график меняется вместе с темой, читается без сети
и без библиотек. Геометрия считается в координатах `viewBox`; по ширине рисунок растягивает
браузер. Подписи значений — текст страницы, а не картинка.
"""

from __future__ import annotations

import html
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sbx.core.report import (
    DETECTOR_NAMES,
    format_number,
    interval_runs,
    model_family,
    model_title,
)

MONTHS_SHORT = ("янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек")
# Класс цвета по семейству модели: эталон, foundation, машинное обучение, ансамбль, наивные.
FAMILY_CLASSES = {
    "reference": "ref",
    "foundation": "found",
    "ml": "amber",
    "ensemble": "acc",
    "naive": "neutral",
    "stats": "stat",
}
LOG_TICKS = (100, 200, 300, 400, 500, 700, 1000, 1500, 2000, 3000, 5000, 8000, 12000, 20000)

SMALL_SIZE = (344, 220)
WIDE_SIZE = (1080, 300)
HORIZON_SIZE = (440, 380)
DELAY_SIZE = (600, 410)
CASE_SIZE = (1080, 330)
CASE_PAD = (44, 16, 38, 56)  # сверху, справа, снизу, слева
DEMO_SIZE = (740, 586)


def _e(value: Any) -> str:
    return html.escape(str(value))


def _n(value: float) -> str:
    """Координата: один знак после запятой, без хвостового нуля."""
    text = f"{value:.1f}"
    return text[:-2] if text.endswith(".0") else text


def _known(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(value)


@dataclass(frozen=True)
class Scale:
    """Отображение значений `low…high` на координаты `start…end`; `log` — в логарифмах."""

    low: float
    high: float
    start: float
    end: float
    log: bool = False

    def __call__(self, value: float) -> float:
        if self.log:
            share = math.log(value / self.low) / math.log(self.high / self.low)
        else:
            share = (value - self.low) / (self.high - self.low)
        return self.start + share * (self.end - self.start)


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    """Деления с круглым шагом (1, 2 или 5 на степень десяти), накрывающие отрезок целиком."""
    if not high > low:
        pad = abs(low) * 0.1 or 1.0
        low, high = low - pad, high + pad
    raw = (high - low) / max(count, 1)
    power = 10.0 ** math.floor(math.log10(raw))
    error = raw / power
    step = power * (
        10 if error >= 50**0.5 else 5 if error >= 10**0.5 else 2 if error >= 2**0.5 else 1
    )
    first = math.floor(low / step + 1e-9)
    last = math.ceil(high / step - 1e-9)
    return [round(i * step, 10) for i in range(first, last + 1)]


def line_path(xs: Sequence[float], ys: Sequence[float | None]) -> str:
    """Ломаная по точкам; на пропуске значения линия обрывается и начинается заново."""
    parts, drawing = [], False
    for x, y in zip(xs, ys, strict=True):
        if not _known(y):
            drawing = False
            continue
        parts.append(f"{'L' if drawing else 'M'}{_n(x)} {_n(float(y))}")  # type: ignore[arg-type]
        drawing = True
    return "".join(parts)


def _band_paths(
    xs: Sequence[float], low: Sequence[Any], high: Sequence[Any], scale: Scale
) -> list[str]:
    """Полоса интервала — отдельным контуром на каждом отрезке, где интервал есть: контур через
    пропуск замыкался бы на соседний отрезок и рисовал клинья."""
    paths = []
    for start, stop in interval_runs(low, high):
        if stop - start < 2:
            continue
        top = [f"{_n(xs[i])} {_n(scale(float(high[i])))}" for i in range(start, stop)]
        bottom = [f"{_n(xs[i])} {_n(scale(float(low[i])))}" for i in reversed(range(start, stop))]
        paths.append("M" + "L".join(top + bottom) + "Z")
    return paths


def month_tick(month: str, with_year: bool = False) -> str:
    """«янв», «янв 2024» — подпись месяца «ГГГГ-ММ» на оси."""
    name = MONTHS_SHORT[int(month[5:7]) - 1]
    return f"{name} {month[:4]}" if with_year else name


def _tick_labels(ticks: Sequence[float], factor: float = 1.0) -> list[str]:
    values = [tick / factor for tick in ticks]
    whole = all(abs(value - round(value)) < 1e-9 for value in values)
    return [
        format_number(round(value) if whole or abs(value) < 1e-9 else value, 1) for value in values
    ]


def _svg(size: tuple[int, int], label: str, body: str, attrs: str = "", role: str = "img") -> str:
    """Корень рисунка. `role="group"` — для рисунка с выбором месяца внутри: роль «картинка»
    скрыла бы элемент выбора от вспомогательных программ."""
    return (
        f'<svg class="ch" viewBox="0 0 {size[0]} {size[1]}" role="{role}" '
        f'aria-label="{_e(label)}"{attrs}>{body}</svg>'
    )


def _text(x: float, y: float, text: str, cls: str = "s11", anchor: str = "") -> str:
    anchor_attr = f' text-anchor="{anchor}"' if anchor else ""
    return f'<text class="{cls}" x="{_n(x)}" y="{_n(y)}"{anchor_attr}>{_e(text)}</text>'


def _y_axis(
    scale: Scale, ticks: Sequence[float], labels: Sequence[str], left: float, right: float
) -> str:
    """Горизонтальные линии сетки и подписи делений слева."""
    parts = []
    for tick, label in zip(ticks, labels, strict=True):
        y = scale(tick)
        parts.append(
            f'<line class="grid" x1="{_n(left)}" x2="{_n(right)}" y1="{_n(y)}" y2="{_n(y)}"/>'
        )
        parts.append(_text(left - 8, y + 4, label, anchor="end"))
    return "".join(parts)


# ── Ряд: факт, прогноз, интервал, тревоги ────────────────────────────────────────────────────


def _series_values(item: Mapping[str, Any]) -> list[float]:
    values = [*item["y"], *item["y_hat"], *(item.get("q_lo") or []), *(item.get("q_hi") or [])]
    return [float(v) for v in values if _known(v)]


def series_unit(item: Mapping[str, Any]) -> str:
    """Единица оси: крупные ряды подписываются в тысячах рублей."""
    return "тыс. ₽" if max(_series_values(item), default=0.0) >= 10_000 else "₽"


def series_chart(
    item: Mapping[str, Any],
    size: tuple[int, int] = SMALL_SIZE,
    alarms: Sequence[str] = (),
    every_month: bool = False,
) -> str:
    """Факт, прогноз и интервал одного ряда по месяцам; тревоги детектора — отметками.

    Тревога рисуется тремя знаками: кольцо вокруг точки факта, пунктир вниз и треугольник у оси —
    так её видно и без цвета.
    """
    width, height = size
    left, right, top, bottom = 44.0, width - 12.0, 12.0, height - 28.0
    months = list(item["ds"])
    values = _series_values(item)
    ticks = nice_ticks(min(values), max(values), 4)
    scale = Scale(ticks[0], ticks[-1], bottom, top)
    factor = 1000.0 if series_unit(item) == "тыс. ₽" else 1.0
    inset = 10.0
    step = (right - left - 2 * inset) / max(len(months) - 1, 1)
    xs = [left + inset + i * step for i in range(len(months))]

    def ys(key: str) -> list[float | None]:
        return [scale(float(v)) if _known(v) else None for v in item.get(key) or []]

    parts = [_y_axis(scale, ticks, _tick_labels(ticks, factor), left, right)]
    parts.append(
        f'<line class="axis" x1="{_n(left)}" x2="{_n(right)}" y1="{_n(bottom)}" y2="{_n(bottom)}"/>'
    )
    for i, month in enumerate(months):
        if every_month or i % 3 == 0:
            parts.append(
                _text(xs[i], bottom + 18, month_tick(month, with_year=i == 0), anchor="middle")
            )
    for band in _band_paths(xs, item.get("q_lo") or [], item.get("q_hi") or [], scale):
        parts.append(f'<path class="band" d="{band}"/>')
    fact = ys("y")
    marked = [months.index(m) for m in dict.fromkeys(str(a)[:7] for a in alarms) if m in months]
    for i in marked:
        parts.append(
            f'<line class="drop" x1="{_n(xs[i])}" x2="{_n(xs[i])}" y1="{_n(fact[i] or top)}" '
            f'y2="{_n(bottom)}"/>'
        )
    parts.append(f'<path class="ln fc" d="{line_path(xs, ys("y_hat"))}"/>')
    parts.append(f'<path class="ln fact" d="{line_path(xs, fact)}"/>')
    for x, y in zip(xs, fact, strict=True):
        if y is not None:
            parts.append(f'<circle class="pt" cx="{_n(x)}" cy="{_n(y)}" r="3"/>')
    for i in marked:
        if fact[i] is not None:
            parts.append(f'<circle class="ring" cx="{_n(xs[i])}" cy="{_n(fact[i])}" r="6.5"/>')
        parts.append(f'<path class="tri" d="M{_n(xs[i])} {_n(bottom + 1)}l-5 8h10z"/>')
    place = item.get("mo") or item.get("region") or ""
    label = f"{place}, {item.get('category', '')}: факт и прогноз по месяцам, {series_unit(item)}"
    if marked:
        label += f"; тревог детектора — {len(marked)}"
    return _svg(size, label, "".join(parts))


# ── MAE по горизонтам ────────────────────────────────────────────────────────────────────────


def _label_lines(title: str) -> list[str]:
    """Подпись линии: длинное имя — в две строки по первому пробелу."""
    if len(title) <= 12 or " " not in title:
        return [title]
    head, tail = title.split(" ", 1)
    return [head, tail]


def _spread(labels: list[dict[str, Any]], top: float, bottom: float, gap: float = 3.0) -> None:
    """Разводит подписи по вертикали, чтобы они не налезали друг на друга."""
    labels.sort(key=lambda item: item["y"])
    for previous, current in zip(labels, labels[1:], strict=False):
        lowest = previous["y"] + previous["height"] + gap
        if current["y"] < lowest and _overlap(previous, current):
            current["y"] = lowest
    if labels:
        overflow = labels[-1]["y"] + labels[-1]["height"] - bottom
        if overflow > 0:
            for item in labels:
                item["y"] = max(top, item["y"] - overflow)


def _overlap(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return a["x"] < b["x"] + b["width"] and b["x"] < a["x"] + a["width"]


def _label_markup(item: Mapping[str, Any]) -> str:
    """Подпись линии из одной-двух строк: первая — полужирная, вторая — обычная."""
    spans = []
    for i, line in enumerate(item["lines"]):
        weight = ' class="b"' if i == 0 or item.get("bold_all") else ""
        dy = ' dy="13"' if i else ""
        spans.append(f'<tspan{weight} x="{_n(item["x"])}"{dy}>{_e(line)}</tspan>')
    return (
        f'<text class="s11 c-{item["cls"]}" x="{_n(item["x"])}" y="{_n(item["y"] + 10)}">'
        f"{''.join(spans)}</text>"
    )


def horizon_chart(
    rows: Sequence[Mapping[str, Any]],
    winners: Sequence[Mapping[str, Any]],
    reference: str = "Prophet",
    extra: Sequence[str] = ("Naive",),
) -> str:
    """MAE в зависимости от горизонта для лидеров, эталона и наивного прогноза.

    Шкала логарифмическая: иначе эталон на длинном горизонте сжимает остальные линии в одну.
    Кольцо — лучшая модель горизонта. Линия модели обрывается там, где модель не определена.
    """
    width, height = HORIZON_SIZE
    left, right, top, bottom = 46.0, width - 112.0, 14.0, height - 30.0
    by_model = {str(row["Модель"]): row for row in rows}
    columns = [key for key in rows[0] if key != "Модель"]
    horizons = [int(key.split("=")[-1]) for key in columns]
    wanted = [str(w["model"]) for w in winners] + [reference, *extra]
    shown = [model for model in dict.fromkeys(wanted) if model in by_model]
    values = {
        model: [float(by_model[model][c]) if _known(by_model[model][c]) else None for c in columns]
        for model in shown
    }
    known = [v for series in values.values() for v in series if v is not None]
    low, high = min(known) * 0.9, max(known) * 1.1
    scale = Scale(low, high, bottom, top, log=True)
    step = (right - left - 24) / max(len(horizons) - 1, 1)
    xs = [left + 12 + i * step for i in range(len(horizons))]

    ticks = [tick for tick in LOG_TICKS if low <= tick <= high]
    parts = [_y_axis(scale, ticks, [format_number(tick) for tick in ticks], left, right)]
    parts.append(
        f'<line class="axis" x1="{_n(left)}" x2="{_n(right)}" y1="{_n(bottom)}" y2="{_n(bottom)}"/>'
    )
    for x, horizon in zip(xs, horizons, strict=True):
        parts.append(_text(x, bottom + 19, f"{horizon} мес.", cls="s12 ink2", anchor="middle"))

    best = {(int(w["horizon"]), str(w["model"])) for w in winners}
    labels: list[dict[str, Any]] = []
    marks = []
    # Эталон и наивный прогноз рисуются первыми: лидеры лежат поверх них.
    for model in sorted(shown, key=lambda m: m in {str(w["model"]) for w in winners}):
        cls = FAMILY_CLASSES[model_family(model, reference)]
        ys = [scale(v) if v is not None else None for v in values[model]]
        dash = " dash" if model == reference else ""
        parts.append(f'<path class="ln s-{cls}{dash}" d="{line_path(xs, ys)}"/>')
        last = max(i for i, y in enumerate(ys) if y is not None)
        for i, y in enumerate(ys):
            if y is None:
                continue
            marks.append(f'<circle class="f-{cls} halo" cx="{_n(xs[i])}" cy="{_n(y)}" r="3.5"/>')
            if (horizons[i], model) in best:
                marks.append(f'<circle class="best" cx="{_n(xs[i])}" cy="{_n(y)}" r="8"/>')
        lines = _label_lines(model_title(model))
        if model == reference:
            lines = [lines[0], "эталон"]
        at_edge = last == len(horizons) - 1
        labels.append(
            {
                "lines": lines,
                "cls": cls,
                "x": xs[last] + 14,
                "y": ys[last] - (7 if at_edge else 2),  # type: ignore[operator]
                "width": 100.0,
                "height": 13.0 * len(lines),
                "bold_all": model != reference,
            }
        )
    _spread(labels, top, bottom)
    parts += marks
    parts += [_label_markup(item) for item in labels]
    label = "MAE по горизонтам прогноза, шкала логарифмическая: " + ", ".join(
        model_title(model) for model in shown
    )
    return _svg(HORIZON_SIZE, label, "".join(parts))


# ── Задержка против частоты ложных тревог ────────────────────────────────────────────────────


def _curve_points(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, float]]]:
    """Детектор → точки (частота ложных тревог, задержка, порог) по возрастанию частоты."""
    out: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        x, y = row.get("false_alarms_per_series_year"), row.get("mean_delay")
        if _known(x) and _known(y):
            point = {"x": float(x), "y": float(y), "threshold": float(row.get("threshold", 0.0))}
            out.setdefault(str(row["detector"]), []).append(point)
    return {name: sorted(points, key=lambda p: p["x"]) for name, points in out.items()}


def _clip(points: Sequence[Mapping[str, float]], limit: float) -> list[tuple[float, float]]:
    """Точки кривой до границы оси; на самой границе — значение по прямой между соседями."""
    kept: list[tuple[float, float]] = []
    for previous, point in zip([None, *points], points, strict=False):
        if point["x"] <= limit:
            kept.append((point["x"], point["y"]))
            continue
        if previous is not None and previous["x"] < limit:
            share = (limit - previous["x"]) / (point["x"] - previous["x"])
            kept.append((limit, previous["y"] + share * (point["y"] - previous["y"])))
        break
    return kept


def delay_chart(
    rows: Sequence[Mapping[str, Any]],
    focus: str,
    match: Mapping[str, Any] | None = None,
    schedule: str = "periodic",
) -> str:
    """Задержка обнаружения против частоты ложных тревог: по кривой на детектор.

    Расписание — пунктирная линия отсчёта, область под ней закрашена: кривая, проходящая ниже,
    при той же частоте ложных тревог срабатывает раньше расписания. Частое расписание даёт в
    разы больше ложных тревог, чем любой детектор, поэтому ось обрезана по детекторам.
    `match` — пороги, взятые для сравнения при равной частоте: их точки соединены отрезком.
    """
    width, height = DELAY_SIZE
    left, right, top, bottom = 46.0, width - 20.0, 16.0, height - 44.0
    curves = _curve_points(rows)
    methods = {name: points for name, points in curves.items() if name != schedule}
    reference = curves.get(schedule) or []
    widest = max((p["x"] for points in methods.values() for p in points), default=0.0)
    limit = (
        1.15 * widest
        if methods and reference
        else 1.05 * max((p["x"] for points in curves.values() for p in points), default=1.0)
    )
    clipped = _clip(reference, limit)
    delays = [p["y"] for points in methods.values() for p in points] + [y for _, y in clipped]
    y_ticks = nice_ticks(0.0, max(delays, default=1.0), 6)
    x_scale = Scale(0.0, limit, left, right)
    y_scale = Scale(0.0, y_ticks[-1], bottom, top)
    x_ticks = [tick for tick in nice_ticks(0.0, limit, 4) if tick <= limit]

    y_labels = _tick_labels(y_ticks)
    y_labels[-1] += " мес."
    parts = [_y_axis(y_scale, y_ticks, y_labels, left, right)]
    for tick, text in zip(x_ticks, _tick_labels(x_ticks), strict=True):
        parts.append(_text(x_scale(tick), bottom + 18, text, anchor="middle"))
    parts.append(
        _text(
            (left + right) / 2,
            bottom + 38,
            "ложных тревог на ряд в год",
            cls="s12 ink2",
            anchor="middle",
        )
    )
    labels: list[dict[str, Any]] = []

    def trace(
        points: Sequence[tuple[float, float]], cls: str, name: Sequence[str], dash: str = ""
    ) -> None:
        xs = [x_scale(x) for x, _ in points]
        ys = [y_scale(y) for _, y in points]
        parts.append(f'<path class="ln s-{cls}{dash}" d="{line_path(xs, ys)}"/>')
        marks.extend(
            f'<circle class="f-{cls} halo" cx="{_n(x)}" cy="{_n(y)}" r="3"/>'
            for x, y in zip(xs, ys, strict=True)
            if x < right - 0.5
        )
        text_width = 6.6 * max(len(line) for line in name)
        # Главная кривая подписана под левым концом, расписание — под правым, внутри закрашенной
        # области; остальные — справа от последней точки.
        if cls == "acc":
            x, y = xs[0], ys[0] + 8
        elif cls == "ref":
            x, y = xs[-1] - text_width - 6, ys[-1] + 12
        else:
            x, y = xs[-1] + 9, ys[-1] - 7
        labels.append(
            {
                "lines": list(name),
                "cls": cls,
                "x": x,
                "y": y,
                "width": text_width,
                "height": 13.0 * len(name),
                "bold_all": cls == "acc",
            }
        )

    marks: list[str] = []
    if clipped:
        outline = "L".join(f"{_n(x_scale(x))} {_n(y_scale(y))}" for x, y in clipped)
        first, last = x_scale(clipped[0][0]), x_scale(clipped[-1][0])
        parts.append(
            f'<path class="ref-area" d="M{outline}L{_n(last)} {_n(bottom)}L{_n(first)} {_n(bottom)}Z"/>'
        )
    parts.append(
        f'<line class="axis" x1="{_n(left)}" x2="{_n(right)}" y1="{_n(bottom)}" y2="{_n(bottom)}"/>'
    )
    for name, points in methods.items():
        if name != focus:
            title = DETECTOR_NAMES.get(name, name)
            trace([(p["x"], p["y"]) for p in points], "neutral", [title])
    if clipped:
        trace(clipped, "ref", ["расписание", "эталон"], " dash")
    if focus in methods:
        title = f"{DETECTOR_NAMES.get(focus, focus)} по остаткам"
        trace([(p["x"], p["y"]) for p in methods[focus]], "acc", [title])
    if match:
        ours = [
            p
            for p in methods.get(str(match.get("detector")), [])
            if p["threshold"] == float(match.get("threshold", math.nan))
        ]
        theirs = [p for p in reference if p["threshold"] == float(match.get("period", math.nan))]
        if ours and theirs and theirs[0]["x"] <= limit:
            parts.append(
                f'<line class="match" x1="{_n(x_scale(ours[0]["x"]))}" y1="{_n(y_scale(ours[0]["y"]))}" '
                f'x2="{_n(x_scale(theirs[0]["x"]))}" y2="{_n(y_scale(theirs[0]["y"]))}"/>'
            )
    _spread(labels, top, bottom - 4)
    parts += marks
    parts += [_label_markup(item) for item in labels]
    label = "Задержка обнаружения против частоты ложных тревог: по кривой на детектор, расписание — пунктиром"
    return _svg(DELAY_SIZE, label, "".join(parts))


# ── Национальный ряд: события, тревоги и окна допуска ────────────────────────────────────────


def case_chart(case: Mapping[str, Any]) -> str:
    """Национальный ряд с размеченными событиями и тревогами офлайн-метода.

    Вокруг каждой тревоги закрашено окно допуска: событие считается найденным, если попало в
    окно. Чем больше закрашено, тем вероятнее «найти» и случайную дату. Найденное событие —
    сплошная линия и закрашенный ромб, ненайденное — пунктир и пустой ромб; номер над ромбом —
    номер события в списке под графиком.
    """
    width, height = CASE_SIZE
    top, right_pad, bottom_pad, left = CASE_PAD
    right, bottom = width - right_pad, height - bottom_pad
    months = [str(m)[:7] for m in case["ds"]]
    values = [float(v) for v in case["values"]]
    step = (right - left) / max(len(months) - 1, 1)
    xs = [left + i * step for i in range(len(months))]
    ticks = nice_ticks(min(values), max(values), 4)
    scale = Scale(ticks[0], ticks[-1], bottom, top)
    position = {month: i for i, month in enumerate(months)}
    margin = int(case.get("margin", 2))
    alarms = [position[m] for m in (str(b)[:7] for b in case.get("breaks", [])) if m in position]

    parts = []
    for i in alarms:
        start, stop = xs[max(i - margin, 0)], xs[min(i + margin, len(months) - 1)]
        parts.append(
            f'<rect class="win" x="{_n(start)}" y="{_n(top)}" width="{_n(stop - start)}" '
            f'height="{_n(bottom - top)}"/>'
        )
    parts.append(_y_axis(scale, ticks, _tick_labels(ticks), left, right))
    parts.append(
        f'<line class="axis" x1="{_n(left)}" x2="{_n(right)}" y1="{_n(bottom)}" y2="{_n(bottom)}"/>'
    )
    for i, month in enumerate(months):
        if month.endswith("-01"):
            parts.append(
                f'<line class="axis" x1="{_n(xs[i])}" x2="{_n(xs[i])}" y1="{_n(bottom)}" y2="{_n(bottom + 5)}"/>'
            )
            parts.append(_text(xs[i], bottom + 30, month[:4], anchor="middle"))
    heads = []
    for number, event in enumerate(case.get("events", []), start=1):
        month = str(event["date"])[:7]
        if month not in position:
            continue
        x = xs[position[month]]
        miss = "" if event.get("detected") else " miss"
        y = top - 12
        parts.append(
            f'<line class="ev{miss}" x1="{_n(x)}" x2="{_n(x)}" y1="{_n(y + 5)}" y2="{_n(bottom)}"/>'
        )
        heads.append(
            f'<rect class="dia{miss}" x="{_n(x - 4)}" y="{_n(y - 4)}" width="8" height="8" '
            f'transform="rotate(45 {_n(x)} {_n(y)})"/>'
        )
        heads.append(_text(x, y - 10, str(number), cls="s11 b c-event", anchor="middle"))
    parts.append(f'<path class="ln fact" d="{line_path(xs, [scale(v) for v in values])}"/>')
    parts += heads
    # Тревоги — под осью: поверх ряда они закрывали бы сам ряд.
    for i in alarms:
        parts.append(f'<path class="tri" d="M{_n(xs[i])} {_n(bottom + 4)}l-5 8h10z"/>')
    found = sum(1 for event in case.get("events", []) if event.get("detected"))
    label = (
        f"Национальный ряд «{case.get('series', '')}»: тревог метода — {len(alarms)}, размеченных "
        f"событий — {len(case.get('events', []))}, найдено — {found}"
    )
    return _svg(CASE_SIZE, label, "".join(parts))


# ── Пример работы детектора ──────────────────────────────────────────────────────────────────


def demo_chart(demo: Mapping[str, Any]) -> str:
    """Три панели на одном ряду бенчмарка: ряд и прогноз, ошибка прогноза, статистика детектора.

    Месяцы от слома до тревоги выделены полосой на каждой панели; над первой панелью — ромб в
    месяце слома и треугольник в месяце тревоги. Сетка месяцев записана в атрибутах рисунка: по
    ней сценарий страницы находит месяц под указателем.
    """
    width, height = DEMO_SIZE
    left, right = 58.0, width - 14.0
    months = list(demo["months"])
    inset = 12.0
    step = (right - left - 2 * inset) / max(len(months) - 1, 1)
    xs = [left + inset + i * step for i in range(len(months))]
    tau, alarm = int(demo["tau"]), int(demo["alarms"][0])
    threshold = float(demo["threshold"])
    name = DETECTOR_NAMES.get(str(demo["detector"]), str(demo["detector"]))

    def known(key: str) -> list[float]:
        return [float(v) for v in demo[key] if _known(v)]

    level = nice_ticks(min(known("y") + known("forecast")), max(known("y") + known("forecast")), 4)
    spread = max((abs(v) for v in known("z")), default=1.0)
    error = nice_ticks(
        min(min(known("z"), default=0.0), -spread * 0.4),
        max(max(known("z"), default=0.0), spread * 0.4),
        4,
    )
    stat = nice_ticks(0.0, max([*known("statistic"), threshold]) * 1.05, 3)
    panels = [
        ("1 · Ряд и прогноз на шаг вперёд, ₽", level, 48.0, 194.0),
        ("2 · Ошибка прогноза, в обычных ошибках", error, 238.0, 368.0),
        (f"3 · Статистика {name} и порог", stat, 412.0, 542.0),
    ]
    first, last = min(tau, alarm), max(tau, alarm)
    band_x, band_width = xs[first] - step / 2, xs[last] - xs[first] + step
    first_top, last_bottom = panels[0][2], panels[-1][3]
    parts = [
        f'<rect class="win" x="{_n(band_x)}" y="{_n(first_top)}" width="{_n(band_width)}" '
        f'height="{_n(last_bottom - first_top)}"/>'
    ]
    for index, (title, ticks, top, bottom) in enumerate(panels):
        scale = Scale(ticks[0], ticks[-1], bottom, top)
        body = [
            _text(left, top - 12, title, cls="s12 b ink"),
            _y_axis(scale, ticks, _tick_labels(ticks), left, right),
            f'<line class="axis" x1="{_n(left)}" x2="{_n(right)}" y1="{_n(bottom)}" y2="{_n(bottom)}"/>',
        ]

        def ys(key: str, scale: Scale = scale) -> list[float | None]:
            return [scale(float(v)) if _known(v) else None for v in demo[key]]

        def mark(key: str, at: int, cls: str, points: list[float | None]) -> str:
            y = points[at]
            return (
                f'<circle class="{cls}" cx="{_n(xs[at])}" cy="{_n(y)}" r="4"/>'
                if y is not None
                else ""
            )

        if index == 0:
            fact = ys("y")
            body.append(f'<path class="ln fc" d="{line_path(xs, ys("forecast"))}"/>')
            body.append(f'<path class="ln fact" d="{line_path(xs, fact)}"/>')
            body.append(mark("y", tau, "pt", fact))
        elif index == 1:
            zero = scale(0.0)
            residual = ys("z")
            body.append(
                f'<line class="axis" x1="{_n(left)}" x2="{_n(right)}" y1="{_n(zero)}" y2="{_n(zero)}"/>'
            )
            body.append(f'<path class="ln fact" d="{line_path(xs, residual)}"/>')
            body.append(mark("z", tau, "pt", residual))
        else:
            line = scale(threshold)
            statistic = ys("statistic")
            body.append(
                f'<line class="thr" x1="{_n(left)}" x2="{_n(right)}" y1="{_n(line)}" y2="{_n(line)}"/>'
            )
            body.append(
                _text(
                    right,
                    line - 6,
                    f"порог {format_number(threshold, 1)}",
                    cls="s11 b ink2",
                    anchor="end",
                )
            )
            body.append(f'<path class="ln fc" d="{line_path(xs, statistic)}"/>')
            body.append(mark("statistic", alarm, "pt alarm", statistic))
        parts.append(f'<g class="panel">{"".join(body)}</g>')

    head = 18.0
    delay = alarm - tau
    when = (
        "тот же месяц"
        if delay == 0
        else f"через {delay} {'месяц' if delay == 1 else 'месяца' if delay < 5 else 'месяцев'}"
    )
    parts.append(_text(xs[tau] - 12, head + 4, "слом внесён", cls="s11 b c-event", anchor="end"))
    parts.append(
        f'<rect class="dia" x="{_n(xs[tau] - 4)}" y="{_n(head - 4)}" width="8" height="8" '
        f'transform="rotate(45 {_n(xs[tau])} {_n(head)})"/>'
    )
    parts.append(f'<path class="tri" d="M{_n(xs[alarm] + 14)} {_n(head - 5)}l-5.5 9h11z"/>')
    parts.append(_text(xs[alarm] + 25, head + 4, f"тревога · {when}", cls="s11 b c-alarm"))
    for i, month in enumerate(months):
        if i % 3 == 0:
            parts.append(
                _text(
                    xs[i],
                    height - 14,
                    month_tick(month, with_year=month.endswith("-01")),
                    anchor="middle",
                )
            )
    parts.append(
        f'<rect class="pick" data-pick x="{_n(xs[alarm] - step / 2)}" y="{_n(first_top)}" '
        f'width="{_n(step)}" height="{_n(last_bottom - first_top)}"/>'
    )
    parts.append(
        f'<rect class="hit" data-hit tabindex="0" role="slider" aria-label="Месяц примера" '
        f'aria-valuemin="0" aria-valuemax="{len(months) - 1}" aria-valuenow="{alarm}" '
        f'x="{_n(left)}" y="{_n(first_top)}" width="{_n(right - left)}" height="{_n(last_bottom - first_top)}"/>'
    )
    label = (
        f"Ряд бенчмарка со сломом в месяце {month_tick(str(demo['break_month']), True)}; тревога "
        f"детектора {name} — {month_tick(months[alarm], True)}"
    )
    geometry = f' data-x0="{_n(xs[0])}" data-step="{step:.3f}"'
    return _svg(DEMO_SIZE, label, "".join(parts), geometry, role="group")
