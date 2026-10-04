"""Сборщики блоков лендинга «данные → расчёт → результат».

Чистые функции: получают уже прочитанные таблицы, конфиги и сводки и возвращают простые
структуры для отрисовки. Числа и выводы на странице — то, что вернули эти функции; текст не
знает результата заранее.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from sbx.core.changepoint.online import cusum_path, page_hinkley_path
from sbx.core.changepoint.residuals import causal_zscores, seasonal_naive_residuals
from sbx.core.features.asof import latest_published_month
from sbx.core.folds import Fold
from sbx.core.regions import region_by_code
from sbx.core.report import (
    CONTRIBUTION_SUBJECTS,
    FOUNDATION_PREFIXES,
    SOURCE_BASES,
    agreeing_windows,
    contribution_statements,
    counted,
    detector_headline,
    difference_brief,
    difference_phrase,
    format_number,
    is_steady,
    windows_label,
)

MONTH_NAMES = (
    "январь",
    "февраль",
    "март",
    "апрель",
    "май",
    "июнь",
    "июль",
    "август",
    "сентябрь",
    "октябрь",
    "ноябрь",
    "декабрь",
)
MONTH_NAMES_GENITIVE = (
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)


def month_label(month: pd.Timestamp) -> str:
    """«май 2024» — месяц в тексте и таблицах страницы."""
    month = pd.Timestamp(month)
    return f"{MONTH_NAMES[month.month - 1]} {month.year}"


def _day_label(day: pd.Timestamp) -> str:
    return f"{day.day} {MONTH_NAMES_GENITIVE[day.month - 1]} {day.year}"


def fold_timeline(folds: Sequence[Fold], months: Sequence[pd.Timestamp]) -> dict[str, Any]:
    """Окна проверки на оси месяцев панели: где обучение, где проверка.

    Позиции — индексы месяцев панели, обе границы включительно. Обучение каждого окна — все
    месяцы от начала панели до cutoff, проверка — следующие `horizon` месяцев.
    """
    months = pd.DatetimeIndex(months)
    position = {month: i for i, month in enumerate(months)}
    rows = []
    for fold in folds:
        last_test = fold.cutoff + pd.DateOffset(months=fold.horizon)
        if fold.cutoff not in position or last_test not in position:
            raise ValueError(f"окно {fold.name} за пределами панели")
        cutoff = position[fold.cutoff]
        rows.append(
            {
                "fold": fold.name,
                "horizon": int(fold.horizon),
                "cutoff": f"{fold.cutoff:%Y-%m}",
                "train": [0, cutoff],
                "test": [cutoff + 1, cutoff + int(fold.horizon)],
            }
        )
    rows.sort(key=lambda row: (row["horizon"], row["cutoff"]))
    horizons = []
    for horizon in sorted({row["horizon"] for row in rows}):
        lengths = [row["train"][1] + 1 for row in rows if row["horizon"] == horizon]
        horizons.append(
            {
                "horizon": horizon,
                "folds": len(lengths),
                "train_months": [min(lengths), max(lengths)],
                "test_months": horizon,
            }
        )
    return {"months": [f"{m:%Y-%m}" for m in months], "rows": rows, "horizons": horizons}


def timing_rows(
    origin: pd.Timestamp,
    lags: Mapping[str, int],
    titles: Mapping[str, str],
    spatial: bool = False,
    population: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Что видит прогноз от месяца `origin`: источник, лаг публикации, последний известный период.

    Последний известный период считает `latest_published_month` — та же функция, по которой
    признаки попадают в расчёт, так что таблица на странице не может разойтись с ним.
    `spatial` добавляет строки о таблицах организаторов уровня МО, `population` (сводка шага
    признаков) — строку о численности населения.

    Для полос на странице у строки есть вид (`kind`): `published` — известен по месяц
    `known_month` включительно, `ahead` — известен заранее на целевые месяцы, `static` — от
    времени не зависит; `tag` — короткая метка лага.
    """
    origin = pd.Timestamp(origin)
    last_day = origin + pd.offsets.MonthEnd(0)

    def published(source: str, lag: str, tag: str, month: pd.Timestamp, known: str = "") -> dict:
        """Источник, известный по месяц `month` включительно."""
        return {
            "source": source,
            "lag": lag,
            "known": known or month_label(month),
            "tag": tag,
            "kind": "published",
            "known_month": f"{pd.Timestamp(month):%Y-%m}",
        }

    def timeless(source: str, lag: str, tag: str, kind: str, known: str) -> dict:
        """Источник без последнего известного месяца: известен заранее или не зависит от времени."""
        return {
            "source": source,
            "lag": lag,
            "known": known,
            "tag": tag,
            "kind": kind,
            "known_month": None,
        }

    rows = [
        published(
            "Новости GDELT (по дате попадания события в базу)",
            "в день события",
            "лаг 0",
            latest_published_month(origin, 0),
        ),
        published(
            "Календарь событий: решения Банка России, паводки, чрезвычайные ситуации",
            "по дате публикации события",
            "по дате публикации",
            origin,
            f"опубликованное до {_day_label(last_day)}",
        ),
    ]
    for lag in sorted(set(lags.values())):
        names = [titles.get(slug, slug) for slug, value in lags.items() if value == lag]
        rows.append(
            published(
                "; ".join(names), f"{lag} дней", f"{lag} дней", latest_published_month(origin, lag)
            )
        )
    rows.append(
        timeless(
            "Производственный календарь: рабочие дни, праздники",
            "известен заранее",
            "известен заранее",
            "ahead",
            "целевой месяц",
        )
    )
    if spatial:
        rows += [
            published(
                "Расходы соседних МО (набор организаторов)",
                "как собственный ряд",
                "как свой ряд",
                origin,
            ),
            timeless(
                "Доступность рынков и расстояния между МО (набор организаторов)",
                "постоянная характеристика",
                "постоянная",
                "static",
                f"расчёт организаторов за {ACCESS_YEAR} год",
            ),
        ]
    if population:
        known = f"оценка на 1 января {int(population['year'])} года"
        if population.get("published"):
            day = _day_label(pd.Timestamp(population["published"]))
            known += f", опубликована к {day} года"
        rows.append(
            timeless(
                "Численность населения МО (Росстат)",
                "постоянная характеристика",
                "постоянная",
                "static",
                known,
            )
        )
    return rows


def _population_timing(population: Mapping[str, Any], first_forecast: str | None) -> str:
    """Когда стала известна численность населения — и раньше ли первого момента прогноза.

    Постоянная характеристика одна на все окна проверки, поэтому сравнивается с самым ранним
    моментом прогноза; «до» — строго раньше его дня.
    """
    text = f"постоянная характеристика: оценка на 1 января {int(population['year'])} года"
    if not population.get("published"):
        return text
    published = pd.Timestamp(population["published"])
    text += f" в редакции, опубликованной к {_day_label(published)} года"
    if not first_forecast:
        return text
    first = pd.Timestamp(first_forecast)
    if published < first:
        return f"{text}, — до первого момента прогноза ({_day_label(first)} года)"
    return f"{text}, — позже части моментов прогноза (первый — {_day_label(first)} года)"


DETECTOR_PATHS = {"page_hinkley": page_hinkley_path, "cusum": cusum_path}
# Задержка, при которой пример ещё читается как «тревога сразу после слома».
DEMO_MAX_DELAY = 2


def _clean(values: np.ndarray) -> list[float | None]:
    return [None if not np.isfinite(v) else float(v) for v in values]


def detector_demo(
    series: pd.DataFrame,
    labels: pd.DataFrame,
    seasonal: pd.DataFrame,
    detector: str,
    threshold: float,
    ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Пример работы детектора на ряду бенчмарка с известным сломом.

    Возвращает ряд, одношаговый прогноз, стандартизованный остаток и статистику детектора с
    порогом — то, что детектор видел на каждом шаге. Берётся ряд со скачком уровня, на котором
    до слома тревог нет, а первая тревога приходит не позже чем через `DEMO_MAX_DELAY` месяца;
    из таких — с наибольшим скачком. Если подходящего ряда нет, возвращается пустой словарь.
    """
    if detector not in DETECTOR_PATHS:
        raise ValueError(f"для детектора {detector!r} нет трассы статистики")
    path = DETECTOR_PATHS[detector]
    index = seasonal.pivot_table(index="category", columns="month", values="index")
    candidates = labels[labels["kind"] == "level_shift"]
    if ids is not None:
        candidates = candidates[candidates["unique_id"].isin(set(ids))]
    candidates = candidates.assign(size=candidates["delta"].abs()).sort_values(
        ["size", "unique_id"], ascending=[False, True]
    )
    by_id = {uid: g.sort_values("ds") for uid, g in series.groupby("unique_id")}
    for label in candidates.itertuples():
        frame = by_id.get(label.unique_id)
        if frame is None:
            continue
        values = frame["y"].to_numpy(dtype=float)
        category = frame["category"].iloc[0]
        table = index.loc[category].to_dict() if category in index.index else {}
        residual = seasonal_naive_residuals(values, frame["ds"].dt.month.to_numpy(), table)
        z = causal_zscores(residual, center=False)
        statistic, alarms = path(np.nan_to_num(z, nan=0.0), threshold=threshold)
        tau = int(label.tau)
        if not alarms or alarms[0] < tau or alarms[0] - tau > DEMO_MAX_DELAY:
            continue
        months = [f"{d:%Y-%m}" for d in frame["ds"]]
        return {
            "unique_id": str(label.unique_id),
            "source_id": str(label.source_id),
            "category": str(category),
            "kind": str(label.kind),
            "delta": float(label.delta),
            "detector": detector,
            "threshold": float(threshold),
            "months": months,
            "y": [float(v) for v in values],
            "forecast": _clean(values - residual),
            "z": _clean(z),
            "statistic": _clean(np.where(np.isfinite(z), statistic, np.nan)),
            "tau": tau,
            "break_month": months[tau],
            "alarms": [int(a) for a in alarms],
            "delay": int(alarms[0] - tau),
        }
    return {}


REFERENCE_MODEL = "Prophet"


def _role(model: str) -> str:
    if model == REFERENCE_MODEL:
        return "reference"
    if model.startswith("Ensemble"):
        return "ensemble"
    return "foundation" if model.startswith(FOUNDATION_PREFIXES) else "other"


def foundation_view(
    horizon_rows: Sequence[Mapping[str, Any]], weights: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Foundation-модели рядом с лучшей из остальных моделей, эталоном и ансамблем.

    Остальные одиночные модели сведены в одну строку — лучшая на каждом горизонте, с именем:
    так видно, где предобученные модели выигрывают у обученных на этих данных, а где нет.
    """
    table = pd.DataFrame(list(horizon_rows))
    if table.empty:
        return {}
    columns = [c for c in table.columns if c != "Модель"]
    horizons = [int(c.split("=")[-1]) for c in columns]
    table = table.assign(role=table["Модель"].map(_role))

    def row(model: str, role: str, values: Sequence[float], **extra: Any) -> dict[str, Any]:
        return {"model": model, "role": role, "values": [float(v) for v in values], **extra}

    rows = [
        row(r["Модель"], "foundation", [r[c] for c in columns])
        for _, r in table[table["role"] == "foundation"].iterrows()
    ]
    others = table[table["role"] == "other"]
    best_foundation = table[table["role"] == "foundation"][columns].min()
    best_other = others[columns].min() if len(others) else pd.Series(np.nan, index=columns)
    if len(others):
        names = [str(others.loc[others[c].idxmin(), "Модель"]) for c in columns]
        rows.append(
            row("Лучшая из остальных одиночных моделей", "best_other", best_other, names=names)
        )
    for role in ("reference", "ensemble"):
        part = table[table["role"] == role]
        if len(part):
            best = part.loc[part[columns].mean(axis=1).idxmin()]
            rows.append(row(str(best["Модель"]), role, [best[c] for c in columns]))
    reference = table.loc[table["role"] == "reference", columns].min()
    return {
        "horizons": horizons,
        "rows": rows,
        "wins": [
            h for h, c in zip(horizons, columns, strict=True) if best_foundation[c] < best_other[c]
        ],
        "beats_reference": [
            h
            for h, c in zip(horizons, columns, strict=True)
            if len(reference) and best_foundation[c] < reference[c]
        ],
        "weights": [
            {
                "repo": str(spec.get("repo_id", name)),
                "revision": str(spec.get("revision", ""))[:8],
                "license": str(spec.get("weights_license", "")),
            }
            for name, spec in weights.items()
        ],
    }


def contribution_view(dm: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Вклад слоёв решения для диаграммы: разница MAE, значимость и направление.

    Порядок — как в `CONTRIBUTION_SUBJECTS`; формулировка — та же, что в отчёте.
    """
    rows = []
    for pair, subject in CONTRIBUTION_SUBJECTS.items():
        entry = dm.get(pair)
        if not entry:
            continue
        delta = float(entry.get("mean_diff", float("nan")))
        p_value = float(entry.get("p_value", float("nan")))
        if not np.isfinite(delta):
            continue
        # Столбец окрашен, только если знак держится по окнам проверки.
        significant = is_steady(entry)
        agree, folds = agreeing_windows(entry) or (None, None)
        rows.append(
            {
                "pair": pair,
                "subject": subject,
                "delta": delta,
                "p_value": p_value,
                "significant": significant,
                "direction": ("better" if delta < 0 else "worse") if significant else "none",
                "windows": windows_label(entry),
                "agree": agree,
                "folds": folds,
                "statement": contribution_statements({pair: entry})[0],
            }
        )
    return rows


def _period(first: str, last: str) -> str:
    return f"{month_label(pd.Timestamp(first))} — {month_label(pd.Timestamp(last))}"


def _gave(dm: Mapping[str, Mapping[str, Any]], pair: str) -> str:
    """Измеренный вклад слоя для таблицы источников — без повторения его названия."""
    return (difference_brief(dm[pair]) if dm.get(pair) else "") or UNMEASURED


def source_statement(sources: Sequence[Mapping[str, Any]], block: str) -> str:
    """Измеренный вклад одного источника: его блок, добавленный к базовой модели и поверх
    лучшего прежнего набора. Не измерено — пустая строка."""
    parts = []
    for row in sources:
        if row.get("block") != block:
            continue
        brief = difference_brief(row)
        if brief:
            base = SOURCE_BASES.get(str(row["against"]), f"к варианту {row['against']}")
            parts.append(f"{base}: {brief}")
    return "; ".join(parts)


def source_view(sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Вклад каждого источника для диаграммы: по столбцу на пару «блок, к чему добавлен»."""
    rows = []
    for row in sources:
        phrase = difference_phrase(row)
        if not phrase:
            continue
        delta, p_value = float(row["mean_diff"]), float(row.get("p_value", float("nan")))
        significant = is_steady(row)
        base = SOURCE_BASES.get(str(row["against"]), f"к варианту {row['against']}")
        subject = f"{row['title']} — {base}"
        agree, folds = agreeing_windows(row) or (None, None)
        rows.append(
            {
                "pair": f"{row['block']}_on_{row['against']}",
                "subject": subject,
                "title": str(row["title"]),
                "base": base,
                "delta": delta,
                "p_value": p_value,
                "significant": significant,
                "direction": ("better" if delta < 0 else "worse") if significant else "none",
                "windows": windows_label(row),
                "agree": agree,
                "folds": folds,
                "statement": f"{subject}: {phrase}.",
            }
        )
    return rows


def source_verdict(sources: Sequence[Mapping[str, Any]]) -> str:
    """Одна фраза о вкладе источников: что устойчиво помогает и что устойчиво вредит.

    Устойчиво — значит, знак разности держится по окнам проверки. Если такого нет ни у одного
    источника, называется ближайший к этому: тот, с которым прогноз точнее в наибольшей доле окон.
    """
    measured = [row for row in sources if difference_phrase(row)]
    if not measured:
        return ""

    def name(row: Mapping[str, Any]) -> str:
        base = SOURCE_BASES.get(str(row["against"]), f"к варианту {row['against']}")
        return f"{row['title']} ({base})"

    better = [name(r) for r in measured if is_steady(r) and float(r["mean_diff"]) < 0]
    worse = [name(r) for r in measured if is_steady(r) and float(r["mean_diff"]) > 0]
    parts = []
    if better:
        parts.append(f"Устойчиво снижают ошибку: {', '.join(better)}.")
    if worse:
        parts.append(f"Устойчиво повышают: {', '.join(worse)}.")
    if parts:
        return " ".join(parts)
    text = "Ни один источник не изменил ошибку устойчиво: знак разности меняется от окна к окну."
    helping = [r for r in measured if float(r["mean_diff"]) < 0 and r.get("folds")]
    if helping:
        best = max(
            helping,
            key=lambda r: (int(r["folds_better"]) / int(r["folds"]), -float(r["mean_diff"])),
        )
        text += f" Ближе всех — {name(best)}: лучше {windows_label(best)}."
    return text


# Индекс доступности рынков организаторы рассчитали «в 2024 году» (описание набора данных).
ACCESS_YEAR = 2024
UNMEASURED = "не измерено в этом прогоне"
ORGANISERS = "набор организаторов, Лаборатория СберИндекс"
# Группы таблицы источников на странице: уровень, который описывает источник.
SOURCE_GROUPS = {
    "spending": "Основа панели · уровень МО",
    "reference": "Основа панели · уровень МО",
    "national": "Уровень страны",
    "calendar": "Уровень страны",
    "news": "Уровень региона и страны",
    "events": "Уровень региона и страны",
    "access": "Уровень МО",
    "neighbours": "Уровень МО",
    "weather": "Уровень МО",
    "population": "Уровень МО",
}


def source_effects(sources: Sequence[Mapping[str, Any]], block: str) -> list[dict[str, Any]]:
    """Измеренный вклад источника числами: к чему добавлен блок, разность MAE, счёт окон, вывод.

    `direction` — `better` или `worse`, только если знак разности держится по окнам проверки;
    иначе `none`. Те же правила, что у формулировки `difference_brief`.
    """
    effects = []
    for row in sources:
        if row.get("block") != block:
            continue
        delta = float(row.get("mean_diff", float("nan")))
        if not np.isfinite(delta):
            continue
        agreeing = agreeing_windows(row)
        effects.append(
            {
                "base": SOURCE_BASES.get(str(row["against"]), f"к варианту {row['against']}"),
                "delta": delta,
                "agree": agreeing[0] if agreeing else None,
                "folds": agreeing[1] if agreeing else None,
                "direction": ("better" if delta < 0 else "worse") if is_steady(row) else "none",
            }
        )
    return effects


def source_inventory(
    facts: Mapping[str, Any],
    dm: Mapping[str, Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Таблица источников: что берём, объём и период, уровень, согласование по времени, вклад.

    `facts` — то, что шаг отчёта прочитал из манифестов, конфигов и артефактов; источника нет
    в `facts` — нет и строки. Вклад — измеренный для самого источника (`sources`: его блок
    против модели без него); если такого измерения нет, берётся тест по парам конфигураций.

    Строки `source`, `timing`, `gave` — для таблицы отчёта. Для страницы те же сведения
    разложены по полям: `title` и `provider` — название и поставщик, `tag` и `note` — метка
    согласования по времени и пояснение к ней, `group` — уровень, `effects` — измеренный вклад
    числами (см. `source_effects`).
    """
    rows: list[dict[str, Any]] = []
    spending = facts.get("spending")
    if spending:
        rows.append(
            {
                "key": "spending",
                "source": "Расходы на уровне МО — набор организаторов (Лаборатория СберИндекс)",
                "what": "цель прогноза: безналичные расходы по шести категориям",
                "volume": (
                    f"{format_number(spending['rows'])} строк: "
                    f"{format_number(spending['territories'])} МО, "
                    f"{counted(spending['series'], 'ряд', 'ряда', 'рядов')}; "
                    f"{_period(spending['first'], spending['last'])}"
                ),
                "level": "МО",
                "timing": "прогноз от месяца использует ряд по этот месяц включительно",
                "gave": "цель прогноза",
                "license": "CC BY-SA 4.0",
                "title": "Расходы на уровне МО",
                "provider": ORGANISERS,
                "tag": "по месяц прогноза",
                "note": "прогноз от месяца использует ряд по этот месяц включительно",
            }
        )
    reference = facts.get("reference")
    if reference:
        rows.append(
            {
                "key": "reference",
                "source": "Справочник территорий: границы и преобразования МО (СберИндекс)",
                "what": "название, тип, регион, ОКТМО и координаты центра МО",
                "volume": (
                    f"{format_number(reference['territories'])} МО панели, редакция "
                    f"{reference['year']} года; координаты центра — у "
                    f"{format_number(reference['with_coordinates'])}"
                ),
                "level": "МО",
                "timing": "от времени не зависит",
                "gave": "регион и тип МО — статические признаки и ключ стыковки новостей",
                "license": "CC BY-SA 4.0",
                "title": "Справочник территорий: границы и преобразования МО",
                "provider": "СберИндекс",
                "tag": "не зависит от времени",
                "note": "",
            }
        )
    national = facts.get("national") or []
    if national:
        first = min(pd.Timestamp(item["first"]) for item in national)
        last = max(pd.Timestamp(item["last"]) for item in national)
        lags = sorted({int(item["lag_days"]) for item in national})
        frequencies = sorted({str(item["frequency"]) for item in national})
        features = facts.get("national_features")
        span = f"{lags[0]}–{lags[-1]}" if len(lags) > 1 else f"{lags[0]}"
        rows.append(
            {
                "key": "national",
                "source": "Национальные ряды СберИндекса: расходы, зарплаты, обороты, ставка",
                "what": (
                    (
                        f"{counted(features, 'макропризнак', 'макропризнака', 'макропризнаков')}; "
                        if features
                        else ""
                    )
                    + "сезонный индекс; проверка детекторов на известных шоках"
                ),
                "volume": (
                    f"{counted(len(national), 'ряд', 'ряда', 'рядов')}; "
                    f"с {MONTH_NAMES_GENITIVE[first.month - 1]} {first.year} по "
                    f"{month_label(last)}; {' и '.join(frequencies)}"
                ),
                "level": "страна",
                "timing": (
                    f"лаг публикации {lags[0]}–{lags[-1]} дней от конца периода"
                    if len(lags) > 1
                    else f"лаг публикации {lags[0]} дней от конца периода"
                ),
                "gave": source_statement(sources, "macro") or _gave(dm, "B_vs_A"),
                "license": "Данные СберИндекса: при копировании необходимо упоминание",
                "title": "Национальные ряды СберИндекса: расходы, зарплаты, обороты, ставка",
                "provider": "",
                "tag": f"лаг {span} дней",
                "note": "лаг публикации от конца периода",
                "effects": source_effects(sources, "macro"),
            }
        )
    calendar = facts.get("calendar")
    if calendar:
        rows.append(
            {
                "key": "calendar",
                "source": "Производственный календарь России",
                "what": "рабочие дни, праздники, сокращённые дни, месяцы распродаж",
                "volume": f"{calendar['first_year']}–{calendar['last_year']} годы",
                "level": "страна",
                "timing": "известен заранее: берётся на целевой месяц",
                "gave": source_statement(sources, "calendar")
                or "входит в сравнение с национальными рядами",
                "license": "официальные сведения; условий использования источник не публикует",
                "title": "Производственный календарь России",
                "provider": "",
                "tag": "известен заранее",
                "note": "берётся на целевой месяц",
                "effects": source_effects(sources, "calendar"),
            }
        )
    news = facts.get("news")
    if news:
        rows.append(
            {
                "key": "news",
                "source": "Новости GDELT 1.0",
                "what": (
                    f"{counted(news['columns'], 'признак', 'признака', 'признаков')}: число "
                    "событий, тон, доли типов событий, новизна"
                ),
                "volume": (
                    f"{format_number(news['events'])} событий, "
                    f"{counted(news['regions'], 'регион', 'региона', 'регионов')}; "
                    f"{_period(news['first'], news['last'])}"
                ),
                "level": "регион, страна",
                "timing": "по дате попадания события в базу; лаг публикации 0",
                "gave": source_statement(sources, "gdelt") or _gave(dm, "C_vs_B"),
                "license": "свободное использование при ссылке на проект GDELT",
                "title": "Новости GDELT 1.0",
                "provider": "",
                "tag": "лаг 0",
                "note": "по дате попадания события в базу",
                "effects": source_effects(sources, "gdelt"),
            }
        )
    events = facts.get("events")
    if events:
        rows.append(
            {
                "key": "events",
                "source": "Календарь событий — составлен авторами по официальным источникам",
                "what": "решения Банка России по ставке, паводки, чрезвычайные ситуации",
                "volume": (
                    f"{counted(events['count'], 'событие', 'события', 'событий')}; "
                    f"{_period(events['first'], events['last'])}"
                ),
                "level": "регион, страна",
                "timing": "по дате публикации события",
                "gave": source_statement(sources, "events") or "входит в сравнение с новостями",
                "license": "составлен авторами решения; у каждой записи ссылка на источник",
                "title": "Календарь событий",
                "provider": "составлен авторами по официальным источникам",
                "tag": "по дате публикации",
                "note": "событие входит с даты его публикации",
                "effects": source_effects(sources, "events"),
            }
        )
    spatial = facts.get("spatial")
    if spatial:
        rows.append(
            {
                "key": "access",
                "source": "Индекс доступности рынков — набор организаторов (Лаборатория СберИндекс)",
                "what": (
                    "логарифм индекса доступности рынков; логарифм среднего расстояния до "
                    f"{spatial['neighbours']} ближайших МО"
                ),
                "volume": (
                    f"{format_number(spatial['with_market_access'])} из "
                    f"{format_number(spatial['territories'])} МО панели; расчёт организаторов за "
                    f"{ACCESS_YEAR} год"
                ),
                "level": "МО",
                "timing": (
                    "постоянная характеристика территории; рассчитана по данным "
                    f"{ACCESS_YEAR} года — позже части моментов прогноза"
                ),
                "gave": source_statement(sources, "access") or UNMEASURED,
                "license": "CC BY-SA 4.0",
                "title": "Индекс доступности рынков",
                "provider": ORGANISERS,
                "tag": "постоянная",
                "note": (
                    f"характеристика территории; рассчитана по данным {ACCESS_YEAR} года — "
                    "позже части моментов прогноза"
                ),
                "effects": source_effects(sources, "access"),
            }
        )
        share = round(100 * float(spatial["cross_region_share"]))
        rows.append(
            {
                "key": "neighbours",
                "source": "Автодорожные связи между МО — набор организаторов (Лаборатория СберИндекс)",
                "what": (
                    f"{spatial['neighbours']} ближайших по автодороге МО панели: прирост их "
                    "расходов той же категории за 1 и 3 месяца и уровень относительно них"
                ),
                "volume": (
                    f"{format_number(spatial['highway_pairs'])} пар МО панели по автодорогам; "
                    f"{spatial['neighbours']} ближайших соседей есть у "
                    f"{format_number(spatial['with_neighbours'])} МО, медианное расстояние "
                    f"{format_number(round(float(spatial['median_distance_km'])))} км, "
                    f"{share}% соседей — в другом регионе"
                ),
                "level": "МО",
                "timing": "расходы соседей — по месяц прогноза включительно, как и свой ряд",
                "gave": source_statement(sources, "neighbours") or UNMEASURED,
                "license": "CC BY-SA 4.0",
                "title": "Автодорожные связи между МО",
                "provider": ORGANISERS,
                "tag": "как свой ряд",
                "note": "расходы соседей — по месяц прогноза включительно",
                "effects": source_effects(sources, "neighbours"),
            }
        )
    weather = facts.get("weather")
    if weather:
        first, last = weather["norm_years"]
        rows.append(
            {
                "key": "weather",
                "source": "Погода NASA POWER (реанализ MERRA-2)",
                "what": (
                    "отклонение температуры и осадков от нормы ячейки сетки за последний "
                    "опубликованный месяц и за три месяца; норма целевого месяца"
                ),
                "volume": (
                    f"{counted(weather['cells'], 'ячейка', 'ячейки', 'ячеек')} сетки 0,5° × 0,625° "
                    f"для {format_number(weather['territories'])} МО; "
                    f"{_period(weather['first'], weather['last'])}, норма — {first}–{last} годы"
                ),
                "level": "МО",
                "timing": (
                    f"лаг публикации {weather['lag_days']} дней: к концу месяца известен "
                    "предыдущий; норма известна заранее"
                ),
                "gave": source_statement(sources, "weather") or UNMEASURED,
                "license": "открытые данные NASA; при использовании указывается источник",
                "title": "Погода NASA POWER",
                "provider": "реанализ MERRA-2",
                "tag": f"лаг {weather['lag_days']} дней",
                "note": "к концу месяца известен предыдущий; норма известна заранее",
                "effects": source_effects(sources, "weather"),
            }
        )
    population = facts.get("population")
    if population:
        share = format_number(100 * float(population["share"]), 1)
        threshold = round(100 * float(population["threshold"]))
        year = int(population["year"])
        rows.append(
            {
                "key": "population",
                "source": f"Росстат: численность населения МО на 1 января {year} года",
                "what": "логарифм численности населения и доля городского населения",
                "volume": (
                    f"{format_number(int(population['matched']))} из "
                    f"{format_number(int(population['territories']))} МО панели ({share}%); "
                    f"привязка по названию внутри региона через бюллетень {year + 1} года"
                ),
                "level": "МО",
                "timing": _population_timing(population, facts.get("first_forecast")),
                "gave": (
                    source_statement(sources, "population") or UNMEASURED
                    if population["used"]
                    else f"привязано меньше {threshold}% территорий — в прогнозе не "
                    "используется, справочный показатель"
                ),
                "license": (
                    "официальная статистика Росстата; условий использования на странице "
                    "бюллетеня нет, источник указывается"
                ),
                "title": f"Росстат: численность населения МО на 1 января {year} года",
                "provider": "",
                "tag": "постоянная",
                "note": _population_timing(population, facts.get("first_forecast")).removeprefix(
                    "постоянная характеристика: "
                ),
                "effects": source_effects(sources, "population") if population["used"] else [],
            }
        )
    for row in rows:
        row.setdefault("effects", [])
        row["group"] = SOURCE_GROUPS[row["key"]]
    return rows


def _join(values: Sequence[Any]) -> str:
    items = [str(v) for v in values]
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} и {items[-1]}"


def _leaders(winners: Sequence[Mapping[str, Any]]) -> str:
    """«1 и 3 мес. — Chronos-2 (covariates); 6 и 12 мес. — Ensemble»."""
    groups: dict[str, list[int]] = {}
    for winner in winners:
        groups.setdefault(str(winner["model"]), []).append(int(winner["horizon"]))
    return "; ".join(f"{_join(horizons)} мес. — {model}" for model, horizons in groups.items())


def chain_summary(
    facts: Mapping[str, Any],
    folds: int,
    models: int,
    winners: Sequence[Mapping[str, Any]],
    contrast: Mapping[str, Any],
    dm: Mapping[str, Mapping[str, Any]],
    horizons: Sequence[int] = (1, 3, 6, 12),
) -> dict[str, dict[str, Any]]:
    """Три колонки «Данные → Расчёт → Результат»: короткие строки со ссылками на разделы.

    Каждая строка собрана из чисел расчёта. Строки без данных не появляются: нет сравнения
    детекторов — нет и строки о нём.
    """
    # Строка колонки — подпись и значение; `text` — они же одной строкой, как в отчёте.
    data: list[tuple[str, str]] = []
    spending = facts.get("spending")
    if spending:
        months = (pd.Period(spending["last"], "M") - pd.Period(spending["first"], "M")).n + 1
        data.append(
            (
                "Расходы МО",
                f"{format_number(spending['territories'])} МО × "
                f"{spending.get('categories', 6)} категорий × "
                f"{counted(months, 'месяц', 'месяца', 'месяцев')}",
            )
        )
    if facts.get("national"):
        count = counted(len(facts["national"]), "ряд", "ряда", "рядов")
        data.append(("Национальные ряды СберИндекса", count))
    if facts.get("news"):
        data.append(("Новости GDELT", f"{format_number(facts['news']['events'])} событий"))
    if facts.get("events"):
        count = counted(facts["events"]["count"], "событие", "события", "событий")
        data.append(("Календарь событий", count))
    if facts.get("calendar"):
        calendar = facts["calendar"]
        data.append(
            ("Производственный календарь", f"{calendar['first_year']}–{calendar['last_year']}")
        )
    if facts.get("spatial"):
        data.append(
            (
                "Таблицы организаторов уровня МО",
                f"доступность рынков и {facts['spatial']['neighbours']} ближайших соседей",
            )
        )
    if facts.get("weather"):
        cells = counted(facts["weather"]["cells"], "ячейка", "ячейки", "ячеек")
        data.append(("Погода NASA POWER", f"{cells} сетки"))
    population = facts.get("population") or {}
    if population.get("used"):
        data.append(
            (
                "Население МО (Росстат)",
                f"{format_number(int(population['matched']))} из "
                f"{format_number(int(population['territories']))} МО",
            )
        )
    data.append(("Все источники, лицензии и подписи", ""))

    # (подпись, значение, строка целиком, якорь)
    method = [
        (
            "Проверка",
            f"{counted(folds, 'окно', 'окна', 'окон')}, горизонты {_join(horizons)} месяцев, "
            "метрика MAE",
            "",
            "#protocol",
        ),
        (
            "Модели",
            f"{counted(models, 'модель', 'модели', 'моделей')} и ансамбль по прошлым ошибкам",
            f"{counted(models, 'модель', 'модели', 'моделей')} и ансамбль по прошлым ошибкам",
            "#pipeline",
        ),
        (
            "Внешние данные",
            "только известные на момент прогноза",
            "Внешние данные — только известные на момент прогноза",
            "#timing",
        ),
        (
            "Сдвиги",
            "ищем в ошибках прогноза, а не в самом ряде",
            "Сдвиги ищем в ошибках прогноза, а не в самом ряде",
            "#detector-demo",
        ),
    ]

    result: list[tuple[str, str]] = []
    if winners:
        beaten = sum(1 for w in winners if w.get("reference_mae") and w["mae"] < w["reference_mae"])
        noun = "горизонте" if beaten == 1 else "горизонтах"
        result.append(
            (f"MAE ниже, чем у Prophet, на {beaten} {noun} из {len(winners)}", "#horizons")
        )
        result.append((f"Лидеры: {_leaders(winners)}", "#horizons"))
    if contrast:
        result.append((detector_headline(contrast), "#changepoints"))
    if winners:
        result.append(("Foundation-модели против остальных — отдельное сравнение", "#foundation"))
    if dm:
        result.append(("Вклад каждого источника: что помогло, а что нет", "#contribution"))
    result.append(("Разборы реальных примеров и ограничения", "#cases"))

    def line(label: str, value: str, text: str, href: str) -> dict[str, str]:
        joined = f"{label}: {value}" if value else label
        return {"text": text or joined, "href": href, "label": label, "value": value}

    return {
        "data": {
            "title": "1. Данные",
            "lines": [line(label, value, "", "#data") for label, value in data],
        },
        "method": {"title": "2. Расчёт", "lines": [line(*item) for item in method]},
        "result": {
            "title": "3. Результат",
            "lines": [line("", "", text, href) for text, href in result],
        },
    }


# Критерий, вес, ссылки на разделы, раздел отчёта, артефакт, короткое имя, основной раздел.
CRITERIA = (
    (
        "Простое и понятное объяснение методологии",
        10,
        (("цепочка", "chain"), ("схема решения", "pipeline"), ("окна проверки", "protocol")),
        "разделы 1–3",
        "reports/methodology.md",
        "Объяснение методологии",
        "pipeline",
    ),
    (
        "Сравнение моделей прогнозирования, Prophet, горизонты 1, 3, 6 и 12 месяцев",
        20,
        (("MAE по горизонтам", "horizons"), ("все метрики", "metrics")),
        "раздел 4",
        "artifacts/metrics/leaderboard_by_horizon.csv",
        "Модели прогноза, Prophet, горизонты 1, 3, 6 и 12",
        "horizons",
    ),
    (
        "Сравнение моделей обнаружения точек структурных изменений",
        20,
        (("сравнение детекторов", "changepoints"), ("как детектор находит сдвиг", "detector-demo")),
        "раздел 5",
        "artifacts/cpd/detector_comparison.csv",
        "Обнаружение структурных изменений",
        "changepoints",
    ),
    (
        "Современные фундаментальные модели временных рядов",
        15,
        (("foundation-модели", "foundation"),),
        "раздел 6",
        "configs/models.yaml",
        "Foundation-модели временных рядов",
        "foundation",
    ),
    (
        "Интеграция новостей и способ их согласования с данными СберИндекса",
        15,
        (
            ("согласование по времени", "timing"),
            ("вклад источников", "contribution"),
            ("источники", "data"),
        ),
        "разделы 7 и 9",
        "configs/features.yaml",
        "Новости и согласование с данными СберИндекса",
        "timing",
    ),
    (
        "Метрики: MAE обязательно, R² опционально",
        10,
        (
            ("определение MAE", "protocol"),
            ("MAE по горизонтам", "horizons"),
            ("все метрики", "metrics"),
        ),
        "раздел 4",
        "artifacts/metrics/summary.csv",
        "Метрики: MAE обязательно, R² опционально",
        "protocol",
    ),
    (
        "Интерпретация, ясность, воспроизводимость и обоснованность выводов",
        10,
        (
            ("разборы примеров", "cases"),
            ("ограничения", "limits"),
            ("воспроизведение", "reproduce"),
        ),
        "разделы 10–12",
        "README.md",
        "Интерпретация, ясность, воспроизводимость",
        "limits",
    ),
)


def criteria_map(present: Collection[str] | None = None) -> list[dict[str, Any]]:
    """Критерий конкурса → разделы страницы → раздел отчёта → артефакт.

    `present` — якоря разделов, которые есть на странице: ссылка на отсутствующий раздел не
    ставится. Без `present` возвращаются все ссылки.

    `code` и `short` — обозначение и короткое имя критерия для оглавления; `primary` — раздел,
    у которого в оглавлении стоит вес критерия (`None`, если раздела на странице нет).
    """
    return [
        {
            "criterion": criterion,
            "weight": weight,
            "links": [
                {"text": text, "href": f"#{anchor}"}
                for text, anchor in links
                if present is None or anchor in present
            ],
            "report": report,
            "artifact": artifact,
            "code": f"К{number}",
            "short": short,
            "primary": primary if present is None or primary in present else None,
        }
        for number, (criterion, weight, links, report, artifact, short, primary) in enumerate(
            CRITERIA, start=1
        )
    ]


def news_timeline(news: pd.DataFrame, events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Число новостей о стране по месяцам и события календаря, попавшие в этот период."""
    if news.empty or "country_news_events" not in news.columns:
        return {}
    monthly = news.drop_duplicates("ds").sort_values("ds")
    months = [f"{d:%Y-%m}" for d in monthly["ds"]]
    marks = []
    for event in events:
        month = f"{pd.Timestamp(event['date']):%Y-%m}"
        if month not in months:
            continue
        scope = str(event.get("region_code") or "all")
        marks.append(
            {
                "month": month,
                "date": str(event["date"]),
                "kind": str(event["kind"]),
                "description": str(event.get("description", "")),
                "scope": "вся страна" if scope == "all" else region_by_code(scope).name,
            }
        )
    return {
        "months": months,
        "counts": [int(v) for v in monthly["country_news_events"]],
        "events": sorted(marks, key=lambda mark: mark["date"]),
    }
