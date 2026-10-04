"""Сборка данных отчёта и лендинга из сохранённых артефактов (чистые преобразования)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from sbx.core.changepoint.residuals import freshest_forecast

RU_METRIC_NAMES = {
    "mae": "MAE, руб.",
    "mae_macro": "MAE macro, руб.",
    "rmse": "RMSE",
    "wape": "WAPE",
    "r2_pooled": "R² pooled",
    "r2_per_series_median": "R² per-series (медиана)",
    "mase": "MASE",
    "f1": "F1",
    "covering": "covering",
    "mean_delay": "средняя задержка, мес.",
    "false_alarms_per_series_year": "ложные тревоги на ряд-год",
    "mean_lead_time": "средний lead time, мес.",
}


def format_number(value: Any, digits: int = 2) -> str:
    """Число для текста: пробел между тысячами, запятая в дроби, знак минуса — «−», не дефис."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, int | np.integer):
        text = f"{abs(int(value)):,}".replace(",", " ")
        return f"−{text}" if int(value) < 0 else text
    text = f"{abs(float(value)):,.{digits}f}".replace(",", " ").replace(".", ",")
    # Ноль после округления знака не носит: «−0,0» читается как ошибка.
    negative = float(value) < 0 and any(ch not in "0, " for ch in text)
    return f"−{text}" if negative else text


def format_delta(value: Any, digits: int = 1) -> str:
    """Разность для столбца таблицы: со знаком, «+18,9» и «−58,7»; ноль знака не носит."""
    text = format_number(None if value is None else float(value), digits)
    positive = text != "—" and float(value) > 0 and any(ch not in "0, " for ch in text)
    return f"+{text}" if positive else text


def interval_runs(low: Sequence[Any], high: Sequence[Any]) -> list[tuple[int, int]]:
    """Отрезки `[начало, конец)`, на которых известны обе границы интервала.

    Полоса интервала рисуется по этим отрезкам: где у прогноза интервала нет, нет и полосы, а
    соседние отрезки не соединяются через пропуск.
    """

    def known(value: Any) -> bool:
        return value is not None and bool(np.isfinite(value))

    runs, start = [], None
    for i, (lo, hi) in enumerate(zip(low, high, strict=True)):
        if known(lo) and known(hi):
            start = i if start is None else start
        elif start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(low)))
    return runs


FEDERAL_CITY_PREFIX = "внутригородская территория города федерального значения "


def short_mo_name(name: Any) -> str:
    """Название МО для подписи: без официального префикса внутригородских территорий Москвы,
    Петербурга и Севастополя, который занимает больше места, чем само название."""
    if name is None or (isinstance(name, float) and not np.isfinite(name)):
        return ""
    text = str(name)
    return text[len(FEDERAL_CITY_PREFIX) :] if text.startswith(FEDERAL_CITY_PREFIX) else text


def counted(number: int, one: str, few: str, many: str) -> str:
    """Число с существительным в нужной форме: «1 ряд», «3 ряда», «12 рядов»."""
    n = abs(int(number))
    if n % 10 == 1 and n % 100 != 11:
        noun = one
    elif n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        noun = few
    else:
        noun = many
    return f"{format_number(int(number))} {noun}"


def horizon_label(months: int) -> str:
    """«1 месяц», «3 месяца», «12 месяцев» — подпись горизонта в тексте и на карточках."""
    return counted(months, "месяц", "месяца", "месяцев")


# Имена моделей для читателя страницы; в артефактах и отчёте остаются имена из расчёта.
MODEL_TITLES = {
    "Chronos-2 (covariates)": "Chronos-2 с ковариатами",
    "Ensemble": "Ансамбль",
    "Ensemble (по категориям)": "Ансамбль по категориям",
    "Naive": "Наивный",
    "SeasonalNaive": "Сезонный наивный",
}


def model_title(model: Any) -> str:
    """Имя модели на странице: «Chronos-2 с ковариатами», «Ансамбль»; без перевода — как есть."""
    return MODEL_TITLES.get(str(model), str(model))


# Семейства моделей — так они сгруппированы в таблице горизонтов и раскрашены на графике.
FOUNDATION_PREFIXES = ("Chronos", "TimesFM")
NAIVE_MODELS = ("Naive", "SeasonalNaive")
ML_MODELS = ("LightGBM", "NHITS", "NBEATSx")


def model_family(model: Any, reference: str = "Prophet") -> str:
    """Семейство модели: `reference`, `foundation`, `ensemble`, `ml`, `naive` или `stats`.

    Эталон определяется именем, а не типом; всё, что не подошло ни под одно правило, —
    статистические модели (AutoARIMA, AutoETS, AutoTheta).
    """
    name = str(model)
    if name == reference:
        return "reference"
    if name.startswith(FOUNDATION_PREFIXES):
        return "foundation"
    if name.startswith("Ensemble"):
        return "ensemble"
    if name in ML_MODELS:
        return "ml"
    return "naive" if name in NAIVE_MODELS else "stats"


def _in_sentence(title: str) -> str:
    """Имя модели внутри предложения: нарицательное («ансамбль») — со строчной буквы."""
    return (
        title[:1].lower() + title[1:]
        if title[:1].isalpha() and "а" <= title[:1].lower() <= "я"
        else title
    )


def _listed(values: Sequence[Any]) -> str:
    items = [str(v) for v in values]
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} и {items[-1]}"


def leaders_statement(winners: Sequence[Mapping[str, Any]], reference: str = "Prophet") -> str:
    """Вывод раздела горизонтов: какая модель точнее на каждом горизонте и что с эталоном.

    «На 1 и 3 месяца точнее всех …, на 6 месяцев — …». Вторая фраза — про эталон: его ошибка
    выше на каждом горизонте, только если это так; иначе — на скольких горизонтах из скольких.
    """
    if not winners:
        return ""
    groups: dict[str, list[int]] = {}
    for winner in winners:
        groups.setdefault(str(winner["model"]), []).append(int(winner["horizon"]))

    def span(horizons: Sequence[int]) -> str:
        return _listed([*horizons[:-1], horizon_label(horizons[-1])])

    parts = []
    for i, (model, horizons) in enumerate(groups.items()):
        title = model_title(model)
        if i == 0:
            parts.append(f"На {span(horizons)} точнее всех {title}")
        else:
            parts.append(f"на {span(horizons)} — {_in_sentence(title)}")
    text = ", ".join(parts) + "."
    compared = [w for w in winners if w.get("reference_mae")]
    if not compared:
        return text
    beaten = sum(1 for w in compared if w["mae"] < w["reference_mae"])
    if beaten == len(compared) and len(compared) > 1:
        return (
            f"{text} У {reference}, эталона из условий конкурса, ошибка выше на каждом из "
            f"{len(compared)} горизонтов."
        )
    noun = "горизонте" if beaten == 1 else "горизонтах"
    return f"{text} MAE ниже, чем у {reference}, на {beaten} {noun} из {len(compared)}."


# Что именно сравнивает пара конфигураций абляции (левая против правой).
CONTRIBUTION_SUBJECTS = {
    "B_vs_A": "Календарь и национальные ряды",
    "C_vs_B": "Новости и календарь событий поверх них",
    "D_vs_B": "Foundation-модели в ансамбле",
    "E_vs_A": "Все слои вместе против базовой модели",
}
SIGNIFICANCE_LEVEL = 0.05
# К чему добавлялся блок источника при измерении его вклада (варианты абляции A и B).
SOURCE_BASES = {
    "A": "к базовой модели",
    "B": "поверх календаря и национальных рядов",
}


def format_p_value(p_value: Any) -> str:
    """p-value для текста: «p = 0,31», «p = 0,004», «p < 0,001»."""
    try:
        value = float(p_value)
    except (TypeError, ValueError):
        return "p не оценено"
    if not np.isfinite(value):
        return "p не оценено"
    if value < 0.001:
        return "p < 0,001"
    return f"p = {format_number(value, 2 if value >= 0.01 else 3)}"


def contribution_statements(dm: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Вывод о вкладе каждого слоя — из знака разницы MAE и её устойчивости по окнам.

    `dm` — сравнения пар конфигураций: `mean_diff` < 0 значит, что конфигурация со слоем
    точнее; `folds`, `folds_better`, `sign_p` — счёт окон проверки. Формулировка следует
    числам: слой может помогать, вредить или не давать устойчивого изменения, и текст не знает
    заранее, что из этого получится.
    """
    lines = []
    for pair, subject in CONTRIBUTION_SUBJECTS.items():
        row = dm.get(pair)
        if not row:
            continue
        phrase = difference_phrase(row)
        if phrase:
            lines.append(f"{subject}: {phrase}.")
    return lines


def _number_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def is_steady(row: Mapping[str, Any]) -> bool:
    """Держится ли знак разности по окнам проверки (знаковый тест, уровень 0,05).

    Значимость по рядам сюда не входит намеренно: все прогнозы окна строит одна модель, и
    разности ошибок у рядов общие (см. `sbx.core.significance.fold_consistency`).
    """
    sign_p = _number_or_nan(row.get("sign_p"))
    return bool(row.get("folds")) and np.isfinite(sign_p) and sign_p < SIGNIFICANCE_LEVEL


def agreeing_windows(row: Mapping[str, Any]) -> tuple[int, int] | None:
    """Окна, согласные со знаком средней разности: (согласных, всего)."""
    diff, folds = _number_or_nan(row.get("mean_diff")), row.get("folds")
    if not folds or not np.isfinite(diff):
        return None
    side = row.get("folds_better") if diff < 0 else row.get("folds_worse")
    return (int(side), int(folds)) if side is not None else None


def windows_label(row: Mapping[str, Any]) -> str:
    """«в 9 окнах из 10» — в скольких окнах проверки знак разности тот же, что в среднем."""
    agreeing = agreeing_windows(row)
    if agreeing is None:
        return ""
    return f"в {counted(agreeing[0], 'окне', 'окнах', 'окнах')} из {agreeing[1]}"


def windows_side(row: Mapping[str, Any]) -> str:
    """«лучше в 8 окнах из 10» или «хуже в 4 окнах из 10»: сторона — по знаку средней разности."""
    windows = windows_label(row)
    if not windows:
        return ""
    return f"{'лучше' if _number_or_nan(row.get('mean_diff')) < 0 else 'хуже'} {windows}"


def difference_phrase(row: Mapping[str, Any]) -> str:
    """Измеренная разница MAE словами: направление — из знака, вывод — из счёта окон."""
    diff = _number_or_nan(row.get("mean_diff"))
    if not np.isfinite(diff):
        return ""
    change = f"MAE {'ниже' if diff < 0 else 'выше'} на {format_number(abs(diff), 1)} руб."
    side = windows_side(row)
    if not side:
        return f"{change}; по окнам проверки не сверено"
    if is_steady(row):
        return f"{change}, {side} — устойчиво"
    return f"устойчивого изменения нет — {change} в среднем, {side}"


def difference_brief(row: Mapping[str, Any]) -> str:
    """Тот же вывод для ячейки таблицы: «−58,7 руб., лучше в 8 окнах из 10 — неустойчиво»."""
    diff = _number_or_nan(row.get("mean_diff"))
    if not np.isfinite(diff):
        return ""
    change = f"{format_delta(diff)} руб."
    side = windows_side(row)
    if not side:
        return f"{change}; по окнам проверки не сверено"
    return f"{change}, {side} — {'устойчиво' if is_steady(row) else 'неустойчиво'}"


def reference_wins(
    significance: Mapping[str, Any], reference: str = "Prophet"
) -> dict[int, tuple[int, int]]:
    """Горизонт → (окон, где лучшая модель точнее эталона; окон всего).

    Горизонта, на котором лучшая модель с эталоном по окнам не сверена, в словаре нет.
    """
    wins: dict[int, tuple[int, int]] = {}
    by_horizon = significance.get("by_horizon") or {}
    for horizon in sorted(by_horizon, key=int):
        rows = [
            row
            for row in by_horizon[horizon].get("dm", [])
            if str(row.get("pair", "")).endswith(f"vs {reference}") and row.get("folds")
        ]
        if rows:
            wins[int(horizon)] = (int(rows[0]["folds_better"]), int(rows[0]["folds"]))
    return wins


def minority_note(significance: Mapping[str, Any], reference: str = "Prophet") -> str:
    """Горизонты, где лучшая модель точнее эталона меньше чем в половине окон проверки.

    Средний перевес по всем окнам может создать одно окно; тогда об этом сказано прямо.
    """
    return " ".join(
        f"На горизонте {horizon_label(horizon)} средний перевес создан меньшинством окон: в "
        f"остальных точнее {reference}."
        for horizon, (better, folds) in reference_wins(significance, reference).items()
        if 2 * better < folds
    )


def reference_windows(significance: Mapping[str, Any], reference: str = "Prophet") -> str:
    """В скольких окнах проверки лучшая модель горизонта точнее эталона.

    Если согласных окон меньше половины, это сказано отдельной фразой (`minority_note`).
    """
    parts = [
        f"{horizon_label(horizon)} — в {counted(better, 'окне', 'окнах', 'окнах')} из {folds}"
        for horizon, (better, folds) in reference_wins(significance, reference).items()
    ]
    if not parts:
        return ""
    text = f"По окнам проверки лучшая модель точнее {reference}: {'; '.join(parts)}."
    note = minority_note(significance, reference)
    return f"{text} {note}" if note else text


# Блоки внешних признаков словами — для текста о составе основной модели.
BLOCK_TITLES = {
    "calendar": "производственный календарь на целевой месяц",
    "macro": "национальные ряды СберИндекса",
    "news": "новости GDELT и календарь событий",
    "gdelt": "новости GDELT",
    "events": "календарь событий",
    "access": "доступность рынков",
    "neighbours": "расходы соседних МО",
    "weather": "погоду NASA POWER",
    "climate": "климатическую норму",
}


def negative_r2_note(board: pd.DataFrame) -> str:
    """У скольких моделей медианный R² по ряду ниже нуля — прогноз хуже среднего самого ряда.

    Считается по сводной таблице моделей (все горизонты, на которых модель определена): на
    коротких горизонтах у лидеров он положителен. Если отрицательных нет или колонки нет,
    утверждения нет.
    """
    if "r2_per_series_median" not in board.columns:
        return ""
    values = pd.to_numeric(board["r2_per_series_median"], errors="coerce").dropna()
    negative, total = int((values < 0).sum()), int(len(values))
    if not negative:
        return ""
    share = f"всех {total}" if negative == total else f"{negative} из {total}"
    return f"в сводке по всем горизонтам R² по отдельным рядам отрицателен у {share} моделей"


def history_limitation(r2_note: str) -> str:
    """Ограничение «короткая история» для списка на странице."""
    consequence = "на горизонте 12 месяцев direct-модели построить нельзя."
    if not r2_note:
        return f"История каждого ряда — 24 месяца: {consequence}"
    return f"История каждого ряда — 24 месяца: {r2_note}, а {consequence}"


def static_caveat(entity: Mapping[str, Any], market_access: bool) -> str:
    """Постоянные характеристики, известные позже части моментов прогноза.

    Тип МО взят из одной редакции справочника; `market_access` — подключены ли таблицы
    организаторов с индексом доступности рынков и расстояниями (расчёт 2024 года). Числа — из
    сводки шага идентификации; нет ни того, ни другого — нет и оговорки.
    """
    parts = []
    changed, year = entity.get("kind_changed_since_previous_year"), entity.get("reference_year")
    if changed and year:
        territories = counted(int(changed), "территории", "территорий", "территорий")
        parts.append(
            f"Тип МО взят из редакции справочника {year} года: у {territories} из "
            f"{format_number(int(entity.get('territories', 0)))} он отличается от редакции "
            f"{int(year) - 1} года, и для окон проверки от месяцев {int(year) - 1} года это "
            "сведение из следующей редакции."
        )
    if market_access:
        parts.append(
            "Индекс доступности рынков и расстояния между МО рассчитаны организаторами по "
            "данным 2024 года — позже части моментов прогноза."
        )
    return " ".join(parts)


def main_model_note(settings: Mapping[str, Any]) -> str:
    """Что получает основной LightGBM и как выбрана его конфигурация.

    Конфигурация выбрана по тем же окнам, на которых модель оценивается; отчёт говорит об этом
    сам. Для модели без внешних признаков (как она была задумана) примечания нет.
    """
    blocks = [str(block) for block in settings.get("feature_blocks") or []]
    if not blocks:
        return ""
    names = [BLOCK_TITLES.get(block, block) for block in blocks]
    listed = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} и {names[-1]}"
    tuning = (
        "гиперпараметры подбираются на последних месяцах перед моментом прогноза"
        if settings.get("tune")
        else "гиперпараметры не подбираются"
    )
    return (
        f"Основной LightGBM получает {listed}; {tuning}. Эта конфигурация выбрана после "
        "сравнения вариантов модели на тех же окнах проверки, поэтому её оценка оптимистична: "
        "лучший из нескольких вариантов на новых данных обычно оказывается хуже, чем на тех, по "
        "которым его выбрали."
    )


def window_limitation(significance: Mapping[str, Any]) -> str:
    """Ограничение «окон проверки мало»: сколько окон стоит за сравнением на каждом горизонте.

    Число окон — то, на котором сравниваются все модели горизонта (общие наблюдения).
    """
    by_horizon = significance.get("by_horizon") or {}
    counts = []
    for horizon in sorted(by_horizon, key=int):
        folds = [row["folds"] for row in by_horizon[horizon].get("dm", []) if row.get("folds")]
        if folds:
            counts.append(f"на {horizon_label(int(horizon))} — {int(folds[0])}")
    if not counts:
        return ""
    text = (
        f"Окон проверки мало: {', '.join(counts)}. Различия моделей измерены в этих окнах; "
        "повторятся ли они в другом периоде, по таким данным сказать нельзя."
    )
    windows = reference_windows(significance)
    return f"{text} {windows}" if windows else text


def hazard_verdict(hazard: Mapping[str, Any]) -> str:
    """Вывод о модели вероятности шока — из её сравнения с детектором при равном числе тревог."""
    model = (hazard.get("model_at_matched_alarm_rate") or {}).get("f1")
    reference = (hazard.get("reference_detector") or {}).get("f1")
    if model is None or reference is None:
        return ""
    scores = f"F1 {format_number(model, 3)} против {format_number(reference, 3)}"
    if float(model) < float(reference):
        return (
            "**Результат отрицательный: модель не входит в итоговую систему.** При равном числе "
            f"тревог она проигрывает простому детектору по остаткам ({scores}), и раннее "
            "предупреждение обеспечивает именно детектор."
        )
    return (
        f"При равном числе тревог модель не уступает детектору по остаткам ({scores}); в "
        "итоговой системе раннее предупреждение по-прежнему даёт детектор — он проще и не "
        "требует обучения."
    )


def hazard_brief(hazard: Mapping[str, Any]) -> str:
    """То же сравнение одной фразой — для списка выводов на странице."""
    model = (hazard.get("model_at_matched_alarm_rate") or {}).get("f1")
    reference = (hazard.get("reference_detector") or {}).get("f1")
    if model is None or reference is None:
        return ""
    scores = f"F1 {format_number(model, 3)} против {format_number(reference, 3)}"
    if float(model) < float(reference):
        return (
            "Модель вероятности шока в итоговую систему не вошла: при равном числе тревог она "
            f"проигрывает простому детектору по остаткам ({scores})."
        )
    return (
        "Модель вероятности шока при равном числе тревог не уступает детектору по остаткам "
        f"({scores}); раннее предупреждение в итоговой системе даёт детектор."
    )


# Ограничения, которые не зависят от прогона; остальные собираются из его чисел.
UNLABELLED_SHOCKS = (
    "Реальных размеченных шоков на уровне МО почти нет: методы обнаружения сравниваются на "
    "полусинтетическом бенчмарке, реальная проверка — только на национальных рядах."
)
PUBLIC_LEADERBOARDS = (
    "Публичные лидерборды foundation-моделей ненадёжны из-за утечки бенчмарков в предобучение, "
    "поэтому выводы делаются только по собственному бэктесту."
)


def limitation_items(
    history: str,
    contributions: Sequence[str] = (),
    window: str = "",
    model_note: str = "",
    caveat: str = "",
    schedule: str = "",
) -> list[dict[str, str]]:
    """Ограничения решения: заголовок, область (прогноз, сдвиги, данные) и текст.

    Тексты — те же формулировки, что в списке ограничений отчёта; необязательный пункт без
    текста пропускается. У вклада слоёв заголовок — сам слой (часть формулировки до двоеточия),
    на странице показываются первые два.
    """
    items = [
        ("Короткая история рядов", "прогноз", history),
        ("На уровне МО шоки почти не размечены", "сдвиги", UNLABELLED_SHOCKS),
        ("Публичным лидербордам не доверяем", "прогноз", PUBLIC_LEADERBOARDS),
        *((text.split(": ", 1)[0], "данные", text) for text in list(contributions)[:2]),
        ("Мало окон проверки", "прогноз", window),
        ("Оценка основной модели оптимистична", "прогноз", model_note),
        ("Часть признаков — из более поздних редакций", "данные", caveat),
        ("F1 не отличает метод от расписания", "сдвиги", schedule),
    ]
    return [{"title": title, "area": area, "text": text} for title, area, text in items if text]


def identification_facts(entity: Mapping[str, Any]) -> dict[str, Any]:
    """Сведения об идентификации территорий по частям — для блока сверки на странице.

    `panel` — состав панели, `same_names` — зачем нужен код, `check` — как устроена сверка с
    идентификацией по названиям; её итог — `wrong` (сколько рядов попадало в чужой регион) или
    `matched` (всё совпало). Числа — из сводки шага идентификации (`entity_report.json`).
    """
    if not entity.get("territories"):
        return {}
    territories = ("территория", "территории", "территорий")
    facts: dict[str, Any] = {
        "panel": (
            f"{counted(entity['territories'], *territories)}, "
            f"{counted(entity.get('series', 0), 'ряд', 'ряда', 'рядов')}, "
            f"{counted(entity.get('regions', 0), 'регион', 'региона', 'регионов')}; название, тип "
            f"и ОКТМО взяты из редакции справочника {entity.get('reference_year')} года."
        ),
        "same_names": (
            f"{counted(entity.get('same_names', 0), 'название', 'названия', 'названий')} "
            "встречаются в нескольких регионах — это "
            f"{counted(entity.get('same_name_territories', 0), *territories)}, и различает их "
            "только код."
        ),
    }
    check = entity.get("heuristic_check") or {}
    if not check:
        return facts
    series = int(check["series"])
    wrong = int(check["series_in_wrong_region"])
    facts["check"] = (
        "Для сверки те же наблюдения сопоставлены с публичной выгрузкой, где территория задана "
        "только названием."
    )
    if wrong == 0:
        facts["matched"] = (
            "Идентификация по названиям совпала с официальными кодами для всех "
            f"{counted(series, 'ряда', 'рядов', 'рядов')}."
        )
        return facts
    facts["wrong"] = {
        "series": wrong,
        "of": series,
        "share": 100 * wrong / series,
        "spread": (
            "ряды из чужого региона попадали в "
            f"{format_number(check['territories_with_foreign_series'])} из "
            f"{format_number(check['territories'])} территорий, "
            f"{counted(check['territories_entirely_in_wrong_region'], *territories)} были "
            "отнесены к чужому региону целиком."
        ),
    }
    return facts


def identification_summary(entity: Mapping[str, Any]) -> list[str]:
    """Абзацы об источнике территорий и о сверке идентификации по названиям.

    Текст собирается из сводки шага идентификации (`identification_facts`): числа и сам вывод
    следуют расчёту, а не зашиты в шаблон.
    """
    facts = identification_facts(entity)
    if not facts:
        return []
    lines = [
        "Основной набор — архив организаторов конкурса: расходы на уровне МО с официальным "
        "кодом территории и справочник территорий. Панель построена по официальным кодам: "
        f"{facts['panel']} {facts['same_names']}"
    ]
    if "check" not in facts:
        return lines
    if "matched" in facts:
        lines.append(f"{facts['check']} {facts['matched']}")
        return lines
    wrong = facts["wrong"]
    lines.append(
        f"{facts['check']} Идентификация по названиям относила к чужому региону "
        f"{format_number(wrong['series'])} из {counted(wrong['of'], 'ряда', 'рядов', 'рядов')} "
        f"({format_number(wrong['share'], 1)}%): {wrong['spread']}"
    )
    return lines


def leaderboard_view(board: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Таблица лидеров с русскими заголовками и отсортированная по MAE."""
    columns = list(
        columns
        or ["model", "mae", "mae_macro", "wape", "r2_pooled", "r2_per_series_median", "mase"]
    )
    view = board[[c for c in columns if c in board.columns]].copy().sort_values("mae")
    # Округление в самой таблице: иначе в отчёте появляются пятнадцать знаков после запятой.
    digits = {
        "mae": 1,
        "mae_macro": 1,
        "wape": 3,
        "r2_pooled": 3,
        "r2_per_series_median": 3,
        "mase": 3,
        "coverage_0.1_0.9": 3,
    }
    for column, places in digits.items():
        if column in view.columns:
            view[column] = view[column].round(places)
    view = view.rename(columns={**RU_METRIC_NAMES, "model": "Модель"})
    return view.reset_index(drop=True)


def model_ranking(board: pd.DataFrame, metric: str = "mae") -> list[tuple[str, float]]:
    ordered = board.sort_values(metric)
    return list(zip(ordered["model"], ordered[metric], strict=True))


def _text(meta: pd.DataFrame, uid: str, column: str) -> str:
    if uid not in meta.index or column not in meta.columns:
        return ""
    value = meta.loc[uid, column]
    if pd.isna(value):
        return ""
    return short_mo_name(value) if column == "mo_name" else str(value)


def series_showcase(
    oof: pd.DataFrame, static: pd.DataFrame, model: str, unique_ids: Sequence[str]
) -> list[dict[str, Any]]:
    """Данные для графиков разбора рядов: факт, прогноз и интервал по месяцам.

    На каждый месяц — один прогноз, самого короткого горизонта (см. `freshest_forecast`):
    иначе перекрывающиеся горизонты рисуют один месяц несколько раз.
    """
    names = static.set_index("unique_id")
    part_all = freshest_forecast(oof[oof["model"] == model])
    out = []
    for uid in unique_ids:
        part = part_all[part_all["unique_id"] == uid].sort_values("ds")
        if part.empty:
            continue
        out.append(
            {
                "unique_id": uid,
                "region": _text(names, uid, "region_name"),
                "mo": _text(names, uid, "mo_name"),
                "category": _text(names, uid, "category"),
                "ds": [d.strftime("%Y-%m") for d in part["ds"]],
                "y": [float(v) for v in part["y"]],
                "y_hat": [float(v) for v in part["y_hat"]],
                "q_lo": [float(v) for v in part["q_0.1"]] if "q_0.1" in part else [],
                "q_hi": [float(v) for v in part["q_0.9"]] if "q_0.9" in part else [],
            }
        )
    return out


def assemble_report_data(
    leaderboard: pd.DataFrame,
    detectors: pd.DataFrame | None,
    ablations: pd.DataFrame | None,
    showcase: Sequence[Mapping[str, Any]] = (),
    examples: Sequence[Mapping[str, Any]] = (),
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Единая структура данных для методологического отчёта и лендинга."""
    data: dict[str, Any] = {
        "meta": dict(meta or {}),
        "leaderboard": leaderboard.to_dict("records"),
        "leaderboard_view": leaderboard_view(leaderboard).to_dict("records"),
        "showcase": list(showcase),
        "examples": list(examples),
    }
    if detectors is not None:
        data["detectors"] = detectors.to_dict("records")
    if ablations is not None:
        data["ablations"] = ablations.to_dict("records")
    return data


def explorer_data(
    oof: pd.DataFrame,
    static: pd.DataFrame,
    model: str,
    alarms: Mapping[str, Sequence[int]] | None = None,
    limit: int = 150,
    seed: int = 0,
) -> dict[str, Any]:
    """Компактный набор рядов для интерактивного выбора на лендинге.

    Страница статическая, поэтому данные встраиваются в HTML. Чтобы она не весила десятки
    мегабайт, берётся ограниченная выборка рядов, равномерно покрывающая регионы и категории:
    по одному ряду на пару «регион × категория», затем случайный добор до `limit`.
    """
    part = oof[oof["model"] == model]
    if part.empty:
        return {"regions": [], "categories": [], "series": []}
    # Горизонты перекрываются по месяцам; без этого месяц попадал бы на график несколько раз,
    # рисуя пилу вместо ряда. Тот же поток видит детектор, поэтому позиции тревог совпадают.
    part = freshest_forecast(part)
    meta = static.set_index("unique_id")
    available = [uid for uid in part["unique_id"].unique() if uid in meta.index]

    frame = pd.DataFrame(
        {
            "unique_id": available,
            "region": [str(meta.loc[uid, "region_name"]) for uid in available],
            "category": [str(meta.loc[uid, "category"]) for uid in available],
        }
    ).sort_values("unique_id")
    # По одному ряду на пару «регион × категория» — так в выпадающих списках нет пустых сочетаний.
    picked = frame.drop_duplicates(["region", "category"])
    if len(picked) > limit:
        picked = picked.sample(limit, random_state=seed).sort_values("unique_id")
    elif len(picked) < limit:
        rest = frame[~frame["unique_id"].isin(set(picked["unique_id"]))]
        extra = rest.sample(min(limit - len(picked), len(rest)), random_state=seed)
        picked = pd.concat([picked, extra]).sort_values("unique_id")

    series = []
    for row in picked.itertuples():
        g = part[part["unique_id"] == row.unique_id].sort_values("ds")
        if g.empty:
            continue
        months = [d.strftime("%Y-%m") for d in g["ds"]]
        positions = list(alarms.get(row.unique_id, [])) if alarms else []
        series.append(
            {
                "id": row.unique_id,
                "region": row.region,
                "mo": _text(meta, row.unique_id, "mo_name"),
                "category": row.category,
                "ds": months,
                "y": [round(float(v), 1) for v in g["y"]],
                "y_hat": [round(float(v), 1) for v in g["y_hat"]],
                "q_lo": [round(float(v), 1) for v in g["q_0.1"]] if "q_0.1" in g else [],
                "q_hi": [round(float(v), 1) for v in g["q_0.9"]] if "q_0.9" in g else [],
                "alarms": [months[p] for p in positions if 0 <= p < len(months)],
            }
        )
        # Отрезки полосы интервала считаются здесь, а не в сценарии страницы.
        series[-1]["band"] = [
            list(run) for run in interval_runs(series[-1]["q_lo"], series[-1]["q_hi"])
        ]
    return {
        "regions": sorted({s["region"] for s in series}),
        "categories": sorted({s["category"] for s in series}),
        "series": series,
    }


def horizon_leaderboard(
    summary: pd.DataFrame, horizons: Sequence[int] = (1, 3, 6, 12), metric: str = "mae"
) -> pd.DataFrame:
    """Таблица «модель × горизонт» для требования критерия конкурса.

    Ожидает строки `summarize(..., slices=("horizon", ...))`, то есть колонки `slice`, `value`,
    `model` и метрику. Сортировка — по самому короткому доступному горизонту: на нём сравнение
    опирается на наибольшее число моделей.

    Пропуск в ячейке означает, что модель на этом горизонте не определена (например, глобальный
    LightGBM при h=12: обучающих origin нет вовсе), а не что она отработала плохо.
    """
    part = summary[summary["slice"] == "horizon"].copy()
    if part.empty:
        return pd.DataFrame()
    part["value"] = part["value"].astype(int)
    wide = part.pivot_table(index="model", columns="value", values=metric, aggfunc="first")
    columns = [h for h in horizons if h in wide.columns]
    wide = wide[columns]
    wide.columns = [f"{metric} h={h}" for h in columns]
    if columns:
        wide = wide.sort_values(f"{metric} h={columns[0]}", na_position="last")
    return wide.reset_index().rename(columns={"model": "Модель"})


def horizon_coverage(summary: pd.DataFrame) -> dict[int, dict[str, Any]]:
    """Сколько моделей и наблюдений стоит за каждым горизонтом.

    Нужно, чтобы таблицу нельзя было прочитать неверно: при h=12 сравнение опирается на
    меньшее число моделей, и это должно быть видно рядом с числами.
    """
    part = summary[summary["slice"] == "horizon"]
    out: dict[int, dict[str, Any]] = {}
    for value, group in part.groupby("value"):
        out[int(value)] = {
            "models": int(group["model"].nunique()),
            "observations": int(group["n"].max()) if "n" in group else None,
        }
    return out


def horizon_winners(
    summary: pd.DataFrame,
    horizons: Sequence[int] = (1, 3, 6, 12),
    reference: str = "Prophet",
    metric: str = "mae",
) -> list[dict[str, Any]]:
    """Лучшая модель на каждом горизонте и её выигрыш у эталона.

    Выбор делается внутри горизонта, а не по сводной таблице: модель, неопределимая на самом
    трудном горизонте (LightGBM при h=12), в смеси горизонтов выглядит лучшей просто потому,
    что этот горизонт в её среднее не попадает.
    """
    part = summary[summary["slice"] == "horizon"].copy()
    if part.empty:
        return []
    part["value"] = part["value"].astype(int)
    out: list[dict[str, Any]] = []
    for horizon in horizons:
        rows = part[(part["value"] == horizon) & part[metric].notna()]
        if rows.empty:
            continue
        best = rows.sort_values(metric).iloc[0]
        ref = rows[rows["model"] == reference]
        ref_value = float(ref.iloc[0][metric]) if len(ref) else None
        value = float(best[metric])
        out.append(
            {
                "horizon": horizon,
                "model": str(best["model"]),
                metric: value,
                f"reference_{metric}": ref_value,
                "gain": (ref_value - value) / ref_value if ref_value else None,
            }
        )
    return out


DETECTOR_NAMES = {
    "page_hinkley": "Page–Hinkley",
    "cusum": "CUSUM",
    "bocpd": "BOCPD",
    "adwin": "ADWIN",
    "conformal": "конформный",
    "pelt_l2": "PELT (L2)",
    "pelt_rbf": "PELT (RBF)",
    "binseg_l2": "BinSeg (L2)",
    "window_l2": "скользящее окно (L2)",
    "kernel_rbf": "ядерный (RBF)",
    "periodic": "тревога по расписанию",
}
# Эталон сравнения детекторов — строка `kind == REFERENCE_KIND`: тревога по расписанию.
REFERENCE_KIND = "baseline"


def detector_rows(detectors: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    """Строки таблицы сравнения: лучшие по F1 и эталон, даже если он ниже границы.

    Ожидает таблицу, отсортированную по убыванию F1. Без эталона F1 методов не с чем
    сравнить, поэтому из таблицы он не выпадает.
    """
    top = detectors.head(limit)
    if "kind" not in detectors.columns:
        return top
    reference = detectors[detectors["kind"] == REFERENCE_KIND]
    return pd.concat([top, reference[~reference.index.isin(top.index)]])


def schedule_reference_statement(detectors: pd.DataFrame) -> str:
    """Эталон сравнения — тревога по расписанию: сколько вариантов «метод × вход» точнее него.

    F1 сам по себе ничего не говорит о том, находит ли метод сломы: тревога в каждом пятом
    месяце, не глядя на данные, тоже попадает в окно допуска части сломов.
    """
    if not {"detector", "kind", "f1", "param"} <= set(detectors.columns):
        return ""
    reference = detectors[detectors["kind"] == REFERENCE_KIND]
    methods = detectors[detectors["kind"] != REFERENCE_KIND]
    if reference.empty or methods.empty:
        return ""
    score, period = float(reference.iloc[0]["f1"]), int(reference.iloc[0]["param"])
    schedule = "каждый месяц" if period == 1 else f"каждые {horizon_label(period)}"
    better = methods[methods["f1"] > score]
    text = (
        f"Эталон сравнения — тревога по расписанию, {schedule} независимо от данных: "
        f"F1 {format_number(score, 3)}. Выше эталона по F1 — {len(better)} из {len(methods)} "
        "вариантов «метод × вход»"
    )
    if better.empty:
        return text + "."
    gain = float(better["f1"].max()) - score
    return f"{text}; лучший превосходит его на {format_number(gain, 3)}."


def detector_contrast(detectors: pd.DataFrame) -> dict[str, Any]:
    """Лучший онлайн-детектор по остаткам прогноза рядом с детекторами по сырому ряду.

    Сравнений два. «Тот же детектор по сырому ряду» (`f1_raw`) показывает, что даёт смена
    входа при том же алгоритме. «Лучший детектор по сырому ряду» (`best_raw`) показывает,
    чего можно добиться без прогноза вовсе: другой алгоритм по сырому ряду может оказаться не
    хуже, и умолчать о нём — значит приукрасить результат.
    """
    needed = {"detector", "kind", "input", "f1"}
    if not needed <= set(detectors.columns):
        return {}
    online = detectors[detectors["kind"] == "online"]
    residual = online[online["input"] == "residual"].sort_values("f1", ascending=False)
    if residual.empty:
        return {}

    def optional(row: pd.Series, column: str) -> float | None:
        return float(row[column]) if column in row.index and pd.notna(row[column]) else None

    best = residual.iloc[0]
    name = str(best["detector"])
    raw = online[online["input"] == "raw"].sort_values("f1", ascending=False)
    same = raw[raw["detector"] == name]
    out: dict[str, Any] = {
        "detector": name,
        "f1_residual": float(best["f1"]),
        "f1_raw": float(same.iloc[0]["f1"]) if len(same) else None,
        "delay_residual": optional(best, "mean_delay"),
        "false_alarms_residual": optional(best, "false_alarms_per_series_year"),
        "threshold": optional(best, "param"),
        "alarmed_breaks": optional(best, "alarmed_break_share"),
        "alarmed_controls": optional(best, "alarmed_control_share"),
    }
    if len(raw):
        top = raw.iloc[0]
        out["best_raw"] = {
            "detector": str(top["detector"]),
            "f1": float(top["f1"]),
            "delay": optional(top, "mean_delay"),
            "false_alarms": optional(top, "false_alarms_per_series_year"),
        }
    schedule = detectors[detectors["kind"] == REFERENCE_KIND]
    if len(schedule) and "param" in schedule.columns:
        row = schedule.iloc[0]
        out["reference"] = {
            "f1": float(row["f1"]),
            "period": int(row["param"]),
            "delay": optional(row, "mean_delay"),
        }
    return out


def control_alarm_statement(contrast: Mapping[str, Any]) -> str:
    """Как часто детектор тревожится на рядах без внесённого слома и на рядах со сломом."""
    breaks, controls = contrast.get("alarmed_breaks"), contrast.get("alarmed_controls")
    if breaks is None or controls is None:
        return ""
    name = DETECTOR_NAMES.get(str(contrast["detector"]), str(contrast["detector"]))
    return (
        f"На рядах без внесённого слома {name} поднимает тревогу у {round(100 * controls)}% "
        f"рядов, на рядах со сломом — у {round(100 * breaks)}%: разметка покрывает только "
        "внесённые сломы, а собственные изменения настоящих рядов в ней не отмечены."
    )


def schedule_match_statement(match: Mapping[str, Any]) -> str:
    """Детектор против расписания с ближайшей частотой ложных тревог, с интервалом разности.

    Сравнение идёт на тестовой половине бенчмарка; период расписания подобран на калибровочной.
    """
    needed = {
        "detector",
        "period",
        "detector_false_alarms",
        "schedule_false_alarms",
        "detector_recall",
        "schedule_recall",
        "detector_delay",
        "schedule_delay",
        "recall_diff",
        "recall_diff_low",
        "recall_diff_high",
    }
    if not needed <= set(match):
        return ""
    name = DETECTOR_NAMES.get(str(match["detector"]), str(match["detector"]))
    period = int(match["period"])
    schedule = "раз в месяц" if period == 1 else f"раз в {horizon_label(period)}"
    low, high = 100 * float(match["recall_diff_low"]), 100 * float(match["recall_diff_high"])
    text = (
        f"Расписание с ближайшей частотой ложных тревог — {schedule} "
        f"({format_number(match['schedule_false_alarms'], 2)} на ряд в год против "
        f"{format_number(match['detector_false_alarms'], 2)} у детектора): оно находит "
        f"{round(100 * float(match['schedule_recall']))}% сломов и срабатывает через "
        f"{format_number(match['schedule_delay'], 1)} мес. после слома; {name} по остаткам "
        f"прогноза — {round(100 * float(match['detector_recall']))}% и "
        f"{format_number(match['detector_delay'], 1)} мес. Разница в полноте — "
        f"{format_number(100 * float(match['recall_diff']), 1)} п.п., "
        f"{round(100 * float(match.get('level', 0.95)))}%-й интервал бутстрепа по рядам — от "
        f"{format_number(low, 1)} до {format_number(high, 1)} п.п."
    )
    if low <= 0 <= high:
        return text + ": он включает ноль, перевес не доказан."
    return text


def schedule_brief(contrast: Mapping[str, Any], match: Mapping[str, Any]) -> str:
    """Детектор против расписания в двух фразах — для первого экрана страницы.

    Первая — про F1, вторая — про сравнение при равной частоте ложных тревог; без эталона
    сказать нечего, без сравнения с подобранным расписанием остаётся одна первая.
    """
    reference = contrast.get("reference")
    if not reference or contrast.get("f1_residual") is None:
        return ""
    ours, theirs = float(contrast["f1_residual"]), float(reference["f1"])
    if ours > theirs:
        text = (
            f"По F1 детектор выше эталона: {format_number(ours, 3)} против "
            f"{format_number(theirs, 3)}."
        )
    else:
        text = "По F1 детектор эталон не превосходит."
    needed = {"detector_recall", "schedule_recall", "detector_delay", "schedule_delay"}
    if not needed <= set(match):
        return text
    return (
        f"{text} При равной частоте ложных тревог он находит "
        f"{round(100 * float(match['detector_recall']))}% сломов против "
        f"{round(100 * float(match['schedule_recall']))}% и срабатывает через "
        f"{format_number(match['detector_delay'], 1)} мес. против "
        f"{format_number(match['schedule_delay'], 1)}."
    )


def schedule_limitation(detectors: pd.DataFrame) -> str:
    """Ограничение решения: пока не каждый вариант выше расписания, F1 детекторов не мера.

    Пустая строка, когда эталона нет или каждый вариант «метод × вход» точнее него.
    """
    if not {"kind", "f1"} <= set(detectors.columns):
        return ""
    reference = detectors[detectors["kind"] == REFERENCE_KIND]
    methods = detectors[detectors["kind"] != REFERENCE_KIND]
    if reference.empty or methods.empty:
        return ""
    score = float(reference.iloc[0]["f1"])
    better = int((methods["f1"] > score).sum())
    if better == len(methods):
        return ""
    return (
        f"Тревога по расписанию, не глядя на данные, даёт F1 {format_number(score, 3)}; выше неё "
        f"— {better} из {len(methods)} вариантов «метод × вход». F1 с окном допуска на этом "
        "бенчмарке не отличает метод от расписания; отличает сравнение при равной частоте "
        "ложных тревог."
    )


def _reference_score(contrast: Mapping[str, Any]) -> str:
    """«; тревога по расписанию — F1 и задержка»: число детектора читается только рядом с ним."""
    reference = contrast.get("reference")
    if not reference:
        return ""
    text = f"; тревога по расписанию — {format_number(reference['f1'], 3)}"
    if reference.get("delay") is not None:
        text += f" и {format_number(reference['delay'], 1)} мес."
    return text


def _detector_score(f1: float, delay: float | None) -> str:
    score = f"F1 {format_number(f1, 3)}"
    return f"{score}, задержка {format_number(delay, 1)} мес." if delay is not None else f"{score}."


def detector_statement(contrast: Mapping[str, Any]) -> str:
    """Вывод о детекторах: лучший по остаткам прогноза, лучший по сырому ряду и тот же
    детектор по сырому ряду — все три числа рядом, без оценочных слов."""
    if not contrast:
        return ""
    name = DETECTOR_NAMES.get(str(contrast["detector"]), str(contrast["detector"]))
    lines = [
        "Лучший онлайн-детектор по остаткам прогноза — "
        f"{name}: {_detector_score(contrast['f1_residual'], contrast.get('delay_residual'))}"
    ]
    best_raw = contrast.get("best_raw")
    if best_raw:
        same = str(best_raw["detector"]) == str(contrast["detector"])
        raw_name = DETECTOR_NAMES.get(str(best_raw["detector"]), str(best_raw["detector"]))
        lines.append(
            f"Лучший по сырому ряду — {'он же' if same else raw_name}: "
            f"{_detector_score(best_raw['f1'], best_raw.get('delay'))}"
        )
        if not same and contrast.get("f1_raw") is not None:
            lines.append(
                f"Тот же {name} по сырому ряду — F1 {format_number(contrast['f1_raw'], 3)}."
            )
    return " ".join(lines)


# Доля месяцев с фактом по одну сторону от прогноза, начиная с которой остатки названы смещёнными.
BIAS_SHARE = 0.6


def live_alarm_statement(live: Mapping[str, Any]) -> str:
    """Тревоги на настоящих остатках ансамбля: сколько их и что они означают.

    Размеченных шоков на настоящих рядах нет, поэтому рядом с частотой тревог стоит доля рядов
    с тревогой и смещение остатков: детектор по смещённым остаткам срабатывает и без сдвига.
    """
    if live.get("alarms") is None:
        return ""
    name = DETECTOR_NAMES.get(str(live.get("detector")), str(live.get("detector")))
    series = live.get("series")
    text = (
        f"На настоящих остатках ансамбля ({format_number(series)} рядов оценочной выборки, один "
        f"прогноз на месяц) {name} поднимает {format_number(live['alarms'])} тревог — "
        f"{format_number(live.get('alarms_per_series_year'), 2)} на ряд в год; конформный "
        f"детектор по интервалам ансамбля — {format_number(live.get('conformal_alarms'))}."
    )
    with_alarm = live.get("series_with_alarm")
    if with_alarm is not None and series:
        share = round(100 * int(with_alarm) / int(series))
        text += (
            f" Хотя бы одна тревога — у {counted(with_alarm, 'ряда', 'рядов', 'рядов')} из "
            f"{format_number(int(series))} ({share}%)."
        )
    above = live.get("above_forecast_share")
    if above is not None and max(float(above), 1 - float(above)) >= BIAS_SHARE:
        side, lag = ("выше", "отставание") if float(above) > 0.5 else ("ниже", "завышение")
        major = format_number(100 * max(float(above), 1 - float(above)), 1)
        text += (
            f" Остатки смещены: факт {side} прогноза в {major}% месяцев, поэтому часть тревог "
            f"отражает систематическое {lag} прогноза, а не смену режима."
        )
    return text


def forecast_bias_statement(live: Mapping[str, Any]) -> str:
    """С какой стороны от прогноза чаще лежит факт — если перевес заметен (см. `BIAS_SHARE`)."""
    above = live.get("above_forecast_share")
    if above is None or max(float(above), 1 - float(above)) < BIAS_SHARE:
        return ""
    share = format_number(100 * max(float(above), 1 - float(above)), 1)
    below, side = ("ниже", "выше") if float(above) > 0.5 else ("выше", "ниже")
    return (
        f"Прогноз чаще {below} факта: на рядах оценочной выборки факт {side} прогноза в "
        f"{share}% месяцев."
    )


def fact_above_statement(fact: Sequence[Any], forecast: Sequence[Any]) -> str:
    """«Факт выше прогноза в 9 месяцах из 12» — подпись под графиком одного ряда.

    Считаются месяцы, где известны и факт, и прогноз; без них подписи нет.
    """
    pairs = [
        (float(y), float(y_hat))
        for y, y_hat in zip(fact, forecast, strict=True)
        if np.isfinite(_number_or_nan(y)) and np.isfinite(_number_or_nan(y_hat))
    ]
    if not pairs:
        return ""
    above = sum(1 for y, y_hat in pairs if y > y_hat)
    return f"Факт выше прогноза в {counted(above, 'месяце', 'месяцах', 'месяцах')} из {len(pairs)}"


def detector_headline(contrast: Mapping[str, Any]) -> str:
    """Одна строка о детекторах для цепочки и карточки: остатки прогноза и сырой ряд."""
    if not contrast:
        return ""
    delay = contrast.get("delay_residual")
    text = f"Сдвиги по остаткам прогноза: F1 {format_number(contrast['f1_residual'], 3)}"
    if delay is not None:
        text += f", задержка {format_number(delay, 1)} мес."
    best_raw = contrast.get("best_raw")
    if not best_raw:
        return text + _reference_score(contrast)
    raw = f"по сырому ряду — {format_number(best_raw['f1'], 3)}"
    if best_raw.get("delay") is not None:
        raw += f" и {format_number(best_raw['delay'], 1)} мес."
    return f"{text}; {raw}{_reference_score(contrast)}"


def detector_card(contrast: Mapping[str, Any]) -> dict[str, str]:
    """Карточка шапки о детекторах: F1 по остаткам прогноза и лучший детектор по сырому ряду."""
    label = "F1 обнаружения сдвигов по остаткам прогноза"
    if contrast.get("delay_residual") is not None:
        label += f", задержка {format_number(contrast['delay_residual'], 1)} мес."
    best_raw = contrast.get("best_raw")
    if best_raw:
        label += f"; лучший детектор по сырому ряду — {format_number(best_raw['f1'], 3)}"
        if best_raw.get("delay") is not None:
            label += f" и {format_number(best_raw['delay'], 1)} мес."
    label += _reference_score(contrast)
    return {"value": format_number(contrast["f1_residual"], 3), "label": label}


def ablation_basis(ablations: Sequence[Mapping[str, Any]]) -> str:
    """Фраза о том, на каких наблюдениях сравниваются конфигурации абляции.

    Возвращается, только если число строк записано и у всех конфигураций одно: тогда разности
    MAE в таблице относятся к одним и тем же наблюдениям.
    """
    counts = {row.get("rows") for row in ablations}
    if len(counts) != 1 or None in counts:
        return ""
    rows = int(float(next(iter(counts))))
    return (
        f"Все конфигурации измерены на одних и тех же {format_number(rows)} наблюдениях. В "
        "таблице — разница общих MAE; в тесте значимости — средняя по рядам разность ошибок, "
        "поэтому эти числа немного расходятся."
    )
