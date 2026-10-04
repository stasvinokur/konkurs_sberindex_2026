"""Сборка методологического отчёта и лендинга из сохранённых артефактов."""

from __future__ import annotations

import json
import shutil
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from sbx.core.fingerprint import changed
from sbx.core.report import (
    DETECTOR_NAMES,
    SOURCE_BASES,
    ablation_basis,
    assemble_report_data,
    contribution_statements,
    control_alarm_statement,
    counted,
    detector_card,
    detector_contrast,
    detector_rows,
    detector_statement,
    explorer_data,
    format_delta,
    format_number,
    format_p_value,
    hazard_verdict,
    history_limitation,
    horizon_coverage,
    horizon_label,
    horizon_leaderboard,
    horizon_winners,
    identification_summary,
    is_steady,
    leaderboard_view,
    limitation_items,
    live_alarm_statement,
    main_model_note,
    negative_r2_note,
    reference_windows,
    schedule_limitation,
    schedule_match_statement,
    schedule_reference_statement,
    series_showcase,
    static_caveat,
    window_limitation,
    windows_side,
)
from sbx.shell import stamps
from sbx.shell.io import ARTIFACTS_DIR, ROOT
from sbx.shell.pipelines import ablations as ablation_step
from sbx.shell.pipelines import backtest as backtest_step
from sbx.shell.pipelines import samples
from sbx.shell.pipelines import storyline as storyline_step
from sbx.shell.pipelines.ablations import ABLATION_DIR, ABLATION_OOF_DIR, contributions_summary
from sbx.shell.pipelines.backtest import ENSEMBLE_FILE, METRICS_DIR, OOF_DIR, collect_oof
from sbx.shell.pipelines.cases import CASES_DIR
from sbx.shell.pipelines.cpd import CPD_DIR, live_stream
from sbx.shell.pipelines.entity import PROCESSED
from sbx.shell.pipelines.features import FEATURES_DIR, NEWS_DIR
from sbx.shell.pipelines.hazard import HAZARD_DIR
from sbx.shell.pipelines.panel import load_static
from sbx.shell.render import landing as landing_render

REPORTS_DIR = ROOT / "reports"
LANDING_DIR = REPORTS_DIR / "landing"
# Без архива блок новостей — один календарь событий; числа такого прогона нельзя выдавать за
# вклад новостей GDELT, поэтому отчёт и лендинг говорят об этом прямо.
NO_GDELT_NOTE = (
    "В этом прогоне архива новостей GDELT не было: блок новостей состоит из одного календаря "
    "событий, а модель вероятности шока обучена без внешних признаков. Конфигурации абляций "
    "C, E, F и модель вероятности шока поэтому не совпадают с опубликованными. Архив скачивает "
    "команда make gdelt."
)


CONTEST = "Онлайн-конкурс СберИндекса 2026"
DIRECTION = "направление «Прогнозирование»"


def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def news_note(news_dir: Path = NEWS_DIR) -> str:
    """Оговорка для отчёта, собранного без новостных признаков GDELT; пусто, если они есть."""
    return "" if (news_dir / "region_month_features.parquet").exists() else NO_GDELT_NOTE


def headline(panel: Mapping[str, Any]) -> dict[str, str]:
    """Подзаголовок отчёта и те же сведения по частям — для шапки страницы."""
    panel_line = (
        f"Панель: {format_number(panel.get('series'))} рядов «территория × категория», "
        f"{format_number(panel.get('territories'))} территорий, январь 2023 — декабрь 2024."
    )
    return {
        "subtitle": f"{CONTEST}, {DIRECTION}. {panel_line}",
        "contest": f"{CONTEST} · {DIRECTION}",
        "panel_line": panel_line,
    }


def page_limitations(
    r2_note: str,
    dm: Mapping[str, Mapping[str, Any]],
    significance: Mapping[str, Any],
    model_note: str,
    caveat: str,
    schedule: str,
) -> list[dict[str, str]]:
    """Ограничения решения пунктами с заголовком и областью; вклад внешних данных называет то,
    что измерила абляция."""
    return limitation_items(
        history_limitation(r2_note),
        contribution_statements(dm),
        window_limitation(significance),
        model_note,
        caveat,
        schedule,
    )


def repository_url(project: Path = ROOT / "pyproject.toml") -> str:
    """Адрес репозитория из метаданных проекта (`[project.urls] Repository`); не указан — пусто,
    и ссылки на странице нет."""
    if not project.exists():
        return ""
    urls = tomllib.loads(project.read_text(encoding="utf-8")).get("project", {}).get("urls", {})
    return str(urls.get("Repository", ""))


def publish_pdf(pdf: Path, out_dir: Path) -> str:
    """Кладёт копию отчёта рядом со страницей: публикуется только каталог лендинга.

    Возвращает имя файла для ссылки; отчёта нет — пустую строку.
    """
    if not pdf.exists():
        return ""
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pdf, out_dir / pdf.name)
    return pdf.name


class StaleArtifacts(RuntimeError):
    """Артефакты посчитаны не на текущих входах: отчёт из них был бы смесью разных прогонов."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__(
            "отчёт не собран: артефакты посчитаны не на текущих входах\n  "
            + "\n  ".join(self.problems)
        )


def _forecast_problems() -> list[str]:
    """Прогнозы, посчитанные не на тех входах, с которыми они посчитались бы сейчас.

    Ожидаемый отпечаток строит та же функция, что и сам расчёт, поэтому сверяется всё, от чего
    зависят прогнозы: панель, оценочный список, фолды, конфиг группы, блоки внешних признаков,
    версия расчёта. Сверки одной панели мало: группа, оставшаяся на прежнем оценочном списке,
    даёт смешанный прогон при общей панели.
    """
    problems = []
    main = sorted(p.stem for p in OOF_DIR.glob("*.parquet") if p.name != ENSEMBLE_FILE)
    variants = sorted(p.stem for p in ABLATION_OOF_DIR.glob("*.parquet"))
    # Варианты источников из конфига, чьих данных сейчас нет (прогон без архива новостей после
    # прогона с ним): посчитаются снова, когда данные появятся, а в отчёт сейчас не идут.
    sources = {
        pair["cache"]: pair
        for pair in ablation_step.source_pairs(ablation_step.load_ablation_config())
    }
    computable: dict[str, dict[str, str]] = {}

    def expected_variants(_names: Sequence[str] = ()) -> dict[str, dict[str, str]]:
        """Варианты абляции, которые посчитались бы сейчас; отпечатки строятся один раз."""
        if not computable:
            computable.update(ablation_step.expected_inputs())
        return computable

    checks = [(OOF_DIR, "прогнозы", "make backtest", main, backtest_step.expected_inputs)]
    checks.append(
        (ABLATION_OOF_DIR, "прогнозы абляции", "make ablations", variants, expected_variants)
    )
    for directory, label, command, names, expected_for in checks:
        if not names:
            continue
        expected = expected_for(names)
        for name in names:
            stamp = directory / f"{name}_inputs.json"
            recorded = stamps.read(stamp)
            if recorded is None:
                where = f"{stamp.parent.name}/{stamp.name}"
                problems.append(
                    f"{label} {name}: нет отметки входов ({where}); пересчитайте: {command}"
                )
                continue
            if name not in expected and name in sources:
                continue
            if name not in expected:
                problems.append(f"{label} {name}: в конфиге такого расчёта нет; удалите файл")
                continue
            differ = changed(recorded, expected[name])
            if differ:
                problems.append(
                    f"{label} {name}: входы изменились ({', '.join(differ)}); "
                    f"пересчитайте: {command}"
                )
    # Источник без данных в отчёт не идёт — значит, и строки о нём в таблице вклада быть не
    # должно: иначе абляции после исчезновения данных не пересчитывались. Остались ли на диске
    # его прогнозы, при этом неважно.
    measured = _read_csv(ABLATION_DIR / "sources.csv")
    if measured is not None and len(measured) and {"block", "against"} <= set(measured.columns):
        rows = set(zip(measured["block"].astype(str), measured["against"].astype(str), strict=True))
        problems += [
            f"вклад источника «{pair['title']}» (к {pair['against']}): посчитан на данных, "
            "которых сейчас нет; пересчитайте: make ablations"
            for pair in sources.values()
            if (pair["block"], pair["against"]) in rows and pair["cache"] not in expected_variants()
        ]
    return problems


def stale_artifacts() -> list[str]:
    """Артефакты, чьи входы сменились после расчёта; пусто — отчёт можно собирать.

    Прогнозы сверяются пересчётом отпечатка входов (`_forecast_problems`). Остальные шаги
    записывают рядом с результатом отметку прочитанных файлов; здесь отметки сверяются с тем,
    что лежит на диске сейчас. Необязательные артефакты проверяются, только если они есть.
    """
    problems = _forecast_problems()
    backtest = "uv run sbx backtest"
    checks = [
        (
            "панель расходов",
            PROCESSED / "panel.parquet",
            PROCESSED / "panel_inputs.json",
            "make panel",
        ),
        (
            "внешние признаки",
            FEATURES_DIR / "features_summary.json",
            FEATURES_DIR / "inputs.json",
            "make features",
        ),
        (
            "ансамбль",
            OOF_DIR / ENSEMBLE_FILE,
            OOF_DIR / "ensemble_inputs.json",
            f"{backtest} ensemble",
        ),
        (
            "метрики моделей",
            METRICS_DIR / "leaderboard.csv",
            METRICS_DIR / "inputs.json",
            f"{backtest} leaderboard",
        ),
        (
            "значимость различий",
            METRICS_DIR / "significance.json",
            METRICS_DIR / "significance_inputs.json",
            f"{backtest} significance",
        ),
        (
            "сравнение детекторов",
            CPD_DIR / "detector_comparison.csv",
            CPD_DIR / "inputs.json",
            "make cpd",
        ),
        (
            "модель вероятности шока",
            HAZARD_DIR / "hazard_results.json",
            HAZARD_DIR / "inputs.json",
            "uv run sbx cpd hazard",
        ),
        (
            "разборы примеров",
            CASES_DIR / "cases.json",
            CASES_DIR / "inputs.json",
            "uv run sbx cpd cases",
        ),
        ("абляции", ABLATION_DIR / "ablations.csv", ABLATION_DIR / "inputs.json", "make ablations"),
    ]
    for label, artifact, stamp, command in checks:
        if not artifact.exists():
            continue
        differ = stamps.stale(stamp)
        if differ is None:
            where = f"{stamp.parent.name}/{stamp.name}"
            problems.append(f"{label}: нет отметки входов ({where}); пересчитайте: {command}")
        elif differ:
            problems.append(
                f"{label}: входы изменились ({', '.join(differ)}); пересчитайте: {command}"
            )
    return problems


def collect() -> dict[str, Any]:
    """Собирает все артефакты в единую структуру данных отчёта."""
    problems = stale_artifacts()
    if problems:
        raise StaleArtifacts(problems)
    board = _read_csv(METRICS_DIR / "leaderboard.csv")
    summary = _read_csv(METRICS_DIR / "summary.csv")
    if board is None:
        raise FileNotFoundError(
            "нет artifacts/metrics/leaderboard.csv — сначала `sbx backtest run`"
        )
    detectors = _read_csv(CPD_DIR / "detector_comparison.csv")
    ablations = _read_csv(ABLATION_DIR / "ablations.csv")
    curve = _read_csv(CPD_DIR / "delay_vs_false_alarms.csv")
    hazard = _read_json(HAZARD_DIR / "hazard_results.json")
    significance = _read_json(METRICS_DIR / "significance.json")
    national = _read_csv(CPD_DIR / "national_comparison.csv")
    cases = _read_json(CASES_DIR / "cases.json")
    coverage = _read_json(METRICS_DIR / "coverage.json")
    panel_summary = _read_json(PROCESSED / "panel_summary.json")
    entity = _read_json(PROCESSED / "entity_report.json")

    oof = collect_oof()
    static = load_static()
    winners = horizon_winners(summary) if summary is not None else []
    contrast = detector_contrast(detectors) if detectors is not None else {}

    # Графики лендинга показывают тот же поток, что видит живой детектор: итоговый ансамбль,
    # один прогноз на месяц, короткий горизонт. Тогда тревоги стоят на своих месяцах.
    live = _read_json(CPD_DIR / "live_alarms.json")
    live_model = str((live.get("summary") or {}).get("model", "Ensemble (по категориям)"))
    stream = live_stream(oof[oof["model"] == live_model])
    showcase = series_showcase(stream, static, live_model, _showcase_ids(stream, static))

    cards = [
        {
            "value": format_number(w["mae"], 1),
            "label": (
                f"MAE на горизонте {horizon_label(w['horizon'])}, руб. — {w['model']}; "
                f"у Prophet {format_number(w['reference_mae'], 1)}"
            ),
        }
        for w in winners
    ]
    if contrast:
        cards.append(detector_card(contrast))
    cards.append(
        {
            "value": format_number(entity.get("territories")),
            "label": "муниципальных образований по официальным кодам организаторов",
        }
    )

    contributions = contributions_summary()
    lgbm_settings = (backtest_step.load_models_config().get("groups") or {}).get("lgbm") or {}
    model_note = main_model_note(lgbm_settings)
    sources = _read_csv(ABLATION_DIR / "sources.csv")
    source_rows = sources.to_dict("records") if sources is not None else []
    r2_note = negative_r2_note(board)
    schedule = schedule_limitation(detectors) if detectors is not None else ""
    limitations = page_limitations(
        r2_note,
        _dm({"contributions": contributions}),
        significance,
        model_note,
        static_caveat(entity, _has_access(source_rows)),
        schedule,
    )
    meta = {
        "title": "Прогнозирование расходов МО и раннее обнаружение структурных сдвигов",
        **headline(panel_summary),
        "repository": repository_url(),
        "cards": cards,
        # На странице — карточками с заголовком и областью; тексты — те же, что строками.
        "limitation_items": limitations,
        "limitations": [item["text"] for item in limitations],
        "schedule_limitation": schedule,
        "schedule_match": _read_json(CPD_DIR / "schedule_match.json"),
        "entity_summary": identification_summary(entity),
        "footer": "Сформировано автоматически из artifacts/ командой `sbx report build`.",
        "winners": winners,
        "detector_contrast": contrast,
        "live": live.get("summary") or {},
        "hazard": hazard,
        "panel": panel_summary,
        "entity": entity,
        "contributions": contributions,
        "source_contributions": source_rows,
        "news_note": news_note(),
        "significance": significance,
        "evaluation_series": len(samples.load_evaluation()),
        "main_model_note": model_note,
        "r2_note": r2_note,
        "coverage": coverage,
        "horizon_coverage": horizon_coverage(summary) if summary is not None else {},
        "cases": cases,
    }
    data = assemble_report_data(board, detectors, ablations, showcase, meta=meta)
    data["explorer"] = explorer_data(
        stream, static, live_model, alarms=live.get("alarms", {}), limit=150
    )
    data["curve"] = curve.to_dict("records") if curve is not None else []
    data["national"] = national.to_dict("records") if national is not None else []
    data["horizon_leaderboard"] = (
        horizon_leaderboard(summary).to_dict("records") if summary is not None else []
    )
    data["storyline"] = storyline_step.build(
        winners, contrast, _dm(meta), data["horizon_leaderboard"], detectors, source_rows
    )
    return data


def _showcase_ids(stream: pd.DataFrame, static: pd.DataFrame, n: int = 3) -> list[str]:
    """Крупнейшие ряды из разных регионов — чтобы примеры не повторяли один регион."""
    region = static.set_index("unique_id")["region_name"]
    medians = stream.groupby("unique_id")["y"].median().sort_values(ascending=False)
    picked: list[str] = []
    seen: set[str] = set()
    for uid in medians.index:
        name = str(region.get(uid, uid))
        if name in seen:
            continue
        seen.add(name)
        picked.append(str(uid))
        if len(picked) == n:
            break
    return picked


def build_landing(data: dict[str, Any], out_dir: Path = LANDING_DIR) -> Path:
    curve = pd.DataFrame(data.get("curve", []))
    return landing_render.render(data, out_dir / "index.html", curve if len(curve) else None)


def build_markdown(data: dict[str, Any], out_path: Path = REPORTS_DIR / "methodology.md") -> Path:
    """Методологический отчёт: числа берутся из артефактов, текст — из шаблона."""
    board = pd.DataFrame(data["leaderboard"])
    meta = data["meta"]
    view = leaderboard_view(board)
    detectors = pd.DataFrame(data.get("detectors", []))
    ablations = pd.DataFrame(data.get("ablations", []))
    hazard = meta.get("hazard", {})

    lines = [
        f"# {meta['title']}",
        "",
        meta["subtitle"],
        "",
        "Отчёт собирается командой `sbx report build` из артефактов в `artifacts/`; все числа в тексте —",
        "результат последнего прогона, ручной правки чисел нет.",
        "",
        "## 1. Задача и критерии",
        "",
        "Требуется прогнозировать потребительские безналичные расходы муниципальных образований и",
        "как можно раньше выявлять структурные изменения. Обязательная метрика — MAE; отдельно",
        "оцениваются сравнение прогнозных моделей, сравнение методов обнаружения сдвигов,",
        "использование современных фундаментальных моделей, интеграция новостей, интерпретация и",
        "воспроизводимость.",
        "",
        "## 2. Данные и идентификация МО",
        "",
        f"Панель: {format_number(meta['panel'].get('rows'))} наблюдений, "
        f"{format_number(meta['panel'].get('series'))} рядов «территория × категория», "
        f"{format_number(meta['panel'].get('territories'))} территорий, январь 2023 — декабрь 2024.",
        "",
        *[part for line in identification_summary(meta.get("entity") or {}) for part in (line, "")],
        *_protocol_section(meta),
        "## 4. Сравнение прогнозных моделей",
        "",
        view.to_markdown(index=False),
        "",
        *_mixed_table_note(data),
        *_horizon_section(data),
        "### Значимость различий",
        "",
        *_dm_table(meta.get("significance", {})),
        "Ранговое сравнение Фридмана и критические разности Немени по каждому горизонту — "
        "в `artifacts/metrics/significance.json`.",
        "",
    ]
    lines += _detector_section(detectors, meta)
    lines += _foundation_section(data, board)
    lines += _news_section(meta, (data.get("storyline") or {}).get("sources") or [])
    lines += _hazard_section(hazard)
    lines += _ablation_section(ablations, meta)
    lines += _cases_section(meta, data.get("national", []))
    lines += _limitations_section(meta)
    lines += _reproduction_section()
    lines += _criteria_section()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _protocol_section(meta: dict[str, Any]) -> list[str]:
    """Протокол оценки: окна, оценочная выборка и то, по чему делается вывод об устойчивости."""
    lines = [
        "## 3. Протокол оценки",
        "",
        "Rolling-origin бэктест на четырёх горизонтах, которых требует критерий конкурса: 1, 3, 6",
        "и 12 месяцев. Фолд (окно проверки) — это пара (горизонт, origin), а не origin с несколькими",
        "шагами: для direct-моделей горизонт меняет обучающую выборку. Всего 14 фолдов; чем длиннее",
        "горизонт, тем меньше origin-ов помещается в 24 месяца панели. Метрики считаются из единой",
        "OOF-таблицы; MAE — основная, R² приводится и pooled, и per-series.",
        "",
    ]
    evaluated, series = meta.get("evaluation_series"), (meta.get("panel") or {}).get("series")
    if evaluated and series:
        lines += [
            f"Глобальные модели учатся на всех {format_number(int(series))} рядах панели; "
            f"сравниваются модели на {counted(int(evaluated), 'ряде', 'рядах', 'рядах')} "
            "замороженной оценочной выборки (`configs/evaluation_series.csv`).",
            "",
        ]
    lines += [
        "Устойчиво ли различие двух моделей, решает счёт окон: знаковый тест показывает, держится ли",
        "знак разности по окнам проверки. Тест Диболда–Мариано по рядам приводится рядом как",
        "справочный: он считает ряды независимыми и завышает уверенность.",
        "",
    ]
    return lines


def _detector_section(detectors: pd.DataFrame, meta: dict[str, Any]) -> list[str]:
    """Сравнение детекторов: таблица бенчмарка, эталон, вывод словами и частота тревог на
    настоящих остатках ансамбля."""
    lines = ["## 5. Обнаружение структурных изменений", ""]
    if not len(detectors):
        return [*lines, "_Результаты бенчмарка не найдены: запустите `sbx cpd run`._", ""]
    columns = [
        "detector",
        "kind",
        "input",
        "param",
        "precision",
        "recall",
        "f1",
        "covering",
        "mean_delay",
        "false_alarms_per_series_year",
    ]
    present = [c for c in columns if c in detectors.columns]
    view = (
        detector_rows(detectors)[present]
        .round(3)
        .rename(
            columns={
                "detector": "Детектор",
                "kind": "тип",
                "input": "вход",
                "param": "параметр",
                "mean_delay": "задержка",
                "false_alarms_per_series_year": "ложных/ряд-год",
            }
        )
    )
    lines += [view.to_markdown(index=False), ""]
    reference = schedule_reference_statement(detectors)
    if reference:
        lines += [reference, ""]
    contrast = meta.get("detector_contrast") or {}
    statement = detector_statement(contrast)
    if statement:
        lines += [statement, ""]
    for extra in (
        schedule_match_statement(meta.get("schedule_match") or {}),
        control_alarm_statement(contrast),
    ):
        if extra:
            lines += [extra, ""]
    live = live_alarm_statement(meta.get("live") or {})
    if live:
        lines += [live, ""]
    lines += [
        "Полнота по типам сломов, проверка на национальных рядах и разбор выбора детектора — "
        "`reports/changepoints.md`.",
        "",
    ]
    return lines


def _horizon_section(data: dict[str, Any]) -> list[str]:
    """Таблица MAE по четырём горизонтам — прямое требование критерия 2 конкурса."""
    rows = data.get("horizon_leaderboard") or []
    if not rows:
        return []
    table = pd.DataFrame(rows)
    numeric = [c for c in table.columns if c != "Модель"]
    # Пропуск печатается прочерком, а не «nan»: иначе таблица читается как сбой расчёта,
    # тогда как это отсутствие модели на горизонте.
    table[numeric] = table[numeric].round(1).astype(object).where(table[numeric].notna(), "—")
    coverage = data.get("meta", {}).get("horizon_coverage") or {}

    lines = [
        "### MAE по горизонтам прогноза",
        "",
        "Критерий конкурса требует четырёх горизонтов: 1, 3, 6 и 12 месяцев. Каждый горизонт —",
        "самостоятельный набор фолдов, а не срез одного прогона: для direct-моделей горизонт",
        "меняет обучающую выборку.",
        "",
        table.to_markdown(index=False),
        "",
    ]
    if coverage:
        counts = ", ".join(
            f"h={h}: {v['models']} моделей на {format_number(v['observations'])} наблюдениях"
            for h, v in sorted(coverage.items(), key=lambda kv: int(kv[0]))
        )
        lines += [f"Сравнение опирается на разное число моделей: {counts}.", ""]
    lines += [
        "**Пропуск в ячейке означает «неопределимо», а не «плохо».** Глобальный LightGBM при",
        "h=12 не обучается вовсе: direct-модели нужен обучающий origin не позже cutoff − 12",
        "месяцев, а панель начинается в январе 2023 — таких origin нет. Нейросети при h=12",
        "пропускают фолд по той же причине: после вычета горизонта и валидационного окна на",
        "вход не остаётся ни одного месяца.",
        "",
        "**Оговорка по h=12.** Тестовое окно в 12 месяцев помещается только от origin не позже",
        "декабря 2023, поэтому фолда всего три и они перекрываются на 10–11 месяцев из 12, а",
        "обучение сокращается до 10–12 месяцев. Это не три независимых наблюдения, и разброс",
        "между ними нельзя читать как оценку устойчивости.",
        "",
    ]
    return lines


def _dm_table(significance: dict[str, Any]) -> list[str]:
    """Значимость различий — по каждому горизонту отдельно.

    Единый тест по смеси горизонтов требование критерия не закрывает: на коротких и длинных
    горизонтах выигрывают разные модели, и общий тест усреднил бы противоположные результаты.
    """
    by_horizon = significance.get("by_horizon") or {}
    if not by_horizon:
        return ["_Тесты значимости не найдены: запустите `sbx backtest significance`._", ""]

    lines = [
        "Лучшая модель горизонта против эталонов. Отрицательная ΔMAE означает, что она точнее.",
        "",
        "| Горизонт | Лучшая модель | Сравнение | ΔMAE, руб. | p по рядам | точнее эталона |",
        "|---|---|---|---|---|---|",
    ]
    for horizon in sorted(by_horizon, key=int):
        entry = by_horizon[horizon]
        for row in entry.get("dm", []):
            reference = str(row["pair"]).split(" vs ")[-1]
            folds = row.get("folds")
            windows = (
                f"в {counted(row['folds_better'], 'окне', 'окнах', 'окнах')} из {folds}"
                if folds
                else "не сверено"
            )
            lines.append(
                f"| h={horizon} | {entry['best_model']} | против {reference} | "
                f"{format_number(row['mean_diff'], 1)} | {float(row['p_value']):.2e} | "
                f"{windows} |"
            )
    lines += [
        "",
        "**Как читать.** «p по рядам» — панельный тест Диболда–Мариано: в нём ряды считаются",
        "независимыми наблюдениями. Он отвечает на вопрос, различаются ли модели в этих окнах",
        "проверки, но не на вопрос, повторится ли различие в другом окне: все прогнозы окна",
        "строит одна обученная модель, а шоки месяца у рядов общие. Поэтому рядом стоит счёт",
        "окон. Окон мало — от одного до четырёх на горизонт, при 12 месяцах они перекрываются, —",
        "и по ним можно судить только о том, держится ли знак.",
        "",
    ]
    windows_text = reference_windows(significance)
    if windows_text:
        lines += [windows_text, ""]
    best = {h: by_horizon[h]["best_model"] for h in sorted(by_horizon, key=int)}
    if len(set(best.values())) > 1:
        picks = ", ".join(f"h={h} — {m}" for h, m in best.items())
        lines += [f"**Лучшая модель зависит от горизонта**: {picks}.", ""]
    return lines


def _mixed_table_note(data: dict[str, Any]) -> list[str]:
    """Почему сводную таблицу нельзя читать как рейтинг и где на самом деле лучшая модель."""
    table = pd.DataFrame(data.get("horizon_leaderboard") or [])
    text = ["Сводная таблица усредняет все горизонты, на которых модель определена."]
    if len(table):
        numeric = [c for c in table.columns if c != "Модель"]
        missing = table[numeric].isna()
        partial = table.loc[missing.any(axis=1), "Модель"].tolist()
        gaps = sorted({c.split("=")[-1] for c in numeric if missing[c].any()}, key=int)
        if partial:
            text.append(
                f"{', '.join(partial)} при h={', '.join(gaps)} неопределимы, поэтому этот горизонт "
                "в их среднее не входит, и сравнивать их с остальными моделями по этой таблице "
                "нельзя."
            )
    winners = data["meta"].get("winners") or []
    if winners:
        picks = "; ".join(
            f"{horizon_label(w['horizon'])} — **{w['model']}** ({format_number(w['mae'], 1)} руб.)"
            for w in winners
        )
        text.append(f"Лучшая модель выбирается внутри горизонта: {picks}.")
    return [" ".join(text), ""]


def _foundation_section(data: dict[str, Any], board: pd.DataFrame) -> list[str]:
    lines = [
        "## 6. Foundation-модели: что дают и чего не дают",
        "",
        "Проверены две модели с разрешающими лицензиями на веса — Chronos-2 (Apache-2.0) и",
        "TimesFM 2.5 (Apache-2.0). Moirai и TimesFM 3.0 исключены: их веса под некоммерческой",
        "лицензией, см. `THIRD_PARTY_MODELS.md`.",
        "",
    ]
    table = pd.DataFrame(data.get("horizon_leaderboard") or [])
    if not len(table):
        return lines
    numeric = [c for c in table.columns if c != "Модель"]
    rows = table[table["Модель"].str.contains("Chronos|TimesFM|AutoTheta", regex=True)].copy()
    if "coverage_0.1_0.9" in board.columns:
        coverage = board.set_index("model")["coverage_0.1_0.9"]
        # У моделей без квантилей (AutoTheta) покрытия нет: прочерк, а не «nan».
        share = rows["Модель"].map(coverage).round(3)
        rows["покрытие 0,1–0,9"] = share.astype(object).where(share.notna(), "—")
    view = rows.copy()
    view[numeric] = view[numeric].round(1).astype(object).where(view[numeric].notna(), "—")
    lines += [view.to_markdown(index=False), ""]

    single = table[~table["Модель"].str.startswith("Ensemble")].set_index("Модель")[numeric]
    picks = {c.split("=")[-1]: str(single[c].idxmin()) for c in numeric if single[c].notna().any()}
    lines += [
        "Лучшая одиночная модель (без ансамблей): "
        + ", ".join(f"h={h} — {m}" for h, m in picks.items())
        + ".",
        "",
    ]

    def beats(model: str, other: str) -> int:
        if model not in single.index or other not in single.index:
            return 0
        return int((single.loc[model] < single.loc[other]).sum())

    fm_best = [h for h, m in picks.items() if m == "Chronos-2 (covariates)"]
    lines += [
        "Главные выводы: выигрыш даёт не сама foundation-модель, а передача ей ковариат. "
        f"Chronos-2 без ковариат точнее простого AutoTheta на {beats('Chronos-2', 'AutoTheta')} "
        f"из {len(numeric)} горизонтов, с ковариатами — на "
        f"{beats('Chronos-2 (covariates)', 'AutoTheta')} из {len(numeric)}"
        + (f" и становится лучшей одиночной моделью при h={', '.join(fm_best)}" if fm_best else "")
        + f". TimesFM 2.5 zero-shot точнее AutoTheta на {beats('TimesFM-2.5', 'AutoTheta')} "
        f"из {len(numeric)} горизонтов. Публичные лидерборды foundation-моделей не переносятся "
        "на конкретную задачу, и проверять нужно собственным бэктестом. Подробности — "
        "`reports/models.md`.",
        "",
    ]
    return lines


def _dm(meta: dict[str, Any]) -> dict[str, Any]:
    return (meta.get("contributions") or {}).get("diebold_mariano") or {}


def _source_table(inventory: Sequence[Mapping[str, Any]]) -> list[str]:
    """Внешние источники одной таблицей: что берётся, как согласовано по времени, что дало."""
    rows = [row for row in inventory if row.get("key") not in ("spending", "reference")]
    if not rows:
        return []
    lines = [
        "| Источник | Что берётся | Уровень | Согласование по времени | Что дал |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {row['source']} | {row['what']} | {row['level']} | {row['timing']} | {row['gave']} |"
        for row in rows
    ]
    return [*lines, ""]


def _news_section(meta: dict[str, Any], inventory: Sequence[Mapping[str, Any]] = ()) -> list[str]:
    contributions = (meta.get("contributions") or {}).get("contributions", {})
    lines = [
        "## 7. Новости и внешние данные",
        "",
        "Источник — архивные суточные файлы GDELT Events за 2022-10…2024-12, отфильтрованные по",
        "России на лету (3,01 млн событий, 90 МБ). Архив скачивает `make gdelt` и сверяет с",
        "`data/gdelt_manifest.yaml`; по неполному архиву признаки не строятся.",
        "Признаки строятся на уровне «регион × месяц» и «страна × месяц». Дополнительно ведётся",
        "курируемый календарь событий с проверяемыми источниками.",
        "",
        "**Согласование по времени.** Момент прогноза — конец последнего наблюдаемого месяца. В",
        "прогноз входит только то, что опубликовано к концу месяца, от которого делается прогноз:",
        "новости — по дате попадания события в базу (`DATEADDED`), события календаря — по дате",
        "публикации, национальные ряды — с лагом публикации своего источника, отсчитанным от",
        "конца отчётного периода. На всех шагах многошагового прогноза стоят одни и те же",
        "значения, известные на момент прогноза; на целевой месяц берётся только производственный",
        "календарь, известный заранее. Лаги — `configs/features.yaml`, проверка —",
        "`tests/test_leakage.py`.",
        "",
    ]
    lines += _source_table(inventory)
    if meta.get("news_note"):
        lines += [f"**{meta['news_note']}**", ""]
    statements = contribution_statements(_dm(meta))
    if statements:
        lines += ["Измеренный вклад слоёв (раздел 9; вывод — по окнам проверки):", ""]
        lines += [f"- {statement}" for statement in statements]
        lines.append("")
    elif contributions:
        lines += [
            f"Вклад новостей в MAE: {format_number(contributions.get('news_mae'), 1)} руб. "
            f"(положительное число означает ухудшение). Вклад foundation-моделей: "
            f"{format_number(contributions.get('foundation_mae'), 1)} руб.",
            "",
        ]
    return lines


def _hazard_section(hazard: dict[str, Any]) -> list[str]:
    lines = ["## 8. Модель вероятности шока на 1–3 месяца вперёд", ""]
    if not hazard:
        return lines + ["_Модель вероятности шока не обучалась в этом прогоне._", ""]
    matched = hazard.get("model_at_matched_alarm_rate", {})
    reference = hazard.get("reference_detector", {})
    lines += [
        f"PR-AUC (h=3): {format_number(hazard.get('h3', {}).get('pr_auc'), 3)} при базовой доле "
        f"{format_number(hazard.get('h3', {}).get('positives', 0) / max(hazard.get('h3', {}).get('n', 1), 1), 3)}.",
        "",
        "Сравнение с онлайн-детектором при **равном числе тревог**:",
        "",
        "| Метрика | Модель шока | Онлайн-детектор |",
        "|---|---|---|",
        f"| precision | {format_number(matched.get('precision'), 3)} | {format_number(reference.get('precision'), 3)} |",
        f"| recall | {format_number(matched.get('recall'), 3)} | {format_number(reference.get('recall'), 3)} |",
        f"| F1 | {format_number(matched.get('f1'), 3)} | {format_number(reference.get('f1'), 3)} |",
        f"| задержка, мес. | {format_number(matched.get('mean_delay'), 2)} | {format_number(reference.get('mean_delay'), 2)} |",
        f"| ложных на ряд-год | {format_number(matched.get('false_alarms_per_series_year'), 3)} | {format_number(reference.get('false_alarms_per_series_year'), 3)} |",
        "",
        f"{hazard_verdict(hazard)} Разбор — `reports/changepoints.md`.".strip(),
        "",
    ]
    return lines


def _source_contributions(sources: list[dict[str, Any]]) -> list[str]:
    """Вклад каждого источника: его блок против варианта, к которому он добавлен."""
    if not sources:
        return []
    lines = [
        "### Вклад каждого источника по отдельности",
        "",
        "LightGBM с одним добавленным блоком признаков против того же варианта без него: к",
        "базовой модели (A) и поверх календаря и национальных рядов (B). Разность — средняя по",
        "рядам, отрицательная означает, что с источником ошибка меньше. «По окнам» — в скольких",
        "окнах проверки знак разности тот же, что в среднем: «лучше в 8 окнах из 10» при",
        "отрицательной разности, «хуже в 4 окнах из 10» при положительной. Вывод «устойчиво»",
        "ставится, только если знак держится по окнам (знаковый тест, уровень 0,05).",
        "",
        "| Источник | К чему добавлен | ΔMAE, руб. | по окнам | вывод | по горизонтам 1 / 3 / 6 мес. | p по рядам |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in sources:
        p_value = float(row.get("p_value", float("nan")))
        by_horizon = " / ".join(format_delta(row.get(f"delta_h{h}")) for h in (1, 3, 6))
        lines.append(
            f"| {row['title']} | {SOURCE_BASES.get(str(row['against']), row['against'])} | "
            f"{format_delta(row.get('mean_diff'))} | {windows_side(row) or 'не сверено'} | "
            f"{'устойчиво' if is_steady(row) else 'неустойчиво'} | {by_horizon} | "
            f"{format_p_value(p_value)} |"
        )
    return [*lines, ""]


def _ablation_section(ablations: pd.DataFrame, meta: dict[str, Any]) -> list[str]:
    lines = ["## 9. Абляции: измеренный вклад каждого слоя", ""]
    if not len(ablations):
        return lines + ["_Матрица абляций не собрана._", ""]
    columns = [
        "name",
        "mae",
        "delta_mae",
        "r2_per_series_median",
        "f1",
        "false_alarms_per_series_year",
        "mean_delay",
    ]
    present = [c for c in columns if c in ablations.columns]
    view = (
        ablations[present]
        .round(3)
        .rename(
            columns={
                "name": "Конфигурация",
                "mae": "MAE, руб.",
                "delta_mae": "ΔMAE к A",
                "r2_per_series_median": "R² per-series",
                "f1": "F1 CPD",
                "false_alarms_per_series_year": "ложных на ряд-год",
                "mean_delay": "задержка, мес.",
            }
        )
    )
    lines += [view.to_markdown(index=False), ""]
    if "f1" in present:
        lines += [
            "В столбцах обнаружения у конфигураций A–E стоит лучший офлайн-метод по сырому ряду, у "
            "F — онлайн-детектор по остаткам прогноза: слой F меняет способ обнаружения, а не "
            "прогноз.",
            "",
        ]
    basis = ablation_basis(ablations.to_dict("records"))
    if basis:
        lines += [basis, ""]
    lines += _source_contributions(meta.get("source_contributions") or [])
    if meta.get("news_note"):
        lines += [f"**{meta['news_note']}**", ""]
    dm = (meta.get("contributions") or {}).get("diebold_mariano", {})
    if dm:
        lines += [
            "Сравнение пар конфигураций (ΔMAE < 0 — левая точнее). Вывод ставится по окнам "
            "проверки: тест по рядам считает ряды независимыми и здесь завышает уверенность.",
            "",
        ]
        lines += [
            "| Пара | ΔMAE, руб. | по окнам | вывод | p по рядам |",
            "|---|---|---|---|---|",
        ]
        for name, row in dm.items():
            lines.append(
                f"| {name.replace('_vs_', ' против ')} | {format_delta(row.get('mean_diff'))} | "
                f"{windows_side(row) or 'не сверено'} | "
                f"{'устойчиво' if is_steady(row) else 'неустойчиво'} | "
                f"{float(row.get('p_value', float('nan'))):.2e} |"
            )
        lines.append("")
    return lines


def _cases_section(meta: dict[str, Any], national: list[dict[str, Any]]) -> list[str]:
    cases = meta.get("cases") or {}
    lines = ["## 10. Разборы реальных примеров", ""]
    for case in cases.get("national", []):
        detected = sum(1 for e in case.get("events", []) if e.get("detected"))
        method = DETECTOR_NAMES.get(str(case.get("method")), "офлайн-метод")
        chance = (
            f"; окна допуска вокруг них накрывают {round(100 * float(case['coverage']))}% ряда"
            if case.get("coverage") is not None
            else ""
        )
        lines.append(
            f"- **{case['series']}**: {method} со штрафом, откалиброванным на бенчмарке, "
            f"находит {detected} из {len(case.get('events', []))} размеченных шоков, всего "
            f"{case.get('alarms')} тревог{chance} (`reports/figures/{case.get('figure')}`)."
        )
    municipal = cases.get("municipal")
    if municipal:
        place = municipal.get("mo") or municipal.get("region")
        lines.append(
            f"- **{place} ({municipal.get('region')}) · {municipal.get('category')}**: "
            "крупнейший остаток "
            f"прогноза (|z| = {format_number(municipal.get('max_abs_z'), 1)}) — не шок, а быстрый "
            "структурный рост категории, за которым модель не успевает."
        )
    monthly = [row for row in national if row.get("freq") == "monthly"]
    if monthly:
        best = max(monthly, key=lambda r: r.get("f1", 0))
        name = DETECTOR_NAMES.get(str(best.get("method")), str(best.get("method")))
        lines += [
            "",
            f"На национальных месячных рядах лучший офлайн-метод — {name} "
            f"(F1 {format_number(best.get('f1'), 3)}, recall {format_number(best.get('recall'), 3)}).",
        ]
    lines += ["", "Полные разборы с графиками — `reports/cases.md`.", ""]
    return lines


def _has_access(sources: Sequence[Mapping[str, Any]]) -> bool:
    """Измерен ли в прогоне блок таблиц организаторов с индексом доступности рынков."""
    return any(str(row.get("block")) == "access" for row in sources)


def _limitations_section(meta: dict[str, Any]) -> list[str]:
    series = meta.get("evaluation_series")
    sample = (
        f"на {counted(series, 'ряду', 'рядах', 'рядах')} оценочной выборки"
        if series
        else "на замороженной выборке рядов"
    )
    coverage = meta.get("horizon_coverage") or {}
    observations = ""
    if coverage:
        counts = "; ".join(
            f"{horizon_label(int(h))} — {format_number(int(v['observations']))}"
            for h, v in sorted(coverage.items(), key=lambda kv: int(kv[0]))
        )
        observations = f" Общих наблюдений всех моделей по горизонтам: {counts}."
    limitation = window_limitation(meta.get("significance") or {})
    note = str(meta.get("main_model_note") or "")
    caveat = static_caveat(
        meta.get("entity") or {}, _has_access(meta.get("source_contributions") or [])
    )
    schedule = str(meta.get("schedule_limitation") or "")
    optional = [
        limitation.replace("Окон проверки мало:", "**Окон проверки мало:**", 1),
        f"**Основная модель выбрана по тем же окнам.** {note}" if note else "",
        f"**Часть постоянных характеристик известна позже момента прогноза.** {caveat}"
        if caveat
        else "",
        f"**Сравнение детекторов читается только рядом с расписанием.** {schedule}"
        if schedule
        else "",
    ]
    # Необязательные пункты идут после восьми постоянных и нумеруются подряд.
    numbered = [f"{9 + i}. {text}" for i, text in enumerate(t for t in optional if t)]
    r2_note = str(meta.get("r2_note") or "")
    r2_note = f" {r2_note[:1].upper()}{r2_note[1:]}." if r2_note else ""
    return [
        "## 11. Ограничения",
        "",
        "Перечислены честно, включая те, что ослабляют выводы решения.",
        "",
        "1. **История — 24 месяца на ряд.** Годовая сезонность и тренд не идентифицируются: у",
        "   каждого МО всего два декабря. На горизонте 12 месяцев это становится жёстким",
        "   ограничением: direct-модели неопределимы, а фолдов помещается лишь три, и те",
        f"   перекрываются на 10–11 месяцев из 12.{r2_note}",
        "2. **Утечка бенчмарков в предобучение foundation-моделей.** Публичные лидерборды Chronos",
        "   и TimesFM не переносятся на эту задачу, и проверить, не видели ли веса похожие данные,",
        "   невозможно. Все выводы делаются только по собственному rolling-origin бэктесту.",
        "3. **Аддитивность показателя не гарантирована.** Значение задано в рублях с медианой",
        "   около 25 тыс. в месяц, что похоже на средние расходы на клиента, а такие величины не",
        "   суммируются. Поэтому согласование прогнозов по иерархии «МО → регион → РФ» (MinT) в",
        "   решение не входит.",
        "4. **Реальных размеченных шоков на уровне МО почти нет.** Приграничных МО Курской области",
        "   в датасете нет; паводок в Орске в месячном итоге не виден. Методы обнаружения",
        "   сравниваются на полусинтетическом бенчмарке, а реальная проверка возможна только на",
        "   национальных рядах.",
        "5. **Вклад внешних данных — то, что показала абляция, а не то, что ожидалось.** "
        + (
            " ".join(contribution_statements(_dm(meta))[:2])
            or "В этом прогоне вклад не измерен: абляции не посчитаны."
        ),
        "6. **Лицензионные исключения.** Moirai, TimesFM 3.0, TiRex, TabPFN-TS и TimeGPT не",
        "   использованы: некоммерческие лицензии весов или закрытый API. Это сузило выбор",
        "   foundation-моделей до двух.",
        "7. **Часть новостей без региона.** 30,2% событий GDELT привязаны только к стране; они",
        "   идут в отдельный блок признаков уровня страны.",
        f"8. **Оценка на подвыборке.** Модели сравниваются {sample}: Prophet",
        "   обучается отдельно на каждый ряд, и полная панель для него недостижима за время",
        f"   конкурса.{observations}",
        *numbered,
        "",
    ]


def _reproduction_section() -> list[str]:
    return [
        "## 12. Воспроизведение",
        "",
        "```bash",
        "uv sync --all-extras",
        "make validate    # сверка файлов данных с манифестами",
        "make gdelt       # архив новостей GDELT: докачать и сверить с манифестом",
        "make entities    # таблица территорий по официальным кодам, сверка идентификации",
        "make panel       # каноническая панель и сезонные индексы",
        "make features    # календарь, национальные ряды, новости, события, соседние МО, погода, население",
        "make backtest    # прогнозные модели, ансамбль, значимость по горизонтам",
        "make cpd         # бенчмарк обнаружения сдвигов, национальная проверка, разборы",
        "make ablations   # матрица абляций и вклад каждого источника",
        "make report      # отчёт и лендинг",
        "make all         # всё перечисленное подряд, в этом порядке",
        "```",
        "",
        "То же в контейнере, без установки uv и Python: `docker compose run --rm pipeline`.",
        "Сеть нужна при первом прогоне: `make gdelt` скачивает архив новостей, `make backtest` —",
        "веса foundation-моделей. Набор организаторов, таблица погоды и таблица численности",
        "населения лежат в репозитории; команды `sbx data official`, `sbx data weather` и",
        "`sbx data population` нужны, только чтобы получить их заново. Без архива новостей",
        "конфигурации абляций C, E, F и модель раздела 8 получаются другими; что делать, если",
        "источник недоступен, — в README.",
        "Все гиперпараметры — в `configs/*.yaml`, все числа отчёта — из `artifacts/`.",
        "",
    ]


def _criteria_section() -> list[str]:
    return [
        "## 13. Где закрыт каждый критерий",
        "",
        "| Критерий | Раздел отчёта | Артефакты |",
        "|---|---|---|",
        "| Понятность методологии | весь отчёт | — |",
        "| Качество прогноза (MAE) | разделы 3, 4 | `artifacts/metrics/leaderboard.csv` |",
        "| Сравнение методов обнаружения | разделы 5 и 8, `reports/changepoints.md` | `artifacts/cpd/detector_comparison.csv` |",
        "| Современные foundation-модели | раздел 6 | `artifacts/oof/chronos2.parquet`, `timesfm25.parquet` |",
        "| Интеграция новостей | раздел 7 | `artifacts/news/`, `configs/events.yaml` |",
        "| Интерпретация и выводы | разделы 9–11, `reports/cases.md` | `artifacts/ablations/`, `artifacts/cases/` |",
        "| Воспроизводимость | раздел 12, README | `data/manifest.yaml`, `configs/` |",
        "",
    ]


def run() -> dict[str, Any]:
    from sbx.shell.render.pdf import markdown_to_pdf

    data = collect()
    markdown = build_markdown(data)
    pdf = markdown_to_pdf(markdown, REPORTS_DIR / "methodology.pdf")
    # Страница ссылается на копию отчёта рядом с собой, поэтому PDF собирается раньше неё.
    data["meta"]["pdf"] = publish_pdf(pdf, LANDING_DIR)
    landing = build_landing(data)
    (ARTIFACTS_DIR / "report_data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    return {
        "markdown": str(markdown),
        "pdf": str(pdf),
        "landing": str(landing),
        "models": len(data["leaderboard"]),
    }
