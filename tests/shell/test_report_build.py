"""Сборка методологического отчёта и его экспорт в PDF."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPORT = Path("reports/methodology.md")
REQUIRED_SECTIONS = [
    "## 1. Задача и критерии",
    "## 2. Данные и идентификация МО",
    "## 3. Протокол оценки",
    "## 4. Сравнение прогнозных моделей",
    "## 5. Обнаружение структурных изменений",
    "## 6. Foundation-модели",
    "## 7. Новости и внешние данные",
    "## 8. Модель вероятности шока",
    "## 9. Абляции",
    "## 10. Разборы реальных примеров",
    "## 11. Ограничения",
    "## 12. Воспроизведение",
    "## 13. Где закрыт каждый критерий",
]


def _text() -> str:
    if not REPORT.exists():
        pytest.skip("отчёт не собран: `uv run sbx report build`")
    return REPORT.read_text(encoding="utf-8")


def test_report_has_every_required_section() -> None:
    text = _text()
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    assert not missing, f"в отчёте нет разделов: {missing}"


def test_report_states_mae_and_compares_with_prophet_at_every_horizon() -> None:
    """Критерий 2 требует MAE, сравнения с Prophet и четырёх горизонтов: 1, 3, 6, 12."""
    text = _text()
    assert "MAE" in text
    assert "p по рядам" in text and "точнее эталона" in text

    block = text.split("### Значимость различий")[1].split("## 5.")[0]
    for horizon in (1, 3, 6, 12):
        row = [ln for ln in block.splitlines() if ln.startswith(f"| h={horizon} |")]
        assert row, f"нет строк значимости для горизонта {horizon}"
        assert any("Prophet" in ln for ln in row), f"на горизонте {horizon} нет сравнения с Prophet"


def test_report_has_a_horizon_table_with_all_four_columns() -> None:
    text = _text()
    assert "### MAE по горизонтам прогноза" in text
    header = [ln for ln in text.splitlines() if ln.startswith("| Модель") and "h=1" in ln]
    assert header, "нет заголовка таблицы по горизонтам"
    for horizon in ("h=1", "h=3", "h=6", "h=12"):
        assert horizon in header[0], f"в таблице нет колонки {horizon}"
    # Пропуск печатается прочерком, а не «nan»: иначе читается как сбой расчёта.
    assert "nan" not in text.lower().replace("значимо", "")


def test_limitations_name_the_uncomfortable_ones() -> None:
    """Раздел ограничений должен называть и те, что ослабляют выводы решения."""
    text = _text()
    for marker in (
        "24 месяца",  # короткие ряды
        "предобучение",  # утечка бенчмарков FM
        "суммируются",  # аддитивность показателя
        "Курской области",  # дефицит реальных региональных шоков
        "лицензи",  # лицензионные исключения
        "нешни",  # внешние данные: что показала абляция, а не что ожидалось
    ):
        assert marker in text, f"в ограничениях не упомянуто: {marker}"


def test_criteria_checklist_maps_every_criterion_to_a_section() -> None:
    text = _text()
    block = text.split("## 13. Где закрыт каждый критерий")[1]
    for criterion in (
        "Понятность методологии",
        "Качество прогноза",
        "Сравнение методов обнаружения",
        "Современные foundation-модели",
        "Интеграция новостей",
        "Интерпретация и выводы",
        "Воспроизводимость",
    ):
        assert criterion in block, f"критерий не отражён в чек-листе: {criterion}"


def test_markdown_to_pdf_produces_a_readable_file(tmp_path) -> None:
    """PDF собирается без pandoc и содержит кириллицу."""
    from sbx.shell.render.pdf import markdown_to_pdf

    source = tmp_path / "doc.md"
    source.write_text(
        "\n".join(
            [
                "# Заголовок отчёта",
                "",
                "Обычный абзац с **выделением** и ссылкой [сюда](http://example.com).",
                "",
                "| Модель | MAE, руб. |",
                "|---|---|",
                "| Ансамбль | 544,7 |",
                "",
                "- пункт списка",
                "",
                "```bash",
                "make all",
                "```",
            ]
        ),
        encoding="utf-8",
    )
    out = markdown_to_pdf(source, tmp_path / "doc.pdf")
    assert out.exists()
    payload = out.read_bytes()
    assert payload.startswith(b"%PDF")
    assert len(payload) > 2000, "подозрительно маленький PDF"


def test_report_says_so_when_it_was_built_without_the_news_archive(tmp_path: Path) -> None:
    """Без архива GDELT блок новостей — один календарь событий: отчёт не должен выдавать
    такие числа за вклад новостей."""
    from sbx.shell.pipelines import report

    assert report.news_note(tmp_path) == report.NO_GDELT_NOTE
    (tmp_path / "region_month_features.parquet").write_bytes(b"")
    assert report.news_note(tmp_path) == ""

    meta = {"news_note": report.NO_GDELT_NOTE}
    ablations = pd.DataFrame({"name": ["A"], "mae": [700.0]})
    assert f"**{report.NO_GDELT_NOTE}**" in report._news_section(meta)
    assert f"**{report.NO_GDELT_NOTE}**" in report._ablation_section(ablations, meta)

    marker = "архива новостей GDELT не было"
    assert not any(marker in line for line in report._news_section({}))
    assert not any(marker in line for line in report._ablation_section(ablations, {}))


def test_report_texts_about_external_data_follow_the_measured_numbers() -> None:
    """Разделы 7, 8 и 11 не содержат зашитого вывода: формулировка следует знаку и значимости."""
    from sbx.shell.pipelines import report

    helps = {
        "contributions": {
            "contributions": {"news_mae": 6.7, "foundation_mae": -45.0},
            "diebold_mariano": {
                "B_vs_A": {
                    "mean_diff": -60.8,
                    "p_value": 2.0e-9,
                    "folds": 10,
                    "folds_better": 9,
                    "folds_worse": 1,
                    "sign_p": 0.021,
                },
                "C_vs_B": {
                    "mean_diff": 6.7,
                    "p_value": 0.31,
                    "folds": 10,
                    "folds_better": 4,
                    "folds_worse": 6,
                    "sign_p": 0.754,
                },
            },
        }
    }
    news = "\n".join(report._news_section(helps))
    limitations = "\n".join(report._limitations_section(helps))
    for text in (news, limitations):
        assert "Календарь и национальные ряды: MAE ниже на 60,8 руб." in text
        assert "устойчивого изменения нет" in text
    assert "значимо ухудшают MAE" not in limitations
    assert "к концу месяца, от которого делается прогноз" in news

    hurts = {
        "contributions": {
            "diebold_mariano": {"B_vs_A": {"mean_diff": 111.9, "p_value": 1e-12}},
        }
    }
    assert "MAE выше на 111,9 руб." in "\n".join(report._limitations_section(hurts))
    assert "не измерен" in "\n".join(report._limitations_section({}))

    worse = {
        "h3": {"pr_auc": 0.14, "positives": 10, "n": 100},
        "model_at_matched_alarm_rate": {"f1": 0.21},
        "reference_detector": {"f1": 0.40},
    }
    assert "не входит в итоговую систему" in "\n".join(report._hazard_section(worse))
    better = {**worse, "model_at_matched_alarm_rate": {"f1": 0.45}}
    assert "не входит в итоговую систему" not in "\n".join(report._hazard_section(better))


def test_ablation_section_says_on_which_rows_the_configurations_are_compared() -> None:
    from sbx.shell.pipelines import report

    ablations = pd.DataFrame(
        {"name": ["A", "D"], "mae": [636.7, 605.6], "delta_mae": [0.0, -31.1], "rows": [31238] * 2}
    )
    text = "\n".join(report._ablation_section(ablations, {}))
    assert "Все конфигурации измерены на одних и тех же 31 238 наблюдениях." in text
    bare = "\n".join(report._ablation_section(ablations.drop(columns="rows"), {}))
    assert "на одних и тех же" not in bare
    # Столбцов обнаружения в таблице нет — нет и пояснения к ним.
    assert "В столбцах обнаружения" not in text
    with_detection = " ".join(
        "\n".join(report._ablation_section(ablations.assign(f1=[0.222, 0.221]), {})).split()
    )
    assert (
        "В столбцах обнаружения у конфигураций A–E стоит лучший офлайн-метод по сырому ряду, у F "
        "— онлайн-детектор по остаткам прогноза: слой F меняет способ обнаружения, а не прогноз."
        in with_detection
    )


def test_ablation_section_lists_the_contribution_of_every_source() -> None:
    from sbx.shell.pipelines import report

    ablations = pd.DataFrame({"name": ["A"], "mae": [636.7], "delta_mae": [0.0]})
    sources = [
        {
            "block": "neighbours",
            "title": "Расходы соседних МО",
            "against": "A",
            "rows": 31701,
            "mean_diff": -12.3,
            "p_value": 0.0004,
            "folds": 10,
            "folds_better": 9,
            "folds_worse": 1,
            "sign_p": 0.021,
            "delta_h1": -20.0,
            "delta_h3": -10.0,
            "delta_h6": 3.5,
        }
    ]
    # Тот же источник поверх набора B: по рядам «значимо», по окнам знак не держится.
    sources.append(
        {**sources[0], "against": "B", "folds_better": 6, "folds_worse": 4, "sign_p": 0.754}
    )
    text = "\n".join(report._ablation_section(ablations, {"source_contributions": sources}))
    assert "### Вклад каждого источника по отдельности" in text
    assert (
        "| Расходы соседних МО | поверх календаря и национальных рядов | −12,3 | лучше в 6 окнах "
        "из 10 | неустойчиво |" in text
    )
    assert (
        "| Расходы соседних МО | к базовой модели | −12,3 | лучше в 9 окнах из 10 | устойчиво "
        "| −20,0 / −10,0 / +3,5 | p < 0,001 |" in text
    )
    # Источник, с которым ошибка выросла: плюс у разности и «хуже» в счёте окон.
    hurts = [{**sources[0], "mean_diff": 42.4, "folds_better": 6, "folds_worse": 4, "sign_p": 0.75}]
    worse = "\n".join(report._ablation_section(ablations, {"source_contributions": hurts}))
    assert "| к базовой модели | +42,4 | хуже в 4 окнах из 10 | неустойчиво |" in worse
    assert text.splitlines()[0] == "## 9. Абляции: измеренный вклад каждого слоя"
    bare = "\n".join(report._ablation_section(ablations, {}))
    assert "Вклад каждого источника" not in bare


def test_significance_table_shows_the_windows_next_to_the_series_level_test() -> None:
    from sbx.shell.pipelines import report

    significance = {
        "by_horizon": {
            "1": {
                "best_model": "Chronos-2 (covariates)",
                "dm": [
                    {
                        "pair": "Chronos-2 (covariates) vs Prophet",
                        "mean_diff": -97.5,
                        "p_value": 2.14e-14,
                        "folds": 4,
                        "folds_better": 1,
                        "folds_worse": 3,
                        "sign_p": 0.625,
                    }
                ],
            },
            "12": {
                "best_model": "Ensemble",
                "dm": [
                    {
                        "pair": "Ensemble vs Prophet",
                        "mean_diff": -3800.5,
                        "p_value": 1.42e-86,
                        "folds": 3,
                        "folds_better": 3,
                        "folds_worse": 0,
                        "sign_p": 0.25,
                    }
                ],
            },
        }
    }
    text = " ".join("\n".join(report._dm_table(significance)).split(" "))
    flat = " ".join(text.split())
    assert (
        "| h=1 | Chronos-2 (covariates) | против Prophet | −97,5 | 2.14e-14 | в 1 окне из 4 |"
        in text
    )
    assert "| h=12 | Ensemble | против Prophet | −3 800,5 | 1.42e-86 | в 3 окнах из 3 |" in text
    assert "ряды считаются независимыми" in flat, "смысл p-value объяснён рядом с таблицей"
    assert "На горизонте 1 месяц средний перевес создан меньшинством окон" in text
    assert "потому что ошибки отдельных моделей" not in text, "объяснение не зашито в текст"


def test_ablation_pairs_table_puts_the_windows_before_the_series_level_p() -> None:
    from sbx.shell.pipelines import report

    ablations = pd.DataFrame({"name": ["A"], "mae": [636.7], "delta_mae": [0.0]})
    pair = {"mean_diff": -27.8, "p_value": 1.2e-17, "verdict": "значимо"}
    shaky = {**pair, "folds": 10, "folds_better": 6, "folds_worse": 4, "sign_p": 0.754}
    meta = {"contributions": {"diebold_mariano": {"B_vs_A": shaky}}}
    text = "\n".join(report._ablation_section(ablations, meta))
    assert "| B против A | −27,8 | лучше в 6 окнах из 10 | неустойчиво | 1.20e-17 |" in text
    assert "_vs_" not in text, "ключи пар в таблицу не идут"
    assert "| значимо |" not in text, "вывод теста по рядам в таблицу не идёт"
    worse = {**shaky, "mean_diff": 7.7, "folds_better": 6, "folds_worse": 4}
    meta = {"contributions": {"diebold_mariano": {"C_vs_B": worse}}}
    text = "\n".join(report._ablation_section(ablations, meta))
    assert "| C против B | +7,7 | хуже в 4 окнах из 10 | неустойчиво | 1.20e-17 |" in text


def test_detector_section_states_the_conclusion_in_words() -> None:
    """Раздел 5 — критерий весом 20%: под таблицей стоит вывод, а не только числа."""
    from sbx.shell.pipelines import report

    detectors = pd.DataFrame(
        {
            "detector": ["cusum", "page_hinkley", "cusum"],
            "kind": ["online"] * 3,
            "input": ["residual", "raw", "raw"],
            "param": [3.0, 5.0, 8.0],
            "f1": [0.221, 0.219, 0.188],
            "mean_delay": [2.275, 3.242, 3.732],
            "false_alarms_per_series_year": [0.889, 0.9, 0.824],
        }
    )
    meta = {
        "detector_contrast": report.detector_contrast(detectors),
        "live": {
            "detector": "cusum",
            "series": 1111,
            "alarms": 3067,
            "alarms_per_series_year": 2.77,
            "conformal_alarms": 444,
            "series_with_alarm": 935,
            "above_forecast_share": 0.705,
        },
    }
    lines = report._detector_section(detectors, meta)
    text = " ".join("\n".join(lines).split())
    assert lines[0] == "## 5. Обнаружение структурных изменений"
    assert "| cusum | online | residual |" in text
    assert "Лучший онлайн-детектор по остаткам прогноза — CUSUM: F1 0,221" in text
    assert "Лучший по сырому ряду — Page–Hinkley: F1 0,219" in text
    assert (
        "На настоящих остатках ансамбля (1 111 рядов оценочной выборки, один прогноз на месяц) "
        "CUSUM поднимает 3 067 тревог — 2,77 на ряд в год; конформный детектор по интервалам "
        "ансамбля — 444." in text
    )
    assert (
        "Хотя бы одна тревога — у 935 рядов из 1 111 (84%). Остатки смещены: факт выше прогноза "
        "в 70,5% месяцев, поэтому часть тревог отражает систематическое отставание прогноза, а "
        "не смену режима." in text
    )
    assert "`reports/changepoints.md`" in text
    older = {**meta, "live": {k: v for k, v in meta["live"].items() if "share" not in k}}
    plain = " ".join("\n".join(report._detector_section(detectors, older)).split())
    assert "Остатки смещены" not in plain and "конформный детектор" in plain
    unbiased = {**meta, "live": {**meta["live"], "above_forecast_share": 0.52}}
    even = " ".join("\n".join(report._detector_section(detectors, unbiased)).split())
    assert "Остатки смещены" not in even, "о смещении говорится, только когда оно есть"
    bare = "\n".join(report._detector_section(pd.DataFrame(), {}))
    assert "sbx cpd run" in bare and "Лучший онлайн-детектор" not in bare
    no_live = " ".join("\n".join(report._detector_section(detectors, {})).split())
    assert "На настоящих остатках" not in no_live


def test_built_report_reads_the_detectors_against_the_schedule() -> None:
    """Эталон и сравнение при равной частоте ложных тревог доходят до собранного отчёта: кривая
    передаётся в раздел 5 при сборке, а не только в тестах раздела."""
    section = _text().split("## 5.")[1].split("## 6.")[0]
    assert "| periodic " in section
    assert "Эталон сравнения — тревога по расписанию" in section
    assert "Расписание с ближайшей частотой ложных тревог" in section
    assert "интервал бутстрепа по рядам" in section
    assert "На рядах без внесённого слома" in section
    cases = _text().split("## 10.")[1].split("## 11.")[0]
    assert "окна допуска вокруг них накрывают" in cases
    limitations = _text().split("## 11.")[1].split("## 12.")[0]
    assert "**Сравнение детекторов читается только рядом с расписанием.**" in limitations
    page = (REPORT.parent / "landing" / "index.html").read_text(encoding="utf-8")
    assert "F1 с окном допуска на этом бенчмарке не отличает метод от расписания" in page


def test_detector_section_puts_the_schedule_reference_next_to_the_methods() -> None:
    """F1 методов читается только рядом с эталоном: тревога по расписанию, не глядя на данные,
    получает почти столько же."""
    from sbx.shell.pipelines import report

    detectors = pd.DataFrame(
        {
            "detector": ["cusum", "page_hinkley", "periodic", "cusum"],
            "kind": ["online", "online", "baseline", "online"],
            "input": ["residual", "raw", "none", "raw"],
            "param": [3.0, 5.0, 5.0, 8.0],
            "f1": [0.221, 0.219, 0.197, 0.188],
        }
    )
    text = " ".join("\n".join(report._detector_section(detectors, {})).split())
    assert "| periodic | baseline | none | 5 |" in text
    assert (
        "Эталон сравнения — тревога по расписанию, каждые 5 месяцев независимо от данных: "
        "F1 0,197. Выше эталона по F1 — 2 из 3 вариантов «метод × вход»; лучший превосходит его на 0,024."
        in text
    )
    # Строк больше, чем помещается в таблицу, а эталон слабее всех: он всё равно показан.
    many = pd.DataFrame(
        {
            "detector": [f"method_{i}" for i in range(13)] + ["periodic"],
            "kind": ["online"] * 13 + ["baseline"],
            "input": ["raw"] * 13 + ["none"],
            "param": [1.0] * 13 + [6.0],
            "f1": [0.5 - 0.01 * i for i in range(13)] + [0.01],
        }
    )
    crowded = "\n".join(report._detector_section(many, {}))
    assert "| periodic" in crowded and "| method_11" in crowded and "| method_12" not in crowded
    plain = "\n".join(report._detector_section(detectors[detectors["kind"] == "online"], {}))
    assert "Эталон сравнения" not in plain


def test_detector_section_compares_the_detector_with_an_equally_noisy_schedule() -> None:
    """Под таблицей — сравнение с расписанием той же частоты ложных тревог и доля рядов без
    слома, получивших тревогу: оба числа из расчёта, а не из текста."""
    from sbx.shell.pipelines import report

    detectors = pd.DataFrame(
        {
            "detector": ["periodic", "cusum"],
            "kind": ["baseline", "online"],
            "input": ["none", "residual"],
            "param": [6.0, 3.0],
            "f1": [0.227, 0.221],
            "alarmed_break_share": [1.0, 0.804],
            "alarmed_control_share": [1.0, 0.714],
        }
    )
    match = {
        "detector": "cusum",
        "period": 8,
        "detector_false_alarms": 0.889,
        "schedule_false_alarms": 0.872,
        "detector_recall": 0.430,
        "schedule_recall": 0.344,
        "detector_delay": 2.275,
        "schedule_delay": 3.082,
        "recall_diff": 0.086,
        "recall_diff_low": 0.031,
        "recall_diff_high": 0.142,
        "level": 0.95,
    }
    meta = {"detector_contrast": report.detector_contrast(detectors), "schedule_match": match}
    text = " ".join("\n".join(report._detector_section(detectors, meta)).split())
    assert "Выше эталона по F1 — 0 из 1 вариантов «метод × вход»." in text
    assert (
        "Расписание с ближайшей частотой ложных тревог — раз в 8 месяцев (0,87 на ряд в год "
        "против 0,89 у детектора): оно находит 34% сломов" in text
    )
    assert "95%-й интервал бутстрепа по рядам — от 3,1 до 14,2 п.п." in text
    assert (
        "На рядах без внесённого слома CUSUM поднимает тревогу у 71% рядов, на рядах со сломом "
        "— у 80%" in text
    )
    bare = {
        "detector_contrast": report.detector_contrast(
            detectors.drop(columns=["alarmed_break_share", "alarmed_control_share"])
        )
    }
    plain = " ".join("\n".join(report._detector_section(detectors, bare)).split())
    assert "Расписание с ближайшей частотой" not in plain
    assert "На рядах без внесённого слома" not in plain


def test_cases_section_names_the_method_and_reads_the_monthly_check() -> None:
    """Разбор называет метод, которым он построен; строка о национальной проверке — о месячных
    рядах, а не о лучшей строке всей таблицы."""
    from sbx.shell.pipelines import report

    meta = {
        "cases": {
            "national": [
                {
                    "series": "Всего",
                    "method": "window_l2",
                    "penalty": 0.5,
                    "alarms": 9,
                    "figure": "case.png",
                    "coverage": 0.473,
                    "margin": 2,
                    "events": [{"detected": True}, {"detected": False}, {"detected": False}],
                }
            ]
        }
    }
    national = [
        {"freq": "weekly", "method": "pelt_rbf", "f1": 0.9, "recall": 0.9},
        {"freq": "monthly", "method": "pelt_l2", "f1": 0.235, "recall": 0.32},
        {"freq": "monthly", "method": "window_l2", "f1": 0.27, "recall": 0.4},
    ]
    text = " ".join("\n".join(report._cases_section(meta, national)).split())
    assert (
        "- **Всего**: скользящее окно (L2) со штрафом, откалиброванным на бенчмарке, находит "
        "1 из 3 размеченных шоков, всего 9 тревог; окна допуска вокруг них накрывают 47% ряда "
        "(`reports/figures/case.png`)." in text
    )
    older = {
        "cases": {
            "national": [{k: v for k, v in meta["cases"]["national"][0].items() if k != "coverage"}]
        }
    }
    without = " ".join("\n".join(report._cases_section(older, national)).split())
    assert "всего 9 тревог (`reports/figures/case.png`)." in without
    assert (
        "На национальных месячных рядах лучший офлайн-метод — скользящее окно (L2) "
        "(F1 0,270, recall 0,400)." in text
    )
    only_weekly = " ".join("\n".join(report._cases_section({}, national[:1])).split())
    assert "На национальных месячных рядах" not in only_weekly


def test_news_section_lists_every_external_source_with_its_timing_and_result() -> None:
    """Раздел 7 показывает цепочку «источник → согласование по времени → что дал» таблицей:
    источники уровня МО описаны в отчёте, а не только строкой в абляциях."""
    from sbx.shell.pipelines import report

    inventory = [
        {
            "key": "spending",
            "source": "Расходы МО — набор организаторов",
            "what": "цель прогноза",
            "level": "МО",
            "timing": "—",
            "gave": "—",
        },
        {
            "key": "population",
            "source": "Росстат: численность населения МО на 1 января 2023 года",
            "what": "логарифм численности населения и доля городского населения",
            "level": "МО",
            "timing": "постоянная характеристика: оценка на 1 января 2023 года",
            "gave": "к базовой модели: −3,0 руб., лучше в 5 окнах из 10 — неустойчиво",
        },
    ]
    text = "\n".join(report._news_section({}, inventory))
    assert "| Источник | Что берётся | Уровень | Согласование по времени | Что дал |" in text
    assert (
        "| Росстат: численность населения МО на 1 января 2023 года | логарифм численности "
        "населения и доля городского населения | МО | постоянная характеристика: оценка на "
        "1 января 2023 года | к базовой модели: −3,0 руб., лучше в 5 окнах из 10 — неустойчиво |"
        in text
    )
    assert "Расходы МО — набор организаторов" not in text, "цель прогноза — не внешний источник"
    assert "| Источник |" not in "\n".join(report._news_section({}))
    measured = {
        "contributions": {
            "diebold_mariano": {
                "B_vs_A": {
                    "mean_diff": -27.8,
                    "p_value": 1e-9,
                    "folds": 10,
                    "folds_better": 6,
                    "folds_worse": 4,
                    "sign_p": 0.75,
                }
            }
        }
    }
    flat = " ".join("\n".join(report._news_section(measured)).split())
    assert "Измеренный вклад слоёв (раздел 9; вывод — по окнам проверки):" in flat
    assert "тест Диболда–Мариано, раздел 9" not in flat


def test_reproduction_section_names_what_each_step_does_now() -> None:
    from sbx.shell.pipelines import report

    text = "\n".join(report._reproduction_section())
    assert "make validate    # сверка файлов данных с манифестами" in text
    assert (
        "make entities    # таблица территорий по официальным кодам, сверка идентификации" in text
    )
    assert "соседние МО, погода, население" in text
    assert "make ablations   # матрица абляций и вклад каждого источника" in text
    assert "sbx data weather" in text and "sbx data population" in text


def test_limitations_count_the_validation_windows_and_the_evaluation_sample() -> None:
    from sbx.shell.pipelines import report

    meta = {
        "significance": {
            "by_horizon": {
                "1": {
                    "best_model": "Chronos-2 (covariates)",
                    "dm": [
                        {
                            "pair": "Chronos-2 (covariates) vs Prophet",
                            "mean_diff": -97.5,
                            "folds": 4,
                            "folds_better": 1,
                            "folds_worse": 3,
                            "sign_p": 0.625,
                        }
                    ],
                }
            }
        },
        "horizon_coverage": {1: {"models": 15, "observations": 4431}},
        "evaluation_series": 1201,
        "main_model_note": "Основной LightGBM получает календарь. Конфигурация выбрана потом.",
    }
    text = " ".join("\n".join(report._limitations_section(meta)).split())
    assert "9. **Окон проверки мало:** на 1 месяц — 4." in text
    assert "средний перевес создан меньшинством окон" in text
    assert "на 1 201 ряду оценочной выборки" in text and "1 месяц — 4 431" in text
    assert "806" not in text, "число строк таблицы прогнозов — не число наблюдений"
    assert "10. **Основная модель выбрана по тем же окнам.** Основной LightGBM получает" in text
    without = " ".join("\n".join(report._limitations_section({})).split())
    assert "Основная модель выбрана" not in without
    assert "отрицателен" not in without, "без таблицы моделей о знаке R² не говорится"
    assert "постоянных характеристик" not in without
    # Необязательные пункты нумеруются подряд, какой бы из них ни отсутствовал.
    static = {
        "entity": {
            "territories": 2190,
            "reference_year": 2024,
            "kind_changed_since_previous_year": 116,
        },
        "source_contributions": [{"block": "access", "against": "A"}],
    }
    only_static = " ".join("\n".join(report._limitations_section(static)).split())
    assert (
        "9. **Часть постоянных характеристик известна позже момента прогноза.** Тип МО взят из "
        "редакции справочника 2024 года: у 116 территорий из 2 190" in only_static
    )
    assert "рассчитаны организаторами по данным 2024 года" in only_static
    everything = " ".join("\n".join(report._limitations_section({**meta, **static})).split())
    assert "11. **Часть постоянных характеристик известна позже момента прогноза.**" in everything
    schedule = {"schedule_limitation": "Тревога по расписанию даёт F1 0,227; выше неё — 0 из 15."}
    with_schedule = " ".join(
        "\n".join(report._limitations_section({**meta, **static, **schedule})).split()
    )
    assert (
        "12. **Сравнение детекторов читается только рядом с расписанием.** Тревога по расписанию "
        "даёт F1 0,227; выше неё — 0 из 15." in with_schedule
    )
    assert "рядом с расписанием" not in everything
    counted_r2 = {
        "r2_note": "в сводке по всем горизонтам R² по отдельным рядам отрицателен у 14 из 15 моделей"
    }
    with_r2 = " ".join("\n".join(report._limitations_section(counted_r2)).split())
    assert (
        "из 12. В сводке по всем горизонтам R² по отдельным рядам отрицателен у 14 из 15 моделей."
        in with_r2
    ), "в начале предложения — с прописной"


def test_protocol_section_says_on_which_series_the_models_are_compared() -> None:
    """Раздел 3 называет оценочную выборку и то, как делается вывод об устойчивости: без этого
    таблицы раздела 4 читаются как сравнение на всей панели по тесту значимости."""
    from sbx.shell.pipelines import report

    meta = {"evaluation_series": 1201, "panel": {"series": 13140}}
    text = " ".join("\n".join(report._protocol_section(meta)).split())
    assert text.startswith("## 3. Протокол оценки")
    assert (
        "Глобальные модели учатся на всех 13 140 рядах панели; сравниваются модели на 1 201 ряде "
        "замороженной оценочной выборки (`configs/evaluation_series.csv`)." in text
    )
    assert "держится ли знак разности по окнам проверки" in text
    bare = " ".join("\n".join(report._protocol_section({})).split())
    assert "оценочной выборки" not in bare and "Всего 14 фолдов" in bare
