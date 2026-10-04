import pandas as pd
import pytest

from sbx.core.report import (
    ablation_basis,
    assemble_report_data,
    contribution_statements,
    control_alarm_statement,
    detector_card,
    detector_contrast,
    detector_headline,
    detector_rows,
    detector_statement,
    difference_brief,
    difference_phrase,
    fact_above_statement,
    forecast_bias_statement,
    format_delta,
    format_number,
    format_p_value,
    hazard_brief,
    hazard_verdict,
    history_limitation,
    horizon_winners,
    identification_facts,
    identification_summary,
    interval_runs,
    is_steady,
    leaderboard_view,
    leaders_statement,
    limitation_items,
    live_alarm_statement,
    main_model_note,
    minority_note,
    model_family,
    model_ranking,
    model_title,
    negative_r2_note,
    reference_windows,
    reference_wins,
    schedule_brief,
    schedule_limitation,
    schedule_match_statement,
    schedule_reference_statement,
    series_showcase,
    static_caveat,
    window_limitation,
    windows_label,
)


def _board() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["Ensemble", "LightGBM", "Prophet"],
            "mae": [600.0, 707.9, 768.2],
            "mae_macro": [610.0, 715.6, 786.2],
            "wape": [0.07, 0.079, 0.086],
            "r2_pooled": [0.99, 0.985, 0.984],
            "r2_per_series_median": [-0.05, -0.098, -0.176],
            "mase": [1.7, 1.85, 2.14],
        }
    )


def test_format_number_uses_russian_separators() -> None:
    assert format_number(1234567) == "1 234 567"
    # Разность печатается со знаком: без плюса «18,9» в столбце разностей читается как уровень.
    assert format_delta(18.94) == "+18,9" and format_delta(-58.66) == "−58,7"
    assert format_delta(0.04) == "0,0" and format_delta(-0.04) == "0,0", "ноль знака не носит"
    assert format_delta(3800.5) == "+3 800,5" and format_delta(float("nan")) == "—"
    assert format_delta(None) == "—" and format_delta(-2.345, 2) == "−2,35"
    assert format_number(707.87, 1) == "707,9"
    assert format_number(float("nan")) == "—"
    assert format_number(None) == "—"


def test_leaderboard_view_is_sorted_and_translated() -> None:
    view = leaderboard_view(_board())
    assert view.iloc[0]["Модель"] == "Ensemble"
    assert "MAE, руб." in view.columns
    assert list(view["Модель"]) == ["Ensemble", "LightGBM", "Prophet"]


def test_ranking() -> None:
    ranking = model_ranking(_board())
    assert [m for m, _ in ranking] == ["Ensemble", "LightGBM", "Prophet"]


def test_series_showcase_collects_plot_data_without_repeated_months() -> None:
    """Месяц предсказан в фолдах h=1 и h=3: на графике он должен быть один раз."""
    ds = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01", "2024-03-01"])
    oof = pd.DataFrame(
        {
            "unique_id": ["a"] * 4,
            "ds": ds,
            "model": ["Ensemble"] * 4,
            "horizon": [1, 3, 3, 3],
            "y": [100.0, 100.0, 110.0, 120.0],
            "y_hat": [101.0, 95.0, 109.0, 118.0],
            "q_0.1": [90.0, 85.0, 95.0, 100.0],
            "q_0.9": [110.0, 105.0, 120.0, 130.0],
        }
    )
    static = pd.DataFrame(
        {
            "unique_id": ["a"],
            "region_name": ["Московская область"],
            "mo_name": ["Городской округ Химки"],
            "category": ["Все категории"],
        }
    )
    out = series_showcase(oof, static, "Ensemble", ["a"])
    assert len(out) == 1
    item = out[0]
    assert item["region"] == "Московская область"
    assert item["mo"] == "Городской округ Химки"
    assert item["ds"] == ["2024-01", "2024-02", "2024-03"]
    assert item["y_hat"][0] == 101.0, "на месяц берётся прогноз самого короткого горизонта"
    assert item["q_lo"] and item["q_hi"]
    assert series_showcase(oof, static, "нет такой модели", ["a"]) == []


def _winner_summary() -> pd.DataFrame:
    rows = [
        ("A", 1, 500.0),
        ("B", 1, 450.0),
        ("Prophet", 1, 600.0),
        ("A", 12, 700.0),
        ("Prophet", 12, 900.0),
    ]
    frame = pd.DataFrame(rows, columns=["model", "value", "mae"])
    frame["slice"] = "horizon"
    overall = pd.DataFrame({"model": ["B"], "value": ["all"], "mae": [10.0], "slice": ["overall"]})
    return pd.concat([frame, overall], ignore_index=True)


def test_horizon_winners_pick_the_best_model_on_each_horizon() -> None:
    """Лучшая модель выбирается внутри горизонта, а не по смеси горизонтов.

    Иначе модель, неопределимая на самом трудном горизонте, выглядит лучшей в среднем: в её
    среднее этот горизонт просто не попадает.
    """
    winners = horizon_winners(_winner_summary(), horizons=(1, 3, 12))
    assert [w["horizon"] for w in winners] == [1, 12], "горизонт без строк пропускается"
    first, last = winners
    assert (first["model"], first["mae"], first["reference_mae"]) == ("B", 450.0, 600.0)
    assert first["gain"] == pytest.approx(0.25)
    assert (last["model"], last["mae"]) == ("A", 700.0)
    assert last["gain"] == pytest.approx(200.0 / 900.0)


def test_horizon_winners_without_reference_leave_the_gain_empty() -> None:
    summary = _winner_summary()
    winners = horizon_winners(summary[summary["model"] != "Prophet"], horizons=(1,))
    assert winners[0]["model"] == "B"
    assert winners[0]["reference_mae"] is None
    assert winners[0]["gain"] is None


DETECTORS = pd.DataFrame(
    {
        "detector": ["cusum", "page_hinkley", "page_hinkley", "cusum", "pelt_l2"],
        "kind": ["online", "online", "online", "online", "offline"],
        "input": ["residual", "residual", "raw", "raw", "raw"],
        "f1": [0.221, 0.220, 0.219, 0.188, 0.30],
        "mean_delay": [2.275, 2.695, 3.242, 3.732, 2.0],
        "false_alarms_per_series_year": [0.889, 1.215, 0.900, 0.824, 1.7],
    }
)


def test_detector_contrast_names_the_best_detector_of_each_input() -> None:
    """Сравнения «тот же детектор по сырому ряду» одного мало: другой детектор по сырому ряду
    может быть не хуже, и умолчать о нём — значит приукрасить результат."""
    contrast = detector_contrast(DETECTORS)
    assert contrast["detector"] == "cusum"
    assert contrast["f1_residual"] == 0.221 and contrast["delay_residual"] == 2.275
    assert contrast["false_alarms_residual"] == 0.889
    assert contrast["f1_raw"] == 0.188, "тот же детектор по сырому ряду"
    assert contrast["best_raw"] == {
        "detector": "page_hinkley",
        "f1": 0.219,
        "delay": 3.242,
        "false_alarms": 0.900,
    }
    assert detector_contrast(DETECTORS[DETECTORS["input"] == "raw"]) == {}


def test_detector_contrast_without_delay_columns_keeps_the_scores() -> None:
    contrast = detector_contrast(DETECTORS[["detector", "kind", "input", "f1"]])
    assert contrast["f1_residual"] == 0.221 and contrast["delay_residual"] is None
    assert contrast["best_raw"]["f1"] == 0.219 and contrast["best_raw"]["delay"] is None


def test_detector_statement_puts_both_comparisons_side_by_side() -> None:
    assert detector_statement(detector_contrast(DETECTORS)) == (
        "Лучший онлайн-детектор по остаткам прогноза — CUSUM: F1 0,221, задержка 2,3 мес. "
        "Лучший по сырому ряду — Page–Hinkley: F1 0,219, задержка 3,2 мес. "
        "Тот же CUSUM по сырому ряду — F1 0,188."
    )
    same = DETECTORS[~((DETECTORS["detector"] == "page_hinkley") & (DETECTORS["input"] == "raw"))]
    assert detector_statement(detector_contrast(same)) == (
        "Лучший онлайн-детектор по остаткам прогноза — CUSUM: F1 0,221, задержка 2,3 мес. "
        "Лучший по сырому ряду — он же: F1 0,188, задержка 3,7 мес."
    )
    bare = detector_statement(detector_contrast(DETECTORS[["detector", "kind", "input", "f1"]]))
    assert bare.startswith("Лучший онлайн-детектор по остаткам прогноза — CUSUM: F1 0,221. ")
    assert detector_statement({}) == ""


def _with_reference(f1: float, period: float = 5.0, delay: float | None = None) -> pd.DataFrame:
    """Сравнение детекторов с эталоном «тревога по расписанию», по убыванию F1."""
    reference = pd.DataFrame(
        {
            "detector": ["periodic"],
            "kind": ["baseline"],
            "input": ["none"],
            "param": [period],
            "f1": [f1],
            "mean_delay": [delay],
        }
    )
    table = pd.concat([DETECTORS.assign(param=3.0), reference], ignore_index=True)
    return table.sort_values("f1", ascending=False).reset_index(drop=True)


def test_schedule_reference_statement_counts_the_methods_that_beat_the_schedule() -> None:
    """Сколько вариантов точнее тревоги по расписанию — мера того, что методы находят."""
    assert schedule_reference_statement(_with_reference(0.2)) == (
        "Эталон сравнения — тревога по расписанию, каждые 5 месяцев независимо от данных: "
        "F1 0,200. Выше эталона по F1 — 4 из 5 вариантов «метод × вход»; лучший превосходит его на 0,100."
    )
    # Равный F1 — не «выше»; период склоняется.
    equal = schedule_reference_statement(_with_reference(0.219, period=3.0))
    assert "каждые 3 месяца независимо" in equal and "— 3 из 5 вариантов" in equal
    assert "каждый месяц независимо" in schedule_reference_statement(_with_reference(0.2, 1.0))
    # Расписание точнее всех методов: так и сказано, без отрицательного «превосходит».
    assert schedule_reference_statement(_with_reference(0.5)).endswith(
        "F1 0,500. Выше эталона по F1 — 0 из 5 вариантов «метод × вход»."
    )


def test_schedule_reference_statement_needs_both_the_reference_and_the_methods() -> None:
    assert schedule_reference_statement(DETECTORS) == ""
    only_reference = _with_reference(0.2)
    assert schedule_reference_statement(only_reference[only_reference["kind"] == "baseline"]) == ""
    assert schedule_reference_statement(pd.DataFrame()) == ""


def test_detector_contrast_carries_the_schedule_reference() -> None:
    """Рядом с F1 детектора стоит F1 расписания: без него число ни о чём не говорит."""
    contrast = detector_contrast(_with_reference(0.227, period=6.0, delay=2.182))
    assert contrast["reference"] == {"f1": 0.227, "period": 6, "delay": 2.182}
    assert contrast["detector"] == "cusum", "эталон не становится лучшим детектором"
    # Порог лучшего детектора — по нему на кривой находится точка для сравнения с расписанием.
    assert contrast["threshold"] == 3.0 and detector_contrast(DETECTORS)["threshold"] is None
    assert detector_contrast(_with_reference(0.227))["reference"]["delay"] is None
    assert "reference" not in detector_contrast(DETECTORS)


def test_headline_and_card_put_the_schedule_next_to_the_detector() -> None:
    contrast = detector_contrast(_with_reference(0.227, period=6.0, delay=2.182))
    assert detector_headline(contrast) == (
        "Сдвиги по остаткам прогноза: F1 0,221, задержка 2,3 мес.; "
        "по сырому ряду — 0,219 и 3,2 мес.; тревога по расписанию — 0,227 и 2,2 мес."
    )
    card = detector_card(contrast)
    assert card["value"] == "0,221"
    assert card["label"].endswith("; тревога по расписанию — 0,227 и 2,2 мес.")
    bare = detector_contrast(_with_reference(0.227))
    assert detector_headline(bare).endswith("; тревога по расписанию — 0,227")
    assert detector_card(bare)["label"].endswith("; тревога по расписанию — 0,227")
    # Детекторов по сырому ряду в таблице нет — эталон в строке всё равно остаётся.
    table = _with_reference(0.227)
    only_residual = detector_contrast(table[table["input"] != "raw"])
    assert detector_headline(only_residual) == (
        "Сдвиги по остаткам прогноза: F1 0,221, задержка 2,3 мес.; тревога по расписанию — 0,227"
    )
    assert "расписани" not in detector_headline(detector_contrast(DETECTORS))
    assert "расписани" not in detector_card(detector_contrast(DETECTORS))["label"]


def test_schedule_limitation_says_what_the_score_cannot_tell() -> None:
    """Пока хоть один вариант не выше расписания, F1 детекторов — не мера качества обнаружения,
    и это ограничение решения, а не подробность таблицы."""
    assert schedule_limitation(_with_reference(0.5)) == (
        "Тревога по расписанию, не глядя на данные, даёт F1 0,500; выше неё — 0 из 5 вариантов "
        "«метод × вход». F1 с окном допуска на этом бенчмарке не отличает метод от расписания; "
        "отличает сравнение при равной частоте ложных тревог."
    )
    assert "выше неё — 4 из 5 вариантов" in schedule_limitation(_with_reference(0.2))
    assert "выше неё — 3 из 5 вариантов" in schedule_limitation(_with_reference(0.219)), (
        "равный — не выше"
    )
    # Каждый вариант точнее расписания — оговаривать нечего.
    assert schedule_limitation(_with_reference(0.01)) == ""
    assert schedule_limitation(DETECTORS) == "" and schedule_limitation(pd.DataFrame()) == ""


def test_detector_contrast_carries_the_share_of_alarmed_series() -> None:
    table = DETECTORS.assign(
        alarmed_break_share=[0.804, 0.98, 0.987, 0.89, 0.784],
        alarmed_control_share=[0.714, 0.979, 0.99, 0.859, 0.74],
    )
    contrast = detector_contrast(table)
    assert (contrast["alarmed_breaks"], contrast["alarmed_controls"]) == (0.804, 0.714)
    bare = detector_contrast(DETECTORS)
    assert bare["alarmed_breaks"] is None and bare["alarmed_controls"] is None


def test_control_alarm_statement_puts_series_without_a_break_next_to_the_rest() -> None:
    """Сколько рядов без внесённого слома получают тревогу — число, без которого полнота
    детектора читается как умение находить сломы."""
    contrast = {"detector": "cusum", "alarmed_breaks": 0.804, "alarmed_controls": 0.714}
    assert control_alarm_statement(contrast) == (
        "На рядах без внесённого слома CUSUM поднимает тревогу у 71% рядов, на рядах со сломом "
        "— у 80%: разметка покрывает только внесённые сломы, а собственные изменения настоящих "
        "рядов в ней не отмечены."
    )
    assert control_alarm_statement({"detector": "cusum", "alarmed_breaks": 0.8}) == ""
    assert control_alarm_statement({"detector": "cusum", "alarmed_controls": 0.7}) == ""
    assert control_alarm_statement({}) == ""


MATCH = {
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


def test_schedule_match_statement_gives_the_gap_with_its_interval() -> None:
    """Детектор против расписания с той же частотой ложных тревог, на тестовой половине, с
    интервалом: одно число разности выглядело бы увереннее, чем позволяют данные."""
    assert schedule_match_statement(MATCH) == (
        "Расписание с ближайшей частотой ложных тревог — раз в 8 месяцев (0,87 на ряд в год "
        "против 0,89 у детектора): оно находит 34% сломов и срабатывает через 3,1 мес. после "
        "слома; CUSUM по остаткам прогноза — 43% и 2,3 мес. Разница в полноте — 8,6 п.п., "
        "95%-й интервал бутстрепа по рядам — от 3,1 до 14,2 п.п."
    )
    unsure = schedule_match_statement({**MATCH, "recall_diff_low": -0.012})
    assert unsure.endswith(
        "95%-й интервал бутстрепа по рядам — от −1,2 до 14,2 п.п.: он включает ноль, перевес не "
        "доказан."
    )
    yearly = schedule_match_statement({**MATCH, "period": 12, "level": 0.9})
    assert "раз в 12 месяцев" in yearly and "90%-й интервал" in yearly
    assert "раз в месяц (" in schedule_match_statement({**MATCH, "period": 1})
    assert schedule_match_statement({}) == ""
    assert schedule_match_statement({"detector": "cusum"}) == ""


def test_detector_rows_keep_the_reference_in_the_table() -> None:
    """Таблица показывает лучшие строки, но эталон остаётся в ней и ниже границы: без него
    F1 методов не с чем сравнить."""
    below = detector_rows(_with_reference(0.01), limit=2)
    assert below["detector"].tolist() == ["pelt_l2", "cusum", "periodic"]
    inside = detector_rows(_with_reference(0.9), limit=2)
    assert inside["detector"].tolist() == ["periodic", "pelt_l2"], "эталон не дублируется"
    ordered = DETECTORS.sort_values("f1", ascending=False)
    assert detector_rows(ordered, limit=2)["detector"].tolist() == ["pelt_l2", "cusum"]
    assert detector_rows(ordered.drop(columns="kind"), limit=2)["f1"].tolist() == [0.30, 0.221]


def test_detector_headline_is_one_line_with_both_inputs() -> None:
    assert detector_headline(detector_contrast(DETECTORS)) == (
        "Сдвиги по остаткам прогноза: F1 0,221, задержка 2,3 мес.; "
        "по сырому ряду — 0,219 и 3,2 мес."
    )
    bare = detector_contrast(DETECTORS[["detector", "kind", "input", "f1"]])
    assert (
        detector_headline(bare) == "Сдвиги по остаткам прогноза: F1 0,221; по сырому ряду — 0,219"
    )
    only_residual = detector_contrast(DETECTORS[DETECTORS["input"] == "residual"])
    assert detector_headline(only_residual) == (
        "Сдвиги по остаткам прогноза: F1 0,221, задержка 2,3 мес."
    )
    assert detector_headline({}) == ""


def test_assemble_report_data_structure() -> None:
    detectors = pd.DataFrame({"detector": ["cusum"], "f1": [0.4]})
    ablations = pd.DataFrame({"name": ["A"], "mae": [700.0]})
    data = assemble_report_data(_board(), detectors, ablations, meta={"title": "t"})
    assert set(data) >= {"meta", "leaderboard", "leaderboard_view", "detectors", "ablations"}
    assert data["meta"]["title"] == "t"
    assert len(data["leaderboard"]) == 3


def _explorer_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    ds = pd.date_range("2024-01-01", periods=3, freq="MS")
    rows = []
    for uid, region, category in [
        ("a", "Московская область", "Все категории"),
        ("b", "Московская область", "Продовольствие"),
        ("c", "Тверская область", "Все категории"),
    ]:
        # Горизонты 1 и 3 предсказывают одни и те же месяцы.
        for horizon in (1, 3):
            for d in ds:
                rows.append(
                    {
                        "unique_id": uid,
                        "ds": d,
                        "fold": f"h{horizon}_2023-12",
                        "horizon": horizon,
                        "model": "Ensemble",
                        "y": 100.0,
                        "y_hat": 98.0,
                        "q_0.1": 90.0,
                        "q_0.9": 110.0,
                        "region": region,
                        "category": category,
                    }
                )
    oof = pd.DataFrame(rows)
    static = (
        oof[["unique_id", "region", "category"]]
        .drop_duplicates("unique_id")
        .rename(columns={"region": "region_name"})
    )
    return oof, static


def test_explorer_data_groups_regions_and_drops_duplicate_months() -> None:
    from sbx.core.report import explorer_data

    oof, static = _explorer_fixture()
    data = explorer_data(oof, static, "Ensemble", alarms={"a": [1]}, limit=10)

    assert data["regions"] == ["Московская область", "Тверская область"]
    assert "Все категории" in data["categories"]
    for item in data["series"]:
        # Горизонты перекрываются по месяцам: месяц не должен повторяться.
        assert len(item["ds"]) == len(set(item["ds"])), item["id"]
        assert len(item["y"]) == len(item["ds"]) == len(item["y_hat"])
    first = next(s for s in data["series"] if s["id"] == "a")
    assert first["alarms"] == ["2024-02"]
    assert first["q_lo"] and first["q_hi"]
    assert first["band"] == [[0, len(first["ds"])]], "отрезки полосы интервала — для страницы"
    # Без названия МО в справочнике подпись пустая, а не падение.
    assert first["mo"] == ""


def test_explorer_data_respects_limit_and_is_deterministic() -> None:
    from sbx.core.report import explorer_data

    oof, static = _explorer_fixture()
    a = explorer_data(oof, static, "Ensemble", limit=2)
    b = explorer_data(oof, static, "Ensemble", limit=2)
    assert len(a["series"]) == 2
    assert [s["id"] for s in a["series"]] == [s["id"] for s in b["series"]]
    assert explorer_data(oof, static, "нет такой модели")["series"] == []


def _horizon_summary() -> pd.DataFrame:
    rows = []
    # LightGBM отсутствует при h=12 — он там неопределим по построению.
    for model, base in [("Ensemble", 500.0), ("Prophet", 700.0), ("LightGBM", 600.0)]:
        for horizon in (1, 3, 6, 12):
            if model == "LightGBM" and horizon == 12:
                continue
            rows.append(
                {
                    "slice": "horizon",
                    "value": horizon,
                    "model": model,
                    "mae": base + horizon * 10,
                    "n": 1000 * horizon,
                }
            )
    rows.append({"slice": "overall", "value": "all", "model": "Ensemble", "mae": 1.0, "n": 1})
    return pd.DataFrame(rows)


def test_horizon_leaderboard_has_a_column_per_required_horizon() -> None:
    from sbx.core.report import horizon_leaderboard

    table = horizon_leaderboard(_horizon_summary())
    assert list(table.columns) == ["Модель", "mae h=1", "mae h=3", "mae h=6", "mae h=12"]
    # Сортировка по самому короткому горизонту: там сравнение опирается на больше моделей.
    assert list(table["Модель"]) == ["Ensemble", "LightGBM", "Prophet"]
    # Пропуск означает «неопределимо», а не «плохо».
    lgbm = table[table["Модель"] == "LightGBM"].iloc[0]
    assert pd.isna(lgbm["mae h=12"])
    assert lgbm["mae h=6"] == 660.0


def test_horizon_leaderboard_is_empty_without_the_slice() -> None:
    from sbx.core.report import horizon_leaderboard

    assert horizon_leaderboard(pd.DataFrame({"slice": ["overall"], "value": ["all"]})).empty


def test_horizon_coverage_reports_model_count_per_horizon() -> None:
    from sbx.core.report import horizon_coverage

    coverage = horizon_coverage(_horizon_summary())
    assert coverage[1]["models"] == 3
    assert coverage[12]["models"] == 2, "при h=12 моделей меньше — это должно быть видно"
    assert coverage[12]["observations"] == 12000


def test_horizon_label_agrees_the_noun_with_the_number() -> None:
    from sbx.core.report import horizon_label

    assert [horizon_label(h) for h in (1, 3, 6, 12, 21, 24)] == [
        "1 месяц",
        "3 месяца",
        "6 месяцев",
        "12 месяцев",
        "21 месяц",
        "24 месяца",
    ]


def test_short_mo_name_drops_the_federal_city_prefix() -> None:
    from sbx.core.report import short_mo_name

    long = "внутригородская территория города федерального значения муниципальный округ Пресненский"
    assert short_mo_name(long) == "муниципальный округ Пресненский"
    assert short_mo_name("Мотыгинский муниципальный район") == "Мотыгинский муниципальный район"
    assert short_mo_name(None) == ""


ENTITY = {
    "territories": 2190,
    "series": 13140,
    "regions": 77,
    "reference_year": 2024,
    "same_names": 49,
    "same_name_territories": 121,
    "heuristic_check": {
        "series": 13140,
        "series_in_wrong_region": 327,
        "territories": 2188,
        "territories_with_foreign_series": 101,
        "territories_entirely_in_wrong_region": 14,
    },
}


def test_identification_summary_is_written_from_the_measured_numbers() -> None:
    text = " ".join(identification_summary(ENTITY))
    assert "2 190 территорий" in text and "13 140 рядов" in text and "77 регион" in text
    assert "49 названий" in text and "121" in text
    assert "327 из 13 140 рядов (2,5%)" in text
    assert "101 из 2 188" in text and "14 территорий" in text
    assert "справочник" in text and "2024" in text


def test_identification_summary_stands_on_its_own() -> None:
    """Сводка сверки не отсылает к другому документу: всё нужное читателю сказано в ней."""
    text = " ".join(identification_summary(ENTITY))
    assert "Разбор —" not in text and ".md" not in text


def test_identification_summary_follows_the_numbers_not_a_template() -> None:
    """Числа не зашиты в текст: другой прогон — другие числа."""
    other = {
        **ENTITY,
        "heuristic_check": {**ENTITY["heuristic_check"], "series_in_wrong_region": 0},
    }
    text = " ".join(identification_summary(other))
    assert "327" not in text and "совпала" in text


def test_identification_summary_without_the_cross_check_states_only_the_source() -> None:
    lines = identification_summary({k: v for k, v in ENTITY.items() if k != "heuristic_check"})
    assert len(lines) == 1 and "2 190 территорий" in lines[0]
    assert identification_summary({}) == []


DM = {
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
    "D_vs_B": {
        "mean_diff": 12.4,
        "p_value": 0.004,
        "folds": 10,
        "folds_better": 1,
        "folds_worse": 9,
        "sign_p": 0.021,
    },
}


def test_p_values_are_printed_the_way_a_reader_expects() -> None:
    assert format_p_value(0.31) == "p = 0,31"
    assert format_p_value(0.004) == "p = 0,004"
    assert format_p_value(2.0e-9) == "p < 0,001"
    assert format_p_value(float("nan")) == "p не оценено"


def test_contribution_statements_follow_the_sign_and_the_windows() -> None:
    """Вывод о вкладе источника не зашит в текст: он складывается из знака разности и из того,
    в скольких окнах проверки этот знак держится."""
    macro, news, foundation = contribution_statements(DM)
    assert macro == (
        "Календарь и национальные ряды: MAE ниже на 60,8 руб., лучше в 9 окнах из 10 — устойчиво."
    )
    assert news == (
        "Новости и календарь событий поверх них: устойчивого изменения нет — MAE выше на "
        "6,7 руб. в среднем, хуже в 6 окнах из 10."
    )
    assert foundation == (
        "Foundation-модели в ансамбле: MAE выше на 12,4 руб., хуже в 9 окнах из 10 — устойчиво."
    )


def test_contribution_statements_change_with_the_numbers() -> None:
    flipped = {
        "B_vs_A": {
            "mean_diff": 111.9,
            "p_value": 1e-12,
            "folds": 10,
            "folds_better": 0,
            "folds_worse": 10,
            "sign_p": 0.002,
        }
    }
    assert contribution_statements(flipped) == [
        "Календарь и национальные ряды: MAE выше на 111,9 руб., хуже в 10 окнах из 10 — устойчиво."
    ]
    assert contribution_statements({}) == []
    unknown = {"X_vs_Y": {"mean_diff": 1.0, "p_value": 0.5}}
    assert contribution_statements(unknown) == [], "пара без названия в текст не попадает"


def test_hazard_verdict_follows_the_matched_comparison() -> None:
    worse = {"model_at_matched_alarm_rate": {"f1": 0.21}, "reference_detector": {"f1": 0.40}}
    assert "не входит в итоговую систему" in hazard_verdict(worse)
    assert "0,210 против 0,400" in hazard_verdict(worse)
    better = {"model_at_matched_alarm_rate": {"f1": 0.45}, "reference_detector": {"f1": 0.40}}
    assert "не входит" not in hazard_verdict(better) and "не уступает" in hazard_verdict(better)
    assert hazard_verdict({}) == ""


def test_negative_numbers_carry_a_minus_sign_not_a_hyphen() -> None:
    assert format_number(-29.5, 1) == "−29,5"
    assert format_number(-1234.5, 1) == "−1 234,5"
    assert format_number(-5) == "−5"
    assert format_number(-0.04, 1) == "0,0", "ноль после округления знака не носит"
    assert format_number(29.5, 1) == "29,5"


def test_interval_runs_are_the_stretches_where_both_bounds_exist() -> None:
    """Полоса интервала рисуется отрезками: там, где у прогноза интервала нет, её нет."""
    nan = float("nan")
    assert interval_runs([1.0, 1.0, nan, nan, 1.0], [2.0, 2.0, nan, 2.0, 2.0]) == [(0, 2), (4, 5)]
    assert interval_runs([1.0, 2.0], [3.0, 4.0]) == [(0, 2)]
    assert interval_runs([1.0, 2.0], [3.0, nan]) == [(0, 1)], "нужны обе границы"
    assert interval_runs([None, 1.0], [None, 2.0]) == [(1, 2)]
    assert interval_runs([], []) == []


def test_detector_card_states_the_score_and_the_best_raw_alternative() -> None:
    assert detector_card(detector_contrast(DETECTORS)) == {
        "value": "0,221",
        "label": (
            "F1 обнаружения сдвигов по остаткам прогноза, задержка 2,3 мес.; "
            "лучший детектор по сырому ряду — 0,219 и 3,2 мес."
        ),
    }
    bare = detector_card({"detector": "cusum", "f1_residual": 0.221, "f1_raw": None})
    assert bare == {"value": "0,221", "label": "F1 обнаружения сдвигов по остаткам прогноза"}


def test_ablation_basis_names_the_rows_every_configuration_shares() -> None:
    rows = [{"name": "A", "rows": 31238}, {"name": "D", "rows": 31238}]
    assert ablation_basis(rows) == (
        "Все конфигурации измерены на одних и тех же 31 238 наблюдениях. В таблице — разница "
        "общих MAE; в тесте значимости — средняя по рядам разность ошибок, поэтому эти числа "
        "немного расходятся."
    )
    assert ablation_basis([{"name": "A", "mae": 1.0}]) == "", "нет числа строк — нет и фразы"
    assert ablation_basis([{"name": "A", "rows": 5}, {"name": "D", "rows": 4}]) == ""
    assert ablation_basis([]) == ""


def _row(diff: float, better: int, folds: int, sign_p: float, p_value: float = 1e-9) -> dict:
    return {
        "mean_diff": diff,
        "p_value": p_value,
        "folds": folds,
        "folds_better": better,
        "folds_worse": folds - better,
        "sign_p": sign_p,
    }


def test_difference_phrase_follows_the_sign_and_the_windows() -> None:
    """Значимость по рядам вывода не делает: все прогнозы окна строит одна модель, и разности у
    рядов общие. Устойчивым считается знак, который держится по окнам проверки."""
    steady = _row(-60.8, better=9, folds=10, sign_p=0.021)
    assert difference_phrase(steady) == "MAE ниже на 60,8 руб., лучше в 9 окнах из 10 — устойчиво"
    shaky = _row(-27.8, better=6, folds=10, sign_p=0.754, p_value=1e-17)
    assert difference_phrase(shaky) == (
        "устойчивого изменения нет — MAE ниже на 27,8 руб. в среднем, лучше в 6 окнах из 10"
    ), "p по рядам меньше 0,001 вывода не меняет"
    one_window = _row(8.0, better=3, folds=4, sign_p=0.625)
    assert difference_phrase(one_window) == (
        "устойчивого изменения нет — MAE выше на 8,0 руб. в среднем, хуже в 1 окне из 4"
    ), "средняя разность создана одним окном"
    assert difference_phrase({"mean_diff": -5.0, "p_value": 0.01}) == (
        "MAE ниже на 5,0 руб.; по окнам проверки не сверено"
    )
    assert difference_phrase({"mean_diff": float("nan")}) == ""
    assert difference_phrase({}) == ""


def test_windows_label_counts_the_windows_that_agree_with_the_average() -> None:
    assert windows_label(_row(-60.8, better=9, folds=10, sign_p=0.021)) == "в 9 окнах из 10"
    assert windows_label(_row(8.0, better=3, folds=4, sign_p=0.625)) == "в 1 окне из 4"
    assert windows_label({"mean_diff": -5.0}) == ""
    assert is_steady(_row(-60.8, better=9, folds=10, sign_p=0.021)) is True
    assert is_steady(_row(-60.8, better=6, folds=10, sign_p=0.754)) is False
    assert is_steady({"mean_diff": -5.0, "p_value": 1e-30}) is False, "без окон вывода нет"


SIGNIFICANCE = {
    "by_horizon": {
        "1": {
            "best_model": "Chronos-2 (covariates)",
            "dm": [
                {"pair": "Chronos-2 (covariates) vs Naive", **_row(-210.3, 2, 4, 1.0)},
                {"pair": "Chronos-2 (covariates) vs Prophet", **_row(-97.5, 1, 4, 0.625)},
            ],
        },
        "3": {
            "best_model": "Chronos-2 (covariates)",
            "dm": [{"pair": "Chronos-2 (covariates) vs Prophet", **_row(-246.6, 3, 4, 0.625)}],
        },
        "6": {"best_model": "Prophet", "dm": []},
        "12": {
            "best_model": "Ensemble",
            "dm": [{"pair": "Ensemble vs Prophet", **_row(-3800.5, 3, 3, 0.25)}],
        },
    }
}


def test_reference_windows_tell_in_how_many_windows_the_leader_beats_prophet() -> None:
    assert reference_windows(SIGNIFICANCE) == (
        "По окнам проверки лучшая модель точнее Prophet: 1 месяц — в 1 окне из 4; 3 месяца — в "
        "3 окнах из 4; 12 месяцев — в 3 окнах из 3. На горизонте 1 месяц средний перевес создан "
        "меньшинством окон: в остальных точнее Prophet."
    )
    steady = {"by_horizon": {"12": SIGNIFICANCE["by_horizon"]["12"]}}
    assert reference_windows(steady) == (
        "По окнам проверки лучшая модель точнее Prophet: 12 месяцев — в 3 окнах из 3."
    )
    assert reference_windows({}) == ""
    bare = {
        "by_horizon": {"1": {"best_model": "X", "dm": [{"pair": "X vs Prophet", "mean_diff": -1}]}}
    }
    assert reference_windows(bare) == "", "прогон без счёта окон — фразы нет"


def test_window_limitation_counts_the_windows_behind_every_horizon() -> None:
    """Ограничение называет число окон, на которых сравниваются модели горизонта, и говорит,
    где средний перевес над эталоном создан меньшинством окон."""
    text = window_limitation(SIGNIFICANCE)
    assert text.startswith(
        "Окон проверки мало: на 1 месяц — 4, на 3 месяца — 4, на 12 месяцев — 3. Различия "
        "моделей измерены в этих окнах; повторятся ли они в другом периоде, по таким данным "
        "сказать нельзя."
    )
    assert "1 месяц — в 1 окне из 4" in text and "меньшинством окон" in text
    assert window_limitation({}) == ""


def test_difference_brief_is_the_same_verdict_in_a_table_cell() -> None:
    assert difference_brief(_row(-60.8, better=9, folds=10, sign_p=0.021)) == (
        "−60,8 руб., лучше в 9 окнах из 10 — устойчиво"
    )
    assert difference_brief(_row(18.9, better=5, folds=10, sign_p=1.0)) == (
        "+18,9 руб., хуже в 5 окнах из 10 — неустойчиво"
    )
    assert difference_brief({"mean_diff": -5.0}) == "−5,0 руб.; по окнам проверки не сверено"
    assert difference_brief({}) == ""


def test_main_model_note_says_what_the_model_takes_and_how_it_was_chosen() -> None:
    """Конфигурация, выбранная по тем же окнам, на которых она оценивается, выглядит лучше, чем
    есть, — отчёт обязан сказать это сам."""
    note = main_model_note({"tune": False, "feature_blocks": ["calendar"]})
    assert note == (
        "Основной LightGBM получает производственный календарь на целевой месяц; "
        "гиперпараметры не подбираются. Эта конфигурация выбрана после сравнения вариантов "
        "модели на тех же окнах проверки, поэтому её оценка оптимистична: лучший из нескольких "
        "вариантов на новых данных обычно оказывается хуже, чем на тех, по которым его выбрали."
    )
    tuned = main_model_note({"tune": True, "feature_blocks": ["calendar", "macro"]})
    assert "календарь на целевой месяц и национальные ряды СберИндекса" in tuned
    assert "гиперпараметры подбираются" in tuned
    assert main_model_note({"tune": True}) == "", "модель без внешних признаков — как задумана"


def test_negative_r2_note_counts_the_models_instead_of_claiming_all() -> None:
    """Утверждение «R² по ряду отрицателен у всех моделей» — число из таблицы, а не из шаблона:
    после смены основной модели оно перестало быть верным."""
    board = _board()
    assert negative_r2_note(board) == (
        "в сводке по всем горизонтам R² по отдельным рядам отрицателен у всех 3 моделей"
    )
    mixed = board.assign(r2_per_series_median=[-0.05, 0.13, -0.176])
    assert negative_r2_note(mixed) == (
        "в сводке по всем горизонтам R² по отдельным рядам отрицателен у 2 из 3 моделей"
    )
    assert negative_r2_note(board.assign(r2_per_series_median=[0.1, 0.2, 0.3])) == ""
    assert negative_r2_note(board.drop(columns="r2_per_series_median")) == ""
    gaps = board.assign(r2_per_series_median=[-0.05, float("nan"), -0.176])
    assert negative_r2_note(gaps).endswith("отрицателен у всех 2 моделей")


def test_history_limitation_carries_the_measured_note() -> None:
    note = "в сводке по всем горизонтам R² по отдельным рядам отрицателен у 14 из 15 моделей"
    assert history_limitation(note) == (
        "История каждого ряда — 24 месяца: в сводке по всем горизонтам R² по отдельным рядам "
        "отрицателен у 14 из 15 моделей, а на горизонте 12 месяцев direct-модели построить нельзя."
    )
    assert history_limitation("") == (
        "История каждого ряда — 24 месяца: на горизонте 12 месяцев direct-модели построить нельзя."
    )


LIVE = {
    "detector": "cusum",
    "series": 1111,
    "alarms": 3067,
    "alarms_per_series_year": 2.77,
    "conformal_alarms": 444,
    "series_with_alarm": 935,
    "above_forecast_share": 0.705,
}


def test_live_alarm_statement_says_how_often_and_why_the_detector_fires() -> None:
    """Частота тревог на настоящих остатках без оговорки о смещении читалась бы как частота
    сдвигов: рядом стоит доля рядов с тревогой и то, в какую сторону ошибается прогноз."""
    assert live_alarm_statement(LIVE) == (
        "На настоящих остатках ансамбля (1 111 рядов оценочной выборки, один прогноз на месяц) "
        "CUSUM поднимает 3 067 тревог — 2,77 на ряд в год; конформный детектор по интервалам "
        "ансамбля — 444. Хотя бы одна тревога — у 935 рядов из 1 111 (84%). Остатки смещены: "
        "факт выше прогноза в 70,5% месяцев, поэтому часть тревог отражает систематическое "
        "отставание прогноза, а не смену режима."
    )
    low = live_alarm_statement({**LIVE, "above_forecast_share": 0.31})
    assert "факт ниже прогноза в 69,0% месяцев" in low and "завышение прогноза" in low
    for share in (0.5, 0.59, 0.41):
        assert "Остатки смещены" not in live_alarm_statement(
            {**LIVE, "above_forecast_share": share}
        )
    assert "Остатки смещены" in live_alarm_statement({**LIVE, "above_forecast_share": 0.6})
    older = {
        k: v for k, v in LIVE.items() if k not in ("series_with_alarm", "above_forecast_share")
    }
    assert live_alarm_statement(older).endswith("конформный детектор по интервалам ансамбля — 444.")
    assert live_alarm_statement({}) == "" and live_alarm_statement({"detector": "cusum"}) == ""
    one = live_alarm_statement({**LIVE, "series_with_alarm": 1, "series": 1})
    assert "Хотя бы одна тревога — у 1 ряда из 1 (100%)." in one


def test_static_caveat_names_the_characteristics_known_later_than_the_forecast() -> None:
    entity = {"territories": 2190, "reference_year": 2024, "kind_changed_since_previous_year": 116}
    assert static_caveat(entity, market_access=True) == (
        "Тип МО взят из редакции справочника 2024 года: у 116 территорий из 2 190 он отличается "
        "от редакции 2023 года, и для окон проверки от месяцев 2023 года это сведение из "
        "следующей редакции. Индекс доступности рынков и расстояния между МО рассчитаны "
        "организаторами по данным 2024 года — позже части моментов прогноза."
    )
    only_kind = static_caveat(entity, market_access=False)
    assert only_kind.endswith("из следующей редакции.") and "доступности" not in only_kind
    unchanged = {**entity, "kind_changed_since_previous_year": 0}
    assert static_caveat(unchanged, market_access=False) == ""
    assert static_caveat(unchanged, market_access=True).startswith("Индекс доступности рынков")
    assert static_caveat({}, market_access=False) == ""
    one = static_caveat({**entity, "kind_changed_since_previous_year": 1}, market_access=False)
    assert "у 1 территории из 2 190" in one


# --- Короткие формулировки для страницы в оформлении редизайна. ---

LEADERS = [
    {"horizon": 1, "model": "Chronos-2 (covariates)", "mae": 581.7, "reference_mae": 683.3},
    {"horizon": 3, "model": "Chronos-2 (covariates)", "mae": 588.6, "reference_mae": 835.9},
    {"horizon": 6, "model": "LightGBM", "mae": 422.0, "reference_mae": 1353.8},
    {"horizon": 12, "model": "Ensemble", "mae": 793.7, "reference_mae": 4596.5},
]


def test_model_title_is_the_name_a_reader_sees() -> None:
    assert model_title("Chronos-2 (covariates)") == "Chronos-2 с ковариатами"
    assert model_title("Ensemble") == "Ансамбль"
    assert model_title("Ensemble (по категориям)") == "Ансамбль по категориям"
    assert model_title("Naive") == "Наивный" and model_title("SeasonalNaive") == "Сезонный наивный"
    assert model_title("LightGBM") == "LightGBM", "имя без перевода остаётся как есть"


def test_leaders_statement_names_the_best_model_of_every_horizon() -> None:
    assert leaders_statement(LEADERS) == (
        "На 1 и 3 месяца точнее всех Chronos-2 с ковариатами, на 6 месяцев — LightGBM, "
        "на 12 месяцев — ансамбль. У Prophet, эталона из условий конкурса, ошибка выше на "
        "каждом из 4 горизонтов."
    )
    one = leaders_statement(LEADERS[:1])
    assert one.startswith("На 1 месяц точнее всех Chronos-2 с ковариатами.")
    assert leaders_statement([]) == ""


def test_leaders_statement_does_not_claim_a_win_over_the_reference_that_did_not_happen() -> None:
    lost = [{"horizon": 1, "model": "Prophet", "mae": 600.0, "reference_mae": 600.0}, LEADERS[2]]
    text = leaders_statement(lost)
    assert text.startswith("На 1 месяц точнее всех Prophet, на 6 месяцев — LightGBM.")
    assert text.endswith("MAE ниже, чем у Prophet, на 1 горизонте из 2.")
    unknown = [{"horizon": 1, "model": "AutoTheta", "mae": 640.0, "reference_mae": None}]
    assert leaders_statement(unknown) == "На 1 месяц точнее всех AutoTheta.", (
        "эталона нет — нет и фразы"
    )


WINS_SIGNIFICANCE = {
    "by_horizon": {
        "1": {"dm": [{"pair": "Chronos-2 (covariates) vs Prophet", "folds": 4, "folds_better": 1}]},
        "12": {"dm": [{"pair": "Ensemble vs Prophet", "folds": 3, "folds_better": 3}]},
        "3": {
            "dm": [
                {"pair": "Chronos-2 (covariates) vs Naive", "folds": 4, "folds_better": 2},
                {"pair": "Chronos-2 (covariates) vs Prophet", "folds": 4, "folds_better": 3},
            ]
        },
        "6": {"dm": [{"pair": "LightGBM vs Naive", "folds": 1, "folds_better": 1}]},
    }
}


def test_reference_wins_count_the_windows_where_the_leader_beats_the_reference() -> None:
    assert reference_wins(WINS_SIGNIFICANCE) == {1: (1, 4), 3: (3, 4), 12: (3, 3)}, (
        "горизонт без сравнения с эталоном в счёт не входит"
    )
    assert reference_wins({}) == {}
    # Формулировка для текста считает те же окна.
    assert "1 месяц — в 1 окне из 4; 3 месяца — в 3 окнах из 4" in reference_windows(
        WINS_SIGNIFICANCE
    )


CONTRAST_WITH_REFERENCE = {
    "detector": "cusum",
    "f1_residual": 0.221,
    "delay_residual": 2.275,
    "reference": {"f1": 0.227, "period": 6, "delay": 2.18},
}
BRIEF_MATCH = {
    "detector": "cusum",
    "period": 8,
    "detector_recall": 0.430,
    "schedule_recall": 0.344,
    "detector_delay": 2.275,
    "schedule_delay": 3.082,
}


def test_schedule_brief_puts_the_detector_against_the_schedule_in_two_sentences() -> None:
    assert schedule_brief(CONTRAST_WITH_REFERENCE, BRIEF_MATCH) == (
        "По F1 детектор эталон не превосходит. При равной частоте ложных тревог он находит "
        "43% сломов против 34% и срабатывает через 2,3 мес. против 3,1."
    )
    better = {**CONTRAST_WITH_REFERENCE, "f1_residual": 0.25}
    assert schedule_brief(better, BRIEF_MATCH).startswith(
        "По F1 детектор выше эталона: 0,250 против 0,227."
    )
    assert schedule_brief(CONTRAST_WITH_REFERENCE, {}) == "По F1 детектор эталон не превосходит."
    assert schedule_brief({"detector": "cusum", "f1_residual": 0.2}, BRIEF_MATCH) == "", (
        "эталона нет"
    )


def test_forecast_bias_statement_says_on_which_side_of_the_forecast_the_fact_lies() -> None:
    assert forecast_bias_statement({"above_forecast_share": 0.704}) == (
        "Прогноз чаще ниже факта: на рядах оценочной выборки факт выше прогноза в 70,4% месяцев."
    )
    assert forecast_bias_statement({"above_forecast_share": 0.30}) == (
        "Прогноз чаще выше факта: на рядах оценочной выборки факт ниже прогноза в 70,0% месяцев."
    )
    assert forecast_bias_statement({"above_forecast_share": 0.55}) == "", (
        "смещения нет — нет и фразы"
    )
    assert forecast_bias_statement({}) == ""


def test_hazard_brief_follows_the_comparison_with_the_detector() -> None:
    lost = {
        "model_at_matched_alarm_rate": {"f1": 0.107},
        "reference_detector": {"f1": 0.172},
    }
    assert hazard_brief(lost) == (
        "Модель вероятности шока в итоговую систему не вошла: при равном числе тревог она "
        "проигрывает простому детектору по остаткам (F1 0,107 против 0,172)."
    )
    won = {"model_at_matched_alarm_rate": {"f1": 0.2}, "reference_detector": {"f1": 0.172}}
    assert hazard_brief(won).startswith(
        "Модель вероятности шока при равном числе тревог не уступает"
    )
    assert hazard_brief({}) == ""


def test_limitation_items_give_every_limitation_a_title_and_an_area() -> None:
    items = limitation_items(
        history="История каждого ряда — 24 месяца: direct-модели построить нельзя.",
        contributions=[
            "Календарь и национальные ряды: устойчивого изменения нет — MAE ниже на 37,8 руб.",
            "Новости и календарь событий поверх них: устойчивого изменения нет — MAE выше.",
            "Foundation-модели в ансамбле: устойчивого изменения нет.",
        ],
        window="Окон проверки мало: на 1 месяц — 4.",
        model_note="Основной LightGBM получает календарь.",
        caveat="Тип МО взят из редакции справочника 2024 года.",
        schedule="Тревога по расписанию даёт F1 0,227.",
    )
    assert [(item["title"], item["area"]) for item in items] == [
        ("Короткая история рядов", "прогноз"),
        ("На уровне МО шоки почти не размечены", "сдвиги"),
        ("Публичным лидербордам не доверяем", "прогноз"),
        ("Календарь и национальные ряды", "данные"),
        ("Новости и календарь событий поверх них", "данные"),
        ("Мало окон проверки", "прогноз"),
        ("Оценка основной модели оптимистична", "прогноз"),
        ("Часть признаков — из более поздних редакций", "данные"),
        ("F1 не отличает метод от расписания", "сдвиги"),
    ]
    assert items[3]["text"].startswith("Календарь и национальные ряды: устойчивого изменения нет")
    assert items[1]["text"].startswith("Реальных размеченных шоков на уровне МО почти нет")
    bare = limitation_items(history="История каждого ряда — 24 месяца.")
    assert [item["title"] for item in bare] == [
        "Короткая история рядов",
        "На уровне МО шоки почти не размечены",
        "Публичным лидербордам не доверяем",
    ], "необязательного ограничения нет — нет и пункта"


def test_model_family_groups_models_the_way_the_page_shows_them() -> None:
    assert model_family("Prophet") == "reference"
    assert model_family("Chronos-2 (covariates)") == "foundation"
    assert model_family("TimesFM-2.5") == "foundation"
    assert model_family("Ensemble (по категориям)") == "ensemble"
    assert {model_family(m) for m in ("LightGBM", "NHITS", "NBEATSx")} == {"ml"}
    assert model_family("Naive") == "naive" and model_family("SeasonalNaive") == "naive"
    assert model_family("AutoTheta") == "stats", "прочие модели — статистические"
    assert model_family("AutoTheta", reference="AutoTheta") == "reference"


def test_fact_above_statement_counts_the_months_where_the_fact_exceeds_the_forecast() -> None:
    fact, forecast = [10.0, 12.0, 9.0, 11.0], [9.0, 11.0, 10.0, 10.5]
    assert fact_above_statement(fact, forecast) == "Факт выше прогноза в 3 месяцах из 4"
    assert fact_above_statement([1.0], [0.5]) == "Факт выше прогноза в 1 месяце из 1"
    gapped = fact_above_statement([10.0, float("nan"), 9.0], [9.0, 11.0, None])
    assert gapped == "Факт выше прогноза в 1 месяце из 1", (
        "месяц без факта или прогноза не считается"
    )
    assert fact_above_statement([], []) == ""


def test_minority_note_names_the_horizons_where_most_windows_favour_the_reference() -> None:
    note = minority_note(SIGNIFICANCE)
    assert note == (
        "На горизонте 1 месяц средний перевес создан меньшинством окон: в остальных точнее Prophet."
    )
    assert reference_windows(SIGNIFICANCE).endswith(note), "та же фраза, что в отчёте"
    assert minority_note({"by_horizon": {"12": SIGNIFICANCE["by_horizon"]["12"]}}) == ""
    assert minority_note({}) == ""


def test_identification_facts_give_the_summary_in_parts() -> None:
    """Блок сверки на странице собран из тех же фраз, что и абзацы отчёта."""
    facts = identification_facts(ENTITY)
    assert facts["panel"] == (
        "2 190 территорий, 13 140 рядов, 77 регионов; название, тип и ОКТМО взяты из редакции "
        "справочника 2024 года."
    )
    assert facts["same_names"] == (
        "49 названий встречаются в нескольких регионах — это 121 территория, и различает их "
        "только код."
    )
    wrong = facts["wrong"]
    assert (wrong["series"], wrong["of"]) == (327, 13140) and round(wrong["share"], 1) == 2.5
    assert wrong["spread"] == (
        "ряды из чужого региона попадали в 101 из 2 188 территорий, 14 территорий были отнесены "
        "к чужому региону целиком."
    )
    summary = " ".join(identification_summary(ENTITY))
    for part in (facts["panel"], facts["same_names"], facts["check"], wrong["spread"]):
        assert part in summary
    exact = {
        **ENTITY,
        "heuristic_check": {**ENTITY["heuristic_check"], "series_in_wrong_region": 0},
    }
    matched = identification_facts(exact)
    assert "wrong" not in matched and "совпала" in matched["matched"]
    bare = identification_facts({k: v for k, v in ENTITY.items() if k != "heuristic_check"})
    assert set(bare) == {"panel", "same_names"}
    assert identification_facts({}) == {}
