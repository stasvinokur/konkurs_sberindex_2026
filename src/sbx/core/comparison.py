"""Сравнение двух прогонов расчёта: что изменилось и на каких наблюдениях это измерено.

Прежний прогон считался на панели, где территории определялись по названиям, новый — на
официальных кодах. Идентификаторы рядов разные, а наблюдения те же, поэтому сравнение идёт по
общим строкам: прежний идентификатор переводится в официальный, и строки двух прогонов
сопоставляются по ряду, месяцу, окну и модели.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

KEYS = ["unique_id", "ds", "fold", "model"]
# Порог «прогнозы совпали»: расхождение меньше копейки.
IDENTICAL_TOLERANCE = 0.01
# Порог «в пределах погрешности вычислений»: предобученная модель считает в одинарной точности,
# и при другом составе пакета рядов последние знаки расходятся. Пять копеек при прогнозах в
# тысячи рублей.
NUMERIC_TOLERANCE = 0.05
VERDICTS = {
    "identical": "совпали до копейки",
    "split_only": "совпали всюду, кроме расклеенных рядов",
    "numeric": "в пределах погрешности вычислений",
    "changed": "изменились",
    "absent": "нет общих строк",
}
PASSING = ("identical", "split_only", "numeric")


def align_oof(old: pd.DataFrame, new: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Строки двух прогонов, относящиеся к одним и тем же наблюдениям.

    `mapping` — официальный `unique_id` и прежний `legacy_unique_id`. Прежнему ряду-склейке
    соответствует несколько официальных рядов: его строки расходятся по ним по месяцам, потому
    что месяц есть только у одного из них.
    """
    renamed = old.rename(columns={"unique_id": "legacy_unique_id"})
    translated = renamed.merge(mapping[["unique_id", "legacy_unique_id"]], on="legacy_unique_id")
    columns = [*KEYS, "y", "y_hat"]
    aligned = translated[[*columns, "horizon"]].merge(
        new[columns], on=KEYS, suffixes=("_old", "_new")
    )
    differ = ~np.isclose(aligned["y_old"], aligned["y_new"], rtol=0, atol=1e-6)
    if differ.any():
        raise ValueError(
            f"у {int(differ.sum())} общих строк факт в двух прогонах разный: сопоставление "
            "рядов неверно"
        )
    return aligned.drop(columns=["y_new"]).rename(columns={"y_old": "y"})


def split_series(mapping: pd.DataFrame) -> list[str]:
    """Официальные ряды, на которые разошёлся прежний ряд-склейка.

    У таких рядов изменилась сама история: прежде преемник продолжал ряд предшественника, теперь
    у каждого своя. Прогнозы по ним совпадать с прежними не обязаны.
    """
    glued = mapping[mapping.duplicated("legacy_unique_id", keep=False)]
    return sorted(glued["unique_id"])


def self_check(
    aligned: pd.DataFrame,
    untouched: Sequence[str],
    split: Sequence[str] = (),
    numeric_tolerance: float = NUMERIC_TOLERANCE,
) -> pd.DataFrame:
    """Сверка моделей, которых исправления не касаются: сколько строк расходится и где.

    Итог по модели: `identical` — совпали все строки; `split_only` — расходятся только строки
    расклеенных рядов; `numeric` — вне них расхождение в пределах погрешности вычислений;
    `changed` — расходятся по-настоящему; `absent` — общих строк нет.
    """
    diff = (aligned["y_hat_new"] - aligned["y_hat_old"]).abs()
    in_split = aligned["unique_id"].isin(set(split))
    rows = []
    for model in untouched:
        mask = aligned["model"] == model
        differing = mask & (diff > IDENTICAL_TOLERANCE)
        elsewhere = diff[mask & ~in_split]
        worst = float(elsewhere.max()) if len(elsewhere) else float("nan")
        if not mask.any():
            verdict = "absent"
        elif not differing.any():
            verdict = "identical"
        elif not (differing & ~in_split).any():
            verdict = "split_only"
        elif worst <= numeric_tolerance:
            verdict = "numeric"
        else:
            verdict = "changed"
        rows.append(
            {
                "model": model,
                "rows": int(mask.sum()),
                "differing": int(differing.sum()),
                "differing_split": int((differing & in_split).sum()),
                "max_diff_elsewhere": worst,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def compare_models(aligned: pd.DataFrame, old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """MAE каждой модели на каждом горизонте в двух прогонах — на общих строках."""
    work = aligned.assign(
        error_old=(aligned["y"] - aligned["y_hat_old"]).abs(),
        error_new=(aligned["y"] - aligned["y_hat_new"]).abs(),
        diff=(aligned["y_hat_new"] - aligned["y_hat_old"]).abs(),
    )
    table = (
        work.groupby(["model", "horizon"])
        .agg(
            rows_common=("diff", "size"),
            mae_old=("error_old", "mean"),
            mae_new=("error_new", "mean"),
            max_abs_diff=("diff", "max"),
        )
        .reset_index()
    )
    for name, frame in (("rows_old", old), ("rows_new", new)):
        counts = frame.groupby(["model", "horizon"]).size().rename(name).reset_index()
        table = table.merge(counts, on=["model", "horizon"], how="left")
    table["delta"] = table["mae_new"] - table["mae_old"]
    table["identical"] = table["max_abs_diff"] <= IDENTICAL_TOLERANCE
    order = [
        "model",
        "horizon",
        "rows_old",
        "rows_new",
        "rows_common",
        "mae_old",
        "mae_new",
        "delta",
        "max_abs_diff",
        "identical",
    ]
    return table[order].sort_values(["horizon", "mae_new"]).reset_index(drop=True)


def compare_tables(
    old: pd.DataFrame, new: pd.DataFrame, keys: Sequence[str], columns: Sequence[str]
) -> pd.DataFrame:
    """Две таблицы результатов рядом: значение прежнего прогона, нового и разница."""
    keys = list(keys)
    merged = old[[*keys, *columns]].merge(
        new[[*keys, *columns]], on=keys, how="outer", suffixes=("_old", "_new")
    )
    for column in columns:
        merged[f"{column}_delta"] = merged[f"{column}_new"] - merged[f"{column}_old"]
    ordered = [
        c for column in columns for c in (f"{column}_old", f"{column}_new", f"{column}_delta")
    ]
    return merged[[*keys, *ordered]]


REFERENCE_MODEL = "Prophet"


def horizon_leaders(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Лучшая модель каждого горизонта и MAE эталона в двух прогонах.

    На входе — таблицы «модель, горизонт, MAE» из отчёта каждого прогона (у каждого свои
    общие строки внутри горизонта).
    """
    rows = []
    horizons = sorted(set(old["horizon"]) | set(new["horizon"]))
    for horizon in horizons:
        row: dict[str, object] = {"horizon": int(horizon)}
        for label, frame in (("old", old), ("new", new)):
            part = frame[frame["horizon"] == horizon].sort_values("mae")
            reference = part.loc[part["model"] == REFERENCE_MODEL, "mae"]
            row[f"leader_{label}"] = part["model"].iloc[0] if len(part) else None
            row[f"mae_{label}"] = float(part["mae"].iloc[0]) if len(part) else float("nan")
            row[f"reference_{label}"] = float(reference.iloc[0]) if len(reference) else float("nan")
        row["leader_changed"] = row["leader_old"] != row["leader_new"]
        rows.append(row)
    return pd.DataFrame(rows)


def tuning_changes(old: dict[str, object], new: dict[str, object]) -> list[dict[str, object]]:
    """Окна, в которых подбор гиперпараметров выбрал другое; сравниваются только общие окна."""
    return [
        {"fold": fold, "old": old[fold], "new": new[fold]}
        for fold in sorted(set(old) & set(new))
        if old[fold] != new[fold]
    ]


def _number(value: object, digits: int = 1, signed: bool = False) -> str:
    """Число для таблицы: запятая, пробелы между тысячами, настоящий минус."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    number = float(value)  # type: ignore[arg-type]
    text = f"{abs(number):,.{digits}f}".replace(",", " ").replace(".", ",")
    if number < 0 and float(text.replace(" ", "").replace(",", ".")) != 0:
        return f"−{text}"
    return f"+{text}" if signed and number > 0 else text


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def _no_data() -> list[str]:
    return ["_нет данных в одном из прогонов_", ""]


def comparison_markdown(data: dict[str, object]) -> str:
    """Документ сравнения двух прогонов для контрольной точки."""
    lines = [
        "# Сравнение прогонов: опубликованный расчёт и расчёт на исправленной основе",
        "",
        "«Было» — опубликованный расчёт (территории по названиям, внешние признаки по целевому",
        "месяцу, сезонный индекс по всему ряду). «Стало» — расчёт на официальных кодах МО с",
        "признаками, известными на момент прогноза. Общие строки — одни и те же наблюдения.",
        "",
    ]
    models = data.get("models")
    lines += ["## 1. Самопроверка перехода", ""]
    check = data.get("self_check")
    checked: list[str] = []
    if isinstance(check, pd.DataFrame) and len(check):
        checked = check["model"].tolist()
        failed = check.loc[~check["verdict"].isin(PASSING), "model"].tolist()
        lines += [
            "Этих моделей исправления не касаются: они не берут ни внешних признаков, ни сезонного",
            "индекса и не зависят от порядка рядов. На общих строках их прогнозы обязаны совпасть.",
            "",
            (
                f"**Самопроверка не пройдена: {', '.join(failed)}.**"
                if failed
                else "**Самопроверка пройдена.**"
            ),
            "",
        ]
        lines += _table(
            [
                "Модель",
                "Общих строк",
                "Расходятся",
                "Из них у расклеенных рядов",
                "Наибольшее расхождение вне их, руб.",
                "Итог",
            ],
            [
                [
                    str(r.model),
                    _number(r.rows, 0),
                    _number(r.differing, 0),
                    _number(r.differing_split, 0),
                    _number(r.max_diff_elsewhere, 2),
                    VERDICTS[r.verdict],
                ]
                for r in check.itertuples()
            ],
        )
        split = data.get("split_series")
        if split:
            lines += [
                f"Расклеенные ряды: {', '.join(split)}. Прежняя панель склеивала их в один "  # type: ignore[arg-type]
                "ряд, и преемник получал историю предшественника; теперь у каждого своя "
                "история, и прогнозы по ней совпадать с прежними не обязаны.",
                "",
            ]
        lines += [
            f"«До копейки» — расхождение не больше {_number(IDENTICAL_TOLERANCE, 2)} руб.; "
            f"погрешность вычислений — не больше {_number(NUMERIC_TOLERANCE, 2)} руб.: "
            "предобученная модель считает в одинарной точности, и при другом составе пакета "
            "рядов последние знаки расходятся.",
            "",
        ]
    else:
        lines += _no_data()
    if isinstance(models, pd.DataFrame) and len(models):
        moved = models[~models["model"].isin(checked)]
        worst = moved.groupby("model")["max_abs_diff"].max().sort_values(ascending=False)
        lines += [
            "Остальные модели исправления затрагивают — напрямую или через порядок рядов: "
            + (
                "; ".join(
                    f"{m} (наибольшее расхождение {_number(v)} руб.)" for m, v in worst.items()
                )
                or "таких нет"
            )
            + ".",
            "",
            "## 2. MAE на общих строках, руб.: было → стало (разница)",
            "",
        ]
        horizons = sorted(models["horizon"].unique())
        cells = {
            (
                r.model,
                r.horizon,
            ): f"{_number(r.mae_old)} → {_number(r.mae_new)} ({_number(r.delta)})"
            for r in models.itertuples()
        }
        order = models.groupby("model")["mae_new"].mean().sort_values().index
        lines += _table(
            ["Модель", *[f"{h} мес." for h in horizons]],
            [[m, *[cells.get((m, h), "—") for h in horizons]] for m in order],
        )
        rows = models.groupby("horizon")[["rows_old", "rows_new", "rows_common"]].max()
        lines += [
            "Строк на модель (было / стало / общих): "
            + "; ".join(
                f"{h} мес. — {int(r.rows_old)} / {int(r.rows_new)} / {int(r.rows_common)}"
                for h, r in rows.iterrows()
            )
            + ".",
            "",
        ]
    else:
        lines += _no_data() + ["## 2. MAE на общих строках, руб.: было → стало (разница)", ""]
        lines += _no_data()

    lines += ["## 3. Лидеры по горизонтам (как в отчёте каждого прогона)", ""]
    leaders = data.get("leaders")
    if isinstance(leaders, pd.DataFrame) and len(leaders):
        lines += _table(
            ["Горизонт", "Было", "Стало", "Prophet: было → стало"],
            [
                [
                    f"{int(r.horizon)} мес.",
                    f"{r.leader_old} ({_number(r.mae_old)})",
                    f"{r.leader_new} ({_number(r.mae_new)})",
                    f"{_number(r.reference_old)} → {_number(r.reference_new)}",
                ]
                for r in leaders.itertuples()
            ],
        )
    else:
        lines += _no_data()

    lines += ["## 4. Значимость против Prophet", ""]
    significance = data.get("significance")
    if significance:
        lines += _table(
            ["Горизонт", "Было: модель, ΔMAE, p", "Стало: модель, ΔMAE, p"],
            [[str(r["horizon"]), str(r["old"]), str(r["new"])] for r in significance],  # type: ignore[union-attr]
        )
        lines += [
            "p — тест по рядам: в нём ряды считаются независимыми, хотя все прогнозы окна "
            "строит одна модель. Поэтому в новом прогоне рядом стоит счёт окон проверки: в "
            "скольких из них лучшая модель точнее Prophet.",
            "",
        ]
    else:
        lines += _no_data()

    lines += ["## 5. Абляции: MAE, руб.", ""]
    ablations = data.get("ablations")
    if isinstance(ablations, pd.DataFrame) and len(ablations):
        lines += _table(
            ["Конфигурация", "Было", "Стало", "Разница"],
            [
                [str(r.name), _number(r.mae_old), _number(r.mae_new), _number(r.mae_delta)]
                for r in ablations.itertuples(index=False)
            ],
        )
        pairs = data.get("ablation_pairs")
        if pairs:
            lines += _table(
                ["Пара", "Было: ΔMAE, p", "Стало: ΔMAE, p"],
                [[str(r["pair"]), str(r["old"]), str(r["new"])] for r in pairs],  # type: ignore[union-attr]
            )
    else:
        lines += _no_data()

    lines += ["## 6. Детекторы сдвигов: F1 (ложных тревог на ряд-год)", ""]
    detectors = data.get("detectors")
    if isinstance(detectors, pd.DataFrame) and len(detectors):
        lines += _table(
            ["Детектор", "Вход", "Было", "Стало"],
            [
                [
                    str(r.detector),
                    str(r.input),
                    f"{_number(r.f1_old, 3)} ({_number(r.false_alarms_per_series_year_old, 2)})",
                    f"{_number(r.f1_new, 3)} ({_number(r.false_alarms_per_series_year_new, 2)})",
                ]
                for r in detectors.itertuples(index=False)
            ],
        )
    else:
        lines += _no_data()

    lines += ["## 7. Живые тревоги по остаткам ансамбля", ""]
    live = data.get("live")
    if live:
        lines += _table(
            ["Показатель", "Было", "Стало"],
            [[str(r["name"]), str(r["old"]), str(r["new"])] for r in live],  # type: ignore[union-attr]
        )
    else:
        lines += _no_data()

    lines += ["## 8. Подобранные гиперпараметры", ""]
    tuning = data.get("tuning")
    if tuning:
        for name, changes in tuning.items():  # type: ignore[union-attr]
            lines.append(f"**{name}:** выбор изменился в {len(changes)} окнах.")
            lines += [f"- {c['fold']}: {c['old']} → {c['new']}" for c in changes]
            lines.append("")
    else:
        lines += _no_data()
    return "\n".join(lines)
