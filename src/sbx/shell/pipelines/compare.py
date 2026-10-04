"""Сравнение текущих артефактов с сохранённой копией прежнего прогона.

Нужно для контрольной точки: прежде чем заменить опубликованные числа, видно, что именно
изменилось и чем это объясняется. Прежний прогон лежит в каталоге с той же раскладкой, что и
`artifacts/` (`oof/`, `metrics/`, `ablations/`, `cpd/`); идентификаторы рядов в нём прежние, и
строки сопоставляются через замороженный оценочный список.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from sbx.core.comparison import (
    PASSING,
    align_oof,
    compare_models,
    compare_tables,
    comparison_markdown,
    horizon_leaders,
    self_check,
    split_series,
    tuning_changes,
)
from sbx.core.report import counted, format_p_value
from sbx.shell.io import ARTIFACTS_DIR
from sbx.shell.pipelines import samples

# Модели, которых исправления не касаются: они не берут ни внешних признаков, ни сезонного
# индекса и прогнозируют каждый ряд отдельно, так что порядок рядов на них не влияет.
UNTOUCHED = (
    "Naive",
    "SeasonalNaive",
    "AutoETS",
    "AutoTheta",
    "AutoARIMA",
    "TimesFM-2.5",
    "Chronos-2",
)


def _oof(directory: Path) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in sorted(directory.glob("*.parquet"))]
    if not frames:
        raise FileNotFoundError(f"нет прогнозов в {directory}")
    return pd.concat(frames, ignore_index=True)


def _csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _difference(row: dict[str, Any] | None) -> str:
    if not row:
        return "—"
    diff = f"{float(row['mean_diff']):+.1f}".replace(".", ",").replace("-", "−")
    text = f"{diff} руб., {format_p_value(row.get('p_value'))}"
    # Счёт окон есть только у прогонов, где он посчитан (см. `fold_consistency`).
    if row.get("folds"):
        better = counted(row["folds_better"], "окне", "окнах", "окнах")
        text += f", точнее в {better} из {row['folds']}"
    return text


def _significance(old: dict[str, Any], new: dict[str, Any]) -> list[dict[str, str]]:
    """Сравнение лучшей модели горизонта с Prophet в двух прогонах."""

    def cell(entry: dict[str, Any] | None) -> str:
        if not entry:
            return "—"
        against = [d for d in entry.get("dm", []) if d["pair"].endswith("vs Prophet")]
        return f"{entry.get('best_model')}: {_difference(against[0] if against else None)}"

    old_h, new_h = old.get("by_horizon", {}), new.get("by_horizon", {})
    return [
        {"horizon": f"{h} мес.", "old": cell(old_h.get(h)), "new": cell(new_h.get(h))}
        for h in sorted(set(old_h) | set(new_h), key=int)
    ]


def _pairs(old: dict[str, Any], new: dict[str, Any]) -> list[dict[str, str]]:
    old_dm, new_dm = old.get("diebold_mariano", {}), new.get("diebold_mariano", {})
    return [
        {"pair": pair, "old": _difference(old_dm.get(pair)), "new": _difference(new_dm.get(pair))}
        for pair in sorted(set(old_dm) | set(new_dm))
    ]


def _live(old: dict[str, Any], new: dict[str, Any]) -> list[dict[str, str]]:
    names = {
        "model": "модель",
        "detector": "детектор",
        "series": "рядов",
        "alarms": "тревог",
        "alarms_per_series_year": "тревог на ряд-год",
        "conformal_alarms": "тревог конформного детектора",
    }
    old_s, new_s = old.get("summary") or {}, new.get("summary") or {}
    return [
        {"name": label, "old": str(old_s.get(key, "—")), "new": str(new_s.get(key, "—"))}
        for key, label in names.items()
        if key in old_s or key in new_s
    ]


def _tuning(baseline: Path, current: Path) -> dict[str, list[dict[str, object]]]:
    out = {}
    for label, file, section, field in (
        ("LightGBM", "lgbm_context.json", "lgbm_tuning", "params"),
        ("Prophet", "prophet_context.json", "prophet_tuning", "best"),
    ):
        old = _json(baseline / "oof" / file).get(section, {})
        new = _json(current / "oof" / file).get(section, {})
        if old and new:
            out[label] = tuning_changes(
                {fold: entry.get(field) for fold, entry in old.items()},
                {fold: entry.get(field) for fold, entry in new.items()},
            )
    return out


def run(baseline: Path, out_dir: Path | None = None) -> dict[str, Any]:
    """Сравнивает `artifacts/` с каталогом прежнего прогона и пишет документ сравнения."""
    baseline, current = Path(baseline), ARTIFACTS_DIR
    if not (baseline / "oof").is_dir():
        raise FileNotFoundError(f"нет каталога прежних прогнозов: {baseline / 'oof'}")
    out_dir = current / "comparison" if out_dir is None else out_dir

    old, new = _oof(baseline / "oof"), _oof(current / "oof")
    mapping = samples.load_evaluation()
    aligned = align_oof(old, new, mapping)
    models = compare_models(aligned, old, new)
    split = split_series(mapping)
    check = self_check(aligned, UNTOUCHED, split)
    data: dict[str, Any] = {"models": models, "self_check": check, "split_series": split}

    headline = "leaderboard_by_horizon.csv"
    old_board, new_board = (
        _csv(baseline / "metrics" / headline),
        _csv(current / "metrics" / headline),
    )
    if old_board is not None and new_board is not None:
        data["leaders"] = horizon_leaders(old_board, new_board)
    significance = _significance(
        _json(baseline / "metrics" / "significance.json"),
        _json(current / "metrics" / "significance.json"),
    )
    if significance:
        data["significance"] = significance

    old_abl, new_abl = (_csv(d / "ablations" / "ablations.csv") for d in (baseline, current))
    if old_abl is not None and new_abl is not None:
        data["ablations"] = compare_tables(old_abl, new_abl, keys=["name"], columns=["mae"])
        data["ablation_pairs"] = _pairs(
            _json(baseline / "ablations" / "contributions.json"),
            _json(current / "ablations" / "contributions.json"),
        )

    old_det, new_det = (_csv(d / "cpd" / "detector_comparison.csv") for d in (baseline, current))
    if old_det is not None and new_det is not None:
        data["detectors"] = compare_tables(
            old_det,
            new_det,
            keys=["detector", "kind", "input"],
            columns=["f1", "false_alarms_per_series_year", "mean_delay"],
        ).sort_values("f1_new", ascending=False)

    live = _live(
        _json(baseline / "cpd" / "live_alarms.json"), _json(current / "cpd" / "live_alarms.json")
    )
    if live:
        data["live"] = live
    tuning = _tuning(baseline, current)
    if tuning:
        data["tuning"] = tuning

    out_dir.mkdir(parents=True, exist_ok=True)
    document = out_dir / "comparison.md"
    document.write_text(comparison_markdown(data), encoding="utf-8")
    models.to_csv(out_dir / "models.csv", index=False)
    by_model = models.groupby("model")["identical"].all()
    return {
        "document": str(document),
        "identical_models": sorted(by_model.index[by_model]),
        "changed_models": sorted(by_model.index[~by_model]),
        "self_check": dict(zip(check["model"], check["verdict"], strict=True)),
        "self_check_passed": bool(check["verdict"].isin(PASSING).all()),
    }
