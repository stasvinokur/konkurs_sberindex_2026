"""Матрица абляций A–F: вклад макро, новостей, foundation-моделей и детектора по остаткам."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from sbx.core.ablation import (
    AblationResult,
    ablation_table,
    contribution,
    on_common_rows,
    significance_note,
    source_contribution,
)
from sbx.core.ensemble import rolling_origin_ensemble
from sbx.core.folds import make_folds
from sbx.core.metrics import summarize
from sbx.core.significance import fold_consistency, panel_dm
from sbx.shell import stamps
from sbx.shell.io import ARTIFACTS_DIR, CONFIG_DIR, read_yaml
from sbx.shell.pipelines.backtest import (
    METRICS_DIR,
    OOF_DIR,
    collect_oof,
    evaluation_series,
    forecast_inputs,
    load_backtest_config,
    load_models_config,
    run_group,
)
from sbx.shell.pipelines.cpd import CPD_DIR
from sbx.shell.pipelines.features import load_feature_blocks
from sbx.shell.pipelines.hazard import HAZARD_DIR
from sbx.shell.pipelines.panel import (
    load_national_spending,
    load_panel,
    load_static,
    seasonal_lag_days,
)

ABLATION_DIR = ARTIFACTS_DIR / "ablations"
# OOF абляционных вариантов лежат отдельно от основных: иначе они попали бы в общий
# leaderboard и в кандидаты итогового ансамбля, и headline-метрика зависела бы от того,
# какие абляции сейчас закэшированы на диске.
ABLATION_OOF_DIR = OOF_DIR / "ablations"


def load_ablation_config(path: Path = CONFIG_DIR / "ablations.yaml") -> dict[str, Any]:
    return dict(read_yaml(path))


def _metrics_for(oof: pd.DataFrame, model: str) -> dict[str, float]:
    part = oof[oof["model"] == model]
    if part.empty:
        return {}
    summary = summarize(part, slices=())
    row = summary.iloc[0]
    return {"mae": float(row["mae"]), "r2_per_series_median": float(row["r2_per_series_median"])}


def source_pairs(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Пары «блок источника, вариант-основа», по которым измеряется вклад каждого источника.

    Блок добавляется к базовому варианту и к лучшему прежнему набору (`sources.base` и
    `sources.on_top_of`); поверх варианта, в который он уже входит, не считается.
    """
    spec = cfg.get("sources") or {}
    variants = cfg.get("feature_variants") or {}
    bases = [k for k in dict.fromkeys((spec.get("base"), spec.get("on_top_of"))) if k in variants]
    pairs = []
    for block, entry in (spec.get("blocks") or {}).items():
        # Запись — название источника или словарь с названием и блоками, из которых он состоит.
        described = isinstance(entry, Mapping)
        title = entry["title"] if described else entry
        own = [str(b) for b in entry.get("blocks", [block])] if described else [str(block)]
        for against in bases:
            base_blocks = list(variants[against].get("blocks", []))
            if set(own) & set(base_blocks):
                continue
            pairs.append(
                {
                    "block": str(block),
                    "title": str(title),
                    "against": str(against),
                    "cache": f"lgbm_src_{against}_{block}",
                    "model": f"LightGBM ({against} + {block})",
                    "base_model": str(variants[against]["model_name"]),
                    "blocks": [*base_blocks, *own],
                }
            )
    return pairs


def _variants(
    cfg: Mapping[str, Any], models_cfg: Mapping[str, Any], blocks: Mapping[str, Any]
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Варианты абляции: имя кэша, конфиг моделей для прогона и блоки признаков варианта.

    Варианту передаются только его блоки: они входят в отпечаток входов кэша, и прогнозы
    считаются заново, когда содержимое блока меняется — например, когда абляции посчитаны без
    архива новостей, а потом архив скачан.

    После вариантов матрицы идут варианты источников (`source_pairs`). Вариант источника
    считается, только когда есть все его блоки: без блока источника он был бы копией основы и
    «измерил» бы нулевой вклад.
    """
    specs = [
        (f"lgbm_{name}", list(spec.get("blocks", [])), spec["model_name"], bool(spec.get("tune")))
        for name, spec in cfg["feature_variants"].items()
    ]
    specs += [
        (pair["cache"], pair["blocks"], pair["model"], False)
        for pair in source_pairs(cfg)
        if set(pair["blocks"]) <= set(blocks)
    ]
    variants = []
    for cache, block_names, model_name, tune in specs:
        group_cfg = dict((models_cfg.get("groups") or {}).get("lgbm", {}))
        group_cfg["feature_blocks"] = block_names
        group_cfg["model_name"] = model_name
        group_cfg["tune"] = tune
        models_for_run = {
            **models_cfg,
            "groups": {**models_cfg.get("groups", {}), "lgbm": group_cfg},
        }
        used = {block: blocks[block] for block in block_names if block in blocks}
        variants.append((cache, models_for_run, used))
    return variants


def expected_inputs(cfg: Mapping[str, Any] | None = None) -> dict[str, dict[str, str]]:
    """Отпечатки входов, с которыми варианты абляции посчитались бы сейчас (для отчёта)."""
    cfg = load_ablation_config() if cfg is None else cfg
    backtest_cfg = load_backtest_config()
    models_cfg = load_models_config()
    panel, static = load_panel(), load_static()
    eval_ids = evaluation_series(static, models_cfg)
    national, lag = load_national_spending(), seasonal_lag_days()
    folds = make_folds(backtest_cfg)
    return {
        name: forecast_inputs(
            "lgbm",
            panel,
            static,
            backtest_cfg,
            models_for_run,
            eval_ids,
            folds,
            national=national,
            national_lag_days=lag,
            context_extra={"feature_blocks": used},
        )
        for name, models_for_run, used in _variants(cfg, models_cfg, load_feature_blocks())
    }


def run_feature_ablations(cfg: Mapping[str, Any], overwrite: bool = False) -> pd.DataFrame:
    """Прогоняет LightGBM с разными наборами внешних признаков (конфигурации A, B, C)."""
    backtest_cfg = load_backtest_config()
    models_cfg = load_models_config()
    panel, static = load_panel(), load_static()
    eval_ids = evaluation_series(static, models_cfg)
    national, lag = load_national_spending(), seasonal_lag_days()
    frames = [
        run_group(
            "lgbm",
            panel,
            static,
            backtest_cfg,
            models_for_run,
            eval_ids=eval_ids,
            out_dir=ABLATION_OOF_DIR,
            overwrite=overwrite,
            context_extra={"feature_blocks": used},
            cache_name=name,
            national=national,
            national_lag_days=lag,
        )
        for name, models_for_run, used in _variants(cfg, models_cfg, load_feature_blocks())
    ]
    return pd.concat(frames, ignore_index=True)


def stamp_inputs(cfg_path: Path) -> list[str]:
    """Входы шага для отметки рядом с результатом (см. `sbx.shell.stamps`)."""
    return [
        stamps.file_key(cfg_path),
        stamps.glob_key(OOF_DIR, "*.parquet"),
        stamps.glob_key(ABLATION_OOF_DIR, "*.parquet"),
        stamps.file_key(CPD_DIR / "detector_comparison.csv"),
        stamps.file_key(HAZARD_DIR / "hazard_results.json"),
    ]


def run(
    cfg_path: Path = CONFIG_DIR / "ablations.yaml", out_dir: Path = ABLATION_DIR
) -> pd.DataFrame:
    """Шаг расчёта: матрица абляций, вклад источников и отметка входов рядом с результатом."""
    cfg = load_ablation_config(cfg_path)
    table = build_table(cfg, out_dir)
    build_sources(cfg, out_dir)
    stamps.write(out_dir / "inputs.json", stamp_inputs(cfg_path))
    return table


SOURCE_COLUMNS = [
    "block",
    "title",
    "against",
    "rows",
    "mae_with",
    "mae_base",
    "delta",
    "mean_diff",
    "statistic",
    "p_value",
    "folds",
    "folds_better",
    "folds_worse",
    "sign_p",
]


def build_sources(
    cfg: Mapping[str, Any], out_dir: Path, blocks: Collection[str] | None = None
) -> pd.DataFrame:
    """Вклад каждого источника: модель с его блоком против варианта, к которому он добавлен.

    Строка есть только у пар, для которых посчитаны оба прогноза и есть данные сейчас
    (`blocks`, по умолчанию — построенные блоки признаков): источник, которого в прогоне не
    было (нет архива новостей, таблица не построена), в таблицу не попадает, даже если на
    диске остались его прогнозы от прошлого прогона.
    """
    available = set(load_feature_blocks() if blocks is None else blocks)
    oof = collect_oof(out_dir=ABLATION_OOF_DIR)
    models = set(oof["model"])
    rows = []
    for pair in source_pairs(cfg):
        if not set(pair["blocks"]) <= available:
            continue
        if pair["model"] not in models or pair["base_model"] not in models:
            continue
        measured = source_contribution(
            oof[oof["model"] == pair["model"]], oof[oof["model"] == pair["base_model"]]
        )
        rows.append({key: pair[key] for key in ("block", "title", "against")} | measured)
    table = pd.DataFrame(rows) if rows else pd.DataFrame(columns=SOURCE_COLUMNS)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "sources.csv", index=False)
    return table


def build_table(cfg: Mapping[str, Any], out_dir: Path) -> pd.DataFrame:
    """Матрица абляций A–F и тесты значимости по парам конфигураций.

    Все конфигурации измеряются на общих наблюдениях. У ансамбля строк меньше, чем у одиночной
    модели (ему нужен прогноз каждой его модели), и на собственных строках каждой конфигурации
    разность MAE сравнивала бы разные наборы рядов.
    """
    backtest_cfg = load_backtest_config()
    fold_order = [f.name for f in make_folds(backtest_cfg) if f.kind == "main"]
    oof = pd.concat([collect_oof(), collect_oof(out_dir=ABLATION_OOF_DIR)], ignore_index=True)
    extra: dict[str, Any] = {}
    # Прогнозы каждой конфигурации — и одиночные модели, и ансамбли — нужны для теста DM:
    # ансамбля нет в исходном OOF, и без этого сравнение «всё против базы» молча пропускалось.
    predictions: dict[str, pd.DataFrame] = {}
    members: dict[str, list[str]] = {}

    for name, spec in cfg["configurations"].items():
        models = list(spec.get("models", []))
        available = [m for m in models if m in set(oof["model"])]
        if not available:
            continue
        if len(available) == 1:
            label = available[0]
            predictions[name] = oof[oof["model"] == available[0]]
        else:
            label = f"Ensemble {name}"
            combined, weights = rolling_origin_ensemble(
                oof[oof["fold"].isin(fold_order)], available, fold_order, name=label
            )
            predictions[name] = combined
            extra[f"weights_{name}"] = {k: v.round(3).to_dict() for k, v in weights.items()}
        members[name] = available
        extra[f"label_{name}"] = label

    predictions = on_common_rows(predictions)
    results = []
    for name, frame in predictions.items():
        metrics: dict[str, Any] = _metrics_for(frame, extra[f"label_{name}"])
        metrics["rows"] = int(len(frame))
        metrics["models"] = ", ".join(members[name])
        results.append(AblationResult(name, metrics))
    extra["rows_common"] = int(len(next(iter(predictions.values())))) if predictions else 0

    cpd_metrics = _cpd_metrics(cfg)
    for result in results:
        result.metrics.update(cpd_metrics.get(result.name, {}))

    table = ablation_table(results, baseline=cfg.get("baseline", "A"))
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "ablations.csv", index=False)

    contributions = {
        "news_mae": contribution(table, "C", "B", "mae"),
        "foundation_mae": contribution(table, "D", "B", "mae"),
        "all_vs_baseline_mae": contribution(table, "E", "A", "mae"),
    }
    dm = {}
    for left_name, right_name in cfg.get("dm_pairs", []):
        left, right = predictions.get(left_name), predictions.get(right_name)
        if left is None or right is None:
            continue
        test = panel_dm(left, right)
        # mean_diff < 0 означает, что левая конфигурация точнее правой.
        dm[f"{left_name}_vs_{right_name}"] = {
            "left": extra.get(f"label_{left_name}"),
            "right": extra.get(f"label_{right_name}"),
            "statistic": test.statistic,
            "p_value": test.p_value,
            "mean_diff": test.mean_diff,
            "verdict": significance_note(test.p_value),
            # Счёт окон: тест по рядам не видит, что знак разности меняется от окна к окну.
            **fold_consistency(left, right).as_dict(),
        }
    (out_dir / "contributions.json").write_text(
        json.dumps(
            {"contributions": contributions, "diebold_mariano": dm, **extra},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    return table


def _cpd_metrics(cfg: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """CPD-метрики конфигураций: офлайн-метод для A–E, детектор по остаткам и hazard — для F."""
    out: dict[str, dict[str, float]] = {}
    comparison = CPD_DIR / "detector_comparison.csv"
    if not comparison.exists():
        return out
    table = pd.read_csv(comparison)
    offline = table[table["kind"] == "offline"].sort_values("f1", ascending=False)
    residual = table[(table["kind"] == "online") & (table["input"] == "residual")].sort_values(
        "f1", ascending=False
    )
    base = offline.iloc[0] if len(offline) else None
    best_residual = residual.iloc[0] if len(residual) else None
    for name in cfg["configurations"]:
        if name != "F" and base is not None:
            out[name] = {
                "f1": float(base["f1"]),
                "false_alarms_per_series_year": float(base["false_alarms_per_series_year"]),
                "mean_delay": float(base["mean_delay"]),
            }
        elif name == "F" and best_residual is not None:
            metrics = {
                "f1": float(best_residual["f1"]),
                "false_alarms_per_series_year": float(
                    best_residual["false_alarms_per_series_year"]
                ),
                "mean_delay": float(best_residual["mean_delay"]),
            }
            hazard_path = HAZARD_DIR / "hazard_results.json"
            if hazard_path.exists():
                hazard = json.loads(hazard_path.read_text(encoding="utf-8"))
                metrics["mean_lead_time"] = float(hazard.get("mean_lead_time", float("nan")))
            out[name] = metrics
    return out


def contributions_summary(path: Path = ABLATION_DIR / "contributions.json") -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def available_models(oof_dir: Path = ABLATION_OOF_DIR) -> Sequence[str]:
    return sorted(collect_oof(out_dir=oof_dir)["model"].unique())
