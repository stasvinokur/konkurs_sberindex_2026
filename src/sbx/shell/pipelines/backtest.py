"""Прогон моделей по фолдам протокола и сборка метрик."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from sbx.core.fingerprint import changed, fingerprint
from sbx.core.folds import BacktestConfig, Fold, check_coverage, make_folds, split
from sbx.core.forecasting import build_oof, stratified_series_sample, validate_forecast
from sbx.core.metrics import leaderboard, mase_scales, summarize
from sbx.core.seasonality import seasonal_index_as_of
from sbx.shell import stamps
from sbx.shell.io import ARTIFACTS_DIR, CONFIG_DIR, ROOT, load_config, read_yaml
from sbx.shell.models import base as model_base
from sbx.shell.pipelines.features import load_feature_blocks
from sbx.shell.pipelines.panel import (
    load_national_spending,
    load_panel,
    load_seasonal,
    load_static,
    seasonal_lag_days,
)

OOF_DIR = ARTIFACTS_DIR / "oof"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
ENSEMBLE_FILE = "ensemble.parquet"
# Увеличивается, когда меняется сам расчёт прогнозов, а не его входы: по входам кэш этого не
# заметит. 2 — внешние признаки и сезонный индекс берутся по состоянию на момент прогноза.
CACHE_VERSION = 2


def load_backtest_config(path: Path = CONFIG_DIR / "backtest.yaml") -> BacktestConfig:
    return load_config(BacktestConfig, path)


def load_models_config(path: Path = CONFIG_DIR / "models.yaml") -> dict[str, Any]:
    return dict(read_yaml(path))


def evaluation_series(static: pd.DataFrame, models_cfg: Mapping[str, Any]) -> list[str] | None:
    """Ряды, на которых сравниваются модели.

    Если в конфиге задан замороженный список (`evaluation_sample.frozen`), берётся он: это
    выборка опубликованного расчёта, и повторный отбор на другой панели сменил бы ряды, а не
    только метрики. Иначе — стратифицированная подвыборка по `n_series` и `seed`.
    """
    sample_cfg = models_cfg.get("evaluation_sample") or {}
    frozen = sample_cfg.get("frozen")
    if frozen:
        ids = sorted(pd.read_csv(ROOT / frozen, dtype=str)["unique_id"])
        absent = sorted(set(ids) - set(static["unique_id"]))
        if absent:
            raise ValueError(
                f"{len(absent)} рядов замороженной оценочной выборки ({frozen}) нет в панели, "
                f"например {absent[:3]}: список составлен для другой панели"
            )
        return ids
    n = int(sample_cfg.get("n_series", 0))
    if not n:
        return None
    return stratified_series_sample(static, n, int(sample_cfg.get("seed", 0)))


def forecast_inputs(
    group: str,
    panel: pd.DataFrame,
    static: pd.DataFrame,
    cfg: BacktestConfig,
    models_cfg: Mapping[str, Any],
    eval_ids: Sequence[str] | None,
    folds: Sequence[Fold],
    seasonal: pd.DataFrame | None = None,
    national: pd.DataFrame | None = None,
    national_lag_days: int | None = None,
    context_extra: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Отпечаток всего, от чего зависят прогнозы группы.

    Им пользуются и сам расчёт (годится ли кэш), и отчёт (посчитаны ли лежащие на диске
    прогнозы на нынешних входах): строится он в одном месте, чтобы эти две проверки не могли
    разойтись.
    """
    if national is not None:
        if national_lag_days is None:
            raise ValueError("для национального ряда нужен лаг публикации national_lag_days")
        seasonal_input: dict[str, Any] = {
            stamps.NATIONAL: national,
            "seasonal_rule": {"lag_days": int(national_lag_days), "per_fold": True},
        }
    else:
        seasonal_input = {stamps.SEASONAL: seasonal}
    return fingerprint(
        {
            stamps.PANEL: panel,
            stamps.STATIC: static,
            **seasonal_input,
            "eval_ids": sorted(eval_ids) if eval_ids is not None else None,
            "folds": [[f.name, str(f.cutoff), f.horizon, f.kind] for f in folds],
            "protocol": {"min_train_obs": cfg.min_train_obs},
            "config": {
                "group": group,
                "settings": dict((models_cfg.get("groups") or {}).get(group, {})),
                "foundation_models": models_cfg.get("foundation_models", {}),
            },
            "context": dict(context_extra or {}),
            "version": CACHE_VERSION,
        }
    )


def group_context(group: str, models_cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    """Блоки внешних признаков, названные в настройках группы (`feature_blocks`).

    Блоки входят и в контекст модели, и в отпечаток входов её прогнозов: изменилось
    содержимое блока — прогнозы считаются заново. Названного блока нет — ошибка: вариант
    абляции без блока просто пропускается, а основная модель молча стала бы другой моделью.
    Группе без внешних признаков таблицы признаков не читаются вовсе.
    """
    names = list(((models_cfg.get("groups") or {}).get(group) or {}).get("feature_blocks") or [])
    if not names:
        return None
    blocks = load_feature_blocks()
    missing = [name for name in names if name not in blocks]
    if missing:
        raise FileNotFoundError(
            f"группе {group} нужны блоки признаков {missing}, а их нет — выполните make features"
        )
    return {"feature_blocks": {name: blocks[name] for name in names}}


def expected_inputs(groups: Sequence[str]) -> dict[str, dict[str, str]]:
    """Отпечатки входов, с которыми основные группы моделей посчитались бы сейчас.

    Отчёт сравнивает их с отметками рядом с прогнозами: так видно группу, оставшуюся на
    прежнем оценочном списке, конфиге или протоколе, даже если панель у всех общая.
    """
    cfg = load_backtest_config()
    models_cfg = load_models_config()
    panel, static = load_panel(), load_static()
    eval_ids = evaluation_series(static, models_cfg)
    national, lag = load_national_spending(), seasonal_lag_days()
    folds = make_folds(cfg)
    return {
        group: forecast_inputs(
            group,
            panel,
            static,
            cfg,
            models_cfg,
            eval_ids,
            folds,
            national=national,
            national_lag_days=lag,
            context_extra=group_context(group, models_cfg),
        )
        for group in groups
    }


def run_group(
    group: str,
    panel: pd.DataFrame,
    static: pd.DataFrame,
    cfg: BacktestConfig,
    models_cfg: Mapping[str, Any],
    eval_ids: Sequence[str] | None = None,
    folds: Sequence[Fold] | None = None,
    out_dir: Path | None = None,
    overwrite: bool = False,
    seasonal: pd.DataFrame | None = None,
    context_extra: Mapping[str, Any] | None = None,
    cache_name: str | None = None,
    national: pd.DataFrame | None = None,
    national_lag_days: int | None = None,
) -> pd.DataFrame:
    """Считает OOF-прогнозы группы моделей по всем фолдам и кэширует их на диск.

    Обучение идёт на всей панели (глобальные модели учатся на всех рядах), а оценка — только
    на рядах `eval_ids`: так все модели сравниваются на одних и тех же наблюдениях.

    Сезонный индекс. Если передан национальный ряд `national`, индекс оценивается для каждого
    окна заново — только по месяцам, опубликованным к его cutoff (`national_lag_days` — лаг
    публикации ряда). Иначе берётся готовая таблица `seasonal`, одна на все окна.
    """
    out_dir = OOF_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    name = cache_name or group
    path = out_dir / f"{name}.parquet"
    stamp = out_dir / f"{name}_inputs.json"
    group_cfg = dict((models_cfg.get("groups") or {}).get(group, {}))
    folds = list(folds or make_folds(cfg))
    if national is None and seasonal is None:
        seasonal = load_seasonal()
    # Кэш годится только для входов, на которых посчитан. Прогнозы без отметки входов
    # (посчитаны неизвестно на чём) и прогнозы другой панели считаются заново.
    inputs = forecast_inputs(
        group,
        panel,
        static,
        cfg,
        models_cfg,
        eval_ids,
        folds,
        seasonal=seasonal,
        national=national,
        national_lag_days=national_lag_days,
        context_extra=context_extra,
    )
    if path.exists() and not overwrite and not changed(stamps.read(stamp), inputs):
        return pd.read_parquet(path)

    eval_set = set(eval_ids) if eval_ids is not None else None
    context: dict[str, Any] = {
        "seasonal_index": seasonal,
        "static": static,
        "eval_ids": sorted(eval_set) if eval_set else None,
        "foundation_models": models_cfg.get("foundation_models", {}),
        **dict(context_extra or {}),
    }
    frames, timings = [], {}
    for fold in folds:
        train, test = split(panel, fold, cfg.min_train_obs)
        # Прогнозируются только «живые» на момент cutoff ряды: если ряд закончился раньше
        # (порог качества СберИндекса), горизонт прогноза не совпал бы с окном фолда.
        active = train.groupby("unique_id")["ds"].max() == fold.cutoff
        active_ids = set(active.index[active])
        train = train[train["unique_id"].isin(active_ids)]
        test = test[test["unique_id"].isin(active_ids)]
        if eval_set is not None:
            test = test[test["unique_id"].isin(eval_set)]
            context["eval_ids"] = sorted(eval_set & active_ids)
        if test.empty:
            continue
        if national is not None:
            seasonal = seasonal_index_as_of(national, fold.cutoff, int(national_lag_days))
            context["seasonal_index"] = seasonal
        started = time.perf_counter()
        preds = model_base.get(group)(train, fold, group_cfg, context)
        timings[fold.name] = round(time.perf_counter() - started, 1)
        if preds.empty:
            continue
        for _model, part in preds.groupby("model"):
            validate_forecast(part, sorted(train["unique_id"].unique()), fold, allow_missing=True)
        train_with_category = (
            train
            if "category" in train.columns
            else train.merge(static[["unique_id", "category"]], on="unique_id", how="left")
        )
        scales = mase_scales(train_with_category, seasonal)
        oof = build_oof(preds, test, fold, static)
        oof["mase_scale"] = oof["unique_id"].map(scales)
        frames.append(oof)
    result = pd.concat(frames, ignore_index=True)
    result.attrs["timings"] = timings
    result.to_parquet(path, index=False)
    stamps.save(stamp, inputs)
    (out_dir / f"{name}_timings.json").write_text(
        json.dumps(timings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # `_skipped` тоже сохраняется: пропуск фолда моделью — это результат («на этом горизонте
    # модель неопределима»), и он должен быть подтверждён артефактом, а не только текстом.
    extras = {
        k: v for k, v in context.items() if k.endswith(("_tuning", "_importance", "_skipped"))
    }
    if extras:
        (out_dir / f"{name}_context.json").write_text(
            json.dumps(extras, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    return result


def collect_oof(groups: Sequence[str] | None = None, out_dir: Path | None = None) -> pd.DataFrame:
    out_dir = OOF_DIR if out_dir is None else out_dir
    paths = (
        sorted(out_dir.glob("*.parquet"))
        if groups is None
        else [out_dir / f"{g}.parquet" for g in groups]
    )
    frames = [pd.read_parquet(p) for p in paths if p.exists()]
    if not frames:
        raise FileNotFoundError(f"нет OOF-файлов в {out_dir}")
    return pd.concat(frames, ignore_index=True)


def collect_base_oof(out_dir: Path | None = None) -> pd.DataFrame:
    """Прогнозы одиночных моделей — без ансамбля, который из них же и строится."""
    out_dir = OOF_DIR if out_dir is None else out_dir
    groups = sorted(p.stem for p in out_dir.glob("*.parquet") if p.name != ENSEMBLE_FILE)
    return collect_oof(groups, out_dir)


def common_rows(oof: pd.DataFrame) -> pd.DataFrame:
    """Наблюдения, для которых есть прогноз каждой модели — внутри каждого горизонта.

    Часть моделей пропускает ряды (Chronos-2 требует регулярную сетку без пропусков), поэтому
    без выравнивания MAE считалась бы на разных наборах и сравнение было бы нечестным.

    Выравнивание идёт ВНУТРИ горизонта, а не по всему OOF сразу. При h=12 глобальный LightGBM
    и нейросети неопределимы по построению, и общее выравнивание выбросило бы весь горизонт 12
    из сравнения. Внутри горизонта сравниваются те модели, которые на нём определены, а их
    число публикуется отдельно (`coverage.json`).
    """
    keys = ["unique_id", "ds", "fold"]
    if "horizon" not in oof.columns:
        counts = oof.groupby(keys)["model"].nunique()
        full = counts[counts == oof["model"].nunique()].index
        return oof.set_index(keys).loc[full].reset_index()

    parts = []
    for _, group in oof.groupby("horizon", sort=True):
        counts = group.groupby(keys)["model"].nunique()
        full = counts[counts == group["model"].nunique()].index
        parts.append(group.set_index(keys).loc[full].reset_index())
    return pd.concat(parts, ignore_index=True)


def build_metrics(
    cfg: BacktestConfig,
    oof: pd.DataFrame,
    metrics_dir: Path | None = None,
    restrict_common: bool = True,
    oof_dir: Path | None = None,
) -> pd.DataFrame:
    """Метрики по OOF-таблице `oof`, собранной из каталога `oof_dir`."""
    metrics_dir = METRICS_DIR if metrics_dir is None else metrics_dir
    oof_dir = OOF_DIR if oof_dir is None else oof_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = len(oof)
    if restrict_common:
        oof = common_rows(oof)
    summary = summarize(oof, slices=cfg.metric_slices, quantiles=cfg.quantiles)
    board = leaderboard(summary)
    timing_files = sorted(oof_dir.glob("*_timings.json"))
    timings = {}
    for f in timing_files:
        timings[f.stem.replace("_timings", "")] = sum(
            json.loads(f.read_text(encoding="utf-8")).values()
        )
    model_group = {}
    for path in sorted(oof_dir.glob("*.parquet")):
        for model in pd.read_parquet(path, columns=["model"])["model"].unique():
            model_group[model] = path.stem
    board["group"] = board["model"].map(model_group)
    board["group_runtime_sec"] = board["group"].map(timings)
    board["rows_common"] = len(oof) // max(oof["model"].nunique(), 1)
    summary.to_csv(metrics_dir / "summary.csv", index=False)
    board.to_csv(metrics_dir / "leaderboard.csv", index=False)

    # Отдельный leaderboard по горизонтам. Общая строка leaderboard.csv усредняет горизонты с
    # очень разным весом (при h=12 наблюдений на порядок больше, чем при h=1), поэтому как
    # headline она вводит в заблуждение; требование критерия закрывает именно этот файл.
    horizons = summary[summary["slice"] == "horizon"]
    if not horizons.empty:
        columns = [c for c in ("model", "value", "n", "mae", "wape", "mase") if c in horizons]
        (
            horizons[columns]
            .rename(columns={"value": "horizon"})
            .astype({"horizon": int})
            .sort_values(["horizon", "mae"])
            .to_csv(metrics_dir / "leaderboard_by_horizon.csv", index=False)
        )
    (metrics_dir / "coverage.json").write_text(
        json.dumps(
            {
                "rows_all_models": int(len(oof)),
                "rows_before_alignment": int(raw_rows),
                "models": sorted(oof["model"].unique()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    stamps.write(metrics_dir / "inputs.json", [stamps.glob_key(oof_dir, "*.parquet")])
    return board


def run(
    groups: Sequence[str],
    cfg_path: Path = CONFIG_DIR / "backtest.yaml",
    overwrite: bool = False,
    limit_series: int | None = None,
) -> pd.DataFrame:
    cfg = load_backtest_config(cfg_path)
    models_cfg = load_models_config()
    panel, static = load_panel(), load_static()
    # Фолд, чьё тестовое окно выходит за данные, молча обрезается и попадает в отчёт под
    # своим горизонтом — например «h=12», посчитанный по трём месяцам. Падаем явно.
    problems = check_coverage(panel, make_folds(cfg))
    if problems:
        raise ValueError("фолды выходят за пределы данных:\n  " + "\n  ".join(problems))
    eval_ids = evaluation_series(static, models_cfg)
    oof_dir, metrics_dir = OOF_DIR, METRICS_DIR
    if limit_series:
        eval_ids = (eval_ids or sorted(static["unique_id"]))[:limit_series]
        panel = panel[panel["unique_id"].isin(eval_ids)]
        # Пробный прогон пишет в свои каталоги: кэш и метрики полного прогона он не трогает.
        oof_dir = OOF_DIR / f"limit_{limit_series}"
        metrics_dir = METRICS_DIR / f"limit_{limit_series}"
    national, lag = load_national_spending(), seasonal_lag_days()
    for group in groups:
        run_group(
            group,
            panel,
            static,
            cfg,
            models_cfg,
            eval_ids=eval_ids,
            out_dir=oof_dir,
            overwrite=overwrite,
            national=national,
            national_lag_days=lag,
            context_extra=group_context(group, models_cfg),
        )
    return build_metrics(cfg, collect_oof(out_dir=oof_dir), metrics_dir, oof_dir=oof_dir)


def build_ensemble(
    cfg: BacktestConfig,
    models: Sequence[str] | None = None,
    name: str = "Ensemble",
    by_category: bool = True,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Ансамбль с весами по OOF: веса фолда подобраны только на предыдущих фолдах.

    Дополнительно считается вариант с отдельными весами по категориям расходов. Ансамбль
    строится только из прогнозов одиночных моделей каталога `out_dir`: собственный прежний
    результат в выравнивание наблюдений не попадает.
    """
    from sbx.core.ensemble import (
        attach_meta,
        combine,
        combine_quantiles,
        fit_weights_by_group,
        pivot_predictions,
        recenter_quantiles,
        rolling_origin_ensemble,
    )

    out_dir = OOF_DIR if out_dir is None else out_dir
    full = common_rows(collect_base_oof(out_dir))
    all_folds = [f.name for f in make_folds(cfg)]
    frames: list[pd.DataFrame] = []
    weights_dump: dict[str, Any] = {}
    rows: list[pd.DataFrame] = []

    # Ансамбль строится отдельно внутри каждого горизонта из моделей, которые на нём
    # определены. Общий набор кандидатов не годится: при h=12 глобальный LightGBM и нейросети
    # отсутствуют, и требование «все модели» обнулило бы весь горизонт.
    for horizon, oof in sorted(full.groupby("horizon"), key=lambda kv: kv[0]):
        fold_order = [f for f in all_folds if f in set(oof["fold"])]
        candidates = list(models or sorted(oof["model"].unique()))
        candidates = [m for m in candidates if not m.startswith("Ensemble")]
        candidates = [m for m in candidates if oof[oof["model"] == m]["y_hat"].notna().any()]
        if not candidates or not fold_order:
            continue

        # `full` — весь OOF по всем горизонтам: к cutoff короткого фолда длинные горизонты
        # уже дали несколько месяцев прогнозов, и это законная история для подбора весов.
        combined, weights = rolling_origin_ensemble(
            oof, candidates, fold_order, name=name, history_pool=full
        )
        frames.append(combined)
        weights_dump[f"h{horizon}"] = {f: w.round(4).to_dict() for f, w in weights.items()}

        if not (by_category and "category" in oof.columns):
            continue
        rows = []  # накопитель строк только этого горизонта
        per_category: dict[str, Any] = {}
        for fold in fold_order:
            current = oof[oof["fold"] == fold]
            preds, y = pivot_predictions(current, candidates)
            # То же правило, что и в rolling_origin_ensemble: история — месяцы, известные к
            # cutoff этого фолда, из всех горизонтов. Горизонты перекрываются, поэтому
            # «предыдущие фолды своего горизонта» и содержат будущее, и слишком малы.
            cutoff = pd.Timestamp(current["cutoff"].iloc[0])
            history = full[(full["ds"] <= cutoff) & (full["fold"] != fold)]
            keep = [m for m in candidates if m in set(history["model"])]
            history = history[history["model"].isin(keep)] if len(keep) >= 2 else history.iloc[:0]
            if history.empty:
                group_weights = {}
            else:
                group_weights = fit_weights_by_group(history, keep, "category")
                group_weights = {
                    k: v.reindex(candidates).fillna(0.0) for k, v in group_weights.items()
                }
            categories = current.drop_duplicates(["unique_id", "ds", "fold"]).set_index(
                ["unique_id", "ds", "fold"]
            )["category"]
            labels = categories.reindex(preds.index)
            values = pd.Series(index=preds.index, dtype=float)
            quantiles = pd.DataFrame(index=preds.index)
            for category in labels.dropna().unique():
                # Маска, а не labels.groupby(...).groups: там лежат метки MultiIndex,
                # и позиционная выборка по ним развалила бы индекс наблюдений.
                index = preds.index[(labels == category).to_numpy()]
                weight = group_weights.get(
                    str(category), pd.Series(1.0 / len(candidates), index=candidates)
                )
                values.loc[index] = combine(preds.loc[index], weight)
                part = recenter_quantiles(
                    combine_quantiles(current, candidates, weight, index), values.loc[index]
                )
                for column in part.columns:
                    quantiles.loc[index, column] = part[column]
            per_category[fold] = {k: v.round(4).to_dict() for k, v in group_weights.items()}
            rows.append(
                pd.DataFrame(
                    {
                        "unique_id": preds.index.get_level_values("unique_id"),
                        "ds": preds.index.get_level_values("ds"),
                        "fold": fold,
                        "model": f"{name} (по категориям)",
                        "y": y.to_numpy(),
                        "y_hat": values.to_numpy(),
                        **{c: quantiles[c].to_numpy() for c in quantiles.columns},
                    }
                )
            )
        if rows:
            frames.append(attach_meta(pd.concat(rows, ignore_index=True), oof))
        weights_dump.setdefault("by_category", {})[f"h{horizon}"] = per_category

    # `cutoff`, `step` и срезы переносятся из исходного OOF в `attach_meta`: ансамбль
    # подчиняется тому же контракту, что и одиночные модели.
    result = pd.concat(frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_dir / ENSEMBLE_FILE, index=False)
    stamps.write(
        out_dir / "ensemble_inputs.json",
        [stamps.glob_key(out_dir, "*.parquet", exclude=[ENSEMBLE_FILE])],
    )
    (out_dir / "ensemble_context.json").write_text(
        json.dumps(
            {"models": candidates, "weights": weights_dump},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return result


def build_significance(
    oof: pd.DataFrame | None = None, metrics_dir: Path | None = None
) -> dict[str, Any]:
    """Тесты значимости — отдельно для каждого горизонта.

    Критерий конкурса требует сравнения с Prophet на четырёх горизонтах, поэтому единый тест
    по смеси горизонтов его не закрывает: на h=1 и h=12 выигрывают разные модели, и общий
    тест усреднил бы противоположные результаты.
    """
    from sbx.core.significance import fold_consistency, friedman_nemenyi, panel_dm

    metrics_dir = METRICS_DIR if metrics_dir is None else metrics_dir
    oof = common_rows(collect_oof()) if oof is None else oof
    out: dict[str, Any] = {"by_horizon": {}}

    for horizon, group in sorted(oof.groupby("horizon"), key=lambda kv: kv[0]):
        errors = (
            group.assign(ae=(group["y"] - group["y_hat"]).abs())
            .pivot_table(index=["unique_id", "ds", "fold"], columns="model", values="ae")
            .dropna()
        )
        if errors.empty or errors.shape[1] < 2:
            continue
        best = str(errors.mean().idxmin())
        entry: dict[str, Any] = {
            "best_model": best,
            "models": sorted(errors.columns),
            "n_observations": int(len(errors)),
            "dm": [],
        }
        for reference in ("Prophet", "Naive", "SeasonalNaive"):
            if reference not in set(group["model"]) or reference == best:
                continue
            left, right = group[group["model"] == best], group[group["model"] == reference]
            test = panel_dm(left, right)
            entry["dm"].append(
                {
                    "pair": f"{best} vs {reference}",
                    "statistic": round(test.statistic, 2),
                    "p_value": test.p_value,
                    "mean_diff": round(test.mean_diff, 1),
                    # Счёт окон: в скольких окнах проверки лучшая модель точнее эталона.
                    **fold_consistency(left, right).as_dict(),
                }
            )
        ranks = friedman_nemenyi(errors)
        entry["friedman_p"] = ranks.friedman_p_value
        entry["critical_difference"] = round(ranks.critical_difference, 3)
        entry["mean_ranks"] = {k: round(v, 2) for k, v in sorted(ranks.mean_ranks.items())}
        out["by_horizon"][str(horizon)] = entry

    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "significance.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    stamps.write(metrics_dir / "significance_inputs.json", [stamps.glob_key(OOF_DIR, "*.parquet")])
    return out
