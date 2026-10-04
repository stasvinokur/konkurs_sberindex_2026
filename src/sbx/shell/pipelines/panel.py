"""Шаг построения панели, статических признаков, сезонных индексов и EDA."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from sbx.core.features.asof import latest_published_month  # noqa: E402
from sbx.core.folds import BacktestConfig, make_folds  # noqa: E402
from sbx.core.panel import (  # noqa: E402
    CATEGORY_SLUGS,
    additivity,
    build_official_panel,
    static_features,
)
from sbx.core.seasonality import (  # noqa: E402
    SEASONAL_START,
    panel_seasonal_index,
    seasonal_amplitude,
    seasonal_index_as_of,
)
from sbx.shell import stamps  # noqa: E402
from sbx.shell.download import official  # noqa: E402
from sbx.shell.io import CONFIG_DIR, DATA_DIR, ROOT, load_config, publication_lags  # noqa: E402
from sbx.shell.pipelines.entity import PROCESSED  # noqa: E402

FIGURES = ROOT / "reports" / "figures"
# Национальный ряд расходов, по которому оценивается сезонный индекс.
SEASONAL_SOURCE = "consumer-spending"


def load_panel() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "panel.parquet")


def load_static() -> pd.DataFrame:
    """Статические признаки рядов; здесь же название МО (`mo_name`) для подписей.

    Без названия МО подпись «регион · категория» не различает муниципалитеты одного региона,
    и разные ряды выглядят дублями.
    """
    return pd.read_parquet(PROCESSED / "static.parquet")


def load_seasonal() -> pd.DataFrame:
    """Статический сезонный индекс: оценён по данным, опубликованным до начала панели."""
    return pd.read_parquet(PROCESSED / "seasonal_index.parquet")


def load_national_spending() -> pd.DataFrame:
    """Национальный ряд расходов по категориям — источник сезонного индекса."""
    return pd.read_parquet(DATA_DIR / "raw" / SEASONAL_SOURCE / f"{SEASONAL_SOURCE}.parquet")


def seasonal_lag_days() -> int:
    """Лаг публикации национального ряда расходов, дней от конца месяца."""
    return publication_lags()[SEASONAL_SOURCE]


def first_forecast_month() -> pd.Timestamp:
    """Месяц самого раннего окна проверки; его конец — самый ранний момент прогноза."""
    config = load_config(BacktestConfig, CONFIG_DIR / "backtest.yaml")
    return min(fold.cutoff for fold in make_folds(config))


def run(out_dir: Path = PROCESSED, figures: Path = FIGURES) -> dict:
    territories = pd.read_parquet(out_dir / "territories.parquet")
    panel = build_official_panel(official.load_consumption(), territories)
    # Размер территории — признак модели на все окна проверки: он оценивается по месяцам не
    # позже самого раннего момента прогноза, иначе в ранние окна попало бы будущее.
    level_until = first_forecast_month()
    static = static_features(panel, territories, level_until=level_until)
    # Этот индекс читают детекторы сдвигов на всём периоде панели, поэтому он оценивается
    # только по месяцам, опубликованным к концу её первого месяца. В бэктесте прогнозов
    # индекс оценивается для каждого окна заново (`backtest.run_group`).
    nat_idx = seasonal_index_as_of(load_national_spending(), panel["ds"].min(), seasonal_lag_days())
    seasonal_period = [
        SEASONAL_START[:7],
        f"{latest_published_month(panel['ds'].min(), seasonal_lag_days()):%Y-%m}",
    ]
    mo_idx = panel_seasonal_index(panel)
    seasonal = nat_idx.merge(
        mo_idx.rename(columns={"index": "panel_index"}), on=["category", "month"], how="left"
    )
    add = additivity(panel)

    out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_dir / "panel.parquet", index=False)
    static.to_parquet(out_dir / "static.parquet", index=False)
    seasonal.to_parquet(out_dir / "seasonal_index.parquet", index=False)

    summary = {
        "rows": int(len(panel)),
        "series": int(panel["unique_id"].nunique()),
        "territories": int(panel["territory_id"].nunique()),
        "complete_series": int(static["is_complete"].sum()),
        "incomplete_series": int((~static["is_complete"]).sum()),
        "series_with_admin_break": int(static["admin_break_at"].notna().sum()),
        "size_level_until": f"{level_until:%Y-%m}",
        "territories_without_size": int(
            static.loc[static["size_level"].isna(), "territory_id"].nunique()
        ),
        "additivity": add,
        "seasonal_index_period": seasonal_period,
        "seasonal_amplitude_national": seasonal_amplitude(nat_idx).round(3).to_dict(),
        "seasonal_amplitude_panel": seasonal_amplitude(mo_idx).round(3).to_dict(),
        "seasonal_corr_national_vs_panel": float(
            seasonal[["index", "panel_index"]].corr().iloc[0, 1]
        ),
        "level_median_by_category": panel.groupby("category")["y"].median().round(0).to_dict(),
    }
    (out_dir / "panel_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    _figures(panel, static, seasonal, figures, seasonal_period)
    stamps.write(out_dir / "panel_inputs.json", stamp_inputs(out_dir))
    return summary


def stamp_inputs(out_dir: Path) -> list[str]:
    """Входы шага для отметки рядом с результатом (см. `sbx.shell.stamps`)."""
    return [
        stamps.file_key(out_dir / "territories.parquet"),
        stamps.file_key(official.ARCHIVE),
        stamps.file_key(DATA_DIR / "raw" / SEASONAL_SOURCE / f"{SEASONAL_SOURCE}.parquet"),
        stamps.file_key(CONFIG_DIR / "features.yaml"),
        stamps.file_key(CONFIG_DIR / "backtest.yaml"),
    ]


def _figures(
    panel: pd.DataFrame,
    static: pd.DataFrame,
    seasonal: pd.DataFrame,
    figures: Path,
    seasonal_period: list[str],
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    months = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, cat in zip(axes.flat, CATEGORY_SLUGS, strict=True):
        s = seasonal[seasonal["category"] == cat].sort_values("month")
        years = f"{seasonal_period[0][:4]}–{seasonal_period[1][:4]}"
        ax.plot(s["month"], s["index"], marker="o", label=f"национальный ({years})")
        ax.plot(s["month"], s["panel_index"], marker="s", label="медиана МО (2023–2024)")
        ax.axhline(1.0, color="grey", lw=0.8)
        ax.set_title(cat)
        ax.set_xticks(range(1, 13), months, rotation=45)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Сезонный индекс по месяцам")
    fig.tight_layout()
    fig.savefig(figures / "eda_seasonality.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for cat in CATEGORY_SLUGS:
        vals = panel.loc[panel["category"] == cat, "y"]
        ax.hist(
            vals[vals > 0].map(lambda v: __import__("math").log10(v)),
            bins=60,
            histtype="step",
            label=cat,
        )
    ax.set_xlabel("log10(расходы, руб.)")
    ax.set_ylabel("число наблюдений")
    ax.set_title("Распределение уровней расходов по категориям")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "eda_levels.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    counts = static["n_obs"].value_counts().sort_index()
    ax.bar(counts.index, counts.values)
    ax.set_yscale("log")
    ax.set_xlabel("число месяцев в ряду")
    ax.set_ylabel("число рядов (лог. шкала)")
    ax.set_title("Полнота рядов «территория × категория»")
    fig.tight_layout()
    fig.savefig(figures / "eda_completeness.png", dpi=130)
    plt.close(fig)

    total = panel[panel["category"] == "Все категории"]
    med = total.groupby("ds")["y"].median()
    q = total.groupby("ds")["y"].quantile([0.1, 0.9]).unstack()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(q.index, q[0.1], q[0.9], alpha=0.25, label="10–90% МО")
    ax.plot(med.index, med.values, marker="o", label="медиана МО")
    ax.set_title("«Все категории»: динамика по МО")
    ax.set_ylabel("руб.")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "eda_total_dynamics.png", dpi=130)
    plt.close(fig)
