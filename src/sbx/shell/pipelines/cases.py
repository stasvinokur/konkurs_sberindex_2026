"""Разборы реальных примеров: ряд, прогноз, интервал, тревоги и совпавшие события.

Три разбора: два национальных шока на длинных рядах (там эффект бесспорен и виден глазом) и
крупнейший найденный слом в рядах МО. Реальных региональных шоков в данных мало, поэтому
третий разбор честно показывает, что именно находит детектор на уровне МО.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from sbx.core.changepoint.metrics import positions_for_dates, window_coverage  # noqa: E402
from sbx.core.changepoint.residuals import residual_frame  # noqa: E402
from sbx.core.report import DETECTOR_NAMES, short_mo_name  # noqa: E402
from sbx.shell import stamps  # noqa: E402
from sbx.shell.cpd_offline import detect_offline  # noqa: E402
from sbx.shell.io import ARTIFACTS_DIR, CONFIG_DIR, ROOT  # noqa: E402
from sbx.shell.pipelines.backtest import OOF_DIR  # noqa: E402
from sbx.shell.pipelines.cpd import CPD_DIR, best_offline, live_stream  # noqa: E402
from sbx.shell.pipelines.national_cpd import load_known_shocks, load_national_series  # noqa: E402
from sbx.shell.pipelines.panel import load_static  # noqa: E402

FIGURES = ROOT / "reports" / "figures"
CASES_DIR = ARTIFACTS_DIR / "cases"


def national_case(series_name: str, shocks: dict[str, Any], method: str, penalty: float) -> dict:
    """Национальный ряд с найденными сломами и размеченными событиями.

    Метод и штраф приходят из калибровки на бенчмарке: разбор показывает, что этот метод
    находит на настоящем ряду с теми настройками, с которыми он оценён.
    """
    monthly = load_national_series()["monthly"]
    part = monthly[monthly["series"] == series_name].sort_values("ds").reset_index(drop=True)
    values = part["value"].to_numpy(dtype=float)
    found = detect_offline(values, method, penalty)

    events = [e for e in shocks.get("events", []) if "monthly" in e.get("series", [])]
    margin = int(shocks.get("margin_months", 2))
    matched = []
    for event in events:
        pos = positions_for_dates(part["ds"], [pd.Timestamp(event["date"])])
        if not pos:
            continue
        near = [p for p in found if abs(p - pos[0]) <= margin]
        matched.append(
            {
                "event": event["name"],
                "date": event["date"],
                "source": event.get("source"),
                "detected": bool(near),
                "detected_at": str(part["ds"].iloc[near[0]].date()) if near else None,
                "offset_months": (near[0] - pos[0]) if near else None,
            }
        )

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(part["ds"], values, color="#1f4e79", lw=1.6, label=series_name)
    for i, position in enumerate(found):
        ax.axvline(
            part["ds"].iloc[position],
            color="#d62728",
            ls="--",
            lw=1.0,
            label="найденный слом" if i == 0 else None,
        )
    for i, event in enumerate(events):
        ax.axvline(
            pd.Timestamp(event["date"]),
            color="#2ca02c",
            ls=":",
            lw=1.6,
            label="размеченное событие" if i == 0 else None,
        )
    label = DETECTOR_NAMES.get(method, method)
    ax.set_title(f"Национальные расходы «{series_name}»: сломы {label} и известные шоки")
    ax.set_ylabel("млрд руб.")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = FIGURES / f"case_national_{_slug(series_name)}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {
        "series": series_name,
        "figure": path.name,
        "method": method,
        "penalty": penalty,
        "alarms": len(found),
        # Доля ряда в окнах допуска вокруг тревог: с такой вероятностью «найденной» оказалась
        # бы случайная дата.
        "coverage": window_coverage(len(values), found, margin),
        "margin": margin,
        "events": matched,
        # Сам ряд и найденные сломы — для интерактивного графика на лендинге.
        "ds": [d.strftime("%Y-%m") for d in part["ds"]],
        "values": [round(float(v), 2) for v in values],
        "breaks": [part["ds"].iloc[p].strftime("%Y-%m") for p in found],
    }


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower())[:30].strip("_")


def municipal_case(model: str = "Ensemble (по категориям)") -> dict | None:
    """Ряд МО с наибольшим остатком прогноза: что именно детектор считает сломом."""
    path = OOF_DIR / "ensemble.parquet"
    if not path.exists():
        return None
    oof = pd.read_parquet(path)
    if model not in set(oof["model"]):
        model = str(oof["model"].iloc[0])
    # Тот же поток, что видит живой детектор: месяц один раз, короткий горизонт. Иначе
    # перекрывающиеся горизонты дают пилу на графике, а крупнейшим остатком оказывается
    # промах фолда h=12, которого детектор не видит.
    oof = live_stream(oof[oof["model"] == model])
    residuals = residual_frame(oof, model)
    worst = residuals.assign(a=residuals["z"].abs()).sort_values("a", ascending=False).iloc[0]
    uid = str(worst["unique_id"])

    part = oof[(oof["model"] == model) & (oof["unique_id"] == uid)].sort_values("ds")
    static = load_static().set_index("unique_id")
    meta = static.loc[uid] if uid in static.index else None

    alarms_path = CPD_DIR / "live_alarms.json"
    alarm_months: list[str] = []
    if alarms_path.exists():
        payload = json.loads(alarms_path.read_text(encoding="utf-8"))
        positions = payload.get("alarms", {}).get(uid, [])
        months = list(part["ds"])
        alarm_months = [str(months[p].date()) for p in positions if p < len(months)]

    events_cfg = yaml.safe_load((CONFIG_DIR / "events.yaml").read_text(encoding="utf-8"))
    region = str(meta["region_code"]) if meta is not None else None
    nearby = [
        {
            "date": e["date"],
            "kind": e["kind"],
            "description": e["description"],
            "source": e["source"],
        }
        for e in events_cfg["events"]
        if str(e["region_code"]) in {region, "all"}
        and any(str(e["date"])[:7] == str(d)[:7] for d in part["ds"])
    ]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(part["ds"], part["y"], color="#1f4e79", lw=1.8, marker="o", ms=3, label="факт")
    ax.plot(part["ds"], part["y_hat"], color="#ff7f0e", lw=1.6, label="прогноз ансамбля")
    if {"q_0.1", "q_0.9"} <= set(part.columns):
        ax.fill_between(
            part["ds"],
            part["q_0.1"],
            part["q_0.9"],
            color="#ff7f0e",
            alpha=0.15,
            label="интервал 0,1–0,9",
        )
    for i, month in enumerate(alarm_months):
        ax.axvline(
            pd.Timestamp(month),
            color="#d62728",
            ls="--",
            lw=1.2,
            label="тревога детектора" if i == 0 else None,
        )
    mo = "" if meta is None else short_mo_name(meta.get("mo_name"))
    title = (
        uid
        if meta is None
        else f"{mo} ({meta.get('region_name', region)}) · {meta.get('category', '')}"
    )
    ax.set_title(f"Крупнейший остаток прогноза: {title}")
    ax.set_ylabel("руб.")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = FIGURES / "case_municipal.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)

    return {
        "unique_id": uid,
        "region": None if meta is None else str(meta.get("region_name", "")),
        "mo": mo,
        "category": None if meta is None else str(meta.get("category", "")),
        "max_abs_z": float(worst["a"]),
        "worst_month": pd.Timestamp(worst["ds"]).strftime("%Y-%m"),
        "figure": path.name,
        "alarms": alarm_months,
        "events_in_window": nearby,
        "ds": [d.strftime("%Y-%m") for d in part["ds"]],
        "y": [round(float(v), 1) for v in part["y"]],
        "y_hat": [round(float(v), 1) for v in part["y_hat"]],
        "q_lo": [round(float(v), 1) for v in part["q_0.1"]] if "q_0.1" in part else [],
        "q_hi": [round(float(v), 1) for v in part["q_0.9"]] if "q_0.9" in part else [],
    }


def stamp_inputs() -> list[str]:
    """Входы шага для отметки рядом с результатом (см. `sbx.shell.stamps`)."""
    return [
        stamps.STATIC,
        stamps.file_key(OOF_DIR / "ensemble.parquet"),
        stamps.file_key(CPD_DIR / "live_alarms.json"),
        stamps.file_key(CPD_DIR / "detector_comparison.csv"),
    ]


def run(out_dir: Path = CASES_DIR) -> dict[str, Any]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    shocks = load_known_shocks()
    comparison = CPD_DIR / "detector_comparison.csv"
    calibrated = best_offline(pd.read_csv(comparison)) if comparison.exists() else None
    cases: dict[str, Any] = {
        "national": [national_case(name, shocks, *calibrated) for name in ("Всего", "Услуги")]
        if calibrated
        else [],
        "municipal": municipal_case(),
    }
    (out_dir / "cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    stamps.write(out_dir / "inputs.json", stamp_inputs())
    return cases
