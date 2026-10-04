"""Проверка офлайн-методов CPD на реальных национальных рядах СберИндекса.

Полусинтетический бенчмарк даёт честную, но искусственную разметку. Здесь методы
проверяются на реальных шоках, про которые заранее известно, что они были: COVID-19, ключевая
ставка 20%, частичная мобилизация. Это проверка на вменяемость, а не основная метрика —
разметка реальных шоков субъективна (известна дата события, но не месяц «слома» ряда), поэтому
окно допуска шире, а штрафы методов берутся с калибровки на синтетике и здесь не подбираются:
подбор порога на тех же пяти событиях, по которым идёт оценка, ничего бы не доказывал.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from sbx.core.changepoint.metrics import f1_margin, positions_for_dates
from sbx.shell.cpd_offline import OFFLINE_METHODS, detect_offline
from sbx.shell.io import CONFIG_DIR, DATA_DIR, read_yaml
from sbx.shell.pipelines.cpd import CPD_DIR

MONTHLY_SLUG = "consumer-spending"
WEEKLY_SLUG = "ver-izmenenie-trat-po-kategoriyam"


def load_known_shocks(path: Path = CONFIG_DIR / "known_shocks.yaml") -> dict[str, Any]:
    return dict(read_yaml(path))


def load_national_series() -> dict[str, pd.DataFrame]:
    """Длинные национальные ряды: месячный с 2018-12 и недельный с 2023-11."""
    monthly = pd.read_parquet(DATA_DIR / "raw" / MONTHLY_SLUG / f"{MONTHLY_SLUG}.parquet")
    monthly = monthly.rename(columns={"period": "ds", "type": "series"})
    monthly["ds"] = pd.to_datetime(monthly["ds"])
    monthly = monthly[["series", "ds", "value"]].sort_values(["series", "ds"])

    weekly = pd.read_parquet(DATA_DIR / "raw" / WEEKLY_SLUG / f"{WEEKLY_SLUG}.parquet")
    weekly = weekly.rename(columns={"period": "ds", "category": "series"})
    weekly["ds"] = pd.to_datetime(weekly["ds"])
    weekly["series"] = weekly["series"].str.strip()
    weekly = weekly[["series", "ds", "value"]].sort_values(["series", "ds"])
    return {"monthly": monthly, "weekly": weekly}


def _penalties_from_benchmark(path: Path = CPD_DIR / "detector_comparison.csv") -> dict[str, float]:
    """Штрафы, подобранные на калибровочной части синтетики; здесь они не пересчитываются."""
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    offline = table[table["kind"] == "offline"]
    return {str(r["detector"]): float(r["param"]) for _, r in offline.iterrows()}


def evaluate(
    shocks: Mapping[str, Any],
    methods: Sequence[str] = OFFLINE_METHODS,
    out_dir: Path = CPD_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Считает F1 методов на национальных рядах и таблицу «событие × метод»."""
    data = load_national_series()
    penalties = _penalties_from_benchmark()
    events = list(shocks.get("events", []))
    margins = {
        "monthly": int(shocks.get("margin_months", 2)),
        "weekly": int(shocks.get("margin_weeks", 3)),
    }

    rows, hits = [], []
    for freq, frame in data.items():
        dates = [pd.Timestamp(e["date"]) for e in events if freq in e.get("series", [])]
        if not dates:
            continue
        margin = margins[freq]
        groups = {name: g.sort_values("ds") for name, g in frame.groupby("series")}
        for method in methods:
            penalty = penalties.get(method)
            if penalty is None:
                continue
            true_by_series, pred_by_series, elapsed = {}, {}, 0.0
            for name, g in groups.items():
                truth = positions_for_dates(g["ds"], dates)
                if not truth:
                    continue
                started = time.perf_counter()
                found = detect_offline(g["value"].to_numpy(dtype=float), method, penalty)
                elapsed += time.perf_counter() - started
                true_by_series[name] = truth
                pred_by_series[name] = found
                for event in events:
                    if freq not in event.get("series", []):
                        continue
                    pos = positions_for_dates(g["ds"], [pd.Timestamp(event["date"])])
                    if not pos:
                        continue
                    detected = any(abs(p - pos[0]) <= margin for p in found)
                    hits.append(
                        {
                            "freq": freq,
                            "series": name,
                            "event": event["name"],
                            "date": event["date"],
                            "method": method,
                            "detected": bool(detected),
                        }
                    )
            if not true_by_series:
                continue
            score = f1_margin(true_by_series, pred_by_series, margin=margin)
            rows.append(
                {
                    "freq": freq,
                    "method": method,
                    "penalty": penalty,
                    "n_series": len(true_by_series),
                    "precision": score.precision,
                    "recall": score.recall,
                    "f1": score.f1,
                    "n_true": score.n_true,
                    "n_alarms": score.n_pred,
                    "runtime_sec": round(elapsed, 2),
                }
            )

    columns = ["freq", "method", "penalty", "n_series", "precision", "recall", "f1", "runtime_sec"]
    comparison = pd.DataFrame(rows, columns=None if rows else columns)
    if rows:
        comparison = comparison.sort_values(["freq", "f1"], ascending=[True, False])
    per_event = pd.DataFrame(hits)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_dir / "national_comparison.csv", index=False)
    if not per_event.empty:
        summary = (
            per_event.groupby(["freq", "event", "date", "method"])["detected"]
            .mean()
            .reset_index()
            .rename(columns={"detected": "share_series_detected"})
        )
        summary.to_csv(out_dir / "national_events.csv", index=False)
    return comparison, per_event


def run(cfg_path: Path = CONFIG_DIR / "known_shocks.yaml", out_dir: Path = CPD_DIR) -> pd.DataFrame:
    shocks = load_known_shocks(cfg_path)
    comparison, _ = evaluate(shocks, out_dir=out_dir)
    (out_dir / "national_context.json").write_text(
        json.dumps(
            {
                "events": [
                    {k: str(e[k]) for k in ("date", "name", "source", "source_name") if k in e}
                    for e in shocks.get("events", [])
                ],
                "margin_months": shocks.get("margin_months"),
                "margin_weeks": shocks.get("margin_weeks"),
                "note": (
                    "Штрафы взяты с калибровки на синтетическом бенчмарке и здесь не "
                    "подбирались: подбор на тех же событиях, по которым идёт оценка, "
                    "ничего бы не доказывал."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return comparison
