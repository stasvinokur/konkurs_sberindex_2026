"""Отчёт собирается только из артефактов, посчитанных на текущих входах.

После смены панели прежние прогнозы, метрики и детекторы остаются на диске. Если отчёт собрать
из них, получится смесь: числа одной панели, подписи другой.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sbx.core.fingerprint import frame_digest
from sbx.shell import stamps
from sbx.shell.pipelines import ablations, cases, cpd, hazard, report

PANEL = pd.DataFrame({"unique_id": ["a", "b"], "y": [1.0, 2.0]})
STATIC = pd.DataFrame({"unique_id": ["a", "b"], "region_code": ["01", "02"]})
SEASONAL = pd.DataFrame({"month": [1, 2], "index": [0.9, 1.1]})


class Tree:
    """Раскладка проекта во временном каталоге с согласованными артефактами."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.processed = root / "data" / "processed"
        self.oof = root / "artifacts" / "oof"
        self.metrics = root / "artifacts" / "metrics"
        self.cpd = root / "artifacts" / "cpd"
        for directory in (self.processed, self.oof, self.metrics, self.cpd):
            directory.mkdir(parents=True)

    def write_tables(self, panel: pd.DataFrame = PANEL) -> None:
        panel.to_parquet(self.processed / "panel.parquet", index=False)
        STATIC.to_parquet(self.processed / "static.parquet", index=False)
        SEASONAL.to_parquet(self.processed / "seasonal_index.parquet", index=False)

    def write_forecasts(self, group: str = "stats", payload: bytes = b"forecasts") -> None:
        (self.oof / f"{group}.parquet").write_bytes(payload)
        stamps.save(
            self.oof / f"{group}_inputs.json",
            {
                stamps.PANEL: frame_digest(PANEL),
                stamps.STATIC: frame_digest(STATIC),
                stamps.SEASONAL: frame_digest(SEASONAL),
                "config": "abc",
            },
        )

    def write_ensemble(self) -> None:
        (self.oof / "ensemble.parquet").write_bytes(b"ensemble")
        stamps.write(
            self.oof / "ensemble_inputs.json",
            [stamps.glob_key(self.oof, "*.parquet", exclude=["ensemble.parquet"])],
        )

    def write_metrics(self) -> None:
        (self.metrics / "leaderboard.csv").write_text("model,mae\n", encoding="utf-8")
        stamps.write(self.metrics / "inputs.json", [stamps.glob_key(self.oof, "*.parquet")])


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Tree:
    tree = Tree(tmp_path)
    monkeypatch.setattr(stamps, "ROOT", tmp_path)
    monkeypatch.setattr(stamps, "PROCESSED", tree.processed)
    monkeypatch.setattr(report, "OOF_DIR", tree.oof)
    monkeypatch.setattr(report, "METRICS_DIR", tree.metrics)
    monkeypatch.setattr(report, "CPD_DIR", tree.cpd)
    monkeypatch.setattr(report, "HAZARD_DIR", tmp_path / "artifacts" / "hazard")
    monkeypatch.setattr(report, "ABLATION_DIR", tmp_path / "artifacts" / "ablations")
    monkeypatch.setattr(report, "CASES_DIR", tmp_path / "artifacts" / "cases")
    monkeypatch.setattr(report, "ABLATION_OOF_DIR", tree.oof / "ablations")
    # Таблицы этого дерева лежат в data/processed, а шаг панели здесь не выполнялся: каталог
    # результатов панели для отчёта отдельный, чтобы проверялось только то, что под тестом.
    monkeypatch.setattr(report, "PROCESSED", tmp_path / "artifacts" / "panel-step")
    monkeypatch.setattr(report, "FEATURES_DIR", tmp_path / "artifacts" / "features")

    def expected(groups: list[str]) -> dict[str, dict[str, str]]:
        """Отпечаток, с которым группа посчиталась бы на таблицах, лежащих на диске сейчас."""
        tables = {
            stamps.PANEL: frame_digest(pd.read_parquet(tree.processed / "panel.parquet")),
            stamps.STATIC: frame_digest(pd.read_parquet(tree.processed / "static.parquet")),
            stamps.SEASONAL: frame_digest(
                pd.read_parquet(tree.processed / "seasonal_index.parquet")
            ),
        }
        return {group: {**tables, "config": "abc"} for group in groups}

    monkeypatch.setattr(report.backtest_step, "expected_inputs", expected)
    tree.write_tables()
    tree.write_forecasts()
    tree.write_ensemble()
    tree.write_metrics()
    return tree


def test_consistent_artifacts_raise_no_objection(tree: Tree) -> None:
    assert report.stale_artifacts() == []


def test_forecasts_of_another_panel_are_refused(tree: Tree) -> None:
    tree.write_tables(PANEL.assign(y=[1.0, 5.0]))
    problems = report.stale_artifacts()
    assert len(problems) == 1
    assert "прогнозы stats" in problems[0] and "panel" in problems[0]
    assert "make backtest" in problems[0]


def test_forecasts_without_a_stamp_are_refused(tree: Tree) -> None:
    (tree.oof / "stats_inputs.json").unlink()
    problems = report.stale_artifacts()
    assert len(problems) == 1 and "нет отметки входов" in problems[0]


def test_an_ensemble_and_metrics_built_from_other_forecasts_are_refused(tree: Tree) -> None:
    """Прогнозы группы пересчитаны, а ансамбль и метрики остались прежними."""
    tree.write_forecasts(payload=b"recomputed forecasts")
    problems = report.stale_artifacts()
    assert [p.split(":")[0] for p in problems] == ["ансамбль", "метрики моделей"]
    assert "sbx backtest ensemble" in problems[0]

    tree.write_ensemble()
    tree.write_metrics()
    assert report.stale_artifacts() == []


def test_optional_artifacts_are_checked_only_when_present(tree: Tree) -> None:
    assert report.stale_artifacts() == [], "детекторов ещё нет — проверять нечего"
    (tree.cpd / "detector_comparison.csv").write_text("detector,f1\n", encoding="utf-8")
    problems = report.stale_artifacts()
    assert len(problems) == 1 and problems[0].startswith("сравнение детекторов")
    stamps.write(
        tree.cpd / "inputs.json", [stamps.PANEL, stamps.file_key(tree.oof / "ensemble.parquet")]
    )
    assert report.stale_artifacts() == []
    tree.write_ensemble()
    (tree.oof / "ensemble.parquet").write_bytes(b"another ensemble")
    assert [p.split(":")[0] for p in report.stale_artifacts()] == [
        "метрики моделей",
        "сравнение детекторов",
    ]


def test_collect_stops_before_reading_stale_artifacts(
    tree: Tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree.write_tables(PANEL.assign(y=[9.0, 9.0]))
    with pytest.raises(report.StaleArtifacts, match="прогнозы stats"):
        report.collect()


def test_cli_report_build_prints_what_to_recompute(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from sbx.shell import cli

    def refuse() -> dict:
        raise report.StaleArtifacts(["прогнозы lgbm: посчитаны на других входах (panel)"])

    monkeypatch.setattr(report, "run", refuse)
    result = CliRunner().invoke(cli.app, ["report", "build"])
    assert result.exit_code == 1
    assert "прогнозы lgbm" in result.output and "не собран" in result.output


def test_every_step_records_the_inputs_it_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Отметка шага перечисляет его входы — по ней и видно, что результат устарел."""
    monkeypatch.setattr(stamps, "ROOT", tmp_path)
    for module, name in ((cpd, "OOF_DIR"), (cases, "OOF_DIR"), (ablations, "OOF_DIR")):
        monkeypatch.setattr(module, name, tmp_path / "artifacts" / "oof")
    for module in (hazard, cases, ablations):
        monkeypatch.setattr(module, "CPD_DIR", tmp_path / "artifacts" / "cpd")
    monkeypatch.setattr(ablations, "ABLATION_OOF_DIR", tmp_path / "artifacts" / "oof" / "ablations")
    monkeypatch.setattr(ablations, "HAZARD_DIR", tmp_path / "artifacts" / "hazard")
    config = tmp_path / "configs" / "step.yaml"

    frozen = {"frozen_benchmark": str(tmp_path / "data" / "benchmark" / "v1")}
    assert cpd.stamp_inputs(frozen, config) == [
        "panel",
        "static",
        "seasonal",
        "file:configs/step.yaml",
        "file:artifacts/oof/ensemble.parquet",
        "file:data/benchmark/v1_series.parquet",
        "file:data/benchmark/v1_labels.parquet",
    ]
    assert cpd.stamp_inputs({}, config)[-1] == "file:artifacts/oof/ensemble.parquet"

    exogenous = {"exogenous": {"news": "artifacts/news/region_month_features.parquet"}}
    assert hazard.stamp_inputs(exogenous, config) == [
        "static",
        "seasonal",
        "file:configs/step.yaml",
        "file:artifacts/cpd/benchmark_series.parquet",
        "file:artifacts/cpd/benchmark_labels.parquet",
        "file:artifacts/cpd/alarms.json",
        "file:artifacts/news/region_month_features.parquet",
    ]
    assert cases.stamp_inputs() == [
        "static",
        "file:artifacts/oof/ensemble.parquet",
        "file:artifacts/cpd/live_alarms.json",
        # Метод и штраф национальных разборов берутся из сравнения детекторов.
        "file:artifacts/cpd/detector_comparison.csv",
    ]
    assert ablations.stamp_inputs(config) == [
        "file:configs/step.yaml",
        "glob:artifacts/oof/*.parquet",
        "glob:artifacts/oof/ablations/*.parquet",
        "file:artifacts/cpd/detector_comparison.csv",
        "file:artifacts/hazard/hazard_results.json",
    ]


def test_cases_step_writes_its_stamp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(stamps, "ROOT", tmp_path)
    monkeypatch.setattr(cases, "FIGURES", tmp_path / "figures")
    monkeypatch.setattr(cases, "OOF_DIR", tmp_path / "artifacts" / "oof")
    monkeypatch.setattr(cases, "CPD_DIR", tmp_path / "artifacts" / "cpd")
    monkeypatch.setattr(cases, "load_known_shocks", lambda: {})
    monkeypatch.setattr(cases, "national_case", lambda name, shocks, *method: {"series": name})
    monkeypatch.setattr(cases, "municipal_case", lambda: None)
    out = tmp_path / "artifacts" / "cases"
    cases.run(out_dir=out)
    assert set(stamps.read(out / "inputs.json")) == set(cases.stamp_inputs())
    assert stamps.stale(out / "inputs.json") == []


# --- Настоящие шаги расчёта: отметки пишет сам код, а не помощники теста ------------------

MONTHS = pd.date_range("2023-01-01", periods=24, freq="MS")


class Project:
    """Маленький проект во временном каталоге: панель из шести рядов и две группы моделей,
    которые считает настоящий `backtest.run`."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from sbx.core import folds
        from sbx.shell.models import base as model_base
        from sbx.shell.pipelines import backtest

        self.backtest = backtest
        self.monkeypatch = monkeypatch
        self.oof = root / "artifacts" / "oof"
        self.metrics = root / "artifacts" / "metrics"
        self.computed: list[str] = []
        self.panel = pd.DataFrame(
            [
                {"unique_id": f"s{i}", "category": "Все категории", "ds": m, "y": 100.0 * (i + 1)}
                for i in range(6)
                for m in MONTHS
            ]
        )
        self.static = pd.DataFrame(
            {"unique_id": [f"s{i}" for i in range(6)], "category": "Все категории", "size_group": 1}
        )
        national = pd.DataFrame(
            {
                "period": pd.date_range("2019-01-01", "2025-12-01", freq="MS"),
                "type": "Всего",
                "value": 1000.0,
            }
        )
        self.eval_ids: list[str] | None = None
        self.models = {"groups": {"first": {"depth": 1}, "second": {"depth": 1}}}
        config = folds.BacktestConfig(
            status="test",
            checked_at="2026-10-02",
            target=folds.TargetSpec("d", "MS", "l"),
            folds=(folds.HorizonFolds(horizon=1, origins=("2024-06-01",)),),
            primary_metric="mae",
            mae_aggregation="micro",
        )
        for group in ("first", "second"):
            monkeypatch.setitem(model_base.REGISTRY, group, self._forecast(group))
        monkeypatch.setattr(stamps, "ROOT", root)
        monkeypatch.setattr(stamps, "PROCESSED", root / "data" / "processed")
        for module in (backtest, report):
            monkeypatch.setattr(module, "OOF_DIR", self.oof)
            monkeypatch.setattr(module, "METRICS_DIR", self.metrics)
        for name in (
            "CPD_DIR",
            "HAZARD_DIR",
            "ABLATION_DIR",
            "CASES_DIR",
            "ABLATION_OOF_DIR",
            "PROCESSED",
            "FEATURES_DIR",
        ):
            monkeypatch.setattr(report, name, root / "artifacts" / name.lower())
        monkeypatch.setattr(backtest, "load_panel", lambda: self.panel)
        monkeypatch.setattr(backtest, "load_static", lambda: self.static)
        monkeypatch.setattr(backtest, "load_national_spending", lambda: national)
        monkeypatch.setattr(backtest, "seasonal_lag_days", lambda: 35)
        monkeypatch.setattr(backtest, "load_backtest_config", lambda path=None: config)
        monkeypatch.setattr(backtest, "load_models_config", lambda: self.models)
        monkeypatch.setattr(backtest, "evaluation_series", lambda static, cfg: self.eval_ids)

    def _forecast(self, group: str):
        def forecast(train, fold, cfg, context) -> pd.DataFrame:
            self.computed.append(group)
            month = fold.cutoff + pd.DateOffset(months=1)
            ids = sorted(train["unique_id"].unique())
            return pd.DataFrame({"unique_id": ids, "ds": month, "model": group, "y_hat": 1.0})

        return forecast

    def run(self, *groups: str) -> None:
        self.backtest.run(list(groups))


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Project:
    return Project(tmp_path, monkeypatch)


def _labels(problems: list[str]) -> list[str]:
    return [problem.split(";")[0] for problem in problems]


def test_a_full_run_leaves_nothing_to_object_to(project: Project) -> None:
    project.run("first", "second")
    assert report.stale_artifacts() == []
    project.run("first", "second")
    assert report.stale_artifacts() == [] and project.computed == ["first", "second"]


def test_a_group_left_on_another_evaluation_list_is_refused(project: Project) -> None:
    """Смешанный прогон: оценочный список сменился, а пересчитана только одна группа. Вторая
    посчитана на прежнем списке, и отчёт обязан это назвать, хотя панель у них общая."""
    project.run("first", "second")
    project.eval_ids = ["s0", "s1", "s2"]
    project.run("first")
    assert _labels(report.stale_artifacts()) == ["прогнозы second: входы изменились (eval_ids)"]
    project.run("second")
    assert report.stale_artifacts() == []


def test_a_group_left_on_other_settings_or_another_protocol_is_refused(project: Project) -> None:
    project.run("first", "second")
    project.models = {"groups": {"first": {"depth": 1}, "second": {"depth": 2}}}
    assert _labels(report.stale_artifacts()) == ["прогнозы second: входы изменились (config)"]
    project.models = {"groups": {"first": {"depth": 1}, "second": {"depth": 1}}}
    assert report.stale_artifacts() == []

    project.monkeypatch.setattr(project.backtest, "CACHE_VERSION", 99)
    assert _labels(report.stale_artifacts()) == [
        "прогнозы first: входы изменились (version)",
        "прогнозы second: входы изменились (version)",
    ]


def test_metrics_written_by_the_real_step_go_stale_with_the_forecasts(project: Project) -> None:
    """Отметку метрик пишет сам шаг; пустая отметка этого бы не заметила."""
    project.run("first", "second")
    assert set(stamps.read(project.metrics / "inputs.json")) == {"glob:artifacts/oof/*.parquet"}
    (project.oof / "second.parquet").write_bytes(b"replaced behind the back")
    labels = _labels(report.stale_artifacts())
    assert "метрики моделей: входы изменились (glob:artifacts/oof/*.parquet)" in labels


def test_ablation_forecasts_made_on_other_features_are_refused(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Шаг признаков пересчитан, абляции — нет: числа B–F в отчёте были бы от прежних признаков."""
    from sbx.core.features.asof import PAST_ONLY, ExogenousBlock
    from sbx.shell.models import base as model_base

    def block(name: str, value: float) -> ExogenousBlock:
        return ExogenousBlock(name, pd.DataFrame({"origin": MONTHS, name: value}), PAST_ONLY)

    blocks = {"macro": block("macro", 16.0), "news": block("news", 120.0)}
    config = {
        "feature_variants": {
            "B": {"blocks": ["macro"], "model_name": "LightGBM (B)"},
            "C": {"blocks": ["macro", "news"], "model_name": "LightGBM (C)"},
        }
    }
    variants = project.oof / "ablations"
    monkeypatch.setitem(model_base.REGISTRY, "lgbm", project._forecast("lgbm"))
    monkeypatch.setattr(ablations, "ABLATION_OOF_DIR", variants)
    monkeypatch.setattr(report, "ABLATION_OOF_DIR", variants)
    monkeypatch.setattr(ablations, "load_ablation_config", lambda path=None: config)
    monkeypatch.setattr(ablations, "load_feature_blocks", lambda: blocks)
    for name in (
        "load_panel",
        "load_static",
        "load_national_spending",
        "seasonal_lag_days",
        "load_backtest_config",
        "load_models_config",
        "evaluation_series",
    ):
        monkeypatch.setattr(ablations, name, getattr(project.backtest, name))
    project.models = {"groups": {"lgbm": {}}}

    ablations.run_feature_ablations(config)
    assert report.stale_artifacts() == []

    blocks["news"] = block("news", 150.0)
    assert _labels(report.stale_artifacts()) == [
        "прогнозы абляции lgbm_C: входы изменились (context)"
    ]
    ablations.run_feature_ablations(config)
    assert report.stale_artifacts() == []

    (variants / "lgbm_Z.parquet").write_bytes(b"left over from a removed variant")
    assert "нет отметки входов" in report.stale_artifacts()[0]
    # Вариант убран из конфига, а его прогнозы с отметкой остались на диске.
    stamps.save(variants / "lgbm_Z_inputs.json", stamps.read(variants / "lgbm_C_inputs.json"))
    assert report.stale_artifacts() == [
        "прогнозы абляции lgbm_Z: в конфиге такого расчёта нет; удалите файл"
    ]
    (variants / "lgbm_Z.parquet").unlink()
    (variants / "lgbm_Z_inputs.json").unlink()

    # Вариант источника, чьих данных сейчас нет (прогон без архива новостей после прогона с
    # ним): он в конфиге, посчитается снова, когда данные появятся, и в отчёт не идёт.
    config["sources"] = {"base": "B", "blocks": {"gdelt": "Новости GDELT"}}
    (variants / "lgbm_src_B_gdelt.parquet").write_bytes(b"from the run with the archive")
    stamps.save(
        variants / "lgbm_src_B_gdelt_inputs.json", stamps.read(variants / "lgbm_C_inputs.json")
    )
    assert report.stale_artifacts() == []

    # Но таблица вклада источников не должна хранить его строку: иначе абляции после
    # исчезновения данных не пересчитывались, и отчёт показал бы число прошлого прогона.
    table = project.oof.parent / "ablations"
    table.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(report, "ABLATION_DIR", table)
    row = {"title": "Новости GDELT", "mean_diff": -2.6}
    pd.DataFrame([{"block": "gdelt", "against": "B", **row}]).to_csv(
        table / "sources.csv", index=False
    )
    assert report.stale_artifacts() == [
        "вклад источника «Новости GDELT» (к B): посчитан на данных, которых сейчас нет; "
        "пересчитайте: make ablations"
    ]
    pd.DataFrame([{"block": "macro", "against": "B", **row}]).to_csv(
        table / "sources.csv", index=False
    )
    assert report.stale_artifacts() == [], "строки других источников отчёту не мешают"

    # Данные источника появились: его вариант снова сверяется по отпечатку входов, как любой
    # другой, а не пропускается только потому, что это вариант источника. Строка о нём в
    # таблице вклада при этом законна.
    pd.DataFrame([{"block": "gdelt", "against": "B", **row}]).to_csv(
        table / "sources.csv", index=False
    )
    blocks["gdelt"] = block("gdelt", 7.0)
    (found,) = _labels(report.stale_artifacts())
    assert found.startswith("прогнозы абляции lgbm_src_B_gdelt: входы изменились")

    # Данных снова нет, и оставшиеся прогнозы удалены вручную: строка в таблице вклада всё
    # равно устаревшая, пока абляции не пересчитаны.
    del blocks["gdelt"]
    (variants / "lgbm_src_B_gdelt.parquet").unlink()
    (variants / "lgbm_src_B_gdelt_inputs.json").unlink()
    assert _labels(report.stale_artifacts()) == [
        "вклад источника «Новости GDELT» (к B): посчитан на данных, которых сейчас нет"
    ]


OPTIONAL = [
    ("значимость различий", "METRICS_DIR", "significance.json", "significance_inputs.json"),
    ("сравнение детекторов", "CPD_DIR", "detector_comparison.csv", "inputs.json"),
    ("модель вероятности шока", "HAZARD_DIR", "hazard_results.json", "inputs.json"),
    ("разборы примеров", "CASES_DIR", "cases.json", "inputs.json"),
    ("абляции", "ABLATION_DIR", "ablations.csv", "inputs.json"),
    ("панель расходов", "PROCESSED", "panel.parquet", "panel_inputs.json"),
    ("внешние признаки", "FEATURES_DIR", "features_summary.json", "inputs.json"),
]


@pytest.mark.parametrize(("label", "directory", "artifact", "stamp"), OPTIONAL)
def test_every_stamped_artifact_is_checked(
    tree: Tree, label: str, directory: str, artifact: str, stamp: str
) -> None:
    """Каждый результат, который читает отчёт, сверяется со своей отметкой: без отметки,
    с пустой отметкой и с изменившимся входом отчёт не собирается."""
    folder = getattr(report, directory)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / artifact).write_bytes(b"result")
    source = tree.root / "source.txt"
    source.write_text("v1", encoding="utf-8")

    assert [p for p in _labels(report.stale_artifacts()) if p.startswith(label)], "нет отметки"
    stamps.save(folder / stamp, {})
    assert [p for p in _labels(report.stale_artifacts()) if p.startswith(label)], "пустая отметка"
    stamps.write(folder / stamp, [stamps.file_key(source)])
    assert not [p for p in report.stale_artifacts() if p.startswith(label)]
    source.write_text("v2", encoding="utf-8")
    assert _labels(report.stale_artifacts()) == [f"{label}: входы изменились (file:source.txt)"]


def _noop(*args: object, **kwargs: object) -> None:
    return None


def test_detector_step_stamps_its_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(stamps, "ROOT", tmp_path)
    monkeypatch.setattr(cpd, "OOF_DIR", tmp_path / "artifacts" / "oof")
    monkeypatch.setattr(cpd, "load_cpd_config", lambda path=None: {"n_series": 5})
    monkeypatch.setattr(cpd, "build", lambda cfg, out_dir: (None, None))
    for name in (
        "calibrate_and_evaluate",
        "delay_vs_false_alarms",
        "schedule_match",
        "pseudo_labels",
        "live_alarms",
    ):
        monkeypatch.setattr(cpd, name, _noop)
    config = tmp_path / "configs" / "cpd.yaml"
    out = tmp_path / "artifacts" / "cpd"
    cpd.run(cfg_path=config, out_dir=out)
    recorded = stamps.read(out / "inputs.json")
    assert list(recorded) == cpd.stamp_inputs({"n_series": 5}, config) and len(recorded) >= 5


def test_hazard_step_stamps_its_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(stamps, "ROOT", tmp_path)
    monkeypatch.setattr(hazard, "CPD_DIR", tmp_path / "artifacts" / "cpd")
    settings = {"exogenous": {"news": "artifacts/news/region_month_features.parquet"}}
    monkeypatch.setattr(hazard, "load_hazard_config", lambda path=None: settings)
    monkeypatch.setattr(hazard, "train_and_compare", lambda cfg, out_dir: {"h3": {}})
    config = tmp_path / "configs" / "hazard.yaml"
    out = tmp_path / "artifacts" / "hazard"
    assert hazard.run(cfg_path=config, out_dir=out) == {"h3": {}}
    recorded = stamps.read(out / "inputs.json")
    assert list(recorded) == hazard.stamp_inputs(settings, config)
    assert "file:artifacts/news/region_month_features.parquet" in recorded


def test_ablation_step_stamps_its_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(stamps, "ROOT", tmp_path)
    for name in ("OOF_DIR", "ABLATION_OOF_DIR", "CPD_DIR", "HAZARD_DIR"):
        monkeypatch.setattr(ablations, name, tmp_path / "artifacts" / name.lower())
    table = pd.DataFrame({"name": ["A"], "mae": [700.0]})
    monkeypatch.setattr(ablations, "load_ablation_config", lambda path=None: {})
    monkeypatch.setattr(ablations, "build_table", lambda cfg, out_dir: table)
    sources: list[Path] = []
    monkeypatch.setattr(ablations, "build_sources", lambda cfg, out_dir: sources.append(out_dir))
    config = tmp_path / "configs" / "ablations.yaml"
    out = tmp_path / "artifacts" / "ablations"
    assert ablations.run(cfg_path=config, out_dir=out) is table
    assert sources == [out], "вклад источников считается тем же шагом"
    recorded = stamps.read(out / "inputs.json")
    assert list(recorded) == ablations.stamp_inputs(config) and len(recorded) == 5
