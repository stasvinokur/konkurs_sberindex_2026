"""Командная строка проекта: `uv run sbx --help`."""

from __future__ import annotations

from typing import Annotated

import typer

import sbx.shell.models.chronos2_  # noqa: F401
import sbx.shell.models.lgbm_  # noqa: F401
import sbx.shell.models.neuralforecast_  # noqa: F401
import sbx.shell.models.prophet_  # noqa: F401  регистрация групп моделей
import sbx.shell.models.statsforecast_  # noqa: F401
import sbx.shell.models.timesfm_  # noqa: F401

app = typer.Typer(
    help="Прогноз расходов МО и раннее обнаружение структурных сдвигов", no_args_is_help=True
)
data_app = typer.Typer(help="Сырые данные СберИндекса", no_args_is_help=True)
app.add_typer(data_app, name="data")


@data_app.command("download")
def data_download() -> None:
    """Докачать отсутствующие датасеты из манифеста (существующие файлы не перезаписываются)."""
    from sbx.shell.download import sberindex

    report = sberindex.sync(sberindex.load_manifest())
    for path in report.downloaded:
        typer.echo(f"скачан: {path}")
    typer.echo(f"в порядке: {len(report.ok)}, скачано: {len(report.downloaded)}")
    for problem in report.problems:
        typer.echo(f"ОШИБКА: {problem}", err=True)
    if report.problems:
        raise typer.Exit(1)


@data_app.command("validate")
def data_validate() -> None:
    """Сверить файлы данных с манифестами: СберИндекс, организаторы, погода, население."""
    from sbx.shell.download import official, power, rosstat, sberindex

    manifest = sberindex.load_manifest()
    official_manifest = official.load_manifest()
    problems = sberindex.validate(manifest) + official.validate(official_manifest)
    problems += power.validate(power.MANIFEST_PATH, power.TABLE_PATH)
    problems += rosstat.validate(rosstat.MANIFEST_PATH, rosstat.TABLE_PATH)
    for problem in problems:
        typer.echo(f"ОШИБКА: {problem}", err=True)
    if problems:
        raise typer.Exit(1)
    total = len(manifest.all_files()) + len(official_manifest.files)
    total += int(power.MANIFEST_PATH.exists()) + int(rosstat.MANIFEST_PATH.exists())
    typer.echo(f"все {total} файлов совпадают с манифестами")


@data_app.command("official")
def data_official(
    check: Annotated[
        bool,
        typer.Option("--check", help="только сверить файлы с манифестом, без обращения к сети"),
    ] = False,
) -> None:
    """Получить отсутствующие файлы организаторов конкурса и сверить их с манифестом."""
    from sbx.shell.download import official

    manifest = official.load_manifest()
    if check:
        problems = official.validate(manifest)
    else:
        report = official.sync(manifest)
        for path in report.downloaded:
            typer.echo(f"получен: {path}")
        problems = report.problems
    for problem in problems:
        typer.echo(f"ОШИБКА: {problem}", err=True)
    if problems:
        raise typer.Exit(1)
    typer.echo(f"все {len(manifest.files)} файлов организаторов совпадают с манифестом")


@data_app.command("manifest")
def data_manifest() -> None:
    """Пересобрать `data/manifest.yaml` по текущим файлам в `data/raw`."""
    from sbx.shell.download import sberindex
    from sbx.shell.io import write_yaml

    write_yaml(sberindex.MANIFEST_PATH, sberindex.build_manifest())
    typer.echo(f"записан {sberindex.MANIFEST_PATH}")


@data_app.command("gdelt")
def data_gdelt(
    check: Annotated[
        bool,
        typer.Option("--check", help="только сверить архив с манифестом, без обращения к сети"),
    ] = False,
) -> None:
    """Докачать недостающие месяцы архива GDELT и сверить его с `data/gdelt_manifest.yaml`."""
    from sbx.shell.download import gdelt

    manifest = gdelt.load_manifest()
    total = len(manifest.months)
    if check:
        problems = gdelt.validate(manifest)
    else:
        if pending := gdelt.pending_months(manifest):
            typer.echo(
                f"нужно скачать месяцев: {len(pending)} из {total}; на месяц уходит 1–2 минуты "
                "и 160–300 МБ трафика, на диске остаётся 3–5 МБ"
            )
        report = gdelt.sync(
            manifest,
            progress=lambda month, rows, seconds: typer.echo(
                f"скачан {month}: {rows} событий за {seconds:.0f} с"
            ),
        )
        problems = report.problems
    for problem in problems:
        typer.echo(f"ОШИБКА: {problem}", err=True)
    if problems:
        typer.echo(
            "Архив GDELT неполон или не совпадает с манифестом.\n"
            "- Месяц не скачан: повторите `make gdelt`, скачанные месяцы не перекачиваются.\n"
            "- Файл месяца на месте, но не совпал: удалите его и повторите, он скачается заново.\n"
            "- Не совпал только что скачанный месяц (лежит рядом как *.downloaded): источник "
            "изменил данные или обновлены pandas и pyarrow; опубликованные числа на таком "
            "архиве не воспроизвести.\n"
            "Как выполнить остальные шаги без новостей — README, раздел «Архив новостей GDELT».",
            err=True,
        )
        raise typer.Exit(1)
    events = sum(spec.rows for spec in manifest.months)
    typer.echo(f"все {total} месяцев архива GDELT совпадают с манифестом ({events} событий)")


@data_app.command("weather")
def data_weather(
    check: Annotated[
        bool,
        typer.Option("--check", help="только сверить таблицу с манифестом, без обращения к сети"),
    ] = False,
) -> None:
    """Скачать погоду NASA POWER по ячейкам сетки и собрать таблицу (меняет эталон сверки!)."""
    from sbx.shell.download import power

    if check:
        if not power.MANIFEST_PATH.exists():
            typer.echo("таблицы погоды нет: источник не подключён (скачать: sbx data weather)")
            return
        problems = power.validate(power.MANIFEST_PATH, power.TABLE_PATH)
        for problem in problems:
            typer.echo(f"ОШИБКА: {problem}", err=True)
        if problems:
            raise typer.Exit(1)
        typer.echo(f"таблица погоды совпадает с манифестом: {power.TABLE_PATH}")
        return

    from sbx.shell.pipelines import features

    def progress(done: int, total: int) -> None:
        if done % 100 == 0 or done == total:
            typer.echo(f"ячеек обработано: {done} из {total}")

    result = power.sync(features.load_territories(), progress=progress)
    typer.echo(
        f"ячеек: {result['cells']}, скачано: {result['downloaded']}, уже было: {result['cached']}"
    )
    if result["failed"]:
        typer.echo(
            f"ОШИБКА: не скачано ячеек: {len(result['failed'])} — источник не ответил; "
            "повторить: sbx data weather (скачанное сохранено)",
            err=True,
        )
        raise typer.Exit(1)
    typer.echo(f"таблица: {power.TABLE_PATH}, строк {result['manifest']['rows']}")


@data_app.command("population")
def data_population(
    check: Annotated[
        bool,
        typer.Option("--check", help="только сверить таблицу с манифестом, без обращения к сети"),
    ] = False,
) -> None:
    """Скачать бюллетени Росстата и собрать таблицу численности населения МО (меняет эталон сверки!)."""
    from sbx.shell.download import rosstat

    if check:
        if not rosstat.MANIFEST_PATH.exists():
            typer.echo("таблицы населения нет: источник не подключён (sbx data population)")
            return
        problems = rosstat.validate(rosstat.MANIFEST_PATH, rosstat.TABLE_PATH)
        for problem in problems:
            typer.echo(f"ОШИБКА: {problem}", err=True)
        if problems:
            raise typer.Exit(1)
        typer.echo(f"таблица населения совпадает с манифестом: {rosstat.TABLE_PATH}")
        return

    from sbx.shell.pipelines import features

    record = rosstat.sync(features.load_territories())
    share = f"{100 * record['share']:.1f}".replace(".", ",")
    typer.echo(f"привязано территорий: {record['matched']} из {record['territories']} ({share}%)")
    typer.echo(f"таблица: {rosstat.TABLE_PATH}")


@data_app.command("gdelt-manifest")
def data_gdelt_manifest() -> None:
    """Пересобрать `data/gdelt_manifest.yaml` по архиву на диске (меняет эталон сверки!)."""
    from sbx.shell.download import gdelt
    from sbx.shell.io import read_yaml, write_yaml

    write_yaml(gdelt.MANIFEST_PATH, gdelt.build_manifest(read_yaml(gdelt.MANIFEST_PATH)))
    typer.echo(f"записан {gdelt.MANIFEST_PATH}")


@data_app.command("oktmo-reference")
def data_oktmo_reference() -> None:
    """Перекачать справочник ОКТМО из Wikidata в `data/reference/` (меняет результат идентификации!)."""
    from sbx.shell.download import oktmo
    from sbx.shell.io import ROOT

    n = oktmo.save_reference(ROOT / "data" / "reference" / "wikidata_oktmo_municipal.csv")
    typer.echo(f"записано {n} объектов")


@data_app.command("entities")
def data_entities() -> None:
    """Территории по официальным кодам организаторов и сверка идентификации по названиям."""
    from sbx.shell.pipelines import entity

    report = entity.run()
    check = report["heuristic_check"]
    typer.echo(
        f"территорий {report['territories']}, рядов {report['series']}, строк {report['rows']}; "
        f"идентификация по названиям ошиблась регионом у {check['series_in_wrong_region']} "
        f"из {check['series']} рядов"
    )


@data_app.command("freeze-samples")
def data_freeze_samples(
    check: Annotated[
        bool,
        typer.Option("--check", help="только сверить файлы с повтором отбора, ничего не записывая"),
    ] = False,
) -> None:
    """Пересобрать замороженные выборки: оценочные ряды и бенчмарк сдвигов (меняет основу сравнения!)."""
    from sbx.shell.pipelines import samples

    if check:
        problems = samples.problems()
        for problem in problems:
            typer.echo(f"ОШИБКА: {problem}", err=True)
        if problems:
            raise typer.Exit(1)
        typer.echo(
            f"замороженные выборки совпадают с отбором опубликованного расчёта: "
            f"оценочных рядов {len(samples.load_evaluation())}, "
            f"рядов бенчмарка {len(samples.load_benchmark()[1])}"
        )
        return
    for path in samples.write(samples.build()):
        typer.echo(f"записан {path}")


@data_app.command("panel")
def data_panel() -> None:
    """Каноническая панель, статические признаки, сезонные индексы и EDA-графики."""
    import json

    from sbx.shell.pipelines import panel

    summary = panel.run()
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


GROUPS_ARG = Annotated[list[str], typer.Argument(help="группы моделей: stats, prophet, lgbm, …")]


@data_app.command("features")
def data_features() -> None:
    """Собрать внешние признаки: календарь, национальные ряды, новости GDELT, события, соседние МО, погода, население."""
    import json

    from sbx.shell.download.gdelt import ArchiveMismatch
    from sbx.shell.pipelines import features

    try:
        summary = features.run()
    except ArchiveMismatch as exc:
        typer.echo(f"ОШИБКА: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    if isinstance(summary["news"], str):
        typer.echo(
            f"ВНИМАНИЕ: {summary['news']}. Без новостных признаков конфигурации абляций C, E, F "
            "и модель вероятности шока получатся не такими, как в опубликованном отчёте.",
            err=True,
        )


backtest_app = typer.Typer(help="Бэктест прогнозных моделей", no_args_is_help=True)
app.add_typer(backtest_app, name="backtest")


@backtest_app.command("run")
def backtest_run(
    groups: GROUPS_ARG,
    overwrite: bool = False,
    limit_series: int = 0,
) -> None:
    """Прогнать группы моделей по фолдам протокола и пересобрать метрики.

    Группы лучше запускать отдельными командами, как это делает `make backtest`: LightGBM и
    PyTorch в одном процессе на macOS могут зависнуть из-за двух библиотек OpenMP.
    """
    from sbx.shell.pipelines import backtest

    board = backtest.run(groups, overwrite=overwrite, limit_series=limit_series or None)
    typer.echo(board.to_string(index=False))


@backtest_app.command("leaderboard")
def backtest_leaderboard() -> None:
    """Пересобрать метрики из сохранённых OOF-прогнозов."""
    from sbx.shell.pipelines import backtest

    cfg = backtest.load_backtest_config()
    board = backtest.build_metrics(cfg, backtest.collect_oof())
    typer.echo(board.to_string(index=False))


@backtest_app.command("ensemble")
def backtest_ensemble() -> None:
    """Построить ансамбль по OOF-прогнозам и пересобрать метрики."""
    from sbx.shell.pipelines import backtest

    cfg = backtest.load_backtest_config()
    backtest.build_ensemble(cfg)
    board = backtest.build_metrics(cfg, backtest.collect_oof())
    typer.echo(board.to_string(index=False))


@backtest_app.command("significance")
def backtest_significance() -> None:
    """Пересчитать тесты значимости по каждому горизонту."""
    import json

    from sbx.shell.pipelines import backtest

    result = backtest.build_significance()
    for horizon, entry in result["by_horizon"].items():
        pairs = "; ".join(f"{d['pair']}: p={d['p_value']:.2e}" for d in entry["dm"])
        typer.echo(f"h={horizon}: лучшая — {entry['best_model']}; {pairs}")
    typer.echo(json.dumps({"горизонтов": len(result["by_horizon"])}, ensure_ascii=False))


@backtest_app.command("compare")
def backtest_compare(
    baseline: Annotated[
        str, typer.Argument(help="каталог артефактов прежнего прогона (с подкаталогом oof)")
    ],
) -> None:
    """Сравнить текущие артефакты с прежним прогоном: MAE на общих строках, лидеры, абляции."""
    from pathlib import Path

    from sbx.shell.pipelines import compare

    try:
        summary = compare.run(Path(baseline))
    except FileNotFoundError as exc:
        typer.echo(f"ОШИБКА: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"документ сравнения: {summary['document']}")
    typer.echo("прогнозы совпали до копейки: " + (", ".join(summary["identical_models"]) or "—"))
    typer.echo("прогнозы изменились: " + (", ".join(summary["changed_models"]) or "—"))
    failed = [m for m, verdict in summary["self_check"].items() if verdict in ("changed", "absent")]
    typer.echo(
        "самопроверка пройдена: модели, которых исправления не касаются, воспроизвелись"
        if summary["self_check_passed"]
        else "самопроверка не пройдена: " + ", ".join(failed)
    )


@backtest_app.command("ablations")
def backtest_ablations(overwrite: bool = False) -> None:
    """Матрица абляций A–F: вклад макро, новостей, FM и детектора по остаткам."""
    from sbx.shell.pipelines import ablations

    cfg = ablations.load_ablation_config()
    ablations.run_feature_ablations(cfg, overwrite=overwrite)
    table = ablations.run()
    typer.echo(table.to_string(index=False))


cpd_app = typer.Typer(help="Обнаружение структурных изменений", no_args_is_help=True)
app.add_typer(cpd_app, name="cpd")


@cpd_app.command("run")
def cpd_run() -> None:
    """Собрать полусинтетический бенчмарк, откалибровать и сравнить детекторы."""
    from sbx.shell.pipelines import cpd

    results = cpd.run()
    typer.echo(results.to_string(index=False))


@cpd_app.command("national")
def cpd_national() -> None:
    """Проверить офлайн-методы на реальных шоках национальных рядов."""
    from sbx.shell.pipelines import national_cpd

    typer.echo(national_cpd.run().to_string(index=False))


@cpd_app.command("cases")
def cpd_cases() -> None:
    """Собрать разборы реальных примеров с графиками."""
    import json

    from sbx.shell.pipelines import cases

    typer.echo(json.dumps(cases.run(), ensure_ascii=False, indent=2, default=str))


@cpd_app.command("hazard")
def cpd_hazard() -> None:
    """Обучить модель вероятности шока и сравнить её с онлайн-детектором."""
    import json

    from sbx.shell.pipelines import hazard

    typer.echo(json.dumps(hazard.run(), ensure_ascii=False, indent=2, default=str))


report_app = typer.Typer(help="Отчёт и лендинг", no_args_is_help=True)
app.add_typer(report_app, name="report")


@report_app.command("build")
def report_build() -> None:
    """Собрать методологический отчёт и статический лендинг из артефактов."""
    import json

    from sbx.shell.pipelines import report

    try:
        summary = report.run()
    except report.StaleArtifacts as exc:
        typer.echo(f"ОШИБКА: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    app()
