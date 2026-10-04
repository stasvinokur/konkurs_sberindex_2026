"""Полный прогон: `make all` строит всё, что читает отчёт, и документация называет те же шаги."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _makefile() -> str:
    return (ROOT / "Makefile").read_text(encoding="utf-8")


def _all_steps() -> list[str]:
    match = re.search(r"^all:(.*)$", _makefile(), flags=re.M)
    assert match, "в Makefile нет цели all"
    return match.group(1).split()


def _recipe(target: str) -> str:
    match = re.search(rf"^{target}:.*\n((?:\t.*\n?)+)", _makefile(), flags=re.M)
    assert match, f"в Makefile нет цели {target}"
    return match.group(1)


def test_make_all_builds_everything_the_report_reads() -> None:
    """Отчёт молча подставляет заглушки вместо отсутствующих артефактов, поэтому пропущенный
    шаг не падает, а даёт другой отчёт: без ансамбля, значимости или абляций."""
    steps = _all_steps()
    backtest = _recipe("backtest")
    assert "sbx backtest ensemble" in backtest
    assert "sbx backtest significance" in backtest
    assert "sbx backtest ablations" in _recipe("ablations")
    # cpd читает OOF ансамбля из backtest, ablations — сравнение детекторов из cpd.
    order = [steps.index(name) for name in ("backtest", "cpd", "ablations", "report")]
    assert order == sorted(order), f"неверный порядок шагов: {steps}"


def test_every_model_group_runs_in_a_process_of_its_own() -> None:
    """LightGBM и PyTorch приносят каждая свою библиотеку OpenMP. В одном процессе на macOS это
    кончается взаимной блокировкой: прогон висит на первом окне нейросетей без нагрузки на
    процессор (так завис полный прогон 2 октября 2026). Поэтому каждая группа моделей —
    отдельная команда, и ни одна не пропущена."""
    runs = re.findall(r"sbx backtest run (.+)", _recipe("backtest"))
    assert all(len(line.split()) == 1 for line in runs), f"несколько групп в одной команде: {runs}"
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    assert sorted(runs) == sorted(models["groups"]), "группы Makefile и configs/models.yaml"
    recipe = _recipe("backtest").splitlines()
    last_run = max(i for i, line in enumerate(recipe) if "sbx backtest run" in line)
    ensemble = next(i for i, line in enumerate(recipe) if "sbx backtest ensemble" in line)
    assert last_run < ensemble, "ансамбль строится после всех групп"


def test_make_all_downloads_the_news_archive_before_building_features() -> None:
    """Без архива GDELT шаг признаков пропускает новости, и абляции с моделью вероятности шока
    выходят не такими, как в опубликованном отчёте."""
    steps = _all_steps()
    assert "sbx data gdelt" in _recipe("gdelt")
    assert steps.index("gdelt") < steps.index("features"), f"неверный порядок шагов: {steps}"


def test_readme_names_one_command_for_a_run_without_the_news_archive() -> None:
    """Если источник GDELT недоступен, остальные шаги выполняются без него — все и в том же порядке."""
    steps = [step for step in _all_steps() if step != "gdelt"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"`make {' '.join(steps)}`" in readme


DOCUMENTS = [
    "README.md",
    "DATA_LICENSES.md",
    "THIRD_PARTY_LICENSES.md",
    "THIRD_PARTY_MODELS.md",
    "reports/methodology.md",
    "reports/models.md",
    "reports/changepoints.md",
    "reports/cases.md",
    "reports/eda.md",
    "reports/landing/index.html",
]
# Образ Docker собирается без файлов, не нужных для запуска (ролик, сами файлы Docker): список —
# в `.dockerignore`, и его самого в образе тоже нет.
IN_IMAGE = not (ROOT / ".dockerignore").exists()
MARKDOWN_LINK = re.compile(r"\]\(\s*(<[^>]*>|[^)\s]+)")
PAGE_LINK = re.compile(r'(?:href|src)="([^"]+)"')


@pytest.mark.parametrize("document", DOCUMENTS)
def test_relative_links_lead_to_files_of_the_repository(document: str) -> None:
    """Ссылка на файл, которого в репозитории нет, у читателя не откроется."""
    path = ROOT / document
    if not path.exists():
        pytest.skip(f"{document} не собран")
    pattern = PAGE_LINK if path.suffix == ".html" else MARKDOWN_LINK
    targets = {target.strip("<>") for target in pattern.findall(path.read_text(encoding="utf-8"))}
    local = {t.split("#")[0] for t in targets if not re.match(r"#|[a-zA-Z][\w+.-]*:", t)}
    broken = sorted(t for t in local if t and not (path.parent / unquote(t)).exists())
    if broken and IN_IMAGE:
        pytest.skip(f"в образе Docker лежит не весь репозиторий: нет {broken}")
    assert not broken, f"{document}: ссылки в никуда {broken}"


@pytest.mark.parametrize("document", ["README.md", "reports/methodology.md"])
def test_documented_full_run_names_every_step_of_make_all(document: str) -> None:
    path = ROOT / document
    if not path.exists():
        pytest.skip(f"{document} не собран")
    text = path.read_text(encoding="utf-8")
    missing = [step for step in _all_steps() if f"make {step}" not in text]
    assert not missing, f"в {document} не названы шаги полного прогона: {missing}"
