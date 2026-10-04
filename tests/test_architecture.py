"""Архитектурное правило Functional Core / Imperative Shell.

Ядро (`sbx.core`) не импортирует оболочку, IO/сетевые библиотеки и тяжёлые ML-фреймворки
и не вызывает функции с побочными эффектами. Проверка статическая (AST).
"""

from __future__ import annotations

import ast
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parents[1] / "src" / "sbx" / "core"

FORBIDDEN_MODULES = {
    "sbx.shell",
    "requests",
    "urllib",
    "httpx",
    "socket",
    "subprocess",
    "logging",
    "shutil",
    "tempfile",
    "yaml",
    "torch",
    "prophet",
    "lightgbm",
    "statsforecast",
    "mlforecast",
    "neuralforecast",
    "chronos",
    "timesfm",
    "ruptures",
    "river",
    "optuna",
    "shap",
    "plotly",
    "matplotlib",
    "jinja2",
    "typer",
}
FORBIDDEN_CALLS = {"open", "print", "input"}
FORBIDDEN_ATTR_CALLS = {
    "read_parquet",
    "read_csv",
    "to_parquet",
    "to_csv",
    "write_text",
    "read_text",
    "now",
    "today",
    "mkdir",
    "unlink",
}


def _module_forbidden(name: str) -> bool:
    return any(name == m or name.startswith(m + ".") for m in FORBIDDEN_MODULES)


def find_violations(source: str, filename: str = "<string>") -> list[str]:
    """Возвращает список нарушений архитектурного правила в исходном коде модуля ядра."""
    tree = ast.parse(source, filename=filename)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _module_forbidden(alias.name):
                    violations.append(f"{filename}:{node.lineno} import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level and module.startswith("shell"):
                violations.append(f"{filename}:{node.lineno} relative import of shell")
            if module and _module_forbidden(module):
                violations.append(f"{filename}:{node.lineno} from {module} import ...")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALLS:
                violations.append(f"{filename}:{node.lineno} call {func.id}()")
            if isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_ATTR_CALLS:
                violations.append(f"{filename}:{node.lineno} call .{func.attr}()")
    return violations


def test_detector_catches_violations() -> None:
    bad = (
        "import torch\n"
        "from sbx.shell.io import read\n"
        "import pandas as pd\n"
        "df = pd.read_parquet('x')\n"
        "with open('f') as fh:\n    pass\n"
        "import datetime\nnow = datetime.datetime.now()\n"
    )
    found = find_violations(bad)
    assert any("torch" in v for v in found)
    assert any("sbx.shell" in v for v in found)
    assert any("read_parquet" in v for v in found)
    assert any("open()" in v for v in found)
    assert any(".now()" in v for v in found)


def test_detector_accepts_pure_code() -> None:
    good = "import numpy as np\nimport pandas as pd\n\ndef f(x):\n    return np.abs(x).mean()\n"
    assert find_violations(good) == []


def test_core_has_no_side_effect_dependencies() -> None:
    files = sorted(CORE_DIR.rglob("*.py"))
    assert files, "core package must exist"
    violations = [v for f in files for v in find_violations(f.read_text(encoding="utf-8"), str(f))]
    assert violations == [], "\n".join(violations)
