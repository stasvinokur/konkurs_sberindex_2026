"""Ревизии весов foundation-моделей закреплены и лицензионно допустимы."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_LICENSES = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause"}
FORBIDDEN_REPOS = ("timesfm-3.0", "moirai", "tirex", "tabpfn")


def test_revisions_are_pinned_commit_shas() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    models = cfg["foundation_models"]
    assert models
    for name, spec in models.items():
        assert re.fullmatch(r"[0-9a-f]{40}", spec["revision"]), name
        assert spec["weights_license"] in ALLOWED_LICENSES, name


def test_no_forbidden_weights_are_loaded() -> None:
    """Запрещённые веса не должны загружаться.

    Проверяется именно загрузка, а не упоминание: отчёт и документация обязаны называть
    Moirai, TimesFM 3.0, TiRex и TabPFN-TS, объясняя, почему они исключены по лицензии.
    Раньше тест запрещал любое вхождение строки и падал на этом объяснении.
    """
    loaders = ("from_pretrained", "repo_id", "model_id", "hf_repo", "snapshot_download")
    offenders = []

    for path in (ROOT / "configs").rglob("*.yaml"):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for name, spec in (cfg.get("foundation_models") or {}).items():
            repo = str(spec.get("repo", "")).lower()
            offenders += [f"{path.name}:{name}" for bad in FORBIDDEN_REPOS if bad in repo]

    for path in (ROOT / "src").rglob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            lowered = line.lower()
            if not any(marker in lowered for marker in loaders):
                continue
            offenders += [f"{path.name}:{number}" for bad in FORBIDDEN_REPOS if bad in lowered]

    assert not offenders, f"запрещённые веса загружаются в: {offenders}"
