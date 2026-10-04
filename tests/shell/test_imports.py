"""Smoke-тест: все библиотеки стека импортируются в окружении проекта."""

import importlib

import pytest

LIBRARIES = [
    "statsforecast",
    "prophet",
    "lightgbm",
    "neuralforecast",
    "chronos",
    "timesfm",
    "ruptures",
    "river",
    "optuna",
]


@pytest.mark.parametrize("name", LIBRARIES)
def test_library_imports(name: str) -> None:
    importlib.import_module(name)


def test_required_model_classes_available() -> None:
    chronos = importlib.import_module("chronos")
    timesfm = importlib.import_module("timesfm")
    assert hasattr(chronos, "Chronos2Pipeline")
    assert hasattr(timesfm, "TimesFM_2p5_200M_torch")
