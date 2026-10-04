.PHONY: sync test test-fast lint format

sync:
	uv sync --all-extras

test:
	uv run pytest

test-fast:
	uv run pytest -m "not slow" tests/core tests/test_architecture.py

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

.PHONY: data validate gdelt
data:
	uv run sbx data download
	uv run sbx data official

validate:
	uv run sbx data validate

# Архив новостей GDELT: докачивает недостающие месяцы и сверяет архив с data/gdelt_manifest.yaml.
gdelt:
	uv run sbx data gdelt

.PHONY: entities
entities:
	uv run sbx data entities

.PHONY: panel
panel:
	uv run sbx data panel

.PHONY: backtest cpd ablations features report all
features:
	uv run sbx data features

# Каждая группа моделей — отдельным процессом. LightGBM и PyTorch приносят каждая свою
# библиотеку OpenMP, и в одном процессе на macOS прогон зависает на нейросетях без нагрузки на
# процессор. Готовые группы берутся из кэша, так что прерванный прогон продолжается с места.
backtest:
	uv run sbx backtest run stats
	uv run sbx backtest run prophet
	uv run sbx backtest run lgbm
	uv run sbx backtest run neural
	uv run sbx backtest run chronos2
	uv run sbx backtest run timesfm25
	uv run sbx backtest ensemble
	uv run sbx backtest significance

cpd:
	uv run sbx cpd run
	uv run sbx cpd national
	uv run sbx cpd hazard
	uv run sbx cpd cases

ablations:
	uv run sbx backtest ablations

report:
	uv run sbx report build

# Порядок важен: features читает архив новостей из gdelt, cpd — OOF ансамбля из backtest,
# ablations — сравнение детекторов из cpd.
all: validate gdelt entities panel features backtest cpd ablations report
