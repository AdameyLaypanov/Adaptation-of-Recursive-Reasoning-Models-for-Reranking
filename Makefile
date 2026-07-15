# Команды повседневной разработки. Все цели работают через uv.

.PHONY: install test lint fmt fmt-check check parity hooks

install:  ## окружение + dev-инструменты
	uv sync --dev

test:  ## быстрые тесты (модели, конфиги, трейнер, майнинг)
	uv run pytest -q

lint:  ## ruff check без изменений файлов
	uv run ruff check src scripts tests

fmt:  ## автоформат + автофиксы
	uv run ruff format src scripts tests
	uv run ruff check --fix src scripts tests

fmt-check:  ## проверка формата (как в CI)
	uv run ruff format --check src scripts tests

check: lint fmt-check test  ## всё, что гоняет CI

parity:  ## сверка с легаси-чекпоинтом: make parity TRM_LEGACY_CHECKPOINT=/path/to/best_mrr.pt
	TRM_LEGACY_CHECKPOINT=$(TRM_LEGACY_CHECKPOINT) uv run pytest tests/test_parity.py -v

hooks:  ## установить pre-commit хуки
	uv run pre-commit install
