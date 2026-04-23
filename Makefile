SHELL := bash
PY := python

.PHONY: help install lint test ci compose-up compose-down serve seed eval airflow-up clean

help:
	@echo "install     - install dev deps"
	@echo "lint        - ruff + black + yamllint"
	@echo "test        - pytest hermetic unit suite"
	@echo "ci          - full CI pipeline (matches GH Actions)"
	@echo "compose-up  - start postgres+pgvector + qdrant + langfuse + ollama"
	@echo "serve       - uvicorn src.rag.api.server:app"
	@echo "seed        - ingest sample docs"
	@echo "eval        - run golden-set eval"
	@echo "airflow-up  - start Airflow with ingest DAGs"
	@echo "clean       - remove build artifacts"

install:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements-dev.txt

lint:
	ruff check .
	black --check .
	yamllint .

test:
	pytest tests/unit -v

ci: lint test

compose-up:
	docker compose up -d

compose-down:
	docker compose down -v

serve:
	uvicorn src.rag.api.server:app --reload --port 8000

seed:
	bash scripts/seed-docs.sh

eval:
	bash scripts/eval-run.sh

airflow-up:
	docker compose --profile airflow up -d

clean:
	rm -rf .pytest_cache .ruff_cache build dist htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
