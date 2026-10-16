.PHONY: install lint format test smoke reproduce

install:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest

smoke:
	tsfm-mre run configs/smoke.yaml

reproduce:
	tsfm-mre run configs/synthetic.yaml --out results/synthetic
	tsfm-mre run configs/fred_h15.yaml --out results/fred_h15
