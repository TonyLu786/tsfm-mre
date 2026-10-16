# Contributing

Contributions of models, tests and documentation are welcome.

## Development setup

```bash
git clone https://github.com/TonyLu786/tsfm-mre.git
cd tsfm-mre
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Before opening a pull request, run:

```bash
ruff format .
ruff check .
pytest
```

Continuous integration runs the same checks on Python 3.11 to 3.14.

## Adding a built-in model

1. Subclass `tsfm_mre.models.Forecaster` and implement `min_history`, `forecast` and
   `params`. `forecast(y, origins, horizon)` returns the predictive distributions of
   `y[t + horizon]` for all origins at once and may use only `y[: t + 1]`.
2. Register the class in `MODEL_REGISTRY`.
3. Add the model to `tests/test_models.py`; the look-ahead test applies to every model in
   `MODELS`.

## Adding an external model

External forecasts, for example from a time-series foundation model, need no code
changes: write them in the [prediction schema](docs/prediction_schema.md) and pass the
file to `tsfm-mre run --predictions`.

## Conventions

- Statistical methods come with a reference in the docstring and in
  `docs/methodology.md`, and with a test of their behaviour under the null hypothesis.
- Code follows the formatting and lint rules in `pyproject.toml`.
- Outputs remain deterministic for a given configuration and seed.
