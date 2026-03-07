# Model Card: TSFM-MRE Baselines

## Scope
This repository provides a minimal reproducible experiment for financial time-series forecasting and risk backtesting. The default baseline is AR(1); an optional tiny Transformer is available when PyTorch is installed.

## Intended Use
- Educational and research benchmarking.
- Offline reproducible risk diagnostics (VaR-focused backtesting).

## Not Intended Use
- Direct production trading/risk decisions without additional validation.
- Regulated reporting without independent model governance review.

## Data
- Default path uses synthetic H.15-like business-day yield panel.
- Optional public data ingestion from FRED.

## Outputs
- Predictive distributions by origin/horizon/series.
- Risk backtest summaries (Kupiec/Christoffersen/Basel traffic-light).
- Figures and an HTML report.

## Limitations
- Statistical tests are simplified and should be validated for formal submissions.
- Synthetic data does not represent full market microstructure or regime complexity.

## Validation
Use:
- `python run_all.py` for end-to-end pipeline.
- `python test/smoke_test.py` or `pytest -q` for smoke validation.
