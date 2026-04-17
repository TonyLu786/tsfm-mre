# TSFM-MRE

A minimal, reproducible evaluation pipeline for time-series forecasting models in finance. The repository is designed for offline-first experiments with probabilistic forecasts, risk backtesting, and audit-friendly reporting.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen)
![Status](https://img.shields.io/badge/status-research%20prototype-informational)

## Overview

TSFM-MRE provides a compact end-to-end workflow for evaluating forecasting models under a setting that is closer to practical financial model review than to a standard point-forecast benchmark. It focuses on three objectives:

- generating distributional forecasts rather than only point estimates;
- testing risk relevance through VaR and ES backtesting;
- preserving reproducibility through deterministic splits, explicit artifacts, and offline execution.

The default configuration runs entirely on synthetic H.15-like data, so the full pipeline can be reproduced without network access. When public data access is available, the same workflow can be run on FRED data with a separate ingestion step.

## Main capabilities

- Offline synthetic data pipeline for fully local reproduction.
- Optional public-data pipeline based on FRED.
- Rolling-origin evaluation on a business-day calendar.
- Probabilistic forecast output with means and quantiles.
- VaR backtesting with Kupiec and Christoffersen diagnostics.
- Basel-style traffic-light reporting based on rolling exception counts.
- Portable outputs in Parquet or CSV, plus figures and an HTML report.
- Metadata and hash files for basic audit and reproducibility checks.

## Repository structure

```text
tsfm-mre/
  src/
    data_ingest.py    # synthetic/FRED ingestion
    preprocess.py     # alignment, scaling, and train/valid/test splits
    forecast.py       # forecasting entry point
    risk.py           # VaR backtesting, traffic-light summary, ES diagnostics
    plots.py          # figures for exceptions and rolling counts
    report.py         # HTML report generation
    stats_tests.py    # optional comparative testing utilities
  data/               # raw and interim data artifacts
  outputs/            # forecasts, risk tables, figures, and reports
  run_all.py          # one-command offline pipeline
  run_all.sh          # optional shell runner
  Makefile
  requirements.txt
  environment.yml
  LICENSE
```

## Installation

### Option 1: pip

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option 2: conda

```bash
conda env create -f environment.yml
conda activate tsfm-mre
```

## Quick start

### Offline pipeline (default)

This is the recommended path for a first run. It requires no internet connection.

```bash
python run_all.py
```

A smaller smoke test can be run with a shorter date range and fewer bootstrap samples:

```bash
python run_all.py \
  --start 2003-01-01 \
  --end 2003-12-31 \
  --train_start 2003-01-01 \
  --valid_start 2003-06-02 \
  --test_start 2003-09-01 \
  --horizons 1 5 10 \
  --B 200 \
  --window 120
```

### Public-data pipeline (FRED)

When internet access is available, the workflow can be run on public macro-financial data:

```bash
python src/data_ingest.py --outdir data/raw --mode fred --start 2003-01-01 --end 2025-06-30
python src/preprocess.py --indir data/raw --outdir data/interim --train_start 2003-01-01 --valid_start 2013-01-02 --test_start 2019-01-02 --end 2025-06-30
python src/forecast.py   --indir data/interim --outdir outputs/forecasts --model ar1 --horizons 1 5 10 20 --context 60
python src/risk.py       --indir data/interim --pred outputs/forecasts/pred_ar1.parquet --outdir outputs/risk --alpha 0.01 --horizon 1 --window 250
python src/plots.py      --indir data/interim --pred outputs/forecasts/pred_ar1.parquet --risk_dir outputs/risk --outdir outputs/figures --alpha 0.01 --horizon 1
python src/report.py     --risk_dir outputs/risk --fig_dir outputs/figures --outdir outputs/report
```

## Outputs

After a successful run, the main artifacts are written under `outputs/`.

| Path | Description |
| --- | --- |
| `outputs/forecasts/pred_*.parquet` | Predictive distributions by origin, horizon, and series; CSV fallback may be used when Parquet support is unavailable. |
| `outputs/risk/risk_summary.csv` | Backtesting summary including p-values, exception counts, and traffic-light classification. |
| `outputs/risk/var_backtest.csv` | Realized values, VaR thresholds, and exceedance indicators used in backtesting. |
| `outputs/figures/*.png` | Figures for VaR overlays and rolling exception counts. |
| `outputs/report/index.html` | Standalone HTML summary for inspection, sharing, or review. |

The pipeline also writes metadata files that record split boundaries and hashes of selected outputs for reproducibility checks.

## Evaluation design

The repository uses a rolling-origin evaluation design on a business-day calendar, with forecast horizons typically set to `1, 5, 10, 20`.

### Forecasting

The baseline workflow supports per-series forecasting with distributional outputs. The default prediction files store at least the following fields:

- `origin`
- `horizon`
- `series`
- `mean`
- `q_0.01`
- `q_0.025`
- `q_0.05`

An AR(1) specification is available as a baseline. Additional model paths can be integrated as long as they write predictions in the same schema.

### Risk backtesting

For each forecast origin and horizon, realized values are aligned to the corresponding business-day target date. The risk module reports:

- VaR exceedance counts;
- Kupiec proportion-of-failures test results;
- Christoffersen independence and conditional coverage diagnostics;
- Basel-style traffic-light classification based on rolling exception windows;
- a lightweight ES-oriented summary.

### Comparative inference

The repository also includes optional utilities for model comparison, including horizon-level loss comparisons and dependence-robust resampling procedures. These components are intended for comparative studies rather than the minimum offline run.

## Reproducibility principles

This project is structured around a small set of reproducibility conventions:

- offline-first execution for the default workflow;
- deterministic date splits and seed control;
- train-only scaling to reduce look-ahead risk;
- explicit file outputs rather than hidden state;
- metadata and hashes for result verification.

The intent is not to provide a large benchmarking framework, but a compact and inspectable experiment that can be reviewed, rerun, and extended with minimal overhead.

## Extending the pipeline

To add a new forecasting model:

1. Generate predictive outputs for each `origin × horizon × series` combination.
2. Save the prediction file with the same schema used by `forecast.py`.
3. Pass that file to `risk.py` and `plots.py` without changing the downstream interfaces.

This interface is deliberately simple so that model development and evaluation remain loosely coupled.

## FAQ

### Can the repository be run without internet access?

Yes. The default workflow is fully offline and uses synthetic data.

### Where should I look first after a successful run?

Start with `outputs/report/index.html` for a human-readable summary, then inspect `outputs/risk/risk_summary.csv` for the main backtesting results.

### What happens if Parquet support is unavailable?

The pipeline can fall back to CSV output for prediction files.

### Does the repository depend on a specific forecasting model?

No. The evaluation code is designed so that new models can be added as long as they export predictions in the expected schema.

## Citation

If you use this repository in academic work, please cite it as software:

```bibtex
@software{tsfm_mre_2025,
  title  = {TSFM-MRE: Minimal Reproducible Experiment for Time-Series Foundation Models in Finance},
  author = {Lu, Tony and Contributors},
  year   = {2025},
  url    = {https://github.com/TonyLu786/tsfm-mre}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
