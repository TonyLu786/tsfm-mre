````markdown
# TSFM-MRE — Minimal Reproducible Experiment for Time-Series Foundation Models in Finance

> **One-line:** Audit-ready, one-click **offline** evaluation of Time-Series Foundation Models (TSFMs) with **distributional forecasts**, **risk backtesting** (VaR/ES, Basel traffic-light), and **dependence-robust** inference — all wrapped in a compact, readable pipeline.

[Getting Started](#quickstart) · [What It Produces](#what-gets-produced) · [Evaluation Protocol](#evaluation-protocol--risk-backtesting) · [Reproducibility](#reproducibility--governance) · [Repo Layout](#repository-layout) · [Advanced Usage](#advanced-usage) · [FAQ](#faq--troubleshooting)

![Build](https://img.shields.io/github/actions/workflow/status/<OWNER>/<REPO>/ci.yml?branch=main&label=CI)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen)
![Reproducibility](https://img.shields.io/badge/reproducible-one--click-success)

> Replace `<OWNER>/<REPO>` with your GitHub path (e.g., `TonyLu786/tsfm-mre`) to activate the CI badge.

---

## Why this repository?

Financial forecasting is only useful when **distributional accuracy**, **risk relevance**, and **statistical validity** travel together. This repository enforces that contract end-to-end under **offline-first** constraints:

- **Distribution →** calibrated predictive distributions (means and quantiles across horizons).
- **Risk →** supervisory backtests (VaR, ES; Kupiec/Christoffersen; Basel traffic-light).
- **Inference →** multiple-comparison-safe model testing with dependence-robust resampling.

Everything runs with a **single command** on **synthetic H.15-like** data (no network), and can be switched to **FRED** when public data access is allowed. Outputs are small, explicit files (Parquet/CSV/PNG/HTML) designed for peer review and audit.

---

## Quickstart

> Choose **one** of the paths below. The offline synthetic path is the default and requires no internet access.

### Path A — Offline synthetic (default)

```bash
# 1) (optional) Create and activate a virtual env
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) One-click run
python run_all.py

# Optional quick smoke run (smaller date span / fewer samples)
python run_all.py --start 2003-01-01 --end 2003-12-31 --train_start 2003-01-01 --valid_start 2003-06-02 --test_start 2003-09-01 --horizons 1 5 10 --B 200 --window 120
```

### Path B — Public data via FRED (optional)

```bash
# Requires internet + pandas_datareader
python src/data_ingest.py --outdir data/raw --mode fred --start 2003-01-01 --end 2025-06-30
python src/preprocess.py --indir data/raw --outdir data/interim --train_start 2003-01-01 --valid_start 2013-01-02 --test_start 2019-01-02 --end 2025-06-30
python src/forecast.py   --indir data/interim --outdir outputs/forecasts --model ar1 --horizons 1 5 10 20 --context 60
python src/risk.py       --indir data/interim --pred outputs/forecasts/pred_ar1.parquet --outdir outputs/risk --alpha 0.01 --horizon 1 --window 250
python src/plots.py      --indir data/interim --pred outputs/forecasts/pred_ar1.parquet --risk_dir outputs/risk --outdir outputs/figures --alpha 0.01 --horizon 1
python src/report.py     --risk_dir outputs/risk --fig_dir outputs/figures --outdir outputs/report
```

**Conda alternative**

```bash
conda env create -f environment.yml
conda activate tsfm-mre
python run_all.py
```

---

## What gets produced

After a successful run you will find the following under `outputs/`:

| Path                               | What it contains                                                                   | How it’s used                                       |
| ---------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------- |
| `outputs/forecasts/pred_*.parquet` (or `.csv` fallback) | Predictive distributions per **origin × horizon × series** (mean, q1%, q2.5%, q5%) | Consumed by risk backtesting and calibration checks |
| `outputs/risk/risk_summary.csv`    | Kupiec/Christoffersen p-values, exception counts, Basel traffic-light zone         | Reported as supervisory diagnostics                 |
| `outputs/risk/var_backtest.csv`    | Realized vs VaR alignment and exceedance indicators                                | Drives exception overlays and rolling counts        |
| `outputs/figures/*.png`            | VaR overlays, rolling 250-day exception counts                                     | Included in the HTML report                         |
| `outputs/report/index.html`        | A portable, human-readable summary                                                 | Attach to reviews/audits; no local tooling required |

The pipeline also emits small **metadata JSON** files with split boundaries and **SHA-256 hashes** of canonical tables so numerical identity can be verified.

---

## Evaluation protocol & risk backtesting

The pipeline implements a **rolling-origin**, business-day evaluation for horizons $h \in \{1,5,10,20\}$:

* **Forecasting.** A per-series **AR(1)** baseline fits residual variance and simulates predictive draws; an optional **tiny Transformer** uses MC dropout for draws when `torch` is available. Forecasts are serialized as **Parquet** (or **CSV fallback** if Parquet engine is unavailable) with origin, horizon, series, mean and quantiles.
* **Proper scoring (optional extension).** CRPS/Pinball can be computed from the stored draws or quantiles (utilities available in extensions).
* **Risk backtesting.** For each origin and horizon, the realized value at $t{+}h$ is aligned on the **business-day calendar**. We test **VaR** exceedances (1% by default) via **Kupiec’s proportion-of-failures** and **Christoffersen’s independence/conditional coverage**, and we compute a Basel-style **traffic-light** zone from 250-day exception counts. A lightweight **ES** estimator is included for completeness.
* **Model comparison (optional extension).** When comparing multiple candidates, we provide HAC-robust **Diebold–Mariano** tests per horizon/series and pooled **Reality Check / SPA** using the **stationary bootstrap**.

The default outputs focus on risk-relevant diagnostics and figures that are easy to interpret by quantitative risk teams and supervisors.

---

## Reproducibility & governance

* **Offline-first.** The default synthetic path requires no network access; the FRED path is isolated behind a single flag and records the source/time bounds in metadata.
* **Determinism.** Seeds are fixed; training-only normalization is used to avoid look-ahead; file-level **SHA-256** hashes are persisted.
* **Audit-ready artifacts.** All claims flow from small, text-oriented artifacts (Parquet/CSV/PNG/HTML). The **HTML report** lists horizons, test levels, p-values, exception counts, and summary figures.
* **Open licensing.** MIT license to encourage inspection and reuse.
* **(Optional) CI.** If you enable `.github/workflows/ci.yml`, each push/PR can run the **offline** pipeline and upload report/figures/risk/forecasts as downloadable artifacts for reviewers.

---

## Repository layout

```text
tsfm-mre/
  src/
    data_ingest.py    # synthetic/FRED ingestion → panel with business-day index
    preprocess.py     # alignment + robust scaling (fit on train only) + splits
    forecast.py       # AR(1) baseline (+ tiny Transformer if torch is installed)
    risk.py           # VaR backtesting (Kupiec/Christoffersen) + traffic-light + ES
    plots.py          # exception overlays and rolling 250d counts
    report.py         # bundles an audit-ready HTML summary
    stats_tests.py    # optional DM / Reality Check / SPA comparisons
  data/               # generated data (raw/interim)
  outputs/            # forecasts / risk / figures / report artifacts
  run_all.py          # one-click runner for the full pipeline
  run_all.sh          # shell runner (optional)
  Makefile            # make all
  requirements.txt    # pip dependencies
  environment.yml     # conda environment
  LICENSE             # MIT
```

---

## Advanced usage

### Switch horizons / context window

```bash
python src/forecast.py --indir data/interim --outdir outputs/forecasts \
  --model ar1 --horizons 1 5 10 --context 60
```

### Change backtesting level and horizon

```bash
python src/risk.py --indir data/interim --pred outputs/forecasts/pred_ar1.parquet \
  --outdir outputs/risk --alpha 0.025 --horizon 5 --window 250
```

### Add a new forecasting model

1. Produce predictive draws or quantiles per **origin × horizon × series**.
2. Serialize to the **same prediction schema** used by `forecast.py` (columns: `origin, horizon, series, mean, q_0.01, q_0.025, q_0.05` at minimum).
3. Point `risk.py` and `plots.py` at your prediction file (`.parquet` or `.csv`); the rest of the pipeline remains unchanged.

### (Optional) Statistical tests for model comparison

If you include the statistical testing utilities, provide a per-origin **loss table** with columns `origin, series, horizon, model, loss` and run HAC-robust **Diebold–Mariano** per horizon/series. For pooled evaluation across horizons/series, run **Reality Check / SPA** with the **stationary bootstrap**. See `src/stats_tests.py` for a runnable reference implementation.

---

## FAQ & Troubleshooting

**Q: I have no internet access — can I still reproduce the paper’s figures?**
Yes. Use the **offline** synthetic path (`python run_all.py`). All figures and tables are produced locally.

**Q: I don’t have `statsmodels` or `torch`.**
The AR(1) baseline uses `statsmodels` when available; a conservative fallback is used otherwise. The tiny Transformer is optional and only runs if `torch` is installed.

**Q: Where are the results?**
Everything is under `outputs/` — forecasts (Parquet), risk tables (CSV), figures (PNG), and the HTML report.

**Q: Can I run this in CI or a container?**
Yes. The offline path is CI-friendly. A minimal Dockerfile can be added to call `python run_all.py` and mount `outputs/` to the host.

---

## Citation

If this repository supports your research, please cite it:

```bibtex
@software{tsfm_mre_2025,
  title        = {TSFM-MRE: Minimal Reproducible Experiment for Time-Series Foundation Models in Finance},
  author       = {Lu, Tony and Contributors},
  year         = {2025},
  url          = {https://github.com/TonyLu786/tsfm-mre},
  note         = {Offline-capable, audit-ready TSFM evaluation pipeline}
}
```

---

## License

This project is released under the **MIT License**. See [`LICENSE`](./LICENSE) for details.

---

## Changelog (high level)

* **v0.1.0** — Public MRE with offline synthetic path, AR(1) baseline, VaR backtesting, Basel traffic-light figures, and HTML report. Optional tiny Transformer and statistical tests for comparative studies.

```
::contentReference[oaicite:0]{index=0}
```
