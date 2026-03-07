#!/usr/bin/env bash
set -euo pipefail

PY=${PYTHON:-python}

echo "[1/6] Data ingest (synthetic)..."
$PY src/data_ingest.py --outdir data/raw --mode synthetic --start 2003-01-01 --end 2025-06-30

echo "[2/6] Preprocess..."
$PY src/preprocess.py --indir data/raw --outdir data/interim --train_start 2003-01-01 --valid_start 2013-01-02 --test_start 2019-01-02 --end 2025-06-30

echo "[3/6] Forecast (AR1 baseline)..."
$PY src/forecast.py --indir data/interim --outdir outputs/forecasts --model ar1 --horizons 1 5 10 20 --context 60

PRED=outputs/forecasts/pred_ar1.parquet
if [ ! -f "$PRED" ] && [ -f "outputs/forecasts/pred_ar1.csv" ]; then
  PRED=outputs/forecasts/pred_ar1.csv
fi

echo "[4/6] Risk backtesting..."
$PY src/risk.py --indir data/interim --pred "$PRED" --outdir outputs/risk --alpha 0.01 --horizon 1 --window 250

echo "[5/6] Plots..."
$PY src/plots.py --indir data/interim --pred "$PRED" --risk_dir outputs/risk --outdir outputs/figures --alpha 0.01 --horizon 1 --window 250

echo "[6/6] Report..."
$PY src/report.py --risk_dir outputs/risk --fig_dir outputs/figures --outdir outputs/report

echo "Done. Open outputs/report/index.html"
