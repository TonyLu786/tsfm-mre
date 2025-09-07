
.PHONY: all data preprocess forecast risk plots report clean

PY=python

all: data preprocess forecast risk plots report

data:
    $(PY) src/data_ingest.py --outdir data/raw --mode synthetic --start 2003-01-01 --end 2025-06-30

preprocess:
    $(PY) src/preprocess.py --indir data/raw --outdir data/interim --train_start 2003-01-01 --valid_start 2013-01-02 --test_start 2019-01-02 --end 2025-06-30

forecast:
    $(PY) src/forecast.py --indir data/interim --outdir outputs/forecasts --model ar1 --horizons 1 5 10 20 --context 60

risk:
    $(PY) src/risk.py --indir data/interim --pred outputs/forecasts/pred_ar1.parquet --outdir outputs/risk --alpha 0.01 --horizon 1 --window 250

plots:
    $(PY) src/plots.py --indir data/interim --pred outputs/forecasts/pred_ar1.parquet --risk_dir outputs/risk --outdir outputs/figures --alpha 0.01 --horizon 1

report:
    $(PY) src/report.py --risk_dir outputs/risk --fig_dir outputs/figures --outdir outputs/report

clean:
    rm -rf data/raw/* data/interim/* outputs/*
    mkdir -p outputs/forecasts outputs/risk outputs/figures outputs/report
