.RECIPEPREFIX := >

.PHONY: all data preprocess forecast risk plots report clean

PY ?= python
PRED ?= outputs/forecasts/pred_ar1.parquet
HORIZONS ?= 1 5 10 20
CONTEXT ?= 60
B ?= 1000
ALPHA ?= 0.01
HORIZON ?= 1
WINDOW ?= 250

all: data preprocess forecast risk plots report

data:
>$(PY) src/data_ingest.py --outdir data/raw --mode synthetic --start 2003-01-01 --end 2025-06-30

preprocess:
>$(PY) src/preprocess.py --indir data/raw --outdir data/interim --train_start 2003-01-01 --valid_start 2013-01-02 --test_start 2019-01-02 --end 2025-06-30

forecast:
>$(PY) src/forecast.py --indir data/interim --outdir outputs/forecasts --model ar1 --horizons $(HORIZONS) --context $(CONTEXT) --B $(B)

risk:
>$(PY) src/risk.py --indir data/interim --pred $(PRED) --outdir outputs/risk --alpha $(ALPHA) --horizon $(HORIZON) --window $(WINDOW)

plots:
>$(PY) src/plots.py --indir data/interim --pred $(PRED) --risk_dir outputs/risk --outdir outputs/figures --alpha $(ALPHA) --horizon $(HORIZON) --window $(WINDOW)

report:
>$(PY) src/report.py --risk_dir outputs/risk --fig_dir outputs/figures --outdir outputs/report

clean:
>$(PY) -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['data/raw','data/interim','outputs']]; [Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/raw','data/interim','outputs/forecasts','outputs/risk','outputs/figures','outputs/report']]"
