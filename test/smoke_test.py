import os, pandas as pd, numpy as np, subprocess, sys
PY = sys.executable

def run(cmd): subprocess.check_call([PY] + cmd)

def test_offline_pipeline():
    # cls
    for p in ["data/raw", "data/interim", "outputs"]:
        os.makedirs(p, exist_ok=True)
    # cli
    run(["src/data_ingest.py","--outdir","data/raw","--mode","synthetic","--start","2003-01-01","--end","2003-12-31"])
    run(["src/preprocess.py","--indir","data/raw","--outdir","data/interim","--train_start","2003-01-01","--valid_start","2003-06-02","--test_start","2003-09-01","--end","2003-12-31"])
    run(["src/forecast.py","--indir","data/interim","--outdir","outputs/forecasts","--model","ar1","--horizons","1","5","10","--context","60","--B","200"])
    run(["src/risk.py","--indir","data/interim","--pred","outputs/forecasts/pred_ar1.parquet","--outdir","outputs/risk","--alpha","0.01","--horizon","1","--window","120"])
    run(["src/plots.py","--indir","data/interim","--pred","outputs/forecasts/pred_ar1.parquet","--risk_dir","outputs/risk","--outdir","outputs/figures","--alpha","0.01","--horizon","1"])
    run(["src/report.py","--risk_dir","outputs/risk","--fig_dir","outputs/figures","--outdir","outputs/report"])

    # key output
    risk = pd.read_csv("outputs/risk/risk_summary.csv")
    assert {"series","Kupiec_p","CC_p","traffic_light"} <= set(risk.columns)
    bt = pd.read_csv("outputs/risk/var_backtest.csv")
    # st
    preds = pd.read_parquet("outputs/forecasts/pred_ar1.parquet")
    assert np.all(preds["q_0.01"] <= preds["q_0.025"])
    assert np.all(preds["q_0.025"] <= preds["q_0.05"])
