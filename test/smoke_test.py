import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PY = sys.executable


def run(cmd):
    subprocess.check_call([PY] + cmd)


def resolve_pred_file(model: str = "ar1") -> Path:
    base = Path("outputs/forecasts") / f"pred_{model}"
    candidates = [base.with_suffix(".parquet"), base.with_suffix(".csv")]
    existing = [p for p in candidates if p.exists()]
    if existing:
        return max(existing, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(f"Forecast output not found for model={model}")


def read_pred(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            csv_fallback = path.with_suffix(".csv")
            if csv_fallback.exists():
                return pd.read_csv(csv_fallback)
            raise
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported prediction format: {path}")


def test_offline_pipeline():
    for p in ["data/raw", "data/interim", "outputs"]:
        os.makedirs(p, exist_ok=True)

    run([
        "src/data_ingest.py",
        "--outdir",
        "data/raw",
        "--mode",
        "synthetic",
        "--start",
        "2003-01-01",
        "--end",
        "2003-12-31",
    ])
    run([
        "src/preprocess.py",
        "--indir",
        "data/raw",
        "--outdir",
        "data/interim",
        "--train_start",
        "2003-01-01",
        "--valid_start",
        "2003-06-02",
        "--test_start",
        "2003-09-01",
        "--end",
        "2003-12-31",
    ])
    run([
        "src/forecast.py",
        "--indir",
        "data/interim",
        "--outdir",
        "outputs/forecasts",
        "--model",
        "ar1",
        "--horizons",
        "1",
        "5",
        "10",
        "--context",
        "60",
        "--B",
        "200",
    ])

    pred_path = resolve_pred_file(model="ar1")

    run([
        "src/risk.py",
        "--indir",
        "data/interim",
        "--pred",
        str(pred_path),
        "--outdir",
        "outputs/risk",
        "--alpha",
        "0.01",
        "--horizon",
        "1",
        "--window",
        "120",
    ])
    run([
        "src/plots.py",
        "--indir",
        "data/interim",
        "--pred",
        str(pred_path),
        "--risk_dir",
        "outputs/risk",
        "--outdir",
        "outputs/figures",
        "--alpha",
        "0.01",
        "--horizon",
        "1",
        "--window",
        "120",
    ])
    run([
        "src/report.py",
        "--risk_dir",
        "outputs/risk",
        "--fig_dir",
        "outputs/figures",
        "--outdir",
        "outputs/report",
    ])

    risk = pd.read_csv("outputs/risk/risk_summary.csv")
    assert {"series", "Kupiec_p", "CC_p", "traffic_light"} <= set(risk.columns)

    preds = read_pred(pred_path)
    assert np.all(preds["q_0.01"] <= preds["q_0.025"])
    assert np.all(preds["q_0.025"] <= preds["q_0.05"])


if __name__ == "__main__":
    test_offline_pipeline()
    print("Smoke test passed.")
