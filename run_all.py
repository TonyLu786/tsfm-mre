import argparse
import os
import subprocess
import sys
from pathlib import Path

PY = os.environ.get("PYTHON", sys.executable)

def run_step(step_idx, total_steps, cmd):
    print(f"[{step_idx}/{total_steps}] Running:", " ".join(cmd))
    subprocess.check_call(cmd)

def resolve_prediction_file(model_name="ar1", outdir="outputs/forecasts"):
    base = Path(outdir) / f"pred_{model_name}"
    candidates = [base.with_suffix(".parquet"), base.with_suffix(".csv")]
    existing = [c for c in candidates if c.exists()]
    if existing:
        newest = max(existing, key=lambda p: p.stat().st_mtime)
        return str(newest)
    raise FileNotFoundError(f"Forecast output not found. Tried: {[str(c) for c in candidates]}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default="2003-01-01")
    ap.add_argument("--end", type=str, default="2025-06-30")
    ap.add_argument("--train_start", type=str, default="2003-01-01")
    ap.add_argument("--valid_start", type=str, default="2013-01-02")
    ap.add_argument("--test_start", type=str, default="2019-01-02")
    ap.add_argument("--model", type=str, default="ar1", choices=["ar1", "tiny_tf"])
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20])
    ap.add_argument("--context", type=int, default=60)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--window", type=int, default=250)
    return ap.parse_args()

def main():
    args = parse_args()
    total_steps = 6
    run_step(
        1,
        total_steps,
        [PY, "src/data_ingest.py", "--outdir", "data/raw", "--mode", "synthetic", "--start", args.start, "--end", args.end],
    )
    run_step(
        2,
        total_steps,
        [
            PY,
            "src/preprocess.py",
            "--indir",
            "data/raw",
            "--outdir",
            "data/interim",
            "--train_start",
            args.train_start,
            "--valid_start",
            args.valid_start,
            "--test_start",
            args.test_start,
            "--end",
            args.end,
        ],
    )
    horizons = [str(h) for h in args.horizons]
    run_step(
        3,
        total_steps,
        [
            PY,
            "src/forecast.py",
            "--indir",
            "data/interim",
            "--outdir",
            "outputs/forecasts",
            "--model",
            args.model,
            "--horizons",
            *horizons,
            "--context",
            str(args.context),
            "--B",
            str(args.B),
        ],
    )
    pred_path = resolve_prediction_file(model_name=args.model, outdir="outputs/forecasts")
    run_step(
        4,
        total_steps,
        [
            PY,
            "src/risk.py",
            "--indir",
            "data/interim",
            "--pred",
            pred_path,
            "--outdir",
            "outputs/risk",
            "--alpha",
            str(args.alpha),
            "--horizon",
            str(args.horizon),
            "--window",
            str(args.window),
        ],
    )
    run_step(
        5,
        total_steps,
        [
            PY,
            "src/plots.py",
            "--indir",
            "data/interim",
            "--pred",
            pred_path,
            "--risk_dir",
            "outputs/risk",
            "--outdir",
            "outputs/figures",
            "--alpha",
            str(args.alpha),
            "--horizon",
            str(args.horizon),
            "--window",
            str(args.window),
        ],
    )
    run_step(
        6,
        total_steps,
        [PY, "src/report.py", "--risk_dir", "outputs/risk", "--fig_dir", "outputs/figures", "--outdir", "outputs/report"],
    )
    print("Done. Open outputs/report/index.html")

if __name__ == "__main__":
    main()
