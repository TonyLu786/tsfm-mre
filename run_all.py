import subprocess, sys, os

PY = os.environ.get("PYTHON", sys.executable)

steps = [
    [PY, "src/data_ingest.py", "--outdir", "data/raw", "--mode", "synthetic", "--start", "2003-01-01", "--end", "2025-06-30"],
    [PY, "src/preprocess.py", "--indir", "data/raw", "--outdir", "data/interim", "--train_start", "2003-01-01", "--valid_start", "2013-01-02", "--test_start", "2019-01-02", "--end", "2025-06-30"],
    [PY, "src/forecast.py", "--indir", "data/interim", "--outdir", "outputs/forecasts", "--model", "ar1", "--horizons", "1", "5", "10", "20", "--context", "60"],
    [PY, "src/risk.py", "--indir", "data/interim", "--pred", "outputs/forecasts/pred_ar1.parquet", "--outdir", "outputs/risk", "--alpha", "0.01", "--horizon", "1", "--window", "250"],
    [PY, "src/plots.py", "--indir", "data/interim", "--pred", "outputs/forecasts/pred_ar1.parquet", "--risk_dir", "outputs/risk", "--outdir", "outputs/figures", "--alpha", "0.01", "--horizon", "1"],
    [PY, "src/report.py", "--risk_dir", "outputs/risk", "--fig_dir", "outputs/figures", "--outdir", "outputs/report"]
]

def main():
    for i, cmd in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] Running:", " ".join(cmd))
        subprocess.check_call(cmd)
    print("Done. Open outputs/report/index.html")

if __name__ == "__main__":
    main()