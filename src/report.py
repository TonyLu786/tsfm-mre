"""
Generate a simple HTML report bundling key tables and plots.

Usage:
python report.py --risk_dir outputs/risk --fig_dir outputs/figures --outdir outputs/report
"""
import argparse
from pathlib import Path
import pandas as pd
from utils import ensure_dir, LOGGER

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk_dir", type=str, default="outputs/risk")
    ap.add_argument("--fig_dir", type=str, default="outputs/figures")
    ap.add_argument("--outdir", type=str, default="outputs/report")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    risk_summary = pd.read_csv(Path(args.risk_dir)/"risk_summary.csv")
    html = ["<html><head><meta charset='utf-8'><title>TSFM-MRE Report</title></head><body>"]
    html.append("<h1>TSFM-MRE Risk Backtesting Summary</h1>")
    html.append(risk_summary.to_html(index=False))
    # images
    html.append("<h2>Figures</h2>")
    for p in sorted(Path(args.fig_dir).glob("*.png")):
        html.append(f"<div><h3>{p.name}</h3><img src='../figures/{p.name}' style='max-width:900px'></div>")
    html.append("</body></html>")
    out = Path(args.outdir)/"index.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    LOGGER.info(f"Wrote report to {out}")

if __name__ == "__main__":
    main()
