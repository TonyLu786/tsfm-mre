"""
Plots: PIT hist, reliability, VaR exceptions, capital multipliers.

Usage:
python plots.py --indir data/interim --pred outputs/forecasts/pred_ar1.parquet --risk_dir outputs/risk --outdir outputs/figures --alpha 0.01 --horizon 1
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import ensure_dir, LOGGER

def rolling_sum(a, window):
    return a.rolling(window).sum()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default="data/interim")
    ap.add_argument("--pred", type=str, required=True)
    ap.add_argument("--risk_dir", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="outputs/figures")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--horizon", type=int, default=1)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    df_true = pd.read_csv(Path(args.indir)/"h15_processed.csv", parse_dates=["date"])
    df_pred = pd.read_parquet(args.pred)
    bt = pd.read_csv(Path(args.risk_dir)/"var_backtest.csv", parse_dates=["date"])
    # VaR exception plot per series
    for s, gs in bt.groupby("series"):
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(gs["date"], gs["y"], label="Realized")
        ax.plot(gs["date"], gs["VaR"], label=f"VaR {int(100*args.alpha)}%")
        ax.fill_between(gs["date"], gs["VaR"], gs["y"], where=gs["y"]<gs["VaR"], color="red", alpha=0.3, label="Exceptions")
        ax.set_title(f"{s} — VaR Backtest (h={args.horizon})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(args.outdir)/f"var_backtest_{s}.png", dpi=160)
        plt.close(fig)
        # capital multiplier proxy: rolling exceptions
        exc = gs.set_index("date")["exceed"]
        roll = exc.rolling(250).sum()
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(roll.index, roll.values)
        ax.axhline(4, linestyle="--")
        ax.axhline(9, linestyle="--")
        ax.set_title(f"{s} — Rolling 250d exceptions")
        fig.tight_layout()
        fig.savefig(Path(args.outdir)/f"exceptions_rolling_{s}.png", dpi=160)
        plt.close(fig)
    LOGGER.info(f"Saved figures to {args.outdir}")

if __name__ == "__main__":
    main()
