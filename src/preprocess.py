"""
Preprocess: business-day alignment, robust scaling, splits.
Generates processed panel and a scalers JSON.
Usage:
python preprocess.py --indir data/raw --outdir data/interim --train_start 2003-01-01 --valid_start 2013-01-02 --test_start 2019-01-02 --end 2025-06-30
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from utils import ensure_dir, robust_zscore_fit, robust_zscore_transform, save_json, sha256_of_dataframe, LOGGER

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default="data/raw")
    ap.add_argument("--outdir", type=str, default="data/interim")
    ap.add_argument("--train_start", type=str, default="2003-01-01")
    ap.add_argument("--valid_start", type=str, default="2013-01-02")
    ap.add_argument("--test_start", type=str, default="2019-01-02")
    ap.add_argument("--end", type=str, default="2025-06-30")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    panel = pd.read_csv(Path(args.indir)/"h15_panel.csv", parse_dates=["date"])
    # Reindex per series to business days in [train_start, end]
    dates = pd.bdate_range(args.train_start, args.end, freq="C")
    full = []
    scalers = {}
    for s, grp in panel.groupby("series"):
        g = grp.set_index("date").reindex(dates).sort_index()
        g["series"] = s
        g["value"] = g["value"].interpolate(limit_direction="both")
        # robust scaler fit on training-only window
        train_mask = (g.index >= args.train_start) & (g.index < args.valid_start)
        params = robust_zscore_fit(g.loc[train_mask, "value"])
        scalers[s] = params
        g["value_z"] = robust_zscore_transform(g["value"], params)
        g = g.reset_index().rename(columns={"index":"date"})
        full.append(g[["date","series","value","value_z"]])
    proc = pd.concat(full, ignore_index=True)
    proc = proc[(proc["date"] >= args.train_start) & (proc["date"] <= args.end)].copy()
    proc = proc.sort_values(["series","date"]).reset_index(drop=True)
    out_csv = Path(args.outdir)/"h15_processed.csv"
    proc.to_csv(out_csv, index=False)
    meta = {
        "train_start": args.train_start,
        "valid_start": args.valid_start,
        "test_start": args.test_start,
        "end": args.end,
        "hash": sha256_of_dataframe(proc)
    }
    save_json(scalers, Path(args.outdir)/"scalers.json")
    save_json(meta, Path(args.outdir)/"processed_meta.json")
    LOGGER.info(f"Wrote {out_csv} with {len(proc)} rows")
    LOGGER.info(f"Scalers saved; meta={meta}")

if __name__ == "__main__":
    main()
