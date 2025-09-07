"""
Data ingestion script for the TSFM-MRE pipeline.

Modes:
- synthetic (default): generate H.15-like yields for DGS3MO, DGS2, DGS10 with crisis slices.
- csv: read user-provided CSVs with ['date','value'] per series from a folder.
- fred: attempt to download from FRED via pandas_datareader if available.

Usage:
python data_ingest.py --outdir data/raw --mode synthetic --start 2003-01-01 --end 2025-06-30
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from utils import ensure_dir, business_days, sha256_of_dataframe, save_json, LOGGER, set_seed

SERIES = ["DGS3MO","DGS2","DGS10"]

def gen_synthetic_h15(start, end, seed=42):
    set_seed(seed)
    dates = business_days(start, end)
    n = len(dates)
    # latent factors: level, slope, curvature (Nelson-Siegel-ish dynamics)
    rng = np.random.default_rng(seed)
    level = np.zeros(n); slope = np.zeros(n); curv = np.zeros(n)
    epsL = rng.normal(0, 0.02, n)
    epsS = rng.normal(0, 0.03, n)
    epsC = rng.normal(0, 0.02, n)
    for t in range(1, n):
        level[t] = 0.98*level[t-1] + epsL[t]
        slope[t] = 0.96*slope[t-1] + epsS[t]
        curv[t]  = 0.95*curv[t-1]  + epsC[t]
    # maturities (in years)
    tau = { "DGS3MO": 0.25, "DGS2": 2.0, "DGS10": 10.0 }
    def yield_from_factors(L,S,C,mat):
        # simple affine combination
        return 0.02 + 0.7*L + (-0.6)*S/(1+mat) + 0.5*C*np.exp(-mat/7)
    data = []
    for name, m in tau.items():
        y = yield_from_factors(level, slope, curv, m)
        # crisis shocks
        crisis = np.zeros(n)
        # 2008: spikes
        crisis[(dates >= "2008-09-01") & (dates <= "2008-12-31")] = rng.normal(0, 0.08, sum((dates >= "2008-09-01") & (dates <= "2008-12-31")))
        # 2020: covid shock
        crisis[(dates >= "2020-03-01") & (dates <= "2020-04-30")] += rng.normal(0, 0.10, sum((dates >= "2020-03-01") & (dates <= "2020-04-30")))
        y = y + crisis
        df = pd.DataFrame({"date": dates, "series": name, "value": y})
        data.append(df)
    out = pd.concat(data, ignore_index=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="data/raw")
    ap.add_argument("--mode", type=str, default="synthetic", choices=["synthetic","csv","fred"])
    ap.add_argument("--start", type=str, default="2003-01-01")
    ap.add_argument("--end", type=str, default="2025-06-30")
    ap.add_argument("--csv_dir", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    if args.mode == "synthetic":
        df = gen_synthetic_h15(args.start, args.end, args.seed)
    elif args.mode == "csv":
        if not args.csv_dir:
            raise ValueError("csv mode requires --csv_dir pointing to files named DGS3MO.csv, DGS2.csv, DGS10.csv")
        data = []
        for s in SERIES:
            p = Path(args.csv_dir)/f"{s}.csv"
            if not p.exists():
                raise FileNotFoundError(p)
            tmp = pd.read_csv(p)
            tmp.columns = [c.lower() for c in tmp.columns]
            assert "date" in tmp.columns and "value" in tmp.columns
            tmp["date"] = pd.to_datetime(tmp["date"])
            tmp = tmp.sort_values("date")
            tmp["series"] = s
            data.append(tmp[["date","series","value"]])
        df = pd.concat(data, ignore_index=True)
    else:  # fred
        try:
            import pandas_datareader.data as web
        except Exception as e:
            raise RuntimeError("pandas_datareader not available; install to use fred mode.") from e
        data = []
        for s in SERIES:
            sdata = web.DataReader(s, "fred", args.start, args.end)
            sdata = sdata.reset_index().rename(columns={s:"value","DATE":"date"})
            sdata["series"] = s
            data.append(sdata[["date","series","value"]])
        df = pd.concat(data, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["series","date"]).reset_index(drop=True)

    out_csv = Path(args.outdir)/"h15_panel.csv"
    df.to_csv(out_csv, index=False)
    meta = {
        "mode": args.mode,
        "start": args.start,
        "end": args.end,
        "series": SERIES,
        "rows": int(len(df)),
        "hash": sha256_of_dataframe(df)
    }
    save_json(meta, Path(args.outdir)/"h15_meta.json")
    LOGGER.info(f"Wrote {out_csv} with {len(df)} rows")
    LOGGER.info(f"Meta: {meta}")

if __name__ == "__main__":
    main()
