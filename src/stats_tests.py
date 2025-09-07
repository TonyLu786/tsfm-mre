"""
Statistical tests: HAC-robust Diebold-Mariano; Reality Check & SPA via dependent bootstrap.
Assumes you have per-origin losses for each model to compare.
For simplicity we compute CRPS-like proxy using quantile losses or provided losses.

Usage:
python stats_tests.py --loss_csv outputs/losses.csv --outdir outputs/stats --benchmark_model ar1
The losses.csv schema:
origin, series, horizon, model, loss
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from utils import ensure_dir, LOGGER

def hac_se(loss_diff: np.ndarray, max_lag: int = None):
    # Newey-West HAC variance for mean
    T = len(loss_diff)
    x = loss_diff - loss_diff.mean()
    if max_lag is None:
        max_lag = int(4 * (T/100)**(2/9)) if T>10 else 1
    gamma0 = np.dot(x, x)/T
    var = gamma0
    for l in range(1, max_lag+1):
        w = 1 - l/(max_lag+1)
        cov = np.dot(x[l:], x[:-l])/T
        var += 2*w*cov
    se = np.sqrt(var/T)
    return se

def dm_test(loss_a: np.ndarray, loss_b: np.ndarray):
    d = loss_a - loss_b
    se = hac_se(d)
    mean = d.mean()
    if se < 1e-12:
        return np.nan, mean, se
    t = mean / se
    # two-sided p-value under approx normal
    from scipy.stats import norm
    p = 2*(1 - norm.cdf(abs(t)))
    return p, mean, se

def stationary_bootstrap_indices(T: int, p: float, B: int):
    rng = np.random.default_rng(42)
    idx = np.zeros((B, T), dtype=int)
    for b in range(B):
        i = rng.integers(0, T)
        for t in range(T):
            idx[b, t] = i
            if rng.random() < p:
                i = rng.integers(0, T)
            else:
                i = (i + 1) % T
    return idx

def reality_check_spa(loss_table: pd.DataFrame, benchmark: str, B: int = 1000, p: float = 0.1):
    """
    loss_table columns: origin, model, loss
    Return RC and SPA p-values per competitor vs benchmark.
    """
    # center by benchmark mean loss
    bench = loss_table[loss_table["model"]==benchmark].groupby("origin")["loss"].mean()
    out = []
    T = len(bench)
    idx = stationary_bootstrap_indices(T, p=p, B=B)
    bench_arr = bench.values
    for model, g in loss_table.groupby("model"):
        if model == benchmark: 
            continue
        d = g.set_index("origin").loc[bench.index, "loss"].values - bench_arr
        # RC p-value: proportion of bootstrap where max(mean(d*)) > mean(d)
        d_center = d - d.mean()
        boot_means = []
        for b in range(B):
            boot_means.append(d_center[idx[b]].mean())
        rc_p = np.mean(np.array(boot_means) >= d.mean())
        # SPA studentized
        std = hac_se(d)
        t_stat = d.mean() / (std + 1e-12)
        boot_t = []
        for b in range(B):
            dd = d_center[idx[b]]
            std_b = dd.std() / np.sqrt(len(dd))
            boot_t.append(dd.mean() / (std_b + 1e-12))
        spa_p = np.mean(np.array(boot_t) >= t_stat)
        out.append({"model": model, "RC_p": float(rc_p), "SPA_p": float(spa_p), "mean_diff": float(d.mean())})
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss_csv", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="outputs/stats")
    ap.add_argument("--benchmark_model", type=str, default="ar1")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--p", type=float, default=0.1, help="stationary bootstrap restart prob")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    losses = pd.read_csv(args.loss_csv, parse_dates=["origin"])
    # DM per horizon & series vs benchmark
    dm_rows = []
    for (series, horizon), g in losses.groupby(["series","horizon"]):
        for model in g["model"].unique():
            if model==args.benchmark_model: continue
            ga = g[g["model"]==model].set_index("origin")["loss"]
            gb = g[g["model"]==args.benchmark_model].set_index("origin")["loss"]
            common = ga.index.intersection(gb.index)
            if len(common)<20: continue
            p, mean, se = dm_test(ga.loc[common].values, gb.loc[common].values)
            dm_rows.append({"series": series, "horizon": horizon, "model": model, "benchmark": args.benchmark_model,
                            "DM_p": p, "mean_diff": mean, "SE": se})
    dm_df = pd.DataFrame(dm_rows)
    dm_df.to_csv(Path(args.outdir)/"dm_results.csv", index=False)
    # RC/SPA pool across horizons & series (average per origin)
    agg = losses.groupby(["origin","model"])["loss"].mean().reset_index()
    rcspa = reality_check_spa(agg, args.benchmark_model, B=args.B, p=args.p)
    rcspa.to_csv(Path(args.outdir)/"rc_spa.csv", index=False)
    LOGGER.info(f"Saved DM and RC/SPA results to {args.outdir}")

if __name__ == "__main__":
    main()
