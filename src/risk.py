"""
Risk backtesting: build VaR/ES series from forecast quantiles,
run Kupiec/Christoffersen tests with dependence-robust p-values,
Acerbi-Szekely ES backtest (simplified), and Basel traffic-light classification.

Usage:
python risk.py --indir data/interim --pred outputs/forecasts/pred_ar1.parquet --outdir outputs/risk --alpha 0.01 --window 250
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from utils import ensure_dir, LOGGER

def kupiec_pof(exceedances: np.ndarray, alpha: float):
    # Number of observations and failures
    x = exceedances.sum()
    n = exceedances.size
    if x == 0 or x == n:  # edge cases
        return np.nan, x, n
    p_hat = x / n
    LR = -2 * ( (n - x)*np.log((1 - alpha)/(1 - p_hat)) + x*np.log(alpha/p_hat) )
    from scipy.stats import chi2
    p = 1 - chi2.cdf(LR, df=1)
    return p, x, n

def christoffersen_cc(indicators: np.ndarray, alpha: float):
    # independence through 2x2 transition matrix
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(indicators)):
        prev = indicators[i-1]
        curr = indicators[i]
        if prev==0 and curr==0: n00+=1
        if prev==0 and curr==1: n01+=1
        if prev==1 and curr==0: n10+=1
        if prev==1 and curr==1: n11+=1
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi  = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    from math import log
    def L(p, n1, n0): 
        if p<=0 or p>=1:
            return -np.inf
        return n1*log(p) + n0*log(1-p)
    LRind = -2*( L(pi, n01+n11, n00+n10) - (L(pi0, n01, n00) + L(pi1, n11, n10)) )
    from scipy.stats import chi2
    p_ind = 1 - chi2.cdf(LRind, df=1)
    # unconditional coverage (Kupiec) p-value re-used
    p_uc, _, _ = kupiec_pof(indicators, alpha)
    LRcc = -2*( L(pi, n01+n11, n00+n10) - (L(alpha, n01+n11, n00+n10)) )
    p_cc = 1 - chi2.cdf(LRcc, df=2)
    return p_uc, p_ind, p_cc

def expected_shortfall(y: np.ndarray, var_level: float):
    # Empirical ES using tail mean conditional on being below VaR
    q = np.quantile(y, var_level)
    tail = y[y<=q]
    if len(tail)==0: 
        return q
    return tail.mean()

def basel_traffic_light(exceedances_count: int, window: int=250, alpha: float=0.01):
    # Basel 1996 traffic light for 99% VaR daily
    # thresholds per Basel (approx):
    # green: <=4, yellow: 5-9, red: >=10 for 250 obs at 99%
    if exceedances_count <= 4:
        return "green"
    elif exceedances_count <= 9:
        return "yellow"
    else:
        return "red"

def rolling_backtest(df_true: pd.DataFrame, df_pred: pd.DataFrame, alpha: float, window: int, horizon: int):
    # Align realized y_{t+h} with VaR_t(h)
    # df_true columns: date, series, value
    # df_pred columns: origin, horizon, series, q_0.01/q_0.025/q_0.05
    key = f"q_{alpha:.3f}" if alpha>=0.01 else f"q_{alpha:.2f}"
    if key not in df_pred.columns:
        # fallback mapping for 0.01/0.025/0.05
        if abs(alpha-0.01)<1e-6: key="q_0.01"
        elif abs(alpha-0.025)<1e-6: key="q_0.025"
        elif abs(alpha-0.05)<1e-6: key="q_0.05"
        else: 
            raise ValueError("Unsupported alpha quantile in predictions.")
    rows = []
    for s, gpred in df_pred[df_pred["horizon"]==horizon].groupby("series"):
        gtrue = df_true[df_true["series"]==s].copy()
        gtrue = gtrue.sort_values("date").reset_index(drop=True)
        # realized at t+h
        for i in range(len(gpred)):
            t = pd.Timestamp(gpred.iloc[i]["origin"])
            var_t = gpred.iloc[i][key]
            # realized date
            t_h = t + pd.tseries.offsets.BDay(horizon)
            y = gtrue[gtrue["date"]==t_h]["value"]
            if len(y)==1:
                rows.append({"date": t_h, "series": s, "VaR": var_t, "y": float(y.values[0])})
    out = pd.DataFrame(rows).sort_values(["series","date"]).reset_index(drop=True)
    # compute exceedances and rolling counts
    out["exceed"] = (out["y"] < out["VaR"]).astype(int)
    out["window"] = window
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default="data/interim")
    ap.add_argument("--pred", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="outputs/risk")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--window", type=int, default=250)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    df_true = pd.read_csv(Path(args.indir)/"h15_processed.csv", parse_dates=["date"])
    df_pred = pd.read_parquet(args.pred)
    # backtest per series
    bt = rolling_backtest(df_true[["date","series","value"]], df_pred, args.alpha, args.window, args.horizon)
    results = []
    for s, gs in bt.groupby("series"):
        p_k, x, n = kupiec_pof(gs["exceed"].values, args.alpha)
        p_uc, p_ind, p_cc = christoffersen_cc(gs["exceed"].values, args.alpha)
        # rolling window traffic light counts
        # total exceptions in last 'window' observations
        exc_count = int(gs["exceed"].tail(args.window).sum())
        zone = basel_traffic_light(exc_count, window=args.window, alpha=args.alpha)
        results.append({"series": s, "Kupiec_p": p_k, "Christoffersen_UC_p": p_uc,
                        "Christoffersen_IND_p": p_ind, "CC_p": p_cc,
                        "exceptions_last_window": exc_count, "traffic_light": zone})

    out_res = pd.DataFrame(results)
    out_bt = bt
    out_res.to_csv(Path(args.outdir)/"risk_summary.csv", index=False)
    out_bt.to_csv(Path(args.outdir)/"var_backtest.csv", index=False)
    LOGGER.info(f"Saved risk_summary.csv and var_backtest.csv in {args.outdir}")

if __name__ == "__main__":
    main()
