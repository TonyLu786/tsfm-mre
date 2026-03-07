"""
Risk backtesting: build VaR/ES series from forecast quantiles,
run Kupiec/Christoffersen tests with dependence-robust p-values,
Acerbi-Szekely ES backtest (simplified), and Basel traffic-light classification.

Usage:
python risk.py --indir data/interim --pred outputs/forecasts/pred_ar1.parquet --outdir outputs/risk --alpha 0.01 --window 250
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2

from utils import ensure_dir, LOGGER, read_prediction_table

EPS = 1e-12


def _safe_loglik_bernoulli(p: float, n1: int, n0: int) -> float:
    p = float(np.clip(p, EPS, 1.0 - EPS))
    return n1 * np.log(p) + n0 * np.log(1.0 - p)


def kupiec_lr(exceedances: np.ndarray, alpha: float):
    x = int(np.sum(exceedances))
    n = int(exceedances.size)
    if n == 0:
        return np.nan, x, n
    p_hat = x / n
    ll0 = _safe_loglik_bernoulli(alpha, x, n - x)
    ll1 = _safe_loglik_bernoulli(p_hat, x, n - x)
    lr = -2.0 * (ll0 - ll1)
    return float(lr), x, n

def kupiec_pof(exceedances: np.ndarray, alpha: float):
    lr, x, n = kupiec_lr(exceedances, alpha)
    if np.isnan(lr):
        return np.nan, x, n
    return float(1.0 - chi2.cdf(lr, df=1)), x, n

def christoffersen_cc(indicators: np.ndarray, alpha: float):
    if len(indicators) < 2:
        return np.nan, np.nan, np.nan

    # Independence through 2x2 transition matrix.
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(indicators)):
        prev = indicators[i - 1]
        curr = indicators[i]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll_ind_null = _safe_loglik_bernoulli(pi, n01 + n11, n00 + n10)
    ll_ind_alt = _safe_loglik_bernoulli(pi0, n01, n00) + _safe_loglik_bernoulli(pi1, n11, n10)
    lr_ind = -2.0 * (ll_ind_null - ll_ind_alt)
    p_ind = float(1.0 - chi2.cdf(lr_ind, df=1))

    lr_uc, _, _ = kupiec_lr(indicators, alpha)
    p_uc, _, _ = kupiec_pof(indicators, alpha)
    if np.isnan(lr_uc):
        return p_uc, p_ind, np.nan
    lr_cc = lr_uc + lr_ind
    p_cc = float(1.0 - chi2.cdf(lr_cc, df=2))
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

def pick_quantile_column(df_pred: pd.DataFrame, alpha: float) -> str:
    quantile_cols = {}
    for col in df_pred.columns:
        if not col.startswith("q_"):
            continue
        try:
            quantile_cols[col] = float(col.split("_", 1)[1])
        except ValueError:
            continue
    if not quantile_cols:
        raise ValueError("Prediction table has no quantile columns like q_0.01.")
    best_col, best_alpha = min(quantile_cols.items(), key=lambda item: abs(item[1] - alpha))
    if abs(best_alpha - alpha) > 1e-9:
        available = ", ".join(sorted(quantile_cols))
        raise ValueError(f"No exact quantile for alpha={alpha}. Available: {available}")
    return best_col

def rolling_backtest(df_true: pd.DataFrame, df_pred: pd.DataFrame, alpha: float, window: int, horizon: int):
    # Align realized y_{t+h} with VaR_t(h)
    # df_true columns: date, series, value
    # df_pred columns: origin, horizon, series, q_0.01/q_0.025/q_0.05
    key = pick_quantile_column(df_pred, alpha)
    pred = df_pred[df_pred["horizon"] == horizon][["origin", "series", key]].copy()
    pred["date"] = pred["origin"] + pd.to_timedelta(0, unit="D")
    pred["date"] = [d + pd.tseries.offsets.BDay(horizon) for d in pred["date"]]
    pred = pred.rename(columns={key: "VaR"})
    true_df = df_true.rename(columns={"value": "y"})
    out = pred.merge(true_df[["date", "series", "y"]], on=["date", "series"], how="inner")
    out = out.sort_values(["series", "date"]).reset_index(drop=True)
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
    df_pred, pred_path = read_prediction_table(args.pred)
    LOGGER.info(f"Using prediction file: {pred_path}")
    # backtest per series
    bt = rolling_backtest(df_true[["date","series","value"]], df_pred, args.alpha, args.window, args.horizon)
    if bt.empty:
        raise RuntimeError(
            "Backtest alignment produced zero rows. Check prediction horizons and available truth dates."
        )
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
