"""
Forecasting script producing distributional forecasts for multiple horizons
via simple AR(1) (per series) and a tiny Transformer (optional, PyTorch).

Outputs a Parquet with columns:
[origin, horizon, series, mean, q_0.01, q_0.025, q_0.05, samples_json]

Usage:
python forecast.py --indir data/interim --outdir outputs/forecasts --model ar1 --horizons 1 5 10 20 --context 60
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
from utils import ensure_dir, LOGGER, robust_zscore_fit, inverse_robust_zscore
from utils import set_seed
import warnings

try:
    import statsmodels.api as sm
except Exception:
    sm = None

def ar1_forecast_dist(y: np.ndarray, h: int, B: int = 1000, rng=None):
    """
    Fit AR(1): y_t = c + phi*y_{t-1} + e_t, e ~ N(0, sigma^2)
    Return bootstrap predictive samples for y_{t+h}.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if sm is None:
        # Fallback: naive RW with Gaussian noise
        mean = y[-1]
        sigma = np.std(np.diff(y[-50:])) + 1e-6
        return rng.normal(mean, sigma*np.sqrt(h), size=B)
    y_lag = y[:-1]
    y_curr = y[1:]
    X = sm.add_constant(y_lag)
    model = sm.OLS(y_curr, X).fit()
    c, phi = model.params
    resid = model.resid
    sigma = resid.std(ddof=1) + 1e-8
    # simulate recursively
    sims = np.empty(B)
    for b in range(B):
        yt = y[-1]
        for _ in range(h):
            eps = rng.normal(0, sigma)
            yt = c + phi*yt + eps
        sims[b] = yt
    return sims

def tiny_transformer_forecast(y: np.ndarray, h: int, B: int=1000, epochs: int=5, context: int=60, rng=None):
    """
    Minimal PyTorch transformer forecaster (if torch available).
    Trains quickly on last 1000 points, returns MC-dropout samples as proxy.
    """
    try:
        import torch
        import torch.nn as nn
    except Exception:
        # fallback to AR1
        return ar1_forecast_dist(y, h, B=B, rng=rng)
    device = "cpu"
    torch.manual_seed(42)
    y_t = torch.tensor(y[-max(1000, context):], dtype=torch.float32).to(device)

    class TinyTF(nn.Module):
        def __init__(self, d_model=32, nhead=4, nlayers=2, dropout=0.1):
            super().__init__()
            self.embed = nn.Linear(1, d_model)
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
            self.head = nn.Linear(d_model, 1)
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x):
            z = self.embed(x)
            z = self.enc(z)
            z = self.dropout(z)
            out = self.head(z)
            return out

    model = TinyTF().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # build supervised windows
    seq = y_t.unsqueeze(-1)  # [T,1]
    X = []
    Y = []
    T = seq.shape[0]
    for t in range(context, T-1):
        X.append(seq[t-context:t, :])
        Y.append(seq[t, :])
    X = torch.stack(X, dim=0)
    Y = torch.stack(Y, dim=0)

    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        opt.step()

    # MC-dropout sampling for h steps
    model.train()  # keep dropout on
    import torch.nn.functional as F
    last = seq[-context:, :].unsqueeze(0)  # [1,context,1]
    samples = []
    for b in range(B):
        cur = last.clone()
        yt = None
        for _ in range(h):
            pred = model(cur)[:, -1:, :]  # next
            yt = pred
            cur = torch.cat([cur[:, 1:, :], yt], dim=1)
        samples.append(yt.item())
    return np.array(samples)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default="data/interim")
    ap.add_argument("--outdir", type=str, default="outputs/forecasts")
    ap.add_argument("--model", type=str, default="ar1", choices=["ar1","tiny_tf"])
    ap.add_argument("--horizons", type=int, nargs="+", default=[1,5,10,20])
    ap.add_argument("--context", type=int, default=60)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    set_seed(args.seed)

    df = pd.read_csv(Path(args.indir)/"h15_processed.csv", parse_dates=["date"])
    meta = json.load(open(Path(args.indir)/"processed_meta.json","r"))
    scalers = json.load(open(Path(args.indir)/"scalers.json","r"))
    valid_start = pd.Timestamp(meta["valid_start"]); test_start = pd.Timestamp(meta["test_start"]); end = pd.Timestamp(meta["end"])

    res = []
    for s, g in df.groupby("series"):
        g = g.sort_values("date").reset_index(drop=True)
        dates = g["date"]
        vals = g["value"].values.astype(float)
        params = scalers[s]
        # rolling origins from valid_start to end- max(h)
        for t_idx in range(len(g)):
            t_date = dates.iloc[t_idx]
            if t_date < valid_start or t_date >= end:
                continue
            # context slice up to t_date inclusive
            hist = g.loc[g["date"] <= t_date, "value"].values.astype(float)
            if len(hist) < args.context + 5:
                continue
            if args.model == "ar1":
                sims = ar1_forecast_dist(hist, h=1, B=args.B)
                # multi-step via iterating
                # store entire horizon samples by iterating using AR1 recursively per sample
                # Simplify: approximate by scaling variance sqrt(h) for speed
                for h in args.horizons:
                    # gaussian approx using AR1 one-step sigma scaling (already in function)
                    sims_h = ar1_forecast_dist(hist, h=h, B=args.B)
                    q01, q025, q05 = np.quantile(sims_h, [0.01,0.025,0.05])
                    mean = sims_h.mean()
                    res.append({"origin": t_date, "horizon": h, "series": s,
                                "mean": mean, "q_0.01": q01, "q_0.025": q025, "q_0.05": q05})
            else:
                for h in args.horizons:
                    sims_h = tiny_transformer_forecast(hist, h=h, B=args.B, epochs=5, context=args.context)
                    q01, q025, q05 = np.quantile(sims_h, [0.01,0.025,0.05])
                    mean = sims_h.mean()
                    res.append({"origin": t_date, "horizon": h, "series": s,
                                "mean": mean, "q_0.01": q01, "q_0.025": q025, "q_0.05": q05})

    out = pd.DataFrame(res)
    out = out.sort_values(["series","origin","horizon"]).reset_index(drop=True)
    out_path = Path(args.outdir)/f"pred_{args.model}.parquet"
    out.to_parquet(out_path, index=False)
    LOGGER.info(f"Wrote forecasts to {out_path} ({len(out)} rows)")

if __name__ == "__main__":
    main()
