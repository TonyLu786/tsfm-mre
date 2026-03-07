"""
Statistical tests for comparing forecasting models.

This module consumes one or more prediction tables (parquet or csv) using the
schema produced by forecast.py:
  origin, horizon, series, mean, q_0.01, q_0.025, q_0.05, ...

It aligns realized values from data/interim/h15_processed.csv, computes loss
series, and reports:
- pairwise Diebold-Mariano p-values
- White Reality Check p-value (simplified)
- SPA p-value (simplified)
"""
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from utils import LOGGER, ensure_dir, read_prediction_table


def infer_model_name(path: Path) -> str:
    m = re.search(r"pred_(.+)$", path.stem)
    if m:
        return m.group(1)
    return path.stem


def extract_quantile_columns(df: pd.DataFrame) -> Dict[float, str]:
    qcols = {}
    for col in df.columns:
        if col.startswith("q_"):
            try:
                qcols[float(col.split("_", 1)[1])] = col
                continue
            except ValueError:
                pass
        m = re.match(r"^q(\d{3})$", col)
        if m:
            qcols[int(m.group(1)) / 1000.0] = col
    return dict(sorted(qcols.items(), key=lambda kv: kv[0]))


def choose_quantile_column(qcols: Dict[float, str], alpha: float) -> str:
    if not qcols:
        raise ValueError("No quantile columns found in predictions.")
    level, col = min(qcols.items(), key=lambda kv: abs(kv[0] - alpha))
    if abs(level - alpha) > 1e-9:
        available = ", ".join([str(k) for k in qcols.keys()])
        raise ValueError(f"No exact quantile for alpha={alpha}. Available levels: {available}")
    return col


def align_predictions_with_truth(pred: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    required_pred = {"origin", "horizon", "series", "mean"}
    missing = required_pred - set(pred.columns)
    if missing:
        raise ValueError(f"Prediction file missing required columns: {sorted(missing)}")

    pred = pred.copy()
    pred["origin"] = pd.to_datetime(pred["origin"])
    pred["horizon"] = pred["horizon"].astype(int)
    pred["target_date"] = [
        origin + pd.tseries.offsets.BDay(int(h)) for origin, h in zip(pred["origin"], pred["horizon"])
    ]

    truth_use = truth.copy()
    truth_use["date"] = pd.to_datetime(truth_use["date"])
    truth_use = truth_use.rename(columns={"date": "target_date", "value": "y"})

    merged = pred.merge(
        truth_use[["target_date", "series", "y"]],
        on=["target_date", "series"],
        how="inner",
    )
    return merged


def crps_from_quantiles(row_q: np.ndarray, y: float, levels: np.ndarray) -> float:
    indicators = (y <= row_q).astype(float)
    return float(np.trapz((indicators - levels) ** 2, levels))


def pinball_loss(y: float, qhat: float, alpha: float) -> float:
    err = y - qhat
    return float(alpha * max(err, 0.0) + (1.0 - alpha) * max(-err, 0.0))


def build_losses(df: pd.DataFrame, metric: str, alpha: float) -> pd.DataFrame:
    out = df[["origin", "series", "horizon", "model", "y", "mean"]].copy()

    if metric == "mse":
        out["loss"] = (out["y"] - out["mean"]) ** 2
    elif metric == "mae":
        out["loss"] = (out["y"] - out["mean"]).abs()
    else:
        qcols = extract_quantile_columns(df)
        if metric == "pinball":
            qcol = choose_quantile_column(qcols, alpha)
            out["loss"] = [pinball_loss(y, q, alpha) for y, q in zip(df["y"], df[qcol])]
        elif metric == "crps":
            levels = np.array(list(qcols.keys()), dtype=float)
            cols = [qcols[l] for l in levels]
            qmat = df[cols].to_numpy(dtype=float)
            out["loss"] = [
                crps_from_quantiles(row_q=qmat[i], y=float(df["y"].iloc[i]), levels=levels)
                for i in range(len(df))
            ]
        else:
            raise ValueError("metric must be one of: crps, pinball, mse, mae")

    return out[["origin", "series", "horizon", "model", "loss"]]


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, hac_lags: int = 10) -> float:
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    t = len(d)
    if t < 5:
        return np.nan

    d_mean = float(np.mean(d))
    gamma0 = float(np.var(d, ddof=1))
    max_lag = max(0, min(hac_lags, t - 1))
    var_hac = gamma0
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1.0)
        cov = float(np.cov(d[:-lag], d[lag:], ddof=0)[0, 1])
        var_hac += 2.0 * w * cov

    if var_hac <= 0:
        return np.nan
    stat = d_mean / np.sqrt(var_hac / t)
    return float(2.0 * (1.0 - stats.norm.cdf(abs(stat))))


def stationary_bootstrap_indices(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)
    for i in range(1, n):
        if rng.random() < p:
            idx[i] = rng.integers(0, n)
        else:
            idx[i] = (idx[i - 1] + 1) % n
    return idx


def white_reality_check(gains: np.ndarray, b: int, p: float, rng: np.random.Generator) -> float:
    if gains.size == 0:
        return 1.0
    t = gains.shape[0]
    stat = float(np.max(np.sqrt(t) * np.mean(gains, axis=0)))

    centered = gains - np.mean(gains, axis=0, keepdims=True)
    boot_stats = np.empty(b, dtype=float)
    for i in range(b):
        idx = stationary_bootstrap_indices(t, p, rng)
        sample = centered[idx, :]
        boot_stats[i] = float(np.max(np.sqrt(t) * np.mean(sample, axis=0)))
    return float(np.mean(boot_stats >= stat))


def spa_test(gains: np.ndarray, b: int, p: float, rng: np.random.Generator) -> float:
    if gains.size == 0:
        return 1.0
    t = gains.shape[0]
    truncated = np.maximum(gains, 0.0)
    stat = float(np.max(np.sqrt(t) * np.mean(truncated, axis=0)))

    centered = truncated - np.mean(truncated, axis=0, keepdims=True)
    boot_stats = np.empty(b, dtype=float)
    for i in range(b):
        idx = stationary_bootstrap_indices(t, p, rng)
        sample = centered[idx, :]
        boot_stats[i] = float(np.max(np.sqrt(t) * np.mean(sample, axis=0)))
    return float(np.mean(boot_stats >= stat))


def summarize(
    df_loss: pd.DataFrame,
    benchmark: str,
    outdir: str,
    bootstrap_b: int,
    bootstrap_p: float,
    hac_lags: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(outdir)
    models = sorted(df_loss["model"].unique())
    if benchmark not in models:
        raise ValueError(f"Benchmark model '{benchmark}' not found in losses. Models={models}")

    keys = ["origin", "series", "horizon"]
    dm_rows: List[Dict[str, object]] = []
    for a in models:
        for b in models:
            if a == b:
                continue
            left = df_loss[df_loss["model"] == a][keys + ["loss"]].rename(columns={"loss": "loss_a"})
            right = df_loss[df_loss["model"] == b][keys + ["loss"]].rename(columns={"loss": "loss_b"})
            merged = left.merge(right, on=keys, how="inner")
            p_dm = dm_test(merged["loss_a"].to_numpy(), merged["loss_b"].to_numpy(), hac_lags=hac_lags)
            dm_rows.append({"A": a, "B": b, "n": int(len(merged)), "p_dm": p_dm})
    dm_tbl = pd.DataFrame(dm_rows)

    pivot = df_loss.pivot_table(index=keys, columns="model", values="loss", aggfunc="mean")
    pivot = pivot.dropna(axis=0, how="any")
    bench_series = pivot[benchmark]
    contenders = [m for m in models if m != benchmark]
    gains = np.column_stack([(bench_series - pivot[m]).to_numpy() for m in contenders]) if contenders else np.zeros((len(bench_series), 0))
    rng = np.random.default_rng(seed)
    rc_p = white_reality_check(gains, b=bootstrap_b, p=bootstrap_p, rng=rng)
    spa_p = spa_test(gains, b=bootstrap_b, p=bootstrap_p, rng=rng)

    mean_loss = df_loss.groupby("model", as_index=False)["loss"].mean().rename(columns={"loss": "mean_loss"})
    bench_mean = float(mean_loss[mean_loss["model"] == benchmark]["mean_loss"].iloc[0])

    summary_rows = []
    rel_col = f"rel_improvement_vs_{benchmark}"
    for _, row in mean_loss.sort_values("mean_loss").iterrows():
        model = row["model"]
        m_loss = float(row["mean_loss"])
        rel = 0.0 if model == benchmark else (bench_mean - m_loss) / bench_mean
        summary_rows.append(
            {
                "model": model,
                "mean_loss": m_loss,
                rel_col: rel,
                "rc_p_value": np.nan if model == benchmark else rc_p,
                "spa_p_value": np.nan if model == benchmark else spa_p,
            }
        )

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(Path(outdir) / "stats_summary.csv", index=False)
    dm_tbl.to_csv(Path(outdir) / "pairwise_dm.csv", index=False)

    with open(Path(outdir) / "stats_summary.md", "w", encoding="utf-8") as f:
        f.write(f"| model | mean_loss | rel_improve_vs_{benchmark} | rc_p | spa_p |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for _, row in sum_df.iterrows():
            rc_str = "" if pd.isna(row["rc_p_value"]) else f"{row['rc_p_value']:.4f}"
            spa_str = "" if pd.isna(row["spa_p_value"]) else f"{row['spa_p_value']:.4f}"
            rel_val = f"{100.0 * float(row[rel_col]):.2f}%"
            f.write(f"| {row['model']} | {row['mean_loss']:.6f} | {rel_val} | {rc_str} | {spa_str} |\n")

    return sum_df, dm_tbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=str, nargs="+", required=True, help="prediction tables across models")
    ap.add_argument("--truth", type=str, default="data/interim/h15_processed.csv")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--metric", type=str, default="crps", choices=["crps", "pinball", "mse", "mae"])
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--benchmark", type=str, default="ar1")
    ap.add_argument("--bootstrap_B", type=int, default=500)
    ap.add_argument("--bootstrap_p", type=float, default=0.1)
    ap.add_argument("--hac_lags", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="outputs/stats")
    args = ap.parse_args()

    truth = pd.read_csv(args.truth, parse_dates=["date"])

    all_losses = []
    for path in args.pred:
        pred_df, resolved = read_prediction_table(path)
        model = infer_model_name(Path(resolved))
        if "model" in pred_df.columns:
            pred_df["model"] = pred_df["model"].astype(str)
        else:
            pred_df["model"] = model

        pred_df = pred_df[pred_df["horizon"] == args.horizon].copy()
        if pred_df.empty:
            LOGGER.warning("Skipping %s: no rows for horizon=%s", resolved, args.horizon)
            continue

        aligned = align_predictions_with_truth(pred_df, truth)
        if aligned.empty:
            LOGGER.warning("Skipping %s: no aligned realized values", resolved)
            continue

        losses = build_losses(aligned, metric=args.metric, alpha=args.alpha)
        all_losses.append(losses)

    if not all_losses:
        raise RuntimeError("No loss rows produced. Check input predictions and truth data range.")

    df_loss = pd.concat(all_losses, ignore_index=True)
    sum_df, dm_tbl = summarize(
        df_loss,
        benchmark=args.benchmark,
        outdir=args.outdir,
        bootstrap_b=args.bootstrap_B,
        bootstrap_p=args.bootstrap_p,
        hac_lags=args.hac_lags,
        seed=args.seed,
    )
    LOGGER.info("Saved stats summary to %s (models=%s, comparisons=%s)", args.outdir, len(sum_df), len(dm_tbl))


if __name__ == "__main__":
    main()
