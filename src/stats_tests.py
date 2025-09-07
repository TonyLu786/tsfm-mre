#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, re, numpy as np, pandas as pd
from typing import List, Dict, Tuple
from scipy import stats

# -----------------------
# 读取 pred_*.parquet 并抽取逐期损失
# 要求列: ["model","series","timestamp","h","mean","q001","q025","q050","q095","q975","q990",...]
# -----------------------
def load_preds(paths: List[str], horizon: int) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        # 仅留指定地平线
        df = df[df["h"] == horizon].copy()
        # 从文件名推断模型名兜底
        if "model" not in df.columns:
            m = re.search(r"pred_(\w+)\.parquet$", p)
            if m:
                df["model"] = m.group(1)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.sort_values(["timestamp","series","model"], inplace=True)
    return out

def crps_from_quantiles(row_q: np.ndarray, row_y: float, qs: np.ndarray) -> float:
    # 简约 CRPS 近似：离散分位积分（论文/工具常用做法）
    # row_q: [Q]  对应 qs: [Q]
    indicators = (row_y <= row_q).astype(float)
    return float(np.trapz((indicators - qs) ** 2, qs))

def qloss(y: float, qhat: float, alpha: float) -> float:
    e = y - qhat
    return float(alpha * max(e,0.0) + (1 - alpha) * max(-e,0.0))

def build_losses(df: pd.DataFrame, metric: str, alpha: float) -> pd.DataFrame:
    # y_true: 需要与仓库保持一致的命名；这里默认 preprocess 生成的 "y" 在 data/interim/test 中，
    # 但 pred 文件中通常不含 y。最小改动：用同一 timestamp/series 去 data/interim/test.parquet 对齐。
    # 若你已有 y 列，可直接合并或在这里读取后 merge。
    # 为自包含，这里给出一个兜底：如果无 y 列，则跳过（你可在本仓库接入 y 的 merge）。
    if "y" not in df.columns:
        raise RuntimeError("pred parquet 需包含真实值列 y（或在此脚本中先 merge test 真值）；请在 forecast.py 写入 y。")
    out = []
    quantile_cols = sorted([c for c in df.columns if c.startswith("q") and c[1:].isdigit()],
                           key=lambda c: int(c[1:]))
    qs = np.array([int(c[1:])/1000.0 for c in quantile_cols], dtype=float)
    for (_, g) in df.groupby(["timestamp","series","model"], sort=False):
        y = float(g["y"].iloc[0])
        rec = dict(timestamp=g["timestamp"].iloc[0], series=g["series"].iloc[0],
                   model=g["model"].iloc[0])
        if metric == "crps":
            qvals = g[quantile_cols].iloc[0].to_numpy(dtype=float)
            rec["loss"] = crps_from_quantiles(qvals, y, qs)
        elif metric == "qloss":
            col = f"q{int(round(alpha*1000)):03d}"
            if col not in g.columns:
                raise RuntimeError(f"缺少分位列 {col}，请在预测阶段产出该分位。")
            rec["loss"] = qloss(y, float(g[col].iloc[0]), alpha)
        else:
            raise ValueError("metric 仅支持 'crps' 或 'qloss'")
        out.append(rec)
    return pd.DataFrame(out)

# -----------------------
# 统计检验：DM / Reality Check / SPA
# -----------------------
def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, block: int = 10) -> float:
    # Diebold-Mariano (HAC/块自助近似)
    d = loss_a - loss_b
    T = len(d)
    d_mean = d.mean()
    # Newey-West 方差估计（滞后 L = block）
    gamma0 = np.var(d, ddof=1)
    s = gamma0
    for lag in range(1, min(block, T-1)+1):
        w = 1.0 - lag/(block+1.0)
        cov = np.cov(d[:-lag], d[lag:])[0,1]
        s += 2*w*cov
    stat = d_mean / np.sqrt(s / T + 1e-12)
    # 双尾
    p = 2*(1 - stats.norm.cdf(abs(stat)))
    return float(p)

def stationary_block_bootstrap(diffs: np.ndarray, B: int = 1000, p: float = 1/10) -> np.ndarray:
    # Politis & Romano (1994) 稳态块自助法；期望块长 = 1/p
    T = len(diffs)
    out = np.zeros(B, dtype=float)
    for b in range(B):
        idx = []
        while len(idx) < T:
            if len(idx)==0 or np.random.rand() < p:
                start = np.random.randint(0, T)
            else:
                start = (idx[-1] + 1) % T
            idx.append(start)
        out[b] = np.mean(diffs[idx[:T]])
    return out

def white_reality_check(loss_mat: np.ndarray, B: int = 2000, p: float = 1/10) -> float:
    # 输入：shape [T, K]，为 (loss_benchmark - loss_model_k)，越大越好
    T, K = loss_mat.shape
    centered = loss_mat - loss_mat.mean(axis=0, keepdims=True)
    # 统计量：max_k sqrt(T)*mean(centered[:,k])
    stat = np.max(np.sqrt(T)*centered.mean(axis=0))
    # bootstrap
    boot_stats = []
    for b in range(B):
        gains = np.stack([stationary_block_bootstrap(centered[:,k], B=1, p=p) for k in range(K)], axis=1)
        boot_stats.append(np.max(np.sqrt(T)*gains.mean(axis=0)))
    pval = np.mean(np.array(boot_stats) >= stat)
    return float(pval)

def spa_test(loss_mat: np.ndarray, B: int = 2000, p: float = 1/10) -> float:
    # Hansen (2005) SPA：与 RC 类似，但对劣质模型做截断修正，减少过度保守
    T, K = loss_mat.shape
    gains = loss_mat.copy()
    # 截断：去除显著劣质的负向“收益”
    gains = np.maximum(gains, 0.0)
    stat = np.max(np.sqrt(T)*gains.mean(axis=0))
    boot_stats = []
    for b in range(B):
        ghat = np.stack([stationary_block_bootstrap(gains[:,k], B=1, p=p) for k in range(K)], axis=1)
        boot_stats.append(np.max(np.sqrt(T)*ghat.mean(axis=0)))
    pval = np.mean(np.array(boot_stats) >= stat)
    return float(pval)

def summarize(df_loss: pd.DataFrame, benchmark: str, outdir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(outdir, exist_ok=True)
    models = sorted(df_loss["model"].unique())
    # 均值损失
    mean_loss = df_loss.groupby("model")["loss"].mean().to_dict()
    # 成对 DM
    dm_rows = []
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if i == j: continue
            la = df_loss[df_loss.model==a]["loss"].to_numpy()
            lb = df_loss[df_loss.model==b]["loss"].to_numpy()
            dm_rows.append({"A": a, "B": b, "p_dm": dm_test(la, lb)})
    dm_tbl = pd.DataFrame(dm_rows)
    # RC / SPA：以基准的差值为正向收益
    bench_loss = df_loss[df_loss.model==benchmark]["loss"].to_numpy()
    K_models = [m for m in models if m != benchmark]
    gains = []
    for m in K_models:
        lm = df_loss[df_loss.model==m]["loss"].to_numpy()
        gains.append(bench_loss - lm)
    gain_mat = np.stack(gains, axis=1) if len(gains)>0 else np.zeros((len(bench_loss),0))
    rc_p = white_reality_check(gain_mat) if gain_mat.shape[1]>0 else 1.0
    spa_p = spa_test(gain_mat) if gain_mat.shape[1]>0 else 1.0

    summary = []
    for m in models:
        summary.append({
            "model": m,
            "mean_loss": mean_loss[m],
            "rel_improvement_vs_%s"%benchmark: (mean_loss[benchmark]-mean_loss[m])/mean_loss[benchmark] if m!=benchmark else 0.0,
            "rc_p_value": rc_p if m!=benchmark else np.nan,
            "spa_p_value": spa_p if m!=benchmark else np.nan
        })
    sum_df = pd.DataFrame(summary).sort_values("mean_loss")
    sum_df.to_csv(os.path.join(outdir, "stats_summary.csv"), index=False)
    dm_tbl.to_csv(os.path.join(outdir, "pairwise_dm.csv"), index=False)

    # Markdown 摘要
    with open(os.path.join(outdir, "stats_summary.md"), "w", encoding="utf-8") as f:
        f.write("| model | mean_loss | rel_improve_vs_%s | rc_p | spa_p |\n" % benchmark)
        f.write("|---|---:|---:|---:|---:|\n")
        for _, r in sum_df.iterrows():
            f.write(f"| {r['model']} | {r['mean_loss']:.6f} | {100*r['rel_improvement_vs_%s'%benchmark]:.2f}% | "
                    f"{'' if pd.isna(r['rc_p_value']) else f'{r['rc_p_value']:.4f}'} | "
                    f"{'' if pd.isna(r['spa_p_value']) else f'{r['spa_p_value']:.4f}'} |\n")
    return sum_df, dm_tbl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=str, nargs="+", required=True,
                    help="list of pred_*.parquet across models")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--metric", type=str, default="crps", choices=["crps","qloss"])
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--benchmark", type=str, default="ar1")
    ap.add_argument("--outdir", type=str, default="outputs/stats")
    args = ap.parse_args()

    df = load_preds(args.pred, args.horizon)
    df_loss = build_losses(df, args.metric, args.alpha)
    summarize(df_loss, args.benchmark, args.outdir)

if __name__ == "__main__":
    main()
