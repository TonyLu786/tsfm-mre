"""Evaluation tables: probabilistic accuracy, risk backtests and model comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tsfm_mre.backtest import (
    TAILS,
    TailForecast,
    acerbi_szekely_test,
    basel_zone,
    christoffersen_test,
    kupiec_test,
)
from tsfm_mre.compare import diebold_mariano, superior_predictive_ability
from tsfm_mre.data import Panel
from tsfm_mre.schema import QUANTILE_COLUMNS, es_column, quantile_column
from tsfm_mre.scoring import crps_from_quantiles, interval_hits, pit_values, tail_quantile_loss

POOLED = "pooled"
INTERVALS = (0.8, 0.9, 0.98)
LOSSES = ("crps", "tail_ql")
"""Loss functions used for model comparison: CRPS and the tail quantile loss at the VaR levels."""


@dataclass(frozen=True)
class EvaluationTables:
    """All evaluation outputs of a run."""

    scored: pd.DataFrame
    metrics: pd.DataFrame
    risk: pd.DataFrame
    pairwise: pd.DataFrame
    joint: pd.DataFrame


def attach_realized(
    predictions: pd.DataFrame,
    panel: Panel,
    var_alpha: float,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Add realised targets ``y``, origin values ``y_origin``, scores and PIT to each forecast."""
    long = panel.values.rename_axis("date").reset_index()
    long = long.melt(id_vars="date", var_name="series", value_name="value")
    long["date"] = pd.to_datetime(long["date"])
    target = long.rename(columns={"date": "target_date", "value": "y"})
    origin = long.rename(columns={"date": "origin", "value": "y_origin"})
    out = predictions.merge(target, on=["series", "target_date"], how="inner")
    out = out.merge(origin, on=["series", "origin"], how="inner")
    if len(out) != len(predictions):
        raise ValueError("some forecasts have no realised value in the panel")
    quantiles = out[list(QUANTILE_COLUMNS)].to_numpy()
    y = out["y"].to_numpy()
    out["crps"] = crps_from_quantiles(y, quantiles)
    out["tail_ql"] = tail_quantile_loss(
        y,
        out[quantile_column(var_alpha)].to_numpy(),
        out[quantile_column(round(1.0 - var_alpha, 6))].to_numpy(),
        var_alpha,
    )
    resolution = float(panel.provenance.get("resolution", 0.0))
    out["pit"] = pit_values(y, quantiles, resolution=resolution, rng=rng)
    for coverage in INTERVALS:
        out[f"in_{round(coverage * 100)}"] = interval_hits(y, quantiles, coverage)
    return out


def accuracy_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Mean scores by model, series and horizon, plus series-pooled rows."""
    frame = scored.assign(
        abs_error=(scored["y"] - scored["q_0.5"]).abs(),
        sq_error=(scored["y"] - scored["mean"]) ** 2,
    )
    columns = {
        "n": ("crps", "count"),
        "crps": ("crps", "mean"),
        "tail_ql": ("tail_ql", "mean"),
        "mae_median": ("abs_error", "mean"),
        "rmse_mean": ("sq_error", "mean"),
        **{f"coverage_{round(c * 100)}": (f"in_{round(c * 100)}", "mean") for c in INTERVALS},
    }
    by_series = frame.groupby(["model", "series", "horizon"], as_index=False).agg(**columns)
    pooled = frame.groupby(["model", "horizon"], as_index=False).agg(**columns)
    pooled.insert(1, "series", POOLED)
    table = pd.concat([by_series, pooled], ignore_index=True)
    table["rmse_mean"] = np.sqrt(table["rmse_mean"])
    return table.sort_values(["series", "horizon", "model"]).reset_index(drop=True)


def tail_losses(frame: pd.DataFrame, tail: str, alpha: float) -> tuple[np.ndarray, ...]:
    """Realised losses, VaR and ES of one tail at level ``alpha``, in loss units."""
    y, base = frame["y"].to_numpy(), frame["y_origin"].to_numpy()
    if tail == "upper":
        level = round(1.0 - alpha, 6)
        losses = y - base
        var = frame[quantile_column(level)].to_numpy() - base
        es = frame[es_column(level)].to_numpy() - base
    else:
        losses = base - y
        var = base - frame[quantile_column(alpha)].to_numpy()
        es = base - frame[es_column(alpha)].to_numpy()
    return losses, var, es


def risk_table(
    scored: pd.DataFrame,
    horizon: int,
    var_alpha: float,
    es_alpha: float,
    basel_window: int,
    es_simulations: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """VaR coverage, independence, Basel zone and Acerbi-Szekely ES tests per tail."""
    rows = []
    subset = scored[scored["horizon"] == horizon]
    for (model, series), frame in subset.groupby(["model", "series"], sort=True):
        frame = frame.sort_values("origin")
        quantiles = frame[list(QUANTILE_COLUMNS)].to_numpy()
        for tail in TAILS:
            losses, var, _ = tail_losses(frame, tail, var_alpha)
            hits = (losses > var).astype(int)
            _, p_uc = kupiec_test(hits, var_alpha)
            markov = christoffersen_test(hits, var_alpha)
            recent = hits[-basel_window:]
            es_losses, es_var, es_values = tail_losses(frame, tail, es_alpha)
            tail_model = TailForecast(frame["y_origin"].to_numpy(), quantiles, tail)
            z2, p_z2 = acerbi_szekely_test(
                es_losses, es_var, es_values, es_alpha, tail_model, es_simulations, rng
            )
            rows.append(
                {
                    "model": model,
                    "series": series,
                    "tail": tail,
                    "horizon": horizon,
                    "n": hits.size,
                    "var_alpha": var_alpha,
                    "exceedances": int(hits.sum()),
                    "exceedance_rate": float(hits.mean()),
                    "kupiec_p": p_uc,
                    "independence_p": markov["p_ind"],
                    "conditional_coverage_p": markov["p_cc"],
                    "basel_window": recent.size,
                    "basel_exceedances": int(recent.sum()),
                    "basel_zone": basel_zone(int(recent.sum()), recent.size, var_alpha),
                    "es_alpha": es_alpha,
                    "es_z2": z2,
                    "es_p": p_z2,
                }
            )
    return pd.DataFrame(rows)


def loss_matrix(scored: pd.DataFrame, loss: str, horizon: int, scope: str) -> pd.DataFrame:
    """Loss by origin (rows) and model (columns) for one series or averaged over series."""
    frame = scored[scored["horizon"] == horizon]
    if scope == POOLED:
        n_series = frame["series"].nunique()
        wide = frame.pivot_table(index=["origin", "series"], columns="model", values=loss)
        wide = wide.dropna()
        counts = wide.groupby(level="origin").size()
        keep = counts[counts == n_series].index
        wide = wide[wide.index.get_level_values("origin").isin(keep)]
        return wide.groupby(level="origin").mean().sort_index()
    frame = frame[frame["series"] == scope]
    return frame.pivot_table(index="origin", columns="model", values=loss).dropna().sort_index()


def comparison_tables(
    scored: pd.DataFrame,
    benchmark: str,
    n_boot: int,
    mean_block_length: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise DM tests and joint RC/SPA tests of every model against the benchmark."""
    pairwise, joint = [], []
    scopes = [*sorted(scored["series"].unique()), POOLED]
    for loss in LOSSES:
        for horizon in sorted(int(h) for h in scored["horizon"].unique()):
            for scope in scopes:
                matrix = loss_matrix(scored, loss, horizon, scope)
                if benchmark not in matrix.columns or matrix.shape[1] < 2:
                    continue
                others = [m for m in matrix.columns if m != benchmark]
                base = matrix[benchmark].to_numpy()
                for model in others:
                    values = matrix[model].to_numpy()
                    stat, p = diebold_mariano(values, base, horizon)
                    pairwise.append(
                        {
                            "loss": loss,
                            "horizon": horizon,
                            "scope": scope,
                            "model": model,
                            "benchmark": benchmark,
                            "n": len(matrix),
                            "loss_model": float(values.mean()),
                            "loss_benchmark": float(base.mean()),
                            "relative_change": float(values.mean() / base.mean() - 1.0),
                            "dm_stat": stat,
                            "dm_p": p,
                        }
                    )
                spa = superior_predictive_ability(
                    base, matrix[others].to_numpy(), n_boot, mean_block_length, rng
                )
                joint.append(
                    {
                        "loss": loss,
                        "horizon": horizon,
                        "scope": scope,
                        "benchmark": benchmark,
                        "models": " ".join(others),
                        "n": len(matrix),
                        **spa,
                    }
                )
    return pd.DataFrame(pairwise), pd.DataFrame(joint)


def evaluate(
    predictions: pd.DataFrame,
    panel: Panel,
    *,
    risk_horizon: int,
    var_alpha: float,
    es_alpha: float,
    basel_window: int,
    es_simulations: int,
    benchmark: str,
    bootstrap_reps: int,
    mean_block_length: float,
    seed: int,
) -> EvaluationTables:
    """Score forecasts against realised values and run all tests."""
    risk_seed, compare_seed, pit_seed = np.random.SeedSequence(seed).spawn(3)
    scored = attach_realized(predictions, panel, var_alpha, np.random.default_rng(pit_seed))
    metrics = accuracy_table(scored)
    risk = risk_table(
        scored,
        risk_horizon,
        var_alpha,
        es_alpha,
        basel_window,
        es_simulations,
        np.random.default_rng(risk_seed),
    )
    pairwise, joint = comparison_tables(
        scored, benchmark, bootstrap_reps, mean_block_length, np.random.default_rng(compare_seed)
    )
    return EvaluationTables(scored, metrics, risk, pairwise, joint)
