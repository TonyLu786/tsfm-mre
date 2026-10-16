"""Figures for the evaluation report."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsfm_mre.backtest import basel_zone_bounds
from tsfm_mre.evaluate import POOLED
from tsfm_mre.schema import quantile_column

PALETTE = ("#1f4e79", "#c0392b", "#2e8b57", "#8e44ad", "#d68910", "#17a2b8", "#7f8c8d")
STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}


def model_colors(models: Sequence[str]) -> dict[str, str]:
    """Fixed colour per model, assigned in configuration order."""
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)
    return path


def predictive_intervals(
    scored: pd.DataFrame,
    models: Sequence[str],
    series: str,
    horizon: int,
    alpha: float,
    path: Path,
) -> Path:
    """Realised yields, central predictive intervals and VaR exceedances per model."""
    lower_col, upper_col = quantile_column(alpha), quantile_column(round(1.0 - alpha, 6))
    coverage = round((1.0 - 2.0 * alpha) * 100)
    colors = model_colors(models)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(models), 1, figsize=(9.5, 2.3 * len(models)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, model in zip(axes, models, strict=True):
            frame = scored[
                (scored["model"] == model)
                & (scored["series"] == series)
                & (scored["horizon"] == horizon)
            ].sort_values("target_date")
            x = frame["target_date"]
            ax.fill_between(
                x,
                frame[lower_col],
                frame[upper_col],
                color=colors[model],
                alpha=0.22,
                linewidth=0,
                label=f"{coverage}% interval",
            )
            ax.plot(x, frame["y"], color="#222222", linewidth=0.7, label="realised")
            up = frame["y"] > frame[upper_col]
            down = frame["y"] < frame[lower_col]
            ax.scatter(
                x[up],
                frame["y"][up],
                marker="^",
                s=16,
                color="#c0392b",
                zorder=3,
                label=f"above q{1 - alpha:g} ({int(up.sum())})",
            )
            ax.scatter(
                x[down],
                frame["y"][down],
                marker="v",
                s=16,
                color="#1f4e79",
                zorder=3,
                label=f"below q{alpha:g} ({int(down.sum())})",
            )
            ax.set_ylabel("percent")
            ax.set_title(f"{model}", loc="left")
            ax.legend(loc="upper left", ncol=4)
        fig.suptitle(
            f"{series}: {horizon}-day-ahead predictive intervals and VaR exceedances",
            x=0.01,
            ha="left",
            fontsize=12,
        )
        fig.tight_layout()
        return _save(fig, path)


def rolling_exceedances(
    scored: pd.DataFrame,
    models: Sequence[str],
    series: str,
    horizon: int,
    alpha: float,
    window: int,
    path: Path,
) -> Path:
    """Rolling exceedance counts with the Basel traffic-light boundaries."""
    yellow, red = basel_zone_bounds(window, alpha)
    colors = model_colors(models)
    tails = (("upper", quantile_column(round(1.0 - alpha, 6))), ("lower", quantile_column(alpha)))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
        for ax, (tail, column) in zip(axes, tails, strict=True):
            for model in models:
                frame = scored[
                    (scored["model"] == model)
                    & (scored["series"] == series)
                    & (scored["horizon"] == horizon)
                ].sort_values("target_date")
                hits = frame["y"] > frame[column] if tail == "upper" else frame["y"] < frame[column]
                count = hits.astype(float).rolling(window).sum()
                ax.plot(
                    frame["target_date"], count, color=colors[model], linewidth=1.2, label=model
                )
            ax.axhline(yellow, color="#d68910", linestyle="--", linewidth=1, label="yellow zone")
            ax.axhline(red, color="#c0392b", linestyle="--", linewidth=1, label="red zone")
            ax.set_title(f"{tail} tail, {window}-day window", loc="left")
            ax.set_ylabel("exceedances")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.02)
        )
        fig.suptitle(
            f"{series}: rolling VaR({1 - alpha:.0%}) exceedances", x=0.01, ha="left", fontsize=12
        )
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        return _save(fig, path)


def scores_by_horizon(
    metrics: pd.DataFrame, models: Sequence[str], benchmark: str, path: Path
) -> Path:
    """Series-pooled CRPS and tail quantile loss by horizon, relative to the benchmark."""
    pooled = metrics[metrics["series"] == POOLED]
    colors = model_colors(models)
    panels = (("crps", "CRPS"), ("tail_ql", "tail quantile loss"))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
        crps = pooled.pivot(index="horizon", columns="model", values="crps") * 100.0
        for model in models:
            axes[0].plot(crps.index, crps[model], marker="o", color=colors[model], label=model)
        axes[0].set_title("CRPS (basis points)", loc="left")
        for ax, (column, label) in zip(axes[1:], panels, strict=True):
            table = pooled.pivot(index="horizon", columns="model", values=column)
            for model in models:
                relative = 100.0 * (table[model] / table[benchmark] - 1.0)
                ax.plot(table.index, relative, marker="o", color=colors[model], label=model)
            ax.axhline(0.0, color="#555555", linewidth=0.8)
            ax.set_title(f"{label} vs {benchmark} (%)", loc="left")
        for ax in axes:
            ax.set_xlabel("horizon (trading days)")
            ax.set_xticks(crps.index)
        axes[0].legend(loc="upper left")
        fig.tight_layout()
        return _save(fig, path)


def pit_histograms(scored: pd.DataFrame, models: Sequence[str], horizon: int, path: Path) -> Path:
    """Histograms of probability integral transforms, pooled over series."""
    colors = model_colors(models)
    bins = np.linspace(0.0, 1.0, 11)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 2.8), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, model in zip(axes, models, strict=True):
            pit = scored.loc[(scored["model"] == model) & (scored["horizon"] == horizon), "pit"]
            ax.hist(pit, bins=bins, density=True, color=colors[model], alpha=0.8, edgecolor="white")
            ax.axhline(1.0, color="#222222", linewidth=0.8, linestyle="--")
            ax.set_title(model, loc="left")
            ax.set_xlabel("PIT")
        axes[0].set_ylabel("density")
        fig.suptitle(f"PIT histograms, {horizon}-day horizon", x=0.01, ha="left", fontsize=12)
        fig.tight_layout()
        return _save(fig, path)
