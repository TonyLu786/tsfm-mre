"""Prediction table schema shared by the built-in models and external forecasts.

Every forecast is one row per (model, series, origin, horizon). A row carries the
predictive mean, quantiles on a fixed symmetric grid and the expected shortfall of
both tails, so that scoring rules, VaR/ES backtests and model comparisons can be
computed from the table alone.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

QUANTILE_LEVELS: tuple[float, ...] = (
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.975,
    0.99,
    0.995,
    0.9975,
    0.999,
)
"""Quantile levels of the predictive distribution (symmetric, includes the median)."""

ES_LEVELS: tuple[float, ...] = (0.01, 0.025, 0.975, 0.99)
"""Expected-shortfall levels: lower tail below 0.5, upper tail above 0.5."""

KEY_COLUMNS: tuple[str, ...] = ("model", "series", "origin", "horizon")


def quantile_column(level: float) -> str:
    """Column name of the predictive quantile at ``level``, e.g. ``q_0.025``."""
    return f"q_{level:g}"


def es_column(level: float) -> str:
    """Column name of the expected shortfall at ``level``, e.g. ``es_0.975``."""
    return f"es_{level:g}"


QUANTILE_COLUMNS: tuple[str, ...] = tuple(quantile_column(x) for x in QUANTILE_LEVELS)
ES_COLUMNS: tuple[str, ...] = tuple(es_column(x) for x in ES_LEVELS)
REQUIRED_COLUMNS: tuple[str, ...] = (
    *KEY_COLUMNS,
    "target_date",
    "mean",
    *QUANTILE_COLUMNS,
    *ES_COLUMNS,
)


class SchemaError(ValueError):
    """Raised when a prediction table does not follow the schema."""


def tail_count(probability: float, n_samples: int) -> int:
    """Number of order statistics that form a tail of the given probability."""
    return max(1, int(np.ceil(round(probability * n_samples, 9))))


def summarize_samples(
    samples: np.ndarray,
    levels: tuple[float, ...] = QUANTILE_LEVELS,
    es_levels: tuple[float, ...] = ES_LEVELS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, quantiles and expected shortfall of predictive samples.

    Parameters
    ----------
    samples:
        Array of shape ``(n_rows, n_samples)`` with draws from each predictive
        distribution.

    Returns
    -------
    mean, quantiles, es:
        Arrays of shape ``(n_rows,)``, ``(n_rows, len(levels))`` and
        ``(n_rows, len(es_levels))``. The lower-tail ES at level ``tau`` is the mean
        of the ``ceil(tau * n)`` smallest draws; the upper-tail ES at ``tau`` is the
        mean of the ``ceil((1 - tau) * n)`` largest draws.
    """
    x = np.sort(np.asarray(samples, dtype=float), axis=1)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("samples must have shape (n_rows, n_samples) with n_samples >= 2")
    n = x.shape[1]
    mean = x.mean(axis=1)
    quantiles = np.quantile(x, levels, axis=1).T
    es = np.empty((x.shape[0], len(es_levels)))
    for j, tau in enumerate(es_levels):
        if tau < 0.5:
            es[:, j] = x[:, : tail_count(tau, n)].mean(axis=1)
        else:
            es[:, j] = x[:, n - tail_count(1.0 - tau, n) :].mean(axis=1)
    return mean, quantiles, es


def validate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Check a prediction table and return a normalised, sorted copy.

    Raises :class:`SchemaError` when columns are missing, values are not finite,
    quantiles decrease with the level, expected shortfall lies inside the VaR,
    targets do not follow origins, or keys are duplicated.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {', '.join(missing)}")
    out = df.loc[:, list(REQUIRED_COLUMNS)].copy()
    out["model"] = out["model"].astype(str)
    out["series"] = out["series"].astype(str)
    try:
        out["origin"] = pd.to_datetime(out["origin"])
        out["target_date"] = pd.to_datetime(out["target_date"])
    except (ValueError, TypeError) as exc:
        raise SchemaError(f"origin and target_date must be dates: {exc}") from exc

    horizon = pd.to_numeric(out["horizon"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(horizon)) or np.any(horizon < 1) or np.any(horizon % 1 != 0):
        raise SchemaError("horizon must be a positive integer")
    out["horizon"] = horizon.astype(np.int64)

    numeric = ["mean", *QUANTILE_COLUMNS, *ES_COLUMNS]
    try:
        values = out[numeric].astype(float).to_numpy()
    except (ValueError, TypeError) as exc:
        raise SchemaError(f"forecast columns must be numeric: {exc}") from exc
    if not np.isfinite(values).all():
        raise SchemaError("forecast columns must be finite")
    out[numeric] = values

    q = out[list(QUANTILE_COLUMNS)].to_numpy()
    tol = 1e-9 * np.maximum(1.0, np.abs(q).max(axis=1))
    if np.any(np.diff(q, axis=1) < -tol[:, None]):
        raise SchemaError("quantiles must be non-decreasing in the level")
    for level in ES_LEVELS:
        es = out[es_column(level)].to_numpy()
        var = out[quantile_column(level)].to_numpy()
        inside = es > var + tol if level < 0.5 else es < var - tol
        if np.any(inside):
            raise SchemaError(f"{es_column(level)} must lie in the tail beyond q_{level:g}")

    if np.any(out["target_date"] <= out["origin"]):
        raise SchemaError("target_date must follow origin")
    duplicated = out.duplicated(list(KEY_COLUMNS))
    if duplicated.any():
        first = out.loc[duplicated, list(KEY_COLUMNS)].iloc[0].to_dict()
        raise SchemaError(f"duplicated forecast keys, e.g. {first}")
    return out.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def read_predictions(path: str | Path) -> pd.DataFrame:
    """Read and validate a prediction table stored as CSV (optionally gzipped) or Parquet."""
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif name.endswith((".csv", ".csv.gz")):
        df = pd.read_csv(path)
    else:
        raise SchemaError(
            f"unsupported prediction file: {path.name} (use .csv, .csv.gz or .parquet)"
        )
    return validate_predictions(df)


def write_predictions(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a prediction table; gzip output is byte-reproducible across runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.loc[:, list(REQUIRED_COLUMNS)].copy()
    for col in ("origin", "target_date"):
        out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d")
    name = path.name.lower()
    if name.endswith(".parquet"):
        out.to_parquet(path, index=False)
        return path
    if not name.endswith((".csv", ".csv.gz")):
        raise SchemaError(f"unsupported prediction file: {path.name}")
    text = out.to_csv(index=False, float_format="%.10g", lineterminator="\n").encode("utf-8")
    path.write_bytes(
        gzip.compress(text, compresslevel=9, mtime=0) if name.endswith(".gz") else text
    )
    return path
