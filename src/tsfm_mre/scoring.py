"""Proper scoring rules and calibration diagnostics for quantile forecasts."""

from __future__ import annotations

import numpy as np

from tsfm_mre.schema import QUANTILE_LEVELS

LEVELS = np.asarray(QUANTILE_LEVELS)


def pinball_loss(y: np.ndarray, q: np.ndarray, tau: np.ndarray | float) -> np.ndarray:
    """Quantile (pinball) loss ``(1{y < q} - tau) * (q - y)``.

    The loss is strictly consistent for the ``tau``-quantile.
    """
    y, q, tau = np.broadcast_arrays(
        np.asarray(y, float), np.asarray(q, float), np.asarray(tau, float)
    )
    diff = y - q
    return np.where(diff >= 0.0, tau * diff, (tau - 1.0) * diff)


def level_weights(levels: np.ndarray = LEVELS) -> np.ndarray:
    """Midpoint-rule weights of the quantile levels on [0, 1] (they sum to one)."""
    levels = np.asarray(levels, dtype=float)
    edges = np.concatenate(([0.0], (levels[1:] + levels[:-1]) / 2.0, [1.0]))
    return np.diff(edges)


def crps_from_quantiles(
    y: np.ndarray, quantiles: np.ndarray, levels: np.ndarray = LEVELS
) -> np.ndarray:
    """Continuous ranked probability score computed from predictive quantiles.

    Uses ``CRPS(F, y) = 2 * integral_0^1 QS_tau(F^-1(tau), y) d tau`` (Gneiting and
    Raftery, 2007), with the integral evaluated by the midpoint rule over the
    quantile levels. The result is a proper scoring rule; on the default grid it
    reproduces the closed-form Gaussian CRPS to within about one percent.
    """
    levels = np.asarray(levels, dtype=float)
    y = np.asarray(y, dtype=float)
    losses = pinball_loss(y[:, None], quantiles, levels[None, :])
    return 2.0 * losses @ level_weights(levels)


def tail_quantile_loss(
    y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> np.ndarray:
    """Average pinball loss of the ``alpha`` and ``1 - alpha`` quantiles (the two VaR levels)."""
    return 0.5 * (pinball_loss(y, lower, alpha) + pinball_loss(y, upper, 1.0 - alpha))


def interval_hits(
    y: np.ndarray, quantiles: np.ndarray, coverage: float, levels: np.ndarray = LEVELS
) -> np.ndarray:
    """Indicator that ``y`` lies inside the central interval of nominal ``coverage``."""
    lower = int(np.argmin(np.abs(levels - (1.0 - coverage) / 2.0)))
    upper = int(np.argmin(np.abs(levels - (1.0 + coverage) / 2.0)))
    if not np.isclose(levels[lower], (1.0 - coverage) / 2.0):
        raise ValueError(f"no quantiles for a {coverage:.0%} central interval")
    y = np.asarray(y, dtype=float)
    return (y >= quantiles[:, lower]) & (y <= quantiles[:, upper])


def predictive_cdf(x: np.ndarray, quantiles: np.ndarray, levels: np.ndarray = LEVELS) -> np.ndarray:
    """Piecewise-linear predictive CDF of each row evaluated at ``x``.

    Values below the lowest (above the highest) quantile are placed at the
    midpoint of the uncovered tail probability.
    """
    x = np.asarray(x, dtype=float)
    rows = np.arange(x.size)
    k = (quantiles <= x[:, None]).sum(axis=1)
    lo = np.clip(k - 1, 0, levels.size - 1)
    hi = np.clip(k, 0, levels.size - 1)
    q_lo, q_hi = quantiles[rows, lo], quantiles[rows, hi]
    width = q_hi - q_lo
    safe = np.where(width > 0.0, width, 1.0)
    weight = np.where(width > 0.0, (x - q_lo) / safe, 0.0)
    u = levels[lo] + np.clip(weight, 0.0, 1.0) * (levels[hi] - levels[lo])
    u = np.where(k == 0, levels[0] / 2.0, u)
    return np.where(k == levels.size, (1.0 + levels[-1]) / 2.0, u)


def pit_values(
    y: np.ndarray,
    quantiles: np.ndarray,
    levels: np.ndarray = LEVELS,
    resolution: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Probability integral transform of the observations.

    With ``resolution > 0`` the observations are treated as rounded to that
    precision (H.15 yields are published in hundredths of a percentage point) and
    the randomised PIT draws uniformly between ``F(y - r/2)`` and ``F(y + r/2)``,
    which is uniform for a calibrated forecast (Czado, Gneiting and Held, 2009).
    """
    y = np.asarray(y, dtype=float)
    if resolution <= 0.0:
        return predictive_cdf(y, quantiles, levels)
    lower = predictive_cdf(y - resolution / 2.0, quantiles, levels)
    upper = predictive_cdf(y + resolution / 2.0, quantiles, levels)
    rng = np.random.default_rng(0) if rng is None else rng
    return lower + rng.random(y.size) * (upper - lower)


def quantile_function(
    quantiles: np.ndarray, rows: np.ndarray, u: np.ndarray, levels: np.ndarray = LEVELS
) -> np.ndarray:
    """Evaluate each row's piecewise-linear quantile function at probability ``u``.

    ``rows`` selects the forecast for every probability in ``u``; probabilities
    beyond the outermost levels take the outermost quantiles.
    """
    u = np.clip(np.asarray(u, dtype=float), levels[0], levels[-1])
    hi = np.clip(np.searchsorted(levels, u, side="left"), 1, levels.size - 1)
    lo = hi - 1
    weight = (u - levels[lo]) / (levels[hi] - levels[lo])
    q_lo = quantiles[rows, lo]
    q_hi = quantiles[rows, hi]
    return q_lo + weight * (q_hi - q_lo)
