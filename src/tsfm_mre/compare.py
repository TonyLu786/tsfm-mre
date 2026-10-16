"""Forecast comparison: Diebold-Mariano, White's Reality Check and Hansen's SPA test."""

from __future__ import annotations

import numpy as np
from scipy import stats


def stationary_bootstrap_indices(
    n: int, n_boot: int, mean_block_length: float, rng: np.random.Generator
) -> np.ndarray:
    """Resampling indices of the stationary bootstrap (Politis and Romano, 1994).

    Blocks have geometric lengths with the given mean and wrap around the sample.
    Returns an integer array of shape ``(n_boot, n)``.
    """
    p = 1.0 / mean_block_length
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    restart = rng.random((n_boot, n)) < p
    starts = rng.integers(0, n, (n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(restart[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def newey_west_variance(d: np.ndarray, lags: int) -> float:
    """Bartlett-kernel long-run variance of a series (Newey and West, 1987)."""
    e = np.asarray(d, dtype=float) - np.mean(d)
    n = e.size
    variance = float(e @ e) / n
    for k in range(1, lags + 1):
        variance += 2.0 * (1.0 - k / (lags + 1.0)) * float(e[k:] @ e[:-k]) / n
    return variance


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int) -> tuple[float, float]:
    """Diebold-Mariano test of equal expected loss, with the HLN adjustment.

    The loss differential ``d = loss_a - loss_b`` is standardised with a Newey-West
    long-run variance using ``max(h - 1, floor(4 (n / 100)^(2/9)))`` lags; the
    statistic is scaled by the Harvey, Leybourne and Newbold (1997) factor and
    compared with a t distribution with ``n - 1`` degrees of freedom. Negative
    statistics favour model ``a``. Returns ``(statistic, two-sided p-value)``.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = d.size
    lags = max(horizon - 1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    variance = newey_west_variance(d, lags)
    if variance <= 0.0:
        return float("nan"), float("nan")
    adjustment = np.sqrt((n + 1.0 - 2.0 * horizon + horizon * (horizon - 1.0) / n) / n)
    statistic = adjustment * d.mean() / np.sqrt(variance / n)
    return float(statistic), float(2.0 * stats.t.sf(abs(statistic), df=n - 1))


def superior_predictive_ability(
    benchmark_loss: np.ndarray,
    model_losses: np.ndarray,
    n_boot: int,
    mean_block_length: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """White's (2000) Reality Check and Hansen's (2005) SPA test against a benchmark.

    Parameters
    ----------
    benchmark_loss:
        Loss series of the benchmark, shape ``(n,)``.
    model_losses:
        Loss series of the competing models, shape ``(n, k)``.

    The null hypothesis is that no competing model has a lower expected loss than
    the benchmark. Relative performance is ``d_kt = L_0t - L_kt``. The Reality
    Check uses ``max_k sqrt(n) mean(d_k)``; the SPA test studentises each model and
    recentres the bootstrap distribution with the consistent (``spa_p``), lower
    (``spa_p_lower``) and upper (``spa_p_upper``) estimators of Hansen (2005).
    """
    d = np.asarray(benchmark_loss, dtype=float)[:, None] - np.asarray(model_losses, dtype=float)
    n, _ = d.shape
    d_bar = d.mean(axis=0)
    idx = stationary_bootstrap_indices(n, n_boot, mean_block_length, rng)
    centred = d[idx].mean(axis=1) - d_bar  # (n_boot, k)
    root_n = np.sqrt(n)

    rc_stat = float(np.max(root_n * d_bar))
    rc_boot = np.max(root_n * centred, axis=1)
    rc_p = float(np.mean(rc_boot >= rc_stat))

    omega = np.sqrt(n * np.mean(centred**2, axis=0))
    omega = np.where(omega > 0.0, omega, np.inf)
    t_stat = root_n * d_bar / omega
    spa_stat = max(0.0, float(np.max(t_stat)))
    threshold = -np.sqrt(2.0 * np.log(np.log(n)))
    recentring = {
        "spa_p_lower": np.minimum(d_bar, 0.0),
        "spa_p": np.where(t_stat <= threshold, d_bar, 0.0),
        "spa_p_upper": np.zeros_like(d_bar),
    }
    result = {"rc_stat": rc_stat, "rc_p": rc_p, "spa_stat": spa_stat}
    for key, mu in recentring.items():
        boot = np.maximum(0.0, np.max(root_n * (centred + mu) / omega, axis=1))
        result[key] = float(np.mean(boot >= spa_stat))
    return result
