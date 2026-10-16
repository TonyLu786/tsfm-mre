"""Value-at-Risk and expected-shortfall backtests.

Risk is evaluated on the loss of a position exposed to one tail of the yield
distribution: rising yields (``upper`` tail, the loss of a long-duration bond
position) or falling yields (``lower`` tail). For an origin value ``y_t`` and a
realised value ``y_{t+h}`` the loss is ``y_{t+h} - y_t`` for the upper tail and
``y_t - y_{t+h}`` for the lower tail; VaR and ES are the corresponding quantile and
tail expectation of the predictive distribution, expressed as losses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.special import xlogy

from tsfm_mre.scoring import LEVELS, quantile_function

TAILS = ("upper", "lower")


def kupiec_test(hits: np.ndarray, alpha: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures likelihood-ratio test. Returns ``(LR, p)``."""
    hits = np.asarray(hits, dtype=int)
    n, x = hits.size, int(hits.sum())
    p_hat = x / n
    ll_null = xlogy(n - x, 1.0 - alpha) + xlogy(x, alpha)
    ll_alt = xlogy(n - x, 1.0 - p_hat) + xlogy(x, p_hat)
    lr = max(0.0, -2.0 * (ll_null - ll_alt))
    return lr, float(stats.chi2.sf(lr, 1))


def christoffersen_test(hits: np.ndarray, alpha: float) -> dict[str, float]:
    """Christoffersen (1998) independence and conditional-coverage tests.

    Returns the likelihood-ratio statistics and p-values of the first-order Markov
    independence test (``ind``) and of the joint conditional-coverage test (``cc``).
    """
    hits = np.asarray(hits, dtype=int)
    prev, curr = hits[:-1], hits[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll_null = xlogy(n00 + n10, 1.0 - pi) + xlogy(n01 + n11, pi)
    ll_alt = xlogy(n00, 1.0 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1.0 - pi11) + xlogy(n11, pi11)
    lr_ind = max(0.0, -2.0 * (ll_null - ll_alt))
    lr_uc, _ = kupiec_test(hits, alpha)
    lr_cc = lr_uc + lr_ind
    return {
        "lr_ind": lr_ind,
        "p_ind": float(stats.chi2.sf(lr_ind, 1)),
        "lr_cc": lr_cc,
        "p_cc": float(stats.chi2.sf(lr_cc, 2)),
    }


def basel_zone_bounds(n: int, alpha: float) -> tuple[int, int]:
    """Smallest exceedance counts of the yellow and red zones.

    Zones follow the Basel Committee (1996) rule: the yellow zone starts where the
    binomial(n, alpha) cumulative probability reaches 95%, the red zone where it
    reaches 99.99%. For n = 250 and alpha = 1% this gives 5 and 10.
    """
    counts = np.arange(n + 1)
    cdf = stats.binom.cdf(counts, n, alpha)
    yellow = int(counts[np.argmax(cdf >= 0.95)])
    red = int(counts[np.argmax(cdf >= 0.9999)])
    return yellow, red


def basel_zone(exceedances: int, n: int, alpha: float) -> str:
    """Traffic-light zone of an exceedance count over ``n`` observations."""
    yellow, red = basel_zone_bounds(n, alpha)
    if exceedances >= red:
        return "red"
    if exceedances >= yellow:
        return "yellow"
    return "green"


def acerbi_szekely_z2(losses: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> float:
    """Acerbi and Szekely (2014) Z2 statistic, written for losses.

    ``Z2 = 1 - sum(L_t * 1{L_t > VaR_t} / ES_t) / (T * alpha)``. Its expectation is
    zero when the ES forecasts are correct; negative values indicate that tail
    risk is underestimated.
    """
    losses = np.asarray(losses, dtype=float)
    hits = losses > var
    return float(1.0 - np.sum(hits * losses / es) / (losses.size * alpha))


@dataclass(frozen=True)
class TailForecast:
    """Predictive loss distributions of one tail for a sequence of forecasts."""

    origin_value: np.ndarray
    quantiles: np.ndarray
    tail: str
    levels: np.ndarray = field(default_factory=lambda: LEVELS.copy())

    def to_loss(self, values: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Convert yield levels into losses of the position exposed to this tail."""
        base = self.origin_value[rows]
        return values - base if self.tail == "upper" else base - values

    def sample_tail(self, rows: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
        """Losses drawn from each forecast's tail beyond its VaR at level ``alpha``."""
        depth = rng.random(rows.size) * alpha
        u = 1.0 - depth if self.tail == "upper" else depth
        return self.to_loss(quantile_function(self.quantiles, rows, u, self.levels), rows)


def acerbi_szekely_test(
    losses: np.ndarray,
    var: np.ndarray,
    es: np.ndarray,
    alpha: float,
    tail: TailForecast,
    n_sim: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Z2 statistic with a Monte Carlo p-value under the forecast distributions.

    Under the null hypothesis each loss is drawn from its predictive distribution:
    it exceeds the VaR with probability ``alpha`` and, if it does, comes from the
    predictive tail, which is reconstructed from the quantile grid. The p-value is
    the share of simulated statistics at or below the observed one.
    """
    if np.any(es <= 0.0):
        raise ValueError(
            "the Z2 test needs positive ES in loss units; use a risk horizon at which the "
            "predictive tail lies beyond the origin value (the reference configurations use 1 day)"
        )
    observed = acerbi_szekely_z2(losses, var, es, alpha)
    n = losses.size
    sim, row = np.nonzero(rng.random((n_sim, n)) < alpha)
    tail_losses = tail.sample_tail(row, alpha, rng)
    totals = np.bincount(sim, weights=tail_losses / es[row], minlength=n_sim)
    simulated = 1.0 - totals / (n * alpha)
    p_value = (1.0 + np.sum(simulated <= observed)) / (n_sim + 1.0)
    return observed, float(p_value)
