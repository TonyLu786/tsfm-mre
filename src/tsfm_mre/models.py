"""Reference forecasters that produce full predictive distributions.

Each model maps a univariate series ``y`` and a set of forecast origins ``t`` to the
predictive distribution of ``y[t + h]`` using only ``y[: t + 1]``. Forecasts for all
origins are computed in a single vectorised pass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

from tsfm_mre.schema import ES_LEVELS, QUANTILE_LEVELS, summarize_samples

_Z_QUANTILES = stats.norm.ppf(np.asarray(QUANTILE_LEVELS))


@dataclass(frozen=True)
class Forecast:
    """Predictive distributions for a batch of origins."""

    mean: np.ndarray
    quantiles: np.ndarray
    es: np.ndarray


def gaussian_forecast(mean: np.ndarray, sd: np.ndarray) -> Forecast:
    """Quantiles and expected shortfall of Gaussian predictive distributions."""
    mean = np.asarray(mean, dtype=float)
    sd = np.asarray(sd, dtype=float)
    quantiles = mean[:, None] + sd[:, None] * _Z_QUANTILES[None, :]
    es = np.empty((mean.size, len(ES_LEVELS)))
    for j, tau in enumerate(ES_LEVELS):
        density = stats.norm.pdf(stats.norm.ppf(tau))
        if tau < 0.5:
            es[:, j] = mean - sd * density / tau
        else:
            es[:, j] = mean + sd * density / (1.0 - tau)
    return Forecast(mean, quantiles, es)


class Forecaster(ABC):
    """Interface of a rolling-origin forecaster."""

    name: str = ""

    @abstractmethod
    def min_history(self, horizon: int) -> int:
        """Smallest origin index for which the model can produce a forecast."""

    @abstractmethod
    def forecast(self, y: np.ndarray, origins: np.ndarray, horizon: int) -> Forecast:
        """Predictive distributions of ``y[t + horizon]`` given ``y[: t + 1]``."""

    def params(self) -> dict[str, Any]:
        """Hyperparameters recorded in run manifests."""
        return {}


class AR1(Forecaster):
    """Gaussian AR(1) estimated by least squares on an expanding or rolling window.

    The h-step predictive distribution is exact under the fitted model:
    mean ``c * sum(phi**j) + phi**h * y_t`` and variance ``sigma**2 * sum(phi**(2j))``
    for ``j = 0, ..., h - 1``.
    """

    name = "ar1"

    def __init__(self, window: int | None = None, min_obs: int = 250):
        if window is not None and window < 10:
            raise ValueError("window must be at least 10 observations")
        self.window = window
        self.min_obs = int(min_obs)

    def params(self) -> dict[str, Any]:
        return {"window": self.window, "min_obs": self.min_obs}

    def min_history(self, horizon: int) -> int:
        return self.min_obs if self.window is None else max(self.min_obs, self.window)

    def forecast(self, y: np.ndarray, origins: np.ndarray, horizon: int) -> Forecast:
        y = np.asarray(y, dtype=float)
        origins = np.asarray(origins, dtype=np.int64)
        shift = y[0]
        x = y[:-1] - shift
        z = y[1:] - shift

        def cumulative(a: np.ndarray) -> np.ndarray:
            return np.concatenate(([0.0], np.cumsum(a)))

        sums = [cumulative(a) for a in (x, z, x * x, x * z, z * z)]
        # Pairs (y[i - 1], y[i]) with i <= t are the first t pairs.
        hi = origins
        lo = np.zeros_like(hi) if self.window is None else np.maximum(hi - self.window, 0)
        n = (hi - lo).astype(float)
        sx, sz, sxx, sxz, szz = (s[hi] - s[lo] for s in sums)

        phi = (n * sxz - sx * sz) / (n * sxx - sx * sx)
        intercept = (sz - phi * sx) / n
        sigma2 = (szz - intercept * sz - phi * sxz) / (n - 2.0)

        powers = phi[:, None] ** np.arange(horizon)[None, :]
        mean = intercept * powers.sum(axis=1) + phi**horizon * (y[origins] - shift) + shift
        sd = np.sqrt(np.maximum(sigma2, 0.0) * (powers**2).sum(axis=1))
        return gaussian_forecast(mean, sd)


class RandomWalkEWMA(Forecaster):
    """Random walk with RiskMetrics exponentially weighted volatility.

    The one-step variance follows ``v_t = decay * v_{t-1} + (1 - decay) * dy_t**2``
    and the h-step predictive distribution is Gaussian with mean ``y_t`` and
    variance ``h * v_t``.
    """

    name = "rw_ewma"

    def __init__(self, decay: float = 0.94, warmup: int = 60):
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must lie in (0, 1)")
        self.decay = float(decay)
        self.warmup = int(warmup)

    def params(self) -> dict[str, Any]:
        return {"decay": self.decay, "warmup": self.warmup}

    def min_history(self, horizon: int) -> int:
        return self.warmup

    def variance_path(self, y: np.ndarray) -> np.ndarray:
        """One-step variance ``v_t`` for every index ``t >= warmup`` (NaN before)."""
        dy2 = np.diff(np.asarray(y, dtype=float)) ** 2
        v = np.full(len(y), np.nan)
        v[self.warmup] = dy2[: self.warmup].mean()
        for t in range(self.warmup + 1, len(y)):
            v[t] = self.decay * v[t - 1] + (1.0 - self.decay) * dy2[t - 1]
        return v

    def forecast(self, y: np.ndarray, origins: np.ndarray, horizon: int) -> Forecast:
        y = np.asarray(y, dtype=float)
        v = self.variance_path(y)
        return gaussian_forecast(y[origins], np.sqrt(horizon * v[origins]))


class HistoricalSimulation(Forecaster):
    """Historical simulation of h-step changes over a rolling window.

    The predictive distribution of ``y[t + h]`` is ``y_t`` plus the empirical
    distribution of the last ``window`` observed h-step changes.
    """

    name = "hs"

    def __init__(self, window: int = 500):
        if window < 20:
            raise ValueError("window must be at least 20 observations")
        self.window = int(window)

    def params(self) -> dict[str, Any]:
        return {"window": self.window}

    def min_history(self, horizon: int) -> int:
        return self.window + horizon - 1

    def forecast(self, y: np.ndarray, origins: np.ndarray, horizon: int) -> Forecast:
        y = np.asarray(y, dtype=float)
        origins = np.asarray(origins, dtype=np.int64)
        changes = y[horizon:] - y[:-horizon]  # changes[k] ends at index k + horizon
        windows = sliding_window_view(changes, self.window)
        first = origins - horizon - self.window + 1  # window ending at index t
        samples = y[origins][:, None] + windows[first]
        mean, quantiles, es = summarize_samples(samples)
        return Forecast(mean, quantiles, es)


MODEL_REGISTRY: dict[str, type[Forecaster]] = {
    "ar1": AR1,
    "rw_ewma": RandomWalkEWMA,
    "hs": HistoricalSimulation,
}


def build_model(spec: dict[str, Any]) -> Forecaster:
    """Instantiate a model from a configuration mapping ``{"name": ..., **params}``.

    An optional ``label`` names the model in outputs, which allows several
    configurations of the same model in one experiment.
    """
    options = dict(spec)
    kind = options.pop("name")
    label = options.pop("label", kind)
    if kind not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"unknown model '{kind}' (available: {known})")
    model = MODEL_REGISTRY[kind](**options)
    model.name = str(label)
    return model
