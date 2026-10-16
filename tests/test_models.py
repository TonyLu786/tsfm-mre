import numpy as np
import pytest
from scipy import integrate, stats

from tests.conftest import simulate_ar1
from tsfm_mre.models import (
    AR1,
    HistoricalSimulation,
    RandomWalkEWMA,
    build_model,
    gaussian_forecast,
)
from tsfm_mre.schema import ES_LEVELS, QUANTILE_LEVELS

MODELS = [AR1(), AR1(window=300), RandomWalkEWMA(), HistoricalSimulation(window=250)]


def test_gaussian_es_matches_numerical_integration():
    fc = gaussian_forecast(np.array([1.0]), np.array([2.0]))
    for j, tau in enumerate(ES_LEVELS):
        q = stats.norm.ppf(tau, 1.0, 2.0)
        if tau < 0.5:
            value = integrate.quad(lambda x: x * stats.norm.pdf(x, 1.0, 2.0), -np.inf, q)[0] / tau
        else:
            value = integrate.quad(lambda x: x * stats.norm.pdf(x, 1.0, 2.0), q, np.inf)[0] / (
                1 - tau
            )
        assert fc.es[0, j] == pytest.approx(value, rel=1e-6)


@pytest.mark.parametrize("window", [None, 300])
def test_ar1_matches_least_squares(window):
    y = simulate_ar1(1500, phi=0.98, c=0.06, sigma=0.05, seed=1)
    origins = np.array([400, 800, 1499 - 5])
    model = AR1(window=window)
    for horizon in (1, 5):
        fc = model.forecast(y, origins, horizon)
        for i, t in enumerate(origins):
            start = 1 if window is None else t - window + 1
            x, z = y[start - 1 : t], y[start : t + 1]
            design = np.column_stack([np.ones_like(x), x])
            (c, phi), *_ = np.linalg.lstsq(design, z, rcond=None)
            sigma2 = np.sum((z - design @ np.array([c, phi])) ** 2) / (len(z) - 2)
            powers = phi ** np.arange(horizon)
            mean = c * powers.sum() + phi**horizon * y[t]
            sd = np.sqrt(sigma2 * np.sum(powers**2))
            assert fc.mean[i] == pytest.approx(mean, rel=1e-9)
            assert fc.quantiles[i, QUANTILE_LEVELS.index(0.5)] == pytest.approx(mean, rel=1e-9)
            assert fc.quantiles[i, -1] - fc.mean[i] == pytest.approx(
                sd * stats.norm.ppf(0.999), rel=1e-7
            )


@pytest.mark.parametrize("model", MODELS, ids=lambda m: f"{m.name}-{m.params()}")
def test_forecasts_use_only_past_data(model):
    y = simulate_ar1(900, phi=0.99, c=0.03, sigma=0.05, seed=2)
    origins = np.arange(600, 880)
    for horizon in (1, 5):
        base = model.forecast(y, origins, horizon)
        k = 40
        t = origins[k]
        perturbed = y.copy()
        perturbed[t + 1 :] += np.random.default_rng(3).normal(0, 1, len(y) - t - 1)
        changed = model.forecast(perturbed, origins, horizon)
        np.testing.assert_allclose(changed.quantiles[: k + 1], base.quantiles[: k + 1])
        np.testing.assert_allclose(changed.es[: k + 1], base.es[: k + 1])


def test_ar1_intervals_are_calibrated_on_an_ar1_process():
    y = simulate_ar1(6000, phi=0.97, c=0.09, sigma=0.05, seed=4)
    origins = np.arange(1000, 5999)
    fc = AR1().forecast(y, origins, 1)
    realised = y[origins + 1]
    inside90 = (realised >= fc.quantiles[:, QUANTILE_LEVELS.index(0.05)]) & (
        realised <= fc.quantiles[:, QUANTILE_LEVELS.index(0.95)]
    )
    exceed = realised > fc.quantiles[:, QUANTILE_LEVELS.index(0.99)]
    assert 0.88 < inside90.mean() < 0.92
    assert 0.005 < exceed.mean() < 0.016


def test_rw_ewma_variance_recursion():
    y = simulate_ar1(400, phi=0.99, c=0.04, sigma=0.05, seed=5)
    model = RandomWalkEWMA(decay=0.9, warmup=30)
    v = model.variance_path(y)
    dy2 = np.diff(y) ** 2
    expected = dy2[:30].mean()
    for t in range(31, 60):
        expected = 0.9 * expected + 0.1 * dy2[t - 1]
    assert v[59] == pytest.approx(expected)
    fc = model.forecast(y, np.array([59]), 4)
    assert fc.mean[0] == y[59]
    assert fc.quantiles[0, -1] - y[59] == pytest.approx(
        np.sqrt(4 * expected) * stats.norm.ppf(0.999)
    )


def test_historical_simulation_uses_the_last_window():
    y = simulate_ar1(700, phi=0.99, c=0.04, sigma=0.05, seed=6)
    model = HistoricalSimulation(window=200)
    t, h = 650, 5
    fc = model.forecast(y, np.array([t]), h)
    changes = y[t - 200 + 1 : t + 1] - y[t - 200 + 1 - h : t + 1 - h]
    expected = np.quantile(y[t] + changes, QUANTILE_LEVELS)
    np.testing.assert_allclose(fc.quantiles[0], expected)
    assert fc.es[0, 0] <= fc.quantiles[0, QUANTILE_LEVELS.index(0.01)]


def test_build_model_with_label_and_errors():
    model = build_model({"name": "ar1", "label": "ar1_rolling", "window": 500})
    assert model.name == "ar1_rolling" and model.window == 500
    with pytest.raises(ValueError, match="unknown model"):
        build_model({"name": "garch"})
    with pytest.raises(TypeError):
        build_model({"name": "hs", "lags": 3})
