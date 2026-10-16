import numpy as np
import pytest
from scipy import stats

from tsfm_mre.models import gaussian_forecast
from tsfm_mre.scoring import (
    LEVELS,
    crps_from_quantiles,
    interval_hits,
    level_weights,
    pinball_loss,
    pit_values,
    quantile_function,
    tail_quantile_loss,
)


def gaussian_crps(y, mu=0.0, sigma=1.0):
    z = (y - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def test_pinball_loss_values():
    assert pinball_loss(1.0, 0.0, 0.9) == pytest.approx(0.9)
    assert pinball_loss(-1.0, 0.0, 0.9) == pytest.approx(0.1)
    assert pinball_loss(0.0, 0.0, 0.3) == 0.0


def test_level_weights_sum_to_one():
    assert level_weights().sum() == pytest.approx(1.0)
    assert np.all(level_weights() > 0)


def test_crps_matches_closed_form_for_gaussian_forecasts():
    y = np.random.default_rng(1).normal(size=50_000)
    fc = gaussian_forecast(np.zeros_like(y), np.ones_like(y))
    approx = crps_from_quantiles(y, fc.quantiles)
    exact = gaussian_crps(y)
    assert approx.mean() == pytest.approx(exact.mean(), rel=0.02)
    assert np.corrcoef(approx, exact)[0, 1] > 0.999


@pytest.mark.parametrize(("shift", "scale"), [(0.25, 1.0), (0.0, 1.3), (0.0, 0.75), (-2.0, 1.0)])
def test_crps_is_proper(shift, scale):
    y = np.random.default_rng(2).normal(size=50_000)
    truth = gaussian_forecast(np.zeros_like(y), np.ones_like(y))
    other = gaussian_forecast(np.full_like(y, shift), np.full_like(y, scale))
    assert (
        crps_from_quantiles(y, truth.quantiles).mean()
        < crps_from_quantiles(y, other.quantiles).mean()
    )


def test_tail_quantile_loss_prefers_correct_var():
    y = np.random.default_rng(3).normal(size=50_000)
    q = stats.norm.ppf([0.01, 0.99])
    correct = tail_quantile_loss(y, np.full_like(y, q[0]), np.full_like(y, q[1]), 0.01).mean()
    narrow = tail_quantile_loss(y, np.full_like(y, 0.7 * q[0]), np.full_like(y, 0.7 * q[1]), 0.01)
    assert correct < narrow.mean()


def test_pit_is_uniform_for_correct_forecasts():
    y = np.random.default_rng(4).normal(size=20_000)
    fc = gaussian_forecast(np.zeros_like(y), np.ones_like(y))
    pit = pit_values(y, fc.quantiles)
    counts, _ = np.histogram(pit, bins=10, range=(0, 1))
    assert stats.chisquare(counts).pvalue > 0.001
    assert np.all((pit > 0) & (pit < 1))


def test_randomised_pit_handles_rounded_observations():
    gen = np.random.default_rng(5)
    y = np.round(gen.normal(size=40_000), 1)  # observations recorded to 0.1
    fc = gaussian_forecast(np.zeros_like(y), np.ones_like(y))
    raw, _ = np.histogram(pit_values(y, fc.quantiles), bins=10, range=(0, 1))
    randomised, _ = np.histogram(
        pit_values(y, fc.quantiles, resolution=0.1, rng=gen), bins=10, range=(0, 1)
    )
    # Chi-square statistics with 9 degrees of freedom: rounding distorts the plain PIT,
    # the randomised PIT stays close to uniform.
    assert stats.chisquare(raw).statistic > 300
    assert stats.chisquare(randomised).statistic < 40


def test_interval_hits_and_quantile_function():
    fc = gaussian_forecast(np.zeros(3), np.ones(3))
    y = np.array([0.0, 2.0, -3.0])
    assert interval_hits(y, fc.quantiles, 0.9).tolist() == [True, False, False]
    rows = np.array([0, 1, 2])
    at_grid = quantile_function(fc.quantiles, rows, np.array([0.025, 0.5, 0.975]))
    assert at_grid == pytest.approx(stats.norm.ppf([0.025, 0.5, 0.975]))
    mid = quantile_function(fc.quantiles, rows[:1], np.array([0.15]))
    expected = 0.5 * (
        fc.quantiles[0, LEVELS.tolist().index(0.1)] + fc.quantiles[0, LEVELS.tolist().index(0.2)]
    )
    assert mid[0] == pytest.approx(expected)
