import numpy as np
import pytest

from tsfm_mre.backtest import (
    TailForecast,
    acerbi_szekely_test,
    acerbi_szekely_z2,
    basel_zone,
    basel_zone_bounds,
    christoffersen_test,
    kupiec_test,
)
from tsfm_mre.models import gaussian_forecast
from tsfm_mre.schema import ES_LEVELS, QUANTILE_LEVELS


def test_basel_zones_reproduce_the_basel_table():
    assert basel_zone_bounds(250, 0.01) == (5, 10)
    assert [basel_zone(k, 250, 0.01) for k in (4, 5, 9, 10)] == ["green", "yellow", "yellow", "red"]
    yellow, red = basel_zone_bounds(500, 0.01)
    assert yellow > 5 and red > 10


def test_kupiec_statistic():
    hits = np.zeros(250, dtype=int)
    hits[:10] = 1
    lr, p = kupiec_test(hits, 0.01)
    expected = -2 * (
        240 * np.log(0.99) + 10 * np.log(0.01) - 240 * np.log(0.96) - 10 * np.log(0.04)
    )
    assert lr == pytest.approx(expected)
    assert p < 0.01
    expected_rate = np.zeros(250, dtype=int)
    expected_rate[:3] = 1
    assert kupiec_test(expected_rate, 0.01)[1] > 0.5


def test_kupiec_and_christoffersen_size(rng):
    rejections = np.zeros(3)
    reps = 1000
    for _ in range(reps):
        hits = (rng.random(1000) < 0.05).astype(int)
        _, p_uc = kupiec_test(hits, 0.05)
        markov = christoffersen_test(hits, 0.05)
        rejections += [p_uc < 0.05, markov["p_ind"] < 0.05, markov["p_cc"] < 0.05]
    rates = rejections / reps
    assert np.all(rates < 0.08)
    assert np.all(rates > 0.02)


def test_christoffersen_detects_clustering():
    hits = np.zeros(1000, dtype=int)
    for start in range(50, 1000, 100):
        hits[start : start + 3] = 1
    assert christoffersen_test(hits, 0.03)["p_ind"] < 0.001


def test_z2_statistic_by_hand():
    losses = np.array([0.0, 3.0, 0.5, 4.0])
    var = np.ones(4)
    es = np.full(4, 2.0)
    expected = 1.0 - (3.0 / 2.0 + 4.0 / 2.0) / (4 * 0.5)
    assert acerbi_szekely_z2(losses, var, es, 0.5) == pytest.approx(expected)


def _es_rejection_rate(true_sd: float, reps: int, rng: np.random.Generator) -> np.ndarray:
    n, alpha = 500, 0.025
    j = ES_LEVELS.index(0.975)
    k = QUANTILE_LEVELS.index(0.975)
    fc = gaussian_forecast(np.zeros(n), np.ones(n))
    tail = TailForecast(np.zeros(n), fc.quantiles, "upper")
    p_values = []
    for _ in range(reps):
        y = rng.normal(0.0, true_sd, n)
        _, p = acerbi_szekely_test(y, fc.quantiles[:, k], fc.es[:, j], alpha, tail, 499, rng)
        p_values.append(p)
    return np.asarray(p_values)


def test_acerbi_szekely_is_calibrated_under_the_null(rng):
    p = _es_rejection_rate(1.0, 300, rng)
    assert 0.02 < np.mean(p < 0.05) < 0.09
    assert 0.42 < p.mean() < 0.58


def test_acerbi_szekely_detects_underestimated_tails(rng):
    p = _es_rejection_rate(1.4, 100, rng)
    assert np.mean(p < 0.05) > 0.9


def test_es_must_be_positive():
    fc = gaussian_forecast(np.zeros(5), np.ones(5))
    tail = TailForecast(np.zeros(5), fc.quantiles, "upper")
    with pytest.raises(ValueError, match="positive"):
        acerbi_szekely_test(
            np.zeros(5), np.ones(5), -np.ones(5), 0.025, tail, 99, np.random.default_rng(0)
        )
