import numpy as np
import pytest

from tsfm_mre.compare import (
    diebold_mariano,
    newey_west_variance,
    stationary_bootstrap_indices,
    superior_predictive_ability,
)


def test_stationary_bootstrap_block_lengths(rng):
    idx = stationary_bootstrap_indices(400, 200, 8.0, rng)
    assert idx.shape == (200, 400)
    assert idx.min() >= 0 and idx.max() < 400
    continues = (idx[:, 1:] == (idx[:, :-1] + 1) % 400).mean()
    assert continues == pytest.approx(1 - 1 / 8.0, abs=0.02)


def test_newey_west_without_lags_is_the_variance():
    x = np.random.default_rng(0).normal(size=500)
    assert newey_west_variance(x, 0) == pytest.approx(np.var(x))


def test_diebold_mariano_size_and_power(rng):
    rejections = sum(
        diebold_mariano(rng.normal(size=300), rng.normal(size=300), 1)[1] < 0.05 for _ in range(500)
    )
    assert rejections / 500 < 0.08
    a = rng.normal(size=500) ** 2
    b = (1.5 * rng.normal(size=500)) ** 2
    stat, p = diebold_mariano(a, b, 1)
    assert stat < 0 and p < 0.01


def _rates(mean_gain: np.ndarray, reps: int, rng: np.random.Generator) -> dict[str, float]:
    keys = ("rc_p", "spa_p", "spa_p_lower", "spa_p_upper")
    rejections = dict.fromkeys(keys, 0)
    for _ in range(reps):
        n = 300
        bench = rng.normal(size=n)
        models = bench[:, None] - mean_gain[None, :] + rng.normal(size=(n, mean_gain.size))
        result = superior_predictive_ability(bench, models, 199, 5.0, rng)
        assert result["spa_p_lower"] <= result["spa_p"] <= result["spa_p_upper"]
        for key in keys:
            rejections[key] += result[key] < 0.05
    return {k: v / reps for k, v in rejections.items()}


def test_rc_and_spa_hold_size_when_no_model_is_better(rng):
    rates = _rates(np.zeros(3), 200, rng)
    assert rates["rc_p"] < 0.08 and rates["spa_p"] < 0.08


def test_rc_and_spa_rarely_reject_with_inferior_models(rng):
    rates = _rates(np.array([-0.3, -0.3, 0.0]), 200, rng)
    assert rates["spa_p"] < 0.08
    assert rates["rc_p"] < 0.08


def test_spa_detects_a_better_model(rng):
    rates = _rates(np.array([0.3, -0.2, -0.2]), 100, rng)
    assert rates["spa_p"] > 0.9
