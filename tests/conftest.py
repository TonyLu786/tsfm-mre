from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


@pytest.fixture
def smoke_config_path() -> Path:
    return CONFIGS / "smoke.yaml"


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


def simulate_ar1(n: int, phi: float, c: float, sigma: float, seed: int) -> np.ndarray:
    """Gaussian AR(1) path started at its mean."""
    gen = np.random.default_rng(seed)
    y = np.empty(n)
    y[0] = c / (1.0 - phi)
    eps = gen.normal(0.0, sigma, n)
    for t in range(1, n):
        y[t] = c + phi * y[t - 1] + eps[t]
    return y
