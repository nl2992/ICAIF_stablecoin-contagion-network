"""Tests for lead-lag cross-correlation and bootstrap inference."""

import numpy as np
import pytest
from stressnet.models.leadlag import cross_correlation_lags, block_bootstrap_pvalue


def _lagged_signal(n: int = 500, true_lag: int = 5, noise: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Generate a signal x and a lagged version y with noise."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    y = np.roll(x, true_lag) + noise * rng.standard_normal(n)
    y[:true_lag] = rng.standard_normal(true_lag)
    return x, y


def test_cross_correlation_detects_lag():
    x, y = _lagged_signal(true_lag=5)
    lags, corrs = cross_correlation_lags(x, y, max_lag=20)
    peak_lag = int(lags[np.argmax(np.abs(corrs))])
    assert peak_lag == 5


def test_cross_correlation_no_lag():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(200)
    lags, corrs = cross_correlation_lags(x, x, max_lag=10)
    peak_lag = int(lags[np.argmax(np.abs(corrs))])
    assert peak_lag == 0


def test_bootstrap_pvalue_small_for_true_lag():
    x, y = _lagged_signal(n=300, true_lag=5, noise=0.05)
    p = block_bootstrap_pvalue(x, y, lag=5, block_size=30, n_reps=200)
    assert p < 0.1  # should be significant


def test_bootstrap_pvalue_large_for_no_lag():
    rng = np.random.default_rng(99)
    x = rng.standard_normal(200)
    y = rng.standard_normal(200)
    p = block_bootstrap_pvalue(x, y, lag=0, block_size=20, n_reps=200)
    assert p > 0.05  # should not be significant


def test_lags_symmetric():
    x, y = _lagged_signal(n=200)
    lags, corrs = cross_correlation_lags(x, y, max_lag=10)
    assert len(lags) == 21  # -10 to +10 inclusive
    assert lags[0] == -10
    assert lags[-1] == 10
