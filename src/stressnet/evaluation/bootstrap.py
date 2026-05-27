"""Block bootstrap confidence intervals for model parameters."""

from __future__ import annotations

import numpy as np
from typing import Callable


def block_bootstrap_ci(
    statistic_fn: Callable[[np.ndarray, np.ndarray], float],
    x: np.ndarray,
    y: np.ndarray,
    block_size: int = 300,
    n_reps: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Compute block-bootstrap CI for a two-sample statistic.

    Args:
        statistic_fn: Function(x, y) → float.
        x: First time series.
        y: Second time series.
        block_size: Block size in samples.
        n_reps: Number of bootstrap replications.
        alpha: Significance level (e.g. 0.05 for 95% CI).

    Returns:
        (observed, ci_lower, ci_upper)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    observed = statistic_fn(x, y)

    boot_stats = np.zeros(n_reps)
    n_blocks = max(1, n // block_size)
    for i in range(n_reps):
        starts = rng.integers(0, max(1, n - block_size), size=n_blocks)
        xb = np.concatenate([x[s : s + block_size] for s in starts])[:n]
        yb = np.concatenate([y[s : s + block_size] for s in starts])[:n]
        boot_stats[i] = statistic_fn(xb, yb)

    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return float(observed), ci_lower, ci_upper
