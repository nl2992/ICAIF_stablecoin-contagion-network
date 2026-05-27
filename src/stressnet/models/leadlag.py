"""Lead-lag cross-correlation with block-bootstrap inference."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def cross_correlation_lags(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cross-correlation between x and y at lags -max_lag to +max_lag.

    Positive lag k means x leads y by k steps (correlation at y[t] ~ x[t-k]).

    Returns:
        lags: Array of lag values from -max_lag to +max_lag.
        corrs: Cross-correlation coefficient at each lag.
    """
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = np.zeros(len(lags))

    x_z = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
    y_z = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-12)

    for i, lag in enumerate(lags):
        if lag > 0:
            xv, yv = x_z[:-lag], y_z[lag:]
        elif lag < 0:
            xv, yv = x_z[-lag:], y_z[:lag]
        else:
            xv, yv = x_z, y_z
        mask = ~(np.isnan(xv) | np.isnan(yv))
        corrs[i] = np.corrcoef(xv[mask], yv[mask])[0, 1] if mask.sum() > 2 else 0.0

    return lags, corrs


def block_bootstrap_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    lag: int,
    block_size: int = 300,
    n_reps: int = 1000,
    rng: np.random.Generator | None = None,
) -> float:
    """Block-bootstrap p-value for the cross-correlation at a given lag.

    Null hypothesis: no cross-correlation between x and y at this lag.
    Block size should be ~5 minutes for 1-second data (block_size=300).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if lag > 0:
        x_obs, y_obs = x[:-lag], y[lag:]
    elif lag < 0:
        x_obs, y_obs = x[-lag:], y[:lag]
    else:
        x_obs, y_obs = x, y

    mask = ~(np.isnan(x_obs) | np.isnan(y_obs))
    x_obs, y_obs = x_obs[mask], y_obs[mask]
    if len(x_obs) < 2:
        return 1.0

    observed = np.corrcoef(x_obs, y_obs)[0, 1]

    n = len(x_obs)
    n_blocks = max(1, n // block_size)
    null_corrs = np.zeros(n_reps)
    for rep in range(n_reps):
        # Shuffle blocks of y, keeping x fixed
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        y_shuffled = np.concatenate([y_obs[s : s + block_size] for s in starts])[:n]
        if len(y_shuffled) < 2:
            continue
        null_corrs[rep] = np.corrcoef(x_obs[: len(y_shuffled)], y_shuffled)[0, 1]

    p_value = np.mean(np.abs(null_corrs) >= abs(observed))
    return float(p_value)


def compute_leadlag_table(
    panel: pl.DataFrame,
    node_pairs: list[tuple[str, str]],
    feature_col: str = "basis_vs_usd",
    max_lag: int = 60,
    block_size: int = 300,
    n_reps: int = 1000,
    ts_col: str = "event_time_seconds",
) -> pl.DataFrame:
    """Compute pairwise lead-lag statistics for a list of (node_i, node_j) pairs.

    Returns a DataFrame suitable for export as table_leadlag_tests.csv.
    """
    results = []
    rng = np.random.default_rng(42)

    for node_i, node_j in node_pairs:
        xi = (
            panel.filter(pl.col("node_id") == node_i)
            .sort(ts_col)[feature_col]
            .to_numpy()
        )
        xj = (
            panel.filter(pl.col("node_id") == node_j)
            .sort(ts_col)[feature_col]
            .to_numpy()
        )
        if len(xi) < 10 or len(xj) < 10:
            continue

        min_len = min(len(xi), len(xj))
        xi, xj = xi[:min_len], xj[:min_len]

        lags, corrs = cross_correlation_lags(xi, xj, max_lag=max_lag)
        peak_idx = np.argmax(np.abs(corrs))
        peak_lag = int(lags[peak_idx])
        peak_corr = float(corrs[peak_idx])

        p_val = block_bootstrap_pvalue(xi, xj, peak_lag, block_size=block_size,
                                       n_reps=n_reps, rng=rng)

        results.append({
            "node_i": node_i,
            "node_j": node_j,
            "peak_lag_steps": peak_lag,
            "peak_corr": peak_corr,
            "p_value": p_val,
            "significant_p01": p_val < 0.01,
        })

    return pl.DataFrame(results)
