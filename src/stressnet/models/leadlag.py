"""Lead-lag cross-correlation with block-bootstrap inference and BH-FDR correction."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

from stressnet.features.synchronization import synchronized_feature_pivot
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def fdr_correct(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction (step-up procedure).

    Returns:
        reject: Boolean array, True where the null is rejected at FDR level alpha.
        adj_p:  BH-adjusted p-values (monotone, clipped to [0, 1]).
    """
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool), np.array([])

    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    thresholds = (np.arange(1, n + 1) / n) * alpha

    below = sorted_p <= thresholds
    reject_sorted = np.zeros(n, dtype=bool)
    if below.any():
        reject_sorted[: np.where(below)[0].max() + 1] = True

    reject = np.zeros(n, dtype=bool)
    reject[sorted_idx] = reject_sorted

    raw_adj = np.minimum(1.0, sorted_p * n / np.arange(1, n + 1))
    raw_adj = np.minimum.accumulate(raw_adj[::-1])[::-1]
    adj_p = np.empty(n)
    adj_p[sorted_idx] = raw_adj

    return reject, adj_p


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
    block_size = max(1, min(block_size, n))
    n_blocks = max(1, int(np.ceil(n / block_size)))
    null_corrs = np.zeros(n_reps)
    for rep in range(n_reps):
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
    grid_seconds: int = 60,
    max_staleness_seconds: int | None = None,
    fdr_alpha: float = 0.05,
) -> pl.DataFrame:
    """Compute pairwise lead-lag statistics with BH-FDR correction.

    Returns a DataFrame suitable for export as table_leadlag_tests.csv.
    Columns: node_i, node_j, peak_lag_steps, peak_corr, p_value,
             significant_p01, p_value_fdr, significant_fdr.
    """
    results = []
    rng = np.random.default_rng(42)
    needed_nodes = sorted({node for pair in node_pairs for node in pair})
    pivot = synchronized_feature_pivot(
        panel,
        node_ids=needed_nodes,
        feature_col=feature_col,
        grid_seconds=grid_seconds,
        max_staleness_seconds=max_staleness_seconds,
        ts_col=ts_col,
    )

    for node_i, node_j in node_pairs:
        if node_i not in pivot.columns or node_j not in pivot.columns:
            continue
        aligned = pivot.select([node_i, node_j]).drop_nulls()
        xi = aligned[node_i].to_numpy()
        xj = aligned[node_j].to_numpy()
        if len(xi) < 10 or len(xj) < 10:
            continue

        lags, corrs = cross_correlation_lags(xi, xj, max_lag=max_lag)
        peak_idx = np.argmax(np.abs(corrs))
        peak_lag = int(lags[peak_idx])
        peak_corr = float(corrs[peak_idx])

        p_val = block_bootstrap_pvalue(
            xi, xj, peak_lag, block_size=block_size, n_reps=n_reps, rng=rng
        )

        results.append({
            "node_i": node_i,
            "node_j": node_j,
            "feature_col": feature_col,
            "grid_seconds": grid_seconds,
            "peak_lag_steps": peak_lag,
            "peak_lag_seconds": peak_lag * grid_seconds,
            "peak_corr": peak_corr,
            "p_value": p_val,
            "significant_p01": p_val < 0.01,
        })

    if not results:
        return pl.DataFrame(schema={
            "node_i": pl.String, "node_j": pl.String,
            "feature_col": pl.String, "grid_seconds": pl.Int64,
            "peak_lag_steps": pl.Int64, "peak_lag_seconds": pl.Int64, "peak_corr": pl.Float64,
            "p_value": pl.Float64, "significant_p01": pl.Boolean,
            "p_value_fdr": pl.Float64, "significant_fdr": pl.Boolean,
            "p_bonferroni": pl.Float64, "significant_bonferroni": pl.Boolean,
        })

    df = pl.DataFrame(results)
    p_arr = df["p_value"].to_numpy()
    reject, adj_p = fdr_correct(p_arr, alpha=fdr_alpha)
    p_bonferroni = np.minimum(p_arr * len(df), 1.0)
    return df.with_columns(
        pl.Series("p_value_fdr", adj_p),
        pl.Series("significant_fdr", reject),
        pl.Series("p_bonferroni", p_bonferroni),
        pl.Series("significant_bonferroni", p_bonferroni < fdr_alpha),
    )
