"""Time-Varying Parameter VAR via rolling windows or exponential forgetting factors.

Supports three window_type modes (from configs/models.yaml):
  rolling           – fit a fresh VAR on each sliding window
  forgetting_factor – recursive least squares with forgetting factor λ
  kalman            – alias for forgetting_factor (same RLS core)
"""

from __future__ import annotations

import numpy as np
import polars as pl

from stressnet.models.var_granger import fit_var, fevd_spillover_table
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Rolling-window approach
# ---------------------------------------------------------------------------

def rolling_var_spillovers(
    data: np.ndarray,
    node_names: list[str],
    timestamps: np.ndarray,
    window_size: int = 3600,
    step_size: int = 300,
    max_lags: int = 5,
    fevd_horizon: int = 10,
) -> pl.DataFrame:
    """Fit independent VAR models on sliding windows and collect FEVD shares.

    Args:
        data: (T, N) array already aligned to the panel grid (NaN for missing).
        node_names: Length-N list of node identifiers.
        timestamps: Length-T array of event_time_seconds values.
        window_size: Number of rows per window.
        step_size: Stride between window starts.
        max_lags: Maximum VAR lag order tried inside each window.
        fevd_horizon: FEVD steps ahead.

    Returns:
        DataFrame with columns: window_center, caused_node, causing_node,
        fevd_share, method.
    """
    n_obs = len(timestamps)
    rows: list[dict] = []

    for start in range(0, max(1, n_obs - window_size), step_size):
        end = start + window_size
        if end > n_obs:
            break
        window = data[start:end].copy()
        center = float(timestamps[start + window_size // 2])

        if np.isnan(window).mean() > 0.3:
            continue

        # Forward-fill NaNs column by column
        for col in range(window.shape[1]):
            mask = np.isnan(window[:, col])
            if mask.all():
                window[:, col] = 0.0
            elif mask.any():
                idx = np.where(~mask)[0]
                window[:, col] = np.interp(np.arange(window.shape[0]), idx, window[idx, col])

        try:
            var_fit = fit_var(window, node_names, max_lags=max_lags)
            fevd_df = fevd_spillover_table(var_fit, horizon=fevd_horizon)
            for row in fevd_df.iter_rows(named=True):
                rows.append({
                    "window_center": center,
                    "caused_node": row["caused_node"],
                    "causing_node": row["causing_node"],
                    "fevd_share": row["fevd_share"],
                    "method": "tvp_var_rolling",
                })
        except Exception as exc:
            logger.debug("Rolling VAR at window start=%d failed: %s", start, exc)

    _SCHEMA = {
        "window_center": pl.Float64,
        "caused_node": pl.String,
        "causing_node": pl.String,
        "fevd_share": pl.Float64,
        "method": pl.String,
    }
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema=_SCHEMA)


# ---------------------------------------------------------------------------
# Forgetting-factor approach (Recursive Least Squares)
# ---------------------------------------------------------------------------

def forgetting_factor_var_spillovers(
    data: np.ndarray,
    node_names: list[str],
    timestamps: np.ndarray,
    forgetting_factor: float = 0.99,
    max_lags: int = 3,
    fevd_horizon: int = 10,
    checkpoint_every: int = 300,
) -> pl.DataFrame:
    """Time-varying VAR via RLS with exponential forgetting.

    Weights observation at time t-s by λ^s, giving a smoothly time-varying
    coefficient estimate. Records FEVD proxy at every `checkpoint_every` steps.
    """
    data = data.copy()
    n_obs, n_vars = data.shape
    p = max_lags

    # Impute with column means so RLS never sees NaN
    col_means = np.nanmean(data, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    for col in range(n_vars):
        nan_mask = np.isnan(data[:, col])
        data[nan_mask, col] = col_means[col]

    n_params = n_vars * p + 1  # p lags × n_vars + intercept
    B = np.zeros((n_params, n_vars))
    P = [np.eye(n_params) * 100.0 for _ in range(n_vars)]
    lam = forgetting_factor

    rows: list[dict] = []

    for t in range(p, n_obs):
        x_t = np.concatenate([data[t - lag - 1] for lag in range(p)] + [np.ones(1)])
        y_t = data[t]

        for eq in range(n_vars):
            Px = P[eq] @ x_t
            denom = lam + x_t @ Px
            K = Px / denom
            e = y_t[eq] - B[:, eq] @ x_t
            B[:, eq] += K * e
            P[eq] = (P[eq] - np.outer(Px, Px) / denom) / lam

        if (t - p) % checkpoint_every == 0:
            center = float(timestamps[t])
            coefs = B[:-1].reshape(p, n_vars, n_vars)
            try:
                for entry in _approx_fevd_from_coefs(coefs, node_names):
                    rows.append({**entry, "window_center": center, "method": "tvp_var_ff"})
            except Exception as exc:
                logger.debug("FF-VAR FEVD at t=%d failed: %s", t, exc)

    _SCHEMA = {
        "window_center": pl.Float64,
        "caused_node": pl.String,
        "causing_node": pl.String,
        "fevd_share": pl.Float64,
        "method": pl.String,
    }
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema=_SCHEMA)


def _approx_fevd_from_coefs(coefs: np.ndarray, node_names: list[str]) -> list[dict]:
    """Normalized absolute coefficient sum as FEVD proxy."""
    abs_sum = np.abs(coefs).sum(axis=0)  # (n_vars, n_vars)
    row_sums = abs_sum.sum(axis=1, keepdims=True) + 1e-12
    shares = abs_sum / row_sums
    return [
        {"caused_node": caused, "causing_node": causing, "fevd_share": float(shares[i, j])}
        for i, caused in enumerate(node_names)
        for j, causing in enumerate(node_names)
    ]


# ---------------------------------------------------------------------------
# Dispatch and summary
# ---------------------------------------------------------------------------

def run_tvp_var(
    data: np.ndarray,
    node_names: list[str],
    timestamps: np.ndarray,
    window_type: str = "rolling",
    window_size: int = 3600,
    step_size: int = 300,
    forgetting_factor: float = 0.99,
    max_lags: int = 5,
    fevd_horizon: int = 10,
    checkpoint_every: int = 300,
) -> pl.DataFrame:
    """Top-level dispatcher: choose rolling or forgetting_factor mode."""
    if window_type in ("forgetting_factor", "kalman"):
        return forgetting_factor_var_spillovers(
            data, node_names, timestamps,
            forgetting_factor=forgetting_factor,
            max_lags=max_lags,
            fevd_horizon=fevd_horizon,
            checkpoint_every=checkpoint_every,
        )
    return rolling_var_spillovers(
        data, node_names, timestamps,
        window_size=window_size,
        step_size=step_size,
        max_lags=max_lags,
        fevd_horizon=fevd_horizon,
    )


def tvp_var_summary(tvp_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-window FEVD shares into a single summary table.

    Returns mean, max, std, and window count per (causing_node, caused_node) pair,
    excluding the diagonal.
    """
    _SCHEMA = {
        "caused_node": pl.String,
        "causing_node": pl.String,
        "fevd_share_mean": pl.Float64,
        "fevd_share_max": pl.Float64,
        "fevd_share_std": pl.Float64,
        "n_windows": pl.UInt32,
        "method": pl.String,
    }
    if tvp_df.is_empty():
        return pl.DataFrame(schema=_SCHEMA)

    off_diag = tvp_df.filter(pl.col("caused_node") != pl.col("causing_node"))
    if off_diag.is_empty():
        return pl.DataFrame(schema=_SCHEMA)

    return (
        off_diag.group_by(["caused_node", "causing_node"])
        .agg([
            pl.col("fevd_share").mean().alias("fevd_share_mean"),
            pl.col("fevd_share").max().alias("fevd_share_max"),
            pl.col("fevd_share").std().alias("fevd_share_std"),
            pl.col("fevd_share").count().cast(pl.UInt32).alias("n_windows"),
            pl.col("method").first().alias("method"),
        ])
        .sort("fevd_share_mean", descending=True)
    )
