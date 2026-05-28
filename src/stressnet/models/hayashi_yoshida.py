"""Hayashi-Yoshida covariance for asynchronous time-series robustness."""

from __future__ import annotations

import numpy as np
import polars as pl


def hayashi_yoshida_covariance(
    t_x: np.ndarray,
    x: np.ndarray,
    t_y: np.ndarray,
    y: np.ndarray,
) -> float:
    """Estimate covariance using overlapping asynchronous return intervals."""
    if len(x) < 2 or len(y) < 2:
        return 0.0

    dx = np.diff(x)
    dy = np.diff(y)
    sx0, sx1 = t_x[:-1], t_x[1:]
    sy0, sy1 = t_y[:-1], t_y[1:]

    cov = 0.0
    j_start = 0
    for i in range(len(dx)):
        while j_start < len(dy) and sy1[j_start] <= sx0[i]:
            j_start += 1
        j = j_start
        while j < len(dy) and sy0[j] < sx1[i]:
            if sx0[i] < sy1[j] and sy0[j] < sx1[i]:
                cov += float(dx[i] * dy[j])
            j += 1
    return cov


def hayashi_yoshida_correlation(
    t_x: np.ndarray,
    x: np.ndarray,
    t_y: np.ndarray,
    y: np.ndarray,
) -> float:
    """Return HY correlation, normalized by asynchronous self-covariances."""
    cov_xy = hayashi_yoshida_covariance(t_x, x, t_y, y)
    cov_xx = hayashi_yoshida_covariance(t_x, x, t_x, x)
    cov_yy = hayashi_yoshida_covariance(t_y, y, t_y, y)
    denom = np.sqrt(max(cov_xx, 0.0) * max(cov_yy, 0.0))
    if denom <= 0:
        return 0.0
    return float(cov_xy / denom)


def compute_hayashi_yoshida_table(
    panel: pl.DataFrame,
    node_pairs: list[tuple[str, str]],
    *,
    feature_col: str = "basis_vs_usd",
    ts_col: str = "event_time_seconds",
) -> pl.DataFrame:
    """Compute pairwise HY correlations for asynchronous node observations."""
    rows = []
    for node_i, node_j in node_pairs:
        xi_df = (
            panel.filter(pl.col("node_id") == node_i)
            .select([ts_col, feature_col])
            .drop_nulls()
            .sort(ts_col)
        )
        xj_df = (
            panel.filter(pl.col("node_id") == node_j)
            .select([ts_col, feature_col])
            .drop_nulls()
            .sort(ts_col)
        )
        if xi_df.height < 3 or xj_df.height < 3:
            continue
        corr = hayashi_yoshida_correlation(
            xi_df[ts_col].to_numpy().astype(float),
            xi_df[feature_col].to_numpy().astype(float),
            xj_df[ts_col].to_numpy().astype(float),
            xj_df[feature_col].to_numpy().astype(float),
        )
        rows.append(
            {
                "node_i": node_i,
                "node_j": node_j,
                "feature_col": feature_col,
                "hy_corr": corr,
                "n_i": xi_df.height,
                "n_j": xj_df.height,
            }
        )
    if not rows:
        return pl.DataFrame(
            schema={
                "node_i": pl.String,
                "node_j": pl.String,
                "feature_col": pl.String,
                "hy_corr": pl.Float64,
                "n_i": pl.Int64,
                "n_j": pl.Int64,
            }
        )
    return pl.DataFrame(rows)
