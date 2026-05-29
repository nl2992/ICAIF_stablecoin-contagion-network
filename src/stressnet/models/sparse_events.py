"""Sparse-event response analysis for mint/burn and settlement flow nodes.

On-chain mint/burn and large settlement flows produce sparse event arrivals:
typically a handful of events per analysis window.  Continuous lead-lag
cross-correlation has low power on such series.  This module implements an
event-arrival conditional response estimator:

  For each mint/burn event arrival at time t, measure the mean of a
  target feature in a response window (t, t + post_seconds) vs. a
  baseline drawn from the pre-event window (t - baseline_seconds, t).

The estimator naturally handles sparse series because it conditions on
actual arrivals rather than treating zero rows as information.

Outputs
-------
table_sparse_events_{event}.csv with columns:
  source_node_id, target_node_id, feature_col,
  n_events, mean_baseline, mean_response,
  mean_diff, pct_change,
  p_value, significant_p05,
  event_id, event_phase

Usage
-----
Designed for nodes whose primary feature column produces the sparse signal,
e.g.:

  source_node  = usdc_mint_burn  (feature: mint_burn_net_1h)
  target_nodes = curve_3pool, usdc_binance  (feature: usdc_net_sold_1h / basis_vs_usd)

The p-value is computed via a permutation test: shuffle the arrival
timestamps within the event window, recompute mean_diff, and report the
fraction of shuffles with |mean_diff| >= observed.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------

def _extract_sparse_arrival_times(
    panel: pl.DataFrame,
    node_id: str,
    feature_col: str,
    *,
    min_abs_value: float = 0.0,
    grid_col: str = "event_time_seconds",
) -> np.ndarray:
    """Return sorted array of event_time_seconds where |feature| > min_abs_value."""
    node_rows = panel.filter(pl.col("node_id") == node_id)
    if node_rows.height == 0 or feature_col not in node_rows.columns:
        return np.array([], dtype=float)
    arr = (
        node_rows
        .filter(pl.col(feature_col).abs() > min_abs_value)
        .filter(pl.col(feature_col).is_not_null())
        .sort(grid_col)[grid_col]
        .to_numpy()
        .astype(float)
    )
    return arr


def _conditional_response(
    panel: pl.DataFrame,
    target_node_id: str,
    feature_col: str,
    arrival_times: np.ndarray,
    *,
    post_seconds: float,
    baseline_seconds: float,
    grid_col: str = "event_time_seconds",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-arrival baseline and response means for the target node.

    Returns
    -------
    baselines : array of length len(arrival_times)
    responses : array of length len(arrival_times)
        NaN where insufficient data for that window.
    """
    target_rows = panel.filter(pl.col("node_id") == target_node_id)
    if target_rows.height == 0 or feature_col not in target_rows.columns:
        nan = np.full(len(arrival_times), np.nan)
        return nan, nan

    times = target_rows.sort(grid_col)[grid_col].to_numpy().astype(float)
    vals  = target_rows.sort(grid_col)[feature_col].to_numpy().astype(float)

    baselines = np.empty(len(arrival_times))
    responses = np.empty(len(arrival_times))

    for i, t in enumerate(arrival_times):
        pre_mask  = (times >= t - baseline_seconds) & (times < t)
        post_mask = (times >= t) & (times < t + post_seconds)
        baselines[i] = np.nanmean(vals[pre_mask])  if pre_mask.any()  else np.nan
        responses[i] = np.nanmean(vals[post_mask]) if post_mask.any() else np.nan

    return baselines, responses


def _permutation_pvalue(
    panel: pl.DataFrame,
    target_node_id: str,
    feature_col: str,
    arrival_times: np.ndarray,
    observed_diff: float,
    *,
    post_seconds: float,
    baseline_seconds: float,
    n_permutations: int,
    rng: np.random.Generator,
    grid_col: str = "event_time_seconds",
) -> float:
    """Permutation null: shuffle arrival times within the analysis span."""
    if len(arrival_times) == 0 or np.isnan(observed_diff):
        return np.nan

    t_min, t_max = arrival_times.min(), arrival_times.max()
    span = t_max - t_min
    if span <= 0:
        return 1.0

    count = 0
    for _ in range(n_permutations):
        shuffled = t_min + rng.random(len(arrival_times)) * span
        b, r = _conditional_response(
            panel, target_node_id, feature_col, np.sort(shuffled),
            post_seconds=post_seconds, baseline_seconds=baseline_seconds,
            grid_col=grid_col,
        )
        diff = np.nanmean(r) - np.nanmean(b)
        if np.isnan(diff):
            continue
        if abs(diff) >= abs(observed_diff):
            count += 1

    return count / n_permutations


# ---------------------------------------------------------------------------
# Public function: compute sparse event response table
# ---------------------------------------------------------------------------

def compute_sparse_response_table(
    panel: pl.DataFrame,
    *,
    source_node_id: str,
    source_feature: str,
    target_node_ids: list[str],
    target_feature_col: str,
    post_seconds: float = 3 * 3600,          # 3-hour response window
    baseline_seconds: float = 12 * 3600,      # 12-hour baseline
    min_abs_source: float = 0.0,
    n_permutations: int = 500,
    grid_col: str = "event_time_seconds",
    phase: str | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Compute conditional response of target nodes to source-node arrivals.

    Parameters
    ----------
    panel:
        Gold feature panel (event-time indexed, node_id column).
    source_node_id:
        Node that emits sparse events (e.g. usdc_mint_burn).
    source_feature:
        Feature column to threshold for arrival detection (e.g. mint_burn_net_1h).
    target_node_ids:
        Nodes whose response to measure (e.g. curve_3pool, usdc_binance).
    target_feature_col:
        Feature to measure on target nodes (e.g. usdc_net_sold_1h, basis_vs_usd).
    post_seconds:
        Duration of post-arrival response window.
    baseline_seconds:
        Duration of pre-arrival baseline window.
    min_abs_source:
        Minimum absolute value for source feature to count as an arrival.
        Default 0 = any non-zero row counts.
    n_permutations:
        Number of permutation replicates for p-value.
    phase:
        If set, filter panel to rows where event_phase == phase first.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    Polars DataFrame with one row per target node.
    """
    rng = np.random.default_rng(seed)

    if phase and "event_phase" in panel.columns:
        panel = panel.filter(pl.col("event_phase") == phase)

    arrivals = _extract_sparse_arrival_times(
        panel, source_node_id, source_feature,
        min_abs_value=min_abs_source, grid_col=grid_col,
    )
    n_events = len(arrivals)
    logger.info(
        "Sparse arrivals for %s.%s: %d events", source_node_id, source_feature, n_events
    )

    rows = []
    for target_id in target_node_ids:
        if target_id == source_node_id:
            continue
        if n_events == 0:
            rows.append({
                "source_node_id":  source_node_id,
                "target_node_id":  target_id,
                "source_feature":  source_feature,
                "feature_col":     target_feature_col,
                "n_events":        0,
                "mean_baseline":   float("nan"),
                "mean_response":   float("nan"),
                "mean_diff":       float("nan"),
                "pct_change":      float("nan"),
                "p_value":         float("nan"),
                "significant_p05": False,
            })
            continue

        baselines, responses = _conditional_response(
            panel, target_id, target_feature_col, arrivals,
            post_seconds=post_seconds, baseline_seconds=baseline_seconds,
            grid_col=grid_col,
        )
        mean_b = float(np.nanmean(baselines))
        mean_r = float(np.nanmean(responses))
        mean_diff = mean_r - mean_b
        pct_change = (
            float("nan") if abs(mean_b) < 1e-12 else mean_diff / abs(mean_b)
        )

        p_val = _permutation_pvalue(
            panel, target_id, target_feature_col, arrivals, mean_diff,
            post_seconds=post_seconds, baseline_seconds=baseline_seconds,
            n_permutations=n_permutations, rng=rng, grid_col=grid_col,
        )

        rows.append({
            "source_node_id":  source_node_id,
            "target_node_id":  target_id,
            "source_feature":  source_feature,
            "feature_col":     target_feature_col,
            "n_events":        n_events,
            "mean_baseline":   mean_b,
            "mean_response":   mean_r,
            "mean_diff":       mean_diff,
            "pct_change":      pct_change,
            "p_value":         float("nan") if np.isnan(p_val) else float(p_val),
            "significant_p05": (not np.isnan(p_val)) and (float(p_val) < 0.05),
        })
        logger.info(
            "  %s → %s  n=%d  diff=%.4g  p=%.3f",
            source_node_id, target_id, n_events, mean_diff,
            p_val if not np.isnan(p_val) else -1.0,
        )

    if not rows:
        return pl.DataFrame()

    return pl.DataFrame(rows)
