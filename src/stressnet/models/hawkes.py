"""Multivariate Hawkes process estimation for stress event contagion."""

from __future__ import annotations

import numpy as np
import polars as pl

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from tick.hawkes import HawkesExpKern
    _HAS_TICK = True
except ImportError:
    _HAS_TICK = False
    logger.warning("tick not installed; Hawkes estimation unavailable. Install with: pip install tick")


def define_stress_events(
    panel: pl.DataFrame,
    node_ids: list[str],
    basis_col: str = "basis_vs_usd",
    threshold_bps: float = 10.0,
    ts_col: str = "event_time_seconds",
) -> dict[str, np.ndarray]:
    """Extract stress event arrival times for each node.

    A stress event at node i is defined as |basis_vs_usd| crossing threshold_bps
    (i.e. a new exceedance that was not present at the previous timestep).

    Returns:
        Dict mapping node_id → 1D array of event arrival times (seconds).
    """
    threshold_logpoints = threshold_bps / 10_000
    events: dict[str, np.ndarray] = {}

    for node_id in node_ids:
        node_df = (
            panel.filter(pl.col("node_id") == node_id)
            .sort(ts_col)
            .select([ts_col, basis_col])
            .drop_nulls()
        )
        if node_df.height < 2:
            events[node_id] = np.array([])
            continue

        ts = node_df[ts_col].to_numpy()
        basis = node_df[basis_col].to_numpy()
        exceedance = np.abs(basis) > threshold_logpoints
        # New crossings only (off→on transitions)
        crossings = exceedance & ~np.roll(exceedance, 1)
        crossings[0] = exceedance[0]  # first observation counts if already exceeded
        events[node_id] = ts[crossings]

    return events


def fit_hawkes(
    events: dict[str, np.ndarray],
    decay: float = 1.0,
    max_iter: int = 1000,
) -> dict:
    """Fit a multivariate exponential Hawkes model.

    Args:
        events: Dict mapping node_id → arrival time array.
        decay: Exponential kernel decay rate.
        max_iter: Maximum EM iterations.

    Returns:
        Dict with keys: node_ids, baselines, kernels, branching_ratios.
    """
    if not _HAS_TICK:
        raise ImportError("tick is required for Hawkes estimation.")

    node_ids = list(events.keys())
    realizations = [[events[n]] for n in node_ids]

    learner = HawkesExpKern(decays=decay, max_iter=max_iter, verbose=False)
    learner.fit(realizations)

    # branching ratios: n_ij = alpha_ij / decay
    kernels = learner.adjacency  # (N, N) array
    branching_ratios = kernels / decay if decay > 0 else kernels

    return {
        "node_ids": node_ids,
        "baselines": learner.baseline.tolist(),
        "kernels": kernels.tolist(),
        "branching_ratios": branching_ratios.tolist(),
        "decay": decay,
    }


def hawkes_results_table(fit: dict) -> pl.DataFrame:
    """Convert Hawkes fit results to a paper-ready DataFrame.

    Returns a DataFrame with columns:
        node_i, node_j, baseline_i, kernel_ij, branching_ratio_ij.
    """
    names = fit["node_ids"]
    br = np.array(fit["branching_ratios"])
    k = np.array(fit["kernels"])
    bl = fit["baselines"]

    rows = []
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i == j:
                continue
            rows.append({
                "node_i": ni,
                "node_j": nj,
                "baseline_i": float(bl[i]),
                "kernel_ij": float(k[i, j]),
                "branching_ratio_ij": float(br[i, j]),
                "contagious": float(br[i, j]) > 0.1,
            })
    return pl.DataFrame(rows).sort("branching_ratio_ij", descending=True)
