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

    A stress event is a new exceedance: |basis| crosses threshold_bps from below.

    Returns:
        Dict mapping node_id → sorted 1D array of event arrival times (seconds).
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
        crossings = exceedance & ~np.roll(exceedance, 1)
        crossings[0] = exceedance[0]
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
        Dict with keys: node_ids, baselines, kernels, branching_ratios, decay.
    """
    if not _HAS_TICK:
        raise ImportError("tick is required for Hawkes estimation.")

    node_ids = list(events.keys())
    realizations = [[events[n]] for n in node_ids]

    learner = HawkesExpKern(decays=decay, max_iter=max_iter, verbose=False)
    learner.fit(realizations)

    kernels = learner.adjacency  # (N, N)
    branching_ratios = kernels / decay if decay > 0 else kernels

    return {
        "node_ids": node_ids,
        "baselines": learner.baseline.tolist(),
        "kernels": kernels.tolist(),
        "branching_ratios": branching_ratios.tolist(),
        "decay": decay,
    }


def hawkes_bootstrap_ci(
    events: dict[str, np.ndarray],
    fit: dict,
    n_bootstraps: int = 100,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict[tuple[str, str], tuple[float, float]]:
    """Non-parametric bootstrap CI for Hawkes branching ratios.

    Resamples event sequences with replacement (block bootstrap within each node),
    refits the Hawkes model, and returns percentile CIs.

    Returns:
        Dict mapping (node_i, node_j) → (lower_ci, upper_ci).
        Returns (NaN, NaN) for all pairs if tick is unavailable.
    """
    node_ids = fit["node_ids"]
    nan_ci = {(ni, nj): (float("nan"), float("nan"))
              for ni in node_ids for nj in node_ids if ni != nj}

    if not _HAS_TICK:
        return nan_ci

    if rng is None:
        rng = np.random.default_rng(42)

    n_nodes = len(node_ids)
    br_samples = np.full((n_bootstraps, n_nodes, n_nodes), np.nan)

    for b in range(n_bootstraps):
        boot_events: dict[str, np.ndarray] = {}
        for nid in node_ids:
            arr = events[nid]
            if len(arr) == 0:
                boot_events[nid] = arr
                continue
            boot_idx = rng.choice(len(arr), size=len(arr), replace=True)
            boot_events[nid] = np.sort(arr[boot_idx])

        try:
            boot_fit = fit_hawkes(boot_events, decay=fit["decay"])
            br_samples[b] = np.array(boot_fit["branching_ratios"])
        except Exception as exc:
            logger.debug("Hawkes bootstrap rep %d failed: %s", b, exc)

    alpha = 1.0 - ci_level
    lower = np.nanpercentile(br_samples, 100 * alpha / 2, axis=0)
    upper = np.nanpercentile(br_samples, 100 * (1 - alpha / 2), axis=0)

    ci: dict[tuple[str, str], tuple[float, float]] = {}
    for i, ni in enumerate(node_ids):
        for j, nj in enumerate(node_ids):
            if ni != nj:
                ci[(ni, nj)] = (float(lower[i, j]), float(upper[i, j]))
    return ci


def hawkes_results_table(
    fit: dict,
    ci: dict[tuple[str, str], tuple[float, float]] | None = None,
) -> pl.DataFrame:
    """Convert Hawkes fit results to a paper-ready DataFrame.

    Args:
        fit: Output of fit_hawkes().
        ci: Optional output of hawkes_bootstrap_ci(). When provided, adds
            branching_ratio_ci_lower and branching_ratio_ci_upper columns.
            Pass None to omit CI columns.

    Returns:
        DataFrame with columns: node_i, node_j, baseline_i, kernel_ij,
        branching_ratio_ij, contagious[, ci_lower, ci_upper, ci_excludes_zero].
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
            br_val = float(br[i, j])
            row: dict = {
                "node_i": ni,
                "node_j": nj,
                "baseline_i": float(bl[i]),
                "kernel_ij": float(k[i, j]),
                "branching_ratio_ij": br_val,
                "contagious": br_val > 0.1,
            }
            if ci is not None:
                lo, hi = ci.get((ni, nj), (float("nan"), float("nan")))
                row["ci_lower"] = lo
                row["ci_upper"] = hi
                row["ci_excludes_zero"] = (not np.isnan(lo)) and lo > 0.0
                row["contagious"] = row["contagious"] and row["ci_excludes_zero"]
            rows.append(row)

    if not rows:
        schema: dict = {
            "node_i": pl.String, "node_j": pl.String,
            "baseline_i": pl.Float64, "kernel_ij": pl.Float64,
            "branching_ratio_ij": pl.Float64, "contagious": pl.Boolean,
        }
        if ci is not None:
            schema.update({
                "ci_lower": pl.Float64, "ci_upper": pl.Float64,
                "ci_excludes_zero": pl.Boolean,
            })
        return pl.DataFrame(schema=schema)

    return pl.DataFrame(rows).sort("branching_ratio_ij", descending=True)
