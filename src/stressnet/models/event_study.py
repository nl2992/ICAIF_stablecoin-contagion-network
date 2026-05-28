"""Event-study layer for stablecoin de-peg events.

Computes abnormal basis (AB) and cumulative abnormal basis (CAB) relative
to a pre-event estimation window.  Statistical significance is assessed via
a block-bootstrap null on the estimation window.

Terminology
-----------
basis_vs_usd   : the peg-deviation feature (e.g. log(price/1.00))
AB_t           : actual[t] - E[actual | estimation window]
CAB(t0, t1)    : sum_{t=t0}^{t1} AB_t  (like a CAR in equity event studies)
estimation window : rows where event_phase == "pre"
event window      : rows where event_phase in {"onset", "panic", "recovery"}

Output columns (per node)
-------------------------
node_id, event_time_seconds, event_phase,
basis_vs_usd, ab (abnormal basis), cab (cumulative),
estimation_mean, estimation_std
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

Phase = Literal["pre", "onset", "panic", "recovery", "post"]

# Phase duration defaults (hours relative to shock onset, i.e. event_time_seconds / 3600)
# event_time_seconds = 0 is the shock onset; negative = pre-event.
PHASE_SECONDS: dict[str, tuple[float, float]] = {
    # (start_offset_s, end_offset_s) relative to shock onset (T=0)
    "onset":    (0,          6 * 3600),
    "panic":    (6 * 3600,  72 * 3600),
    "recovery": (72 * 3600, 168 * 3600),   # 3–7 days post-onset
    "post":     (168 * 3600, 1e12),         # anything beyond 7 days
}


# ---------------------------------------------------------------------------
# Phase labelling
# ---------------------------------------------------------------------------

def label_phases(
    event_time_seconds: np.ndarray,
    shock_onset_offset: float = 0.0,
    _unused: float = 0.0,
) -> np.ndarray:
    """Assign a phase label (str) to each value of event_time_seconds.

    Parameters
    ----------
    event_time_seconds : Seconds relative to shock onset (T=0 at onset).
                         Negative values = pre-event.
    shock_onset_offset : Unused, kept for API compatibility (always 0).
    """
    phases = np.full(len(event_time_seconds), "pre", dtype=object)
    for phase, (start_s, end_s) in PHASE_SECONDS.items():
        mask = (event_time_seconds >= (shock_onset_offset + start_s)) & \
               (event_time_seconds <  (shock_onset_offset + end_s))
        phases[mask] = phase
    return phases


# ---------------------------------------------------------------------------
# Per-node abnormal basis
# ---------------------------------------------------------------------------

def _block_bootstrap_null(
    series: np.ndarray,
    n_reps: int = 1000,
    block_size: int = 60,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Return (mean, std) of block-bootstrap CAB null distribution.

    Each rep shuffles contiguous blocks and computes the sum.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(series)
    n_blocks = max(1, n // block_size)
    null_sums = np.empty(n_reps)
    for i in range(n_reps):
        order = rng.permutation(n_blocks)
        shuffled = np.concatenate([series[b * block_size: (b + 1) * block_size] for b in order])[:n]
        null_sums[i] = shuffled.sum()
    return float(null_sums.mean()), float(null_sums.std())


def compute_node_event_study(
    ts: np.ndarray,
    values: np.ndarray,
    phases: np.ndarray,
    n_bootstrap: int = 500,
    block_size: int = 60,
    min_pre_obs: int = 30,
    rng: np.random.Generator | None = None,
) -> dict[str, object]:
    """Compute AB, CAB, and significance for one node.

    Returns
    -------
    dict with keys:
        ts, values, phases (passed through),
        ab          : np.ndarray (same length as ts),
        cab         : np.ndarray (cumulative, reset at start of event window),
        est_mean    : float  (pre-event mean),
        est_std     : float  (pre-event std),
        cab_event   : float  (total CAB over event window: onset+panic+recovery),
        p_value     : float  (one-sided: how often |null| >= |cab_event|),
        significant : bool,
        has_baseline: bool   (True if sufficient pre-event observations exist),
    """
    if rng is None:
        rng = np.random.default_rng(42)

    pre_mask = phases == "pre"
    event_mask = np.isin(phases, ["onset", "panic", "recovery"])

    pre_vals = values[pre_mask]
    pre_vals_clean = pre_vals[~np.isnan(pre_vals)]
    has_baseline = len(pre_vals_clean) >= min_pre_obs
    est_mean = float(np.nanmean(pre_vals_clean)) if has_baseline else float("nan")
    est_std  = float(np.nanstd(pre_vals_clean))  if has_baseline else float("nan")

    if not has_baseline:
        ab  = np.full_like(values, float("nan"))
        cab = np.full_like(values, float("nan"))
        return {
            "ts": ts, "values": values, "phases": phases,
            "ab": ab, "cab": cab,
            "est_mean": float("nan"), "est_std": float("nan"),
            "cab_event": float("nan"), "p_value": float("nan"),
            "significant": False, "has_baseline": False,
        }

    ab = values - est_mean

    # CAB resets to 0 at start of event window
    cab = np.full_like(ab, float("nan"))
    if event_mask.any():
        event_idx = np.where(event_mask)[0]
        running = 0.0
        for idx in sorted(event_idx):
            running += float(ab[idx]) if not np.isnan(ab[idx]) else 0.0
            cab[idx] = running

    cab_event = float(np.nansum(cab[event_mask])) if event_mask.any() else 0.0

    # Block-bootstrap p-value against pre-event series
    if len(pre_vals_clean) >= block_size and event_mask.any():
        event_len = int(event_mask.sum())
        null_sums = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            if len(pre_vals_clean) >= event_len:
                start = rng.integers(0, len(pre_vals_clean) - event_len + 1)
                null_sums[i] = float(
                    np.nansum(pre_vals_clean[start: start + event_len]) - est_mean * event_len
                )
            else:
                null_sums[i] = 0.0
        p_value = float(np.mean(np.abs(null_sums) >= abs(cab_event)))
    else:
        p_value = 1.0

    return {
        "ts": ts, "values": values, "phases": phases,
        "ab": ab, "cab": cab,
        "est_mean": est_mean, "est_std": est_std,
        "cab_event": cab_event, "p_value": p_value,
        "significant": p_value < 0.05,
        "has_baseline": True,
    }


# ---------------------------------------------------------------------------
# Timing of first significant deviation
# ---------------------------------------------------------------------------

def _first_significant_hour(
    ts: np.ndarray,
    ab: np.ndarray,
    phases: np.ndarray,
    est_std: float,
    threshold_sigma: float = 2.0,
) -> float | None:
    """Return Unix timestamp of first |AB| > threshold_sigma * est_std in event window."""
    event_mask = np.isin(phases, ["onset", "panic", "recovery"])
    if not event_mask.any() or est_std <= 0:
        return None
    event_ts = ts[event_mask]
    event_ab = ab[event_mask]
    order = np.argsort(event_ts)
    for i in order:
        if abs(event_ab[i]) > threshold_sigma * est_std:
            return float(event_ts[i])
    return None


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def compute_event_study_table(
    panel: pl.DataFrame,
    node_ids: list[str],
    shock_onset_ts: float = 0.0,   # kept for API compatibility; unused
    analysis_start_ts: float = 0.0,
    feature_col: str = "basis_vs_usd",
    ts_col: str = "event_time_seconds",
    n_bootstrap: int = 500,
    block_size: int = 60,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute event-study AB/CAB for all nodes.

    ``event_time_seconds`` in the panel is seconds relative to shock onset
    (T=0), so ``shock_onset_ts`` and ``analysis_start_ts`` are not needed and
    are ignored.

    Returns
    -------
    timeseries_df : row per (node, timestep) — AB and CAB time series
    summary_df    : one row per node — est_mean, est_std, cab_event, p_value,
                    significant, first_deviation_ts, transmission_rank
    """
    rng = np.random.default_rng(42)
    ts_rows: list[dict] = []
    summary_rows: list[dict] = []

    for node_id in node_ids:
        node_data = (
            panel.filter(pl.col("node_id") == node_id)
            .sort(ts_col)
            .select([ts_col, feature_col])
            .drop_nulls()
        )
        if node_data.height < 10:
            logger.debug("Skipping %s — fewer than 10 non-null rows.", node_id)
            continue

        ts_arr  = node_data[ts_col].to_numpy().astype(float)
        val_arr = node_data[feature_col].to_numpy().astype(float)
        # event_time_seconds = 0 at shock onset; negative = pre-event
        phase_arr = label_phases(ts_arr)

        result = compute_node_event_study(
            ts_arr, val_arr, phase_arr,
            n_bootstrap=n_bootstrap,
            block_size=block_size,
            rng=rng,
        )

        first_dev_ts = _first_significant_hour(
            ts_arr, result["ab"], phase_arr, result["est_std"]
        )

        # Time-series rows
        for i in range(len(ts_arr)):
            ts_rows.append({
                "node_id":             node_id,
                ts_col:                int(ts_arr[i]),
                "event_phase":         str(phase_arr[i]),
                feature_col:           float(val_arr[i]),
                "ab":                  float(result["ab"][i]),
                "cab":                 float(result["cab"][i]),
                "estimation_mean":     result["est_mean"],
                "estimation_std":      result["est_std"],
            })

        cab_str = f"{result['cab_event']:.4f}" if result["has_baseline"] else "n/a (no pre-event data)"
        p_str   = f"{result['p_value']:.3f}"  if result["has_baseline"] else "n/a"
        logger.info(
            "%-25s  CAB=%s  p=%s  %s",
            node_id, cab_str, p_str,
            "✓" if result["significant"] else ("–" if result["has_baseline"] else "⚠ no baseline"),
        )
        summary_rows.append({
            "node_id":               node_id,
            "has_baseline":          result["has_baseline"],
            "estimation_mean":       result["est_mean"],
            "estimation_std":        result["est_std"],
            "cab_event":             result["cab_event"],
            "p_value":               result["p_value"],
            "significant_p05":       result["significant"],
            "first_deviation_ts":    first_dev_ts if first_dev_ts is not None else float("nan"),
            "n_pre_obs":             int((phase_arr == "pre").sum()),
            "n_event_obs":           int(np.isin(phase_arr, ["onset", "panic", "recovery"]).sum()),
        })

    if not summary_rows:
        empty_ts  = pl.DataFrame(schema={
            "node_id": pl.String, ts_col: pl.Int64, "event_phase": pl.String,
            feature_col: pl.Float64, "ab": pl.Float64, "cab": pl.Float64,
            "estimation_mean": pl.Float64, "estimation_std": pl.Float64,
        })
        empty_sum = pl.DataFrame(schema={
            "node_id": pl.String, "has_baseline": pl.Boolean,
            "estimation_mean": pl.Float64, "estimation_std": pl.Float64,
            "cab_event": pl.Float64, "p_value": pl.Float64,
            "significant_p05": pl.Boolean, "first_deviation_ts": pl.Float64,
            "n_pre_obs": pl.Int64, "n_event_obs": pl.Int64,
            "transmission_rank": pl.Int32,
        })
        return empty_ts, empty_sum

    ts_df = pl.DataFrame(ts_rows).sort([ts_col, "node_id"])
    sum_df = pl.DataFrame(summary_rows)

    # Transmission rank: order by first_deviation_ts (NaN last)
    sum_df = sum_df.with_columns(
        pl.col("first_deviation_ts")
        .rank(method="ordinal", descending=False)
        .cast(pl.Int32)
        .alias("transmission_rank")
    )

    return ts_df, sum_df
