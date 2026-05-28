"""Robustness check orchestration: alternative grids, subsamples, definitions."""

from __future__ import annotations

from typing import Any, Callable

import polars as pl

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_GRIDS = [1, 5, 60]          # seconds
_DEFAULT_THRESHOLDS = [10, 50]        # bps


def run_grid_robustness(
    panel: pl.DataFrame,
    estimation_fn: Callable[[pl.DataFrame, int], Any],
    grids: list[int] | None = None,
    ts_col: str = "event_time_seconds",
) -> dict[int, Any]:
    """Re-run an estimation function on differently downsampled panels.

    For each grid size (in seconds), downsample by taking the last observation
    in each window and re-run estimation_fn.
    """
    if grids is None:
        grids = _DEFAULT_GRIDS

    results = {}
    for grid in grids:
        resampled = (
            panel
            .with_columns(((pl.col(ts_col) // grid) * grid).alias("_grid_ts"))
            .group_by(["event_id", "node_id", "_grid_ts"])
            .last()
            .drop(ts_col)           # remove original before renaming bucket
            .rename({"_grid_ts": ts_col})
            .sort(["node_id", ts_col])
        )
        logger.info("Running with grid=%ds (%d rows)", grid, len(resampled))
        results[grid] = estimation_fn(resampled, grid)

    return results


def subsample_without_dominant(
    panel: pl.DataFrame,
    dominant_node: str = "usdt_binance",
    node_col: str = "node_id",
) -> pl.DataFrame:
    """Return panel with dominant venue removed for robustness check."""
    return panel.filter(pl.col(node_col) != dominant_node)


def subsample_cex_only(
    panel: pl.DataFrame,
    layer_col: str = "layer",
) -> pl.DataFrame:
    """Return panel with only CEX nodes for robustness check."""
    return panel.filter(pl.col(layer_col) == "CEX")
