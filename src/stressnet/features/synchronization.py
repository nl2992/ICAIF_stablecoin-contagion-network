"""Event-time synchronization utilities for asynchronous node panels."""

from __future__ import annotations

import itertools

import polars as pl


def synchronize_node_feature(
    panel: pl.DataFrame,
    *,
    node_ids: list[str],
    feature_col: str,
    grid_seconds: int,
    max_staleness_seconds: int | None = None,
    ts_col: str = "event_time_seconds",
) -> pl.DataFrame:
    """Put one node feature onto a common event-time grid.

    Observations are bucketed to ``grid_seconds`` and then forward-filled within
    each node up to ``max_staleness_seconds``. Values older than that limit are
    nulled and flagged with ``is_stale`` so pairwise methods cannot silently
    align stale CEX/DEX/flow observations.
    """
    if max_staleness_seconds is None:
        max_staleness_seconds = grid_seconds
    if grid_seconds <= 0:
        raise ValueError("grid_seconds must be positive")
    if feature_col not in panel.columns:
        raise ValueError(f"feature column not found: {feature_col}")
    if ts_col not in panel.columns:
        raise ValueError(f"timestamp column not found: {ts_col}")

    base = (
        panel.filter(pl.col("node_id").is_in(node_ids))
        .select(["node_id", ts_col, feature_col])
        .drop_nulls([ts_col])
        .with_columns(
            ((pl.col(ts_col) / grid_seconds).floor().cast(pl.Int64) * grid_seconds)
            .alias("_grid_ts")
        )
        .group_by(["node_id", "_grid_ts"])
        .agg(
            pl.col(feature_col).drop_nulls().last().alias(feature_col),
            pl.col("_grid_ts").last().alias("_observed_ts"),
        )
        .sort(["node_id", "_grid_ts"])
    )
    if base.height == 0:
        return pl.DataFrame(
            schema={
                "node_id": pl.String,
                ts_col: pl.Int64,
                feature_col: pl.Float64,
                "stale_seconds": pl.Float64,
                "is_stale": pl.Boolean,
            }
        )

    min_ts = int(base["_grid_ts"].min())
    max_ts = int(base["_grid_ts"].max())
    grid_values = list(range(min_ts, max_ts + grid_seconds, grid_seconds))
    grid = pl.DataFrame(
        {
            "node_id": [node for node, _ in itertools.product(node_ids, grid_values)],
            "_grid_ts": [ts for _, ts in itertools.product(node_ids, grid_values)],
        }
    )

    synced = (
        grid.join(base, on=["node_id", "_grid_ts"], how="left")
        .sort(["node_id", "_grid_ts"])
        .with_columns(
            pl.col(feature_col).forward_fill().over("node_id").alias("_value_ffill"),
            pl.col("_observed_ts").forward_fill().over("node_id").alias("_observed_ffill"),
        )
        .with_columns(
            (pl.col("_grid_ts") - pl.col("_observed_ffill")).alias("stale_seconds")
        )
        .with_columns(
            (
                pl.col("_observed_ffill").is_null()
                | (pl.col("stale_seconds") > max_staleness_seconds)
            ).alias("is_stale")
        )
        .with_columns(
            pl.when(pl.col("is_stale"))
            .then(None)
            .otherwise(pl.col("_value_ffill"))
            .alias(feature_col)
        )
        .rename({"_grid_ts": ts_col})
        .select(["node_id", ts_col, feature_col, "stale_seconds", "is_stale"])
    )
    return synced


def synchronized_feature_pivot(
    panel: pl.DataFrame,
    *,
    node_ids: list[str],
    feature_col: str,
    grid_seconds: int,
    max_staleness_seconds: int | None = None,
    ts_col: str = "event_time_seconds",
) -> pl.DataFrame:
    """Return a wide node-by-time pivot after bounded synchronization."""
    synced = synchronize_node_feature(
        panel,
        node_ids=node_ids,
        feature_col=feature_col,
        grid_seconds=grid_seconds,
        max_staleness_seconds=max_staleness_seconds,
        ts_col=ts_col,
    )
    if synced.height == 0:
        return pl.DataFrame()
    return (
        synced.select([ts_col, "node_id", feature_col])
        .pivot(values=feature_col, index=ts_col, on="node_id")
        .sort(ts_col)
    )
