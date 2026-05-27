"""On-chain flow feature computation: exchange flows, bridge flows, mint/burn."""

from __future__ import annotations

import polars as pl

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def add_gas_features(
    panel: pl.DataFrame,
    gas_df: pl.DataFrame,
    ts_col: str = "wall_clock_utc",
    gas_col: str = "base_fee_gwei",
    window_seconds: int = 3600,
) -> pl.DataFrame:
    """Join hourly gas/base-fee data onto the main feature panel."""
    gas_windowed = gas_df.with_columns(
        ((pl.col(ts_col).cast(pl.Int64) // window_seconds) * window_seconds).alias("ts_window")
    ).group_by("ts_window").agg(pl.col(gas_col).mean().alias("gas_base_fee_gwei"))

    return panel.with_columns(
        ((pl.col(ts_col).cast(pl.Int64) // window_seconds) * window_seconds).alias("_ts_window")
    ).join(
        gas_windowed.rename({"ts_window": "_ts_window"}),
        on="_ts_window",
        how="left",
    ).drop("_ts_window")


def join_flow_features(
    panel: pl.DataFrame,
    flow_df: pl.DataFrame,
    node_id: str,
    ts_col: str = "ts_window",
    flow_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Join aggregated on-chain flow features onto the panel for a specific node."""
    if flow_cols is None:
        flow_cols = ["exchange_inflow_1h", "exchange_outflow_1h", "exchange_netflow_1h",
                     "bridge_inflow_1h", "bridge_outflow_1h", "mint_burn_net_1h"]

    available = [c for c in flow_cols if c in flow_df.columns]
    if not available:
        logger.warning("No flow columns found for node %s; returning panel unchanged.", node_id)
        return panel

    node_flows = flow_df.select([ts_col] + available)
    return panel.join(node_flows, on=ts_col, how="left")
