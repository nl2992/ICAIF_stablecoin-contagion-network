"""On-chain flow aggregation: exchange netflows, bridge flows, mint/burn."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import polars as pl

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

# Known Ethereum null addresses for mint/burn detection
_NULL_ADDRESSES = frozenset({
    "0x0000000000000000000000000000000000000000",
    "0x000000000000000000000000000000000000dead",
})


def detect_mint_burn(
    transfers: pl.DataFrame,
    from_col: str = "from_address",
    to_col: str = "to_address",
    value_col: str = "value_norm",
) -> pl.DataFrame:
    """Add mint_flag and burn_flag columns to a transfer DataFrame.

    A mint is a Transfer from the null address; a burn is a Transfer to the null address.
    """
    null_set = list(_NULL_ADDRESSES)
    return transfers.with_columns([
        pl.col(from_col).is_in(null_set).alias("mint_flag"),
        pl.col(to_col).is_in(null_set).alias("burn_flag"),
    ])


def aggregate_exchange_flows(
    transfers: pl.DataFrame,
    exchange_labels: dict[str, str],
    ts_col: str = "block_ts",
    value_col: str = "value_norm",
    window_seconds: int = 3600,
) -> pl.DataFrame:
    """Aggregate exchange inflows and outflows into rolling windows.

    Args:
        transfers: Token transfer DataFrame with address and value columns.
        exchange_labels: Dict mapping address → exchange name.
        ts_col: Timestamp column (Unix seconds).
        value_col: Normalised USD value column.
        window_seconds: Rolling window size in seconds.

    Returns:
        DataFrame with columns: ts_window, exchange, inflow, outflow, netflow.
    """
    label_map = pl.DataFrame(
        {"address": list(exchange_labels.keys()), "exchange": list(exchange_labels.values())}
    )

    # Tag inflows (to = exchange) and outflows (from = exchange)
    inflows = (
        transfers.join(label_map.rename({"address": "to_address"}), on="to_address", how="inner")
        .select([ts_col, "exchange", pl.col(value_col).alias("inflow")])
    )
    outflows = (
        transfers.join(label_map.rename({"address": "from_address"}), on="from_address", how="inner")
        .select([ts_col, "exchange", pl.col(value_col).alias("outflow")])
    )

    # Floor timestamps to window
    def floor_ts(df: pl.DataFrame, col: str) -> pl.DataFrame:
        return df.with_columns(
            ((pl.col(col) // window_seconds) * window_seconds).alias("ts_window")
        )

    inflows_agg = (
        floor_ts(inflows, ts_col)
        .group_by(["ts_window", "exchange"])
        .agg(pl.col("inflow").sum())
    )
    outflows_agg = (
        floor_ts(outflows, ts_col)
        .group_by(["ts_window", "exchange"])
        .agg(pl.col("outflow").sum())
    )

    flows = (
        inflows_agg.join(outflows_agg, on=["ts_window", "exchange"], how="full", coalesce=True)
        .fill_null(0.0)
        .with_columns((pl.col("inflow") - pl.col("outflow")).alias("netflow"))
        .sort(["exchange", "ts_window"])
    )
    return flows


def aggregate_mint_burn(
    transfers: pl.DataFrame,
    ts_col: str = "block_ts",
    value_col: str = "value_norm",
    window_seconds: int = 3600,
) -> pl.DataFrame:
    """Aggregate mint and burn totals into rolling windows.

    Assumes transfers already have mint_flag and burn_flag columns
    (add via detect_mint_burn).
    """
    if "mint_flag" not in transfers.columns:
        transfers = detect_mint_burn(transfers)

    mints = (
        transfers.filter(pl.col("mint_flag"))
        .with_columns(((pl.col(ts_col) // window_seconds) * window_seconds).alias("ts_window"))
        .group_by("ts_window")
        .agg(pl.col(value_col).sum().alias("mint_usd"))
    )
    burns = (
        transfers.filter(pl.col("burn_flag"))
        .with_columns(((pl.col(ts_col) // window_seconds) * window_seconds).alias("ts_window"))
        .group_by("ts_window")
        .agg(pl.col(value_col).sum().alias("burn_usd"))
    )

    return (
        mints.join(burns, on="ts_window", how="full", coalesce=True)
        .fill_null(0.0)
        .with_columns((pl.col("mint_usd") - pl.col("burn_usd")).alias("net_mint_usd"))
        .sort("ts_window")
    )
