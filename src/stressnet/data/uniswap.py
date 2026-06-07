"""Uniswap v3 data via The Graph subgraph GraphQL API.

Uses the generic ``thegraph.query_subgraph`` client for all HTTP calls.
Requires ``THE_GRAPH_API_KEY`` environment variable.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from stressnet.data.thegraph import query_subgraph
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

# Uniswap v3 subgraph deployment ID on The Graph hosted service
_UNISWAP_V3_SUBGRAPH = "FbCGRftH4a3yZugY7TnbYgPJVEv2LvMT6oF1fxPe9aJM"

_SWAP_QUERY = """
query Swaps($pool: String!, $skip: Int!, $first: Int!, $startTime: Int!, $endTime: Int!) {
  swaps(
    where: { pool: $pool, timestamp_gte: $startTime, timestamp_lte: $endTime }
    skip: $skip
    first: $first
    orderBy: timestamp
    orderDirection: asc
  ) {
    id
    timestamp
    transaction { blockNumber gasUsed gasPrice }
    sqrtPriceX96
    tick
    liquidity
    amount0
    amount1
    amountUSD
  }
}
"""

# Fallback: pool-hour snapshots (pre-aggregated — faster, Tier B by nature)
_POOL_HOUR_QUERY = """
query PoolHours($pool: String!, $skip: Int!, $first: Int!, $startTime: Int!, $endTime: Int!) {
  poolHourDatas(
    where: { pool: $pool, periodStartUnix_gte: $startTime, periodStartUnix_lte: $endTime }
    skip: $skip
    first: $first
    orderBy: periodStartUnix
    orderDirection: asc
  ) {
    periodStartUnix
    sqrtPrice
    tick
    volumeToken0
    volumeToken1
    volumeUSD
    tvlUSD
    txCount
    open
    high
    low
    close
  }
}
"""


def parse_graph_decimal(value: Any, default: float = 0.0) -> float:
    """Parse a The Graph BigDecimal field without token-decimal rescaling.

    Uniswap v3 subgraph swap amounts are already decimal-normalised strings.
    Dividing them by token decimals again would understate swap-flow proxies by
    orders of magnitude.
    """
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def fetch_pool_swaps(
    pool_address: str,
    start_ts: int,
    end_ts: int,
    first: int = 1_000,
) -> list[dict[str, Any]]:
    """Fetch Uniswap v3 Swap events for a pool between two Unix timestamps.

    Paginates automatically via ``thegraph.query_subgraph``.
    Returns a flat list of swap dicts, or an empty list when
    ``THE_GRAPH_API_KEY`` is absent.
    """
    results = query_subgraph(
        subgraph_id=_UNISWAP_V3_SUBGRAPH,
        query=_SWAP_QUERY,
        variables={
            "pool": pool_address.lower(),
            "startTime": start_ts,
            "endTime": end_ts,
        },
        data_key="swaps",
        first=first,
    )
    logger.info("Fetched %d swaps for pool %s", len(results), pool_address[:10])
    return results


def fetch_pool_hour_datas(
    pool_address: str,
    start_ts: int,
    end_ts: int,
    first: int = 1_000,
) -> list[dict[str, Any]]:
    """Fetch pre-aggregated hourly pool snapshots from The Graph.

    These are already 1-hour buckets so no client-side aggregation is needed.
    Used as a fallback when individual swap data is unavailable or too sparse
    for very quiet pools.

    Returns a flat list of poolHourData dicts.
    """
    results = query_subgraph(
        subgraph_id=_UNISWAP_V3_SUBGRAPH,
        query=_POOL_HOUR_QUERY,
        variables={
            "pool": pool_address.lower(),
            "startTime": start_ts,
            "endTime": end_ts,
        },
        data_key="poolHourDatas",
        first=first,
    )
    logger.info(
        "Fetched %d poolHourDatas for pool %s", len(results), pool_address[:10]
    )
    return results


# ---------------------------------------------------------------------------
# Range ingestion (writes bronze parquet)
# ---------------------------------------------------------------------------

def ingest_uniswap_pool_swaps(
    pool_address: str,
    start_ts: int,
    end_ts: int,
    out_dir: Path,
    event_id: str,
    node_id: str,
) -> tuple[Path | None, str]:
    """Fetch Uniswap v3 swap events and write a bronze pool-state parquet.

    Primary path: raw Swap events → client-side 1-hour aggregation.
    Fallback: ``poolHourDatas`` snapshots when swaps are unavailable.

    Output columns include ``usdc_net_sold_1h`` (matching the Curve node
    convention) for direct cross-venue lead-lag comparability.

    Tier B: The Graph is a pre-indexed intermediary, not a raw on-chain
    log query, so we cannot guarantee execution-grade provenance.

    Requires ``THE_GRAPH_API_KEY`` environment variable.

    Returns:
        ``(parquet_path, 'B')`` on success.
        ``(None, 'fixture_non_empirical')`` when the API key is absent or
        no data is found.
    """
    if not os.environ.get("THE_GRAPH_API_KEY"):
        logger.info("THE_GRAPH_API_KEY not set; skipping Uniswap ingest for %s", node_id)
        return None, "fixture_non_empirical"

    _POOL_SIZE_NORMALISER = 200_000_000.0  # typical USDC/USDT 0.05% pool size (USD)

    # ── Primary path: individual Swap events ────────────────────────────────
    swaps = fetch_pool_swaps(pool_address, start_ts, end_ts)

    if swaps:
        rows = []
        usdc_net_cum = 0.0
        for swap in swaps:
            ts = int(swap.get("timestamp", 0))
            sqrt_x96 = int(swap.get("sqrtPriceX96", 0))
            implied = (sqrt_x96 / (2 ** 96)) ** 2 if sqrt_x96 > 0 else None

            # amount0 = USDC (token0), amount1 = USDT (token1).
            # The Graph returns BigDecimal strings already decimal-adjusted.
            amt0 = parse_graph_decimal(swap.get("amount0", 0))
            usdc_net_cum += amt0

            rows.append({
                "block_ts":           ts,
                "wall_clock_utc":     datetime.fromtimestamp(ts, tz=timezone.utc),
                "implied_pool_price": implied,
                "usdc_net_sold_1h":   amt0,         # matches Curve node column name
                "usdc_net_sold_cum":  usdc_net_cum,
                "amount_usd":         parse_graph_decimal(swap.get("amountUSD", 0)),
            })

        df = pl.DataFrame(rows).with_columns(
            pl.col("wall_clock_utc").cast(pl.Datetime("us", "UTC"))
        )
        df_agg = (
            df.with_columns(
                ((pl.col("block_ts") // 3600) * 3_600_000_000)
                .cast(pl.Datetime("us")).dt.replace_time_zone("UTC").alias("wall_clock_utc"),
            )
            .group_by("wall_clock_utc")
            .agg(
                pl.col("implied_pool_price").last(),
                pl.col("usdc_net_sold_1h").sum(),
                pl.col("usdc_net_sold_cum").last(),
                pl.col("amount_usd").sum(),
            )
            .with_columns(
                (pl.col("usdc_net_sold_cum") / _POOL_SIZE_NORMALISER).alias("reserve_imbalance"),
            )
            .sort("wall_clock_utc")
        )

    else:
        # ── Fallback: poolHourDatas snapshots ────────────────────────────────
        logger.info(
            "No swaps found; falling back to poolHourDatas for %s", node_id
        )
        hour_datas = fetch_pool_hour_datas(pool_address, start_ts, end_ts)
        if not hour_datas:
            logger.warning("No Uniswap data found for pool %s", pool_address[:10])
            return None, "fixture_non_empirical"

        rows = []
        for h in hour_datas:
            ts = int(h.get("periodStartUnix", 0))
            sqrt_price = parse_graph_decimal(h.get("sqrtPrice", 0))
            implied = (sqrt_price / (2 ** 96)) ** 2 if sqrt_price > 0 else None
            vol0 = parse_graph_decimal(h.get("volumeToken0", 0))

            rows.append({
                "wall_clock_utc":     datetime.fromtimestamp(ts, tz=timezone.utc),
                "implied_pool_price": implied,
                "usdc_net_sold_1h":   vol0,   # hourly volume token0 (USDC) as flow proxy
                "amount_usd":         parse_graph_decimal(h.get("volumeUSD", 0)),
                "reserve_imbalance":  parse_graph_decimal(h.get("tvlUSD", 0)) / _POOL_SIZE_NORMALISER,
            })

        df_agg = (
            pl.DataFrame(rows)
            .with_columns(pl.col("wall_clock_utc").cast(pl.Datetime("us", "UTC")))
            .sort("wall_clock_utc")
        )

    if df_agg.height == 0:
        return None, "fixture_non_empirical"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{node_id}_pool_events.parquet"
    df_agg.write_parquet(out_path)
    logger.info(
        "Wrote %d hourly Uniswap rows for %s → %s (Tier B)",
        df_agg.height, node_id, out_path.name,
    )
    return out_path, "B"
