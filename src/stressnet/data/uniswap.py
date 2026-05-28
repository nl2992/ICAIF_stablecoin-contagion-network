"""Uniswap v3 data via The Graph subgraph GraphQL API."""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import requests

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_GRAPH_BASE = "https://gateway.thegraph.com/api"
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


def fetch_pool_swaps(
    pool_address: str,
    start_ts: int,
    end_ts: int,
    first: int = 1000,
) -> list[dict[str, Any]]:
    """Fetch Uniswap v3 Swap events for a pool between two Unix timestamps.

    Paginates automatically. Returns a flat list of swap dicts.
    """
    api_key = os.environ.get("THE_GRAPH_API_KEY", "")
    url = f"{_GRAPH_BASE}/{api_key}/subgraphs/id/{_UNISWAP_V3_SUBGRAPH}"

    results = []
    skip = 0
    while True:
        payload = {
            "query": _SWAP_QUERY,
            "variables": {
                "pool": pool_address.lower(),
                "skip": skip,
                "first": first,
                "startTime": start_ts,
                "endTime": end_ts,
            },
        }
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json().get("data", {}).get("swaps", [])
        results.extend(data)
        if len(data) < first:
            break
        skip += first

    logger.info("Fetched %d swaps for pool %s", len(results), pool_address[:10])
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

    Derives implied_pool_price from sqrtPriceX96 and a simple reserve_imbalance
    proxy from swap amounts.  Tier A for raw events; Tier B for derived features.

    Requires THE_GRAPH_API_KEY environment variable.

    Returns:
        (parquet_path, 'B') on success, (None, 'fixture_non_empirical') otherwise.
    """
    if not os.environ.get("THE_GRAPH_API_KEY"):
        logger.info("THE_GRAPH_API_KEY not set; skipping Uniswap ingest for %s", node_id)
        return None, "fixture_non_empirical"

    swaps = fetch_pool_swaps(pool_address, start_ts, end_ts)
    if not swaps:
        logger.warning("No Uniswap swaps found for pool %s", pool_address[:10])
        return None, "fixture_non_empirical"

    rows = []
    usdc_net = 0.0
    for swap in swaps:
        ts = int(swap.get("timestamp", 0))
        sqrt_x96 = int(swap.get("sqrtPriceX96", 0))
        implied = (sqrt_x96 / (2 ** 96)) ** 2 if sqrt_x96 > 0 else None

        # amount0 = USDC (token0), amount1 = USDT (token1)
        # Both have 6 decimals; positive = in, negative = out
        try:
            amt0 = float(swap.get("amount0", 0)) / 1e6
        except (ValueError, TypeError):
            amt0 = 0.0

        usdc_net += amt0  # net USDC into pool (positive = more USDC in pool)

        rows.append({
            "block_ts": ts,
            "wall_clock_utc": datetime.fromtimestamp(ts, tz=timezone.utc),
            "implied_pool_price": implied,
            "usdc_net": amt0,
            "usdc_net_cum": usdc_net,
            "amount_usd": float(swap.get("amountUSD", 0) or 0),
        })

    if not rows:
        return None, "fixture_non_empirical"

    # Aggregate to 1-hour windows
    _POOL_SIZE_NORMALISER = 200_000_000.0  # typical USDC/USDT 0.05% pool size
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
            pl.col("usdc_net").sum(),
            pl.col("usdc_net_cum").last(),
            pl.col("amount_usd").sum(),
        )
        .with_columns(
            (pl.col("usdc_net_cum") / _POOL_SIZE_NORMALISER).alias("reserve_imbalance"),
        )
        .sort("wall_clock_utc")
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{node_id}_pool_events.parquet"
    df_agg.write_parquet(out_path)
    logger.info(
        "Wrote %d hourly Uniswap pool states for %s → %s (Tier B proxy)",
        df_agg.height, node_id, out_path.name,
    )
    return out_path, "B"
