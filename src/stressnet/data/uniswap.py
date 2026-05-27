"""Uniswap v3 data via The Graph subgraph GraphQL API."""

from __future__ import annotations

import os
from typing import Any

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
