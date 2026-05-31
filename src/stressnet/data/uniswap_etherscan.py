"""Uniswap v3 pool event ingestion via Etherscan getLogs.

Alternative to The Graph when THE_GRAPH_API_KEY is unavailable.
Uses Etherscan's eth_getLogs endpoint to fetch Swap events from Uniswap v3
pool contracts, then aggregates to configurable time-window features.

Swap event ABI
--------------
Swap(address indexed sender, address indexed recipient,
     int256 amount0, int256 amount1,
     uint160 sqrtPriceX96, uint128 liquidity, int24 tick)

topic0 = keccak256("Swap(address,address,int256,int256,uint160,uint128,int24)")
       = 0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67

Non-indexed fields (packed into data):
  [0]  int256   amount0        (32 bytes)
  [1]  int256   amount1        (32 bytes)
  [2]  uint160  sqrtPriceX96   (32 bytes)
  [3]  uint128  liquidity      (32 bytes)
  [4]  int24    tick           (32 bytes)

For the USDC/USDT 0.05% pool (0x3416cF6C708Da44DB2624D63ea0AAef7113527C6):
  token0 = USDC (6 decimals)   amount0 < 0 ⟹ USDC leaving pool (sold)
  token1 = USDT (6 decimals)   amount1 > 0 ⟹ USDT entering pool

Provenance: Etherscan getLogs → Tier A (same provenance as Curve TokenExchange logs).

Feature ``usdc_net_sold_1h`` is the grid-window sum of (−amount0) for
TokenExchange-equivalent swaps, matching the Curve feature definition exactly.
"""

from __future__ import annotations

import os
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from stressnet.data.etherscan import get_all_logs
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Swap event topic0 (keccak256 of canonical ABI signature)
TOPIC_UNISWAP_V3_SWAP = (
    "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
)

# Pool configs: (token0_symbol, token0_decimals, token1_symbol, token1_decimals, stress_token)
# stress_token is the token whose "leaving the pool" flow we track (positive → stress)
_POOL_CONFIGS: dict[str, dict[str, Any]] = {
    # USDC/USDT 0.05%
    "0x3416cf6c708da44db2624d63ea0aaef7113527c6": {
        "token0": "USDC", "decimals0": 6,
        "token1": "USDT", "decimals1": 6,
        "stress_token": "USDC",   # track USDC flow (matches Curve 3pool feature)
        "pool_size_usd": 300_000_000,
    },
    # USDT/WETH 0.3% — useful for cross-protocol comparison
    "0x4e68ccd3e89f51c3074ca5072bbac773960dfa36": {
        "token0": "WETH", "decimals0": 18,
        "token1": "USDT", "decimals1": 6,
        "stress_token": "USDT",
        "pool_size_usd": 200_000_000,
    },
}


# ---------------------------------------------------------------------------
# ABI decoding
# ---------------------------------------------------------------------------

def _decode_int256(data_hex: str, slot: int) -> int:
    """Decode a signed int256 from ABI-encoded data at slot index."""
    # Use [2:] not lstrip("0x") — lstrip strips ALL leading '0'/'x' chars,
    # which would eat meaningful zero-padding in the ABI encoding.
    data = data_hex[2:] if data_hex.startswith("0x") else data_hex
    start = slot * 64
    chunk = data[start: start + 64].zfill(64)
    raw = int(chunk, 16)
    # Two's complement for int256
    if raw >= (1 << 255):
        raw -= 1 << 256
    return raw


def decode_uniswap_swap(data_hex: str, pool_address: str) -> dict[str, Any] | None:
    """Decode a Uniswap v3 Swap event data payload.

    Returns:
        dict with ``amount0``, ``amount1``, ``usdc_net_sold`` (signed),
        ``tick``, and token symbols — or None on decode failure.
    """
    cfg = _POOL_CONFIGS.get(pool_address.lower())
    if cfg is None:
        logger.debug("No config for pool %s", pool_address)
        return None

    try:
        amount0_raw = _decode_int256(data_hex, 0)
        amount1_raw = _decode_int256(data_hex, 1)
        # tick is at slot 4 (int24 packed in int256 slot)
        tick_raw = _decode_int256(data_hex, 4)
    except (ValueError, IndexError):
        return None

    dec0, dec1 = cfg["decimals0"], cfg["decimals1"]
    amount0 = amount0_raw / (10 ** dec0)
    amount1 = amount1_raw / (10 ** dec1)

    # stress_token flow: if stress_token is token0, "sold" = amount0 < 0 (USDC leaves pool)
    # We follow Curve convention: positive = stress token leaving pool → upward pressure
    if cfg["stress_token"] == cfg["token0"]:
        usdc_net_sold = -amount0  # USDC leaving pool → positive
    else:
        usdc_net_sold = -amount1  # USDT/other leaving pool → positive

    return {
        "amount0": amount0,
        "amount1": amount1,
        "usdc_net_sold": usdc_net_sold,
        "tick": int(tick_raw),
        "token0": cfg["token0"],
        "token1": cfg["token1"],
    }


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------

def ingest_uniswap_pool_events(
    pool_address: str,
    start_block: int,
    end_block: int,
    out_dir: Path,
    event_id: str,
    node_id: str,
    grid_seconds: int = 3600,
    save_raw: bool = False,
) -> tuple[Path | None, str]:
    """Download all Uniswap v3 Swap events and write a bronze pool-events parquet.

    Mirrors ``stressnet.data.curve.ingest_curve_pool_events`` exactly so the
    two data sources are interchangeable in the feature pipeline.

    Feature tier:
      - ``usdc_net_sold_1h`` : **Tier A** — direct sum from on-chain Swap logs.
      - ``reserve_imbalance``: **Tier B** — approximate; uses hardcoded pool size.

    Args:
        pool_address:  Uniswap v3 pool contract address.
        start_block:   Inclusive start block.
        end_block:     Inclusive end block.
        out_dir:       Bronze output directory.
        event_id:      Stress event ID (for logging).
        node_id:       Node ID string for the output filename.
        grid_seconds:  Time-bucket width in seconds (default 3600).
        save_raw:      Also write per-event parquet at block resolution.

    Returns:
        ``(parquet_path, 'A')`` on success, ``(None, 'fixture_non_empirical')``
        on failure.
    """
    if not os.environ.get("ETHERSCAN_API_KEY"):
        logger.info("ETHERSCAN_API_KEY not set; skipping Uniswap ingest for %s", node_id)
        return None, "fixture_non_empirical"

    cfg = _POOL_CONFIGS.get(pool_address.lower())
    if cfg is None:
        logger.warning("No pool config for %s; cannot ingest.", pool_address)
        return None, "fixture_non_empirical"

    logs = get_all_logs(pool_address, TOPIC_UNISWAP_V3_SWAP, start_block, end_block)
    logger.info("  Swap: %d events from %s", len(logs), pool_address[:10] + "…")

    if not logs:
        logger.warning("No Uniswap Swap events found for %s in %s", node_id, event_id)
        return None, "fixture_non_empirical"

    logs.sort(key=lambda x: (
        int(x.get("blockNumber", "0x0"), 16),
        int(x.get("logIndex", "0x0"), 16),
    ))

    rows = []
    net_sold_cum = 0.0

    for log in logs:
        block_ts_hex = log.get("timeStamp", "0x0")
        block_ts = int(block_ts_hex, 16) if isinstance(block_ts_hex, str) else int(block_ts_hex)
        data_hex = log.get("data", "0x")

        decoded = decode_uniswap_swap(data_hex, pool_address)
        if decoded is None:
            continue

        delta = decoded["usdc_net_sold"]
        net_sold_cum += delta

        rows.append({
            "block_ts": block_ts,
            "wall_clock_utc": datetime.fromtimestamp(block_ts, tz=timezone.utc) if block_ts else None,
            "event_type": "Swap",
            "usdc_net_sold": delta,
            "usdc_net_sold_cum": net_sold_cum,
            "amount0": decoded["amount0"],
            "amount1": decoded["amount1"],
            "tick": decoded["tick"],
        })

    if not rows:
        return None, "fixture_non_empirical"

    df_raw = pl.DataFrame(rows).with_columns(
        pl.col("wall_clock_utc").cast(pl.Datetime("us", "UTC"))
    )

    if save_raw:
        raw_path = out_dir / f"{node_id}_pool_events_raw.parquet"
        df_raw.write_parquet(raw_path)
        logger.info("Saved %d raw Swap events to %s", df_raw.height, raw_path.name)

    # Aggregate to grid_seconds buckets
    grid_us = grid_seconds * 1_000_000
    df_agg = (
        df_raw.with_columns(
            ((pl.col("block_ts") // grid_seconds) * grid_us).cast(pl.Datetime("us"))
            .dt.replace_time_zone("UTC").alias("wall_clock_utc"),
        )
        .group_by("wall_clock_utc")
        .agg(
            pl.col("usdc_net_sold").sum().alias("usdc_net_sold_1h"),
            pl.col("usdc_net_sold_cum").last().alias("usdc_net_sold_cum"),
            pl.col("event_type").count().alias("n_events"),
        )
        .sort("wall_clock_utc")
    )

    # Tier-B derived proxy: reserve_imbalance
    pool_size = cfg["pool_size_usd"]
    df_agg = df_agg.with_columns(
        (pl.col("usdc_net_sold_cum") / pool_size).alias("reserve_imbalance"),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{node_id}_pool_events.parquet"
    df_agg.write_parquet(out_path)
    logger.info(
        "Wrote %d %ds-grid Uniswap pool state rows for %s → %s "
        "(Tier A: usdc_net_sold_1h; Tier B proxy: reserve_imbalance)",
        df_agg.height, grid_seconds, node_id, out_path.name,
    )
    return out_path, "A"
