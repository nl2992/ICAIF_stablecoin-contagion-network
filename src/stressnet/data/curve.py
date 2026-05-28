"""Curve pool event fetching and ingestion.

Fetches TokenExchange / AddLiquidity / RemoveLiquidity events from Etherscan
and decodes key fields via minimal ABI parsing (no web3.py dependency).
Tier A: events are directly on-chain; reserve reconstruction is approximate
(Tier B) because we cannot cheaply call get_balances() at historical blocks
without a full archive node.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from stressnet.data.etherscan import get_all_logs, get_block_number_by_timestamp
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

# Curve 3pool (DAI/USDC/USDT) on Ethereum
CURVE_3POOL_ADDRESS = "0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7"

# Event topic0 hashes
TOPIC_TOKEN_EXCHANGE = "0x8b3e96f2b889fa771c53c981b40daf005f63f637f1869f707052d15a3dd97140"
TOPIC_ADD_LIQUIDITY  = "0x189f0db4e11e3d2a5dd7bfbcd5e79e4ae43d59db8b2668a7fc4f92a2f2c2a5e2"
TOPIC_REMOVE_LIQUIDITY = "0x9878ca375e106f2a43c3b599fc624568131c4c9a4ba66a14563715763be9d59d"
TOPIC_REMOVE_LIQUIDITY_IMBALANCE = "0x2b5508378d7e19e0d5fa338419034731416c4f5b219a10379956f764317fd47e"

# Token index → (symbol, decimals) for Curve 3pool
_3POOL_TOKENS = {0: ("DAI", 18), 1: ("USDC", 6), 2: ("USDT", 6)}


# ---------------------------------------------------------------------------
# ABI decoding helpers (no web3.py dependency)
# ---------------------------------------------------------------------------

def _decode_uint256(data_hex: str, slot: int) -> int | None:
    """Read a uint256 from ABI-encoded data at 32-byte slot `slot` (0-indexed)."""
    try:
        raw = bytes.fromhex(data_hex.lstrip("0x"))
        start = slot * 32
        if len(raw) < start + 32:
            return None
        return int.from_bytes(raw[start : start + 32], "big")
    except (ValueError, TypeError):
        return None


def _decode_int128(data_hex: str, slot: int) -> int | None:
    """Read a signed int128 from ABI-encoded data at 32-byte slot `slot`."""
    try:
        raw = bytes.fromhex(data_hex.lstrip("0x"))
        start = slot * 32
        if len(raw) < start + 32:
            return None
        return int.from_bytes(raw[start : start + 32], "big", signed=True)
    except (ValueError, TypeError):
        return None


def decode_token_exchange(data_hex: str) -> dict[str, Any]:
    """Decode Curve TokenExchange event data field.

    ABI layout (non-indexed): int128 sold_id | uint256 tokens_sold |
                               int128 bought_id | uint256 tokens_bought
    """
    sold_id     = _decode_int128(data_hex, 0)
    tokens_sold = _decode_uint256(data_hex, 1)
    bought_id   = _decode_int128(data_hex, 2)
    tokens_bought = _decode_uint256(data_hex, 3)
    if sold_id is None or tokens_sold is None:
        return {}
    _, sold_dec    = _3POOL_TOKENS.get(sold_id,   ("UNK", 18))
    _, bought_dec  = _3POOL_TOKENS.get(bought_id, ("UNK", 18))
    return {
        "sold_id":          sold_id,
        "sold_symbol":      _3POOL_TOKENS.get(sold_id, ("UNK",))[0],
        "tokens_sold_raw":  tokens_sold,
        "tokens_sold":      tokens_sold / (10 ** sold_dec),
        "bought_id":        bought_id,
        "bought_symbol":    _3POOL_TOKENS.get(bought_id, ("UNK",))[0],
        "tokens_bought_raw": tokens_bought,
        "tokens_bought":    (tokens_bought or 0) / (10 ** bought_dec),
    }


def decode_add_liquidity(data_hex: str) -> dict[str, Any]:
    """Decode AddLiquidity event data: uint256[3] token_amounts | uint256[3] fees | ..."""
    amounts = [_decode_uint256(data_hex, i) for i in range(3)]
    decimals = [18, 6, 6]  # DAI, USDC, USDT
    amounts_norm = [
        (a or 0) / (10 ** d) for a, d in zip(amounts, decimals)
    ]
    return {"dai_in": amounts_norm[0], "usdc_in": amounts_norm[1], "usdt_in": amounts_norm[2]}


def decode_remove_liquidity(data_hex: str) -> dict[str, Any]:
    """Decode RemoveLiquidity event data."""
    amounts = [_decode_uint256(data_hex, i) for i in range(3)]
    decimals = [18, 6, 6]
    amounts_norm = [
        (a or 0) / (10 ** d) for a, d in zip(amounts, decimals)
    ]
    return {"dai_out": amounts_norm[0], "usdc_out": amounts_norm[1], "usdt_out": amounts_norm[2]}


# ---------------------------------------------------------------------------
# Original single-page fetcher (unchanged interface)
# ---------------------------------------------------------------------------

def fetch_3pool_events(
    start_block: int,
    end_block: int,
    event_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch Curve 3pool events (all types) from Etherscan logs."""
    topic_map = {
        "TokenExchange": TOPIC_TOKEN_EXCHANGE,
        "AddLiquidity":  TOPIC_ADD_LIQUIDITY,
        "RemoveLiquidity": TOPIC_REMOVE_LIQUIDITY,
    }
    if event_types is None:
        event_types = list(topic_map.keys())

    all_logs: list[dict[str, Any]] = []
    for event_type in event_types:
        topic0 = topic_map.get(event_type)
        if not topic0:
            continue
        from stressnet.data.etherscan import get_logs
        logs = get_logs(
            contract_address=CURVE_3POOL_ADDRESS,
            topic0=topic0,
            from_block=start_block,
            to_block=end_block,
        )
        for log in logs:
            log["_event_type"] = event_type
        all_logs.extend(logs)

    return sorted(
        all_logs,
        key=lambda x: (
            int(x.get("blockNumber", "0x0"), 16),
            int(x.get("logIndex", "0x0"), 16),
        ),
    )


# ---------------------------------------------------------------------------
# Paginated ingestion (new — writes bronze parquet)
# ---------------------------------------------------------------------------

def ingest_curve_pool_events(
    contract_address: str,
    start_block: int,
    end_block: int,
    out_dir: Path,
    event_id: str,
    node_id: str,
) -> tuple[Path | None, str]:
    """Download all Curve pool events and write a bronze pool-events parquet.

    Decodes TokenExchange events to compute running USDC net-sold proxy.
    Tier is 'A' for the raw events; 'B' for derived reserve estimates.

    Returns:
        (parquet_path, 'B') on success, (None, 'fixture_non_empirical') on failure.
    """
    if not os.environ.get("ETHERSCAN_API_KEY"):
        logger.info("ETHERSCAN_API_KEY not set; skipping Curve ingest for %s", node_id)
        return None, "fixture_non_empirical"

    topic_map = {
        "TokenExchange":  TOPIC_TOKEN_EXCHANGE,
        "AddLiquidity":   TOPIC_ADD_LIQUIDITY,
        "RemoveLiquidity": TOPIC_REMOVE_LIQUIDITY,
    }

    all_logs: list[dict[str, Any]] = []
    for event_type, topic0 in topic_map.items():
        logs = get_all_logs(contract_address, topic0, start_block, end_block)
        for log in logs:
            log["_event_type"] = event_type
        all_logs.extend(logs)
        logger.info("  %s: %d events", event_type, len(logs))

    if not all_logs:
        logger.warning("No Curve pool events found for %s", node_id)
        return None, "fixture_non_empirical"

    # Sort by block/logIndex
    all_logs.sort(
        key=lambda x: (
            int(x.get("blockNumber", "0x0"), 16),
            int(x.get("logIndex", "0x0"), 16),
        )
    )

    rows = []
    usdc_net_sold_cumsum = 0.0  # running proxy: positive = USDC pressure on pool

    for log in all_logs:
        block_ts_hex = log.get("timeStamp", "0x0")
        block_ts = int(block_ts_hex, 16) if isinstance(block_ts_hex, str) else int(block_ts_hex)
        data_hex  = log.get("data", "0x")
        evt_type  = log.get("_event_type", "")

        usdc_delta = 0.0
        if evt_type == "TokenExchange":
            decoded = decode_token_exchange(data_hex)
            if decoded:
                if decoded.get("sold_symbol") == "USDC":
                    usdc_delta = decoded.get("tokens_sold", 0.0)   # USDC into pool → +
                elif decoded.get("bought_symbol") == "USDC":
                    usdc_delta = -decoded.get("tokens_bought", 0.0)  # USDC out → -
        elif evt_type == "AddLiquidity":
            decoded = decode_add_liquidity(data_hex)
            usdc_delta = decoded.get("usdc_in", 0.0)
        elif evt_type == "RemoveLiquidity":
            decoded = decode_remove_liquidity(data_hex)
            usdc_delta = -decoded.get("usdc_out", 0.0)

        usdc_net_sold_cumsum += usdc_delta

        rows.append({
            "block_ts":          block_ts,
            "wall_clock_utc":    datetime.fromtimestamp(block_ts, tz=timezone.utc) if block_ts else None,
            "event_type":        evt_type,
            "usdc_net_sold":     usdc_delta,        # per-event USDC flow
            "usdc_net_sold_cum": usdc_net_sold_cumsum,  # running total
        })

    # Build DataFrame and resample to 1-hour windows to get reserve pressure proxy
    df_raw = pl.DataFrame(rows).with_columns(
        pl.col("wall_clock_utc").cast(pl.Datetime("us", "UTC"))
    )

    # Aggregate to 1-hour buckets
    df_agg = (
        df_raw.with_columns(
            ((pl.col("block_ts") // 3600) * 3_600_000_000).cast(pl.Datetime("us"))
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

    # Compute reserve_imbalance proxy: cumulative USDC sold / reference pool size
    # Use 3pool typical size ~$500M as normaliser (approximate, Tier B claim)
    _POOL_SIZE_NORMALISER = 500_000_000.0
    df_agg = df_agg.with_columns(
        (pl.col("usdc_net_sold_cum") / _POOL_SIZE_NORMALISER).alias("reserve_imbalance"),
        (
            pl.col("usdc_net_sold_cum") / _POOL_SIZE_NORMALISER
        ).map_elements(lambda x: 1.0 / (1.0 + abs(x)) if x is not None else None,
                       return_dtype=pl.Float64)
        .alias("implied_pool_price"),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{node_id}_pool_events.parquet"
    df_agg.write_parquet(out_path)
    logger.info(
        "Wrote %d hourly Curve pool state rows for %s → %s (Tier B proxy)",
        df_agg.height, node_id, out_path.name,
    )
    return out_path, "B"


