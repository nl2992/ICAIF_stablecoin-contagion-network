"""Curve pool event fetching and ingestion.

Fetches TokenExchange / AddLiquidity / RemoveLiquidity events from Etherscan
and decodes key fields via minimal ABI parsing (no web3.py dependency).

Tier assignment:
  - ``usdc_net_sold_1h``: Tier A — direct hourly sum from on-chain TokenExchange logs.
  - ``reserve_imbalance``, ``implied_pool_price``: Tier B — derived proxy.
    The normaliser is hardcoded per pool and the price formula is approximate
    (true balances require an archive-RPC call to get_balances()).

Pool-type notes:
  - Classic 3pool and meta-pools: amounts in events use the token's *native*
    decimals (e.g. 6 for USDC/USDT, 18 for DAI/UST/3CRV).
  - StableSwap-ng pools (e.g. crvUSD/USDT): amounts are emitted in a uniform
    18-decimal internal representation.  The ``ng_scaled`` flag in
    ``_POOL_CONFIGS`` controls which normaliser to apply.
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

# ---------------------------------------------------------------------------
# Contract addresses
# ---------------------------------------------------------------------------

# Curve 3pool (DAI/USDC/USDT) on Ethereum
CURVE_3POOL_ADDRESS    = "0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7"
CURVE_UST_WORMHOLE     = "0xCEAF7747579696A2F0bb206a14210e3c9e6fB269"
CURVE_CRVUSD_USDT      = "0x390f3595bCa2Df7d23783dFd126427CCeb997BF4"

# Event topic0 hashes
TOPIC_TOKEN_EXCHANGE = "0x8b3e96f2b889fa771c53c981b40daf005f63f637f1869f707052d15a3dd97140"
TOPIC_ADD_LIQUIDITY  = "0x189f0db4e11e3d2a5dd7bfbcd5e79e4ae43d59db8b2668a7fc4f92a2f2c2a5e2"
TOPIC_REMOVE_LIQUIDITY = "0x9878ca375e106f2a43c3b599fc624568131c4c9a4ba66a14563715763be9d59d"
TOPIC_REMOVE_LIQUIDITY_IMBALANCE = "0x2b5508378d7e19e0d5fa338419034731416c4f5b219a10379956f764317fd47e"

# Token index → (symbol, decimals) for Curve 3pool
_3POOL_TOKENS = {0: ("DAI", 18), 1: ("USDC", 6), 2: ("USDT", 6)}

# ---------------------------------------------------------------------------
# Per-pool configuration
# ---------------------------------------------------------------------------

class PoolConfig:
    """Static config for one Curve pool."""

    __slots__ = (
        "tokens",           # {idx: (symbol, native_decimals)}
        "stablecoin_symbol",  # symbol to track as stress indicator
        "pool_size_usd",    # approximate TVL at time of events (normaliser)
        "ng_scaled",        # True → amounts in events are 18-dec internal units
    )

    def __init__(
        self,
        tokens: dict[int, tuple[str, int]],
        stablecoin_symbol: str,
        pool_size_usd: float,
        ng_scaled: bool = False,
    ) -> None:
        self.tokens           = tokens
        self.stablecoin_symbol = stablecoin_symbol
        self.pool_size_usd    = pool_size_usd
        self.ng_scaled        = ng_scaled

    def normalize_amount(self, token_idx: int, raw_amount: int) -> float:
        """Return human-scale token amount given a raw ABI uint256."""
        if self.ng_scaled:
            # StableSwap-ng emits amounts in 18-dec internal units
            return raw_amount / 1e18
        _, dec = self.tokens.get(token_idx, ("UNK", 18))
        return raw_amount / (10 ** dec)


_POOL_CONFIGS: dict[str, PoolConfig] = {
    CURVE_3POOL_ADDRESS.lower(): PoolConfig(
        tokens           = {0: ("DAI", 18), 1: ("USDC", 6), 2: ("USDT", 6)},
        stablecoin_symbol = "USDC",
        pool_size_usd    = 500_000_000,   # 3pool peak TVL ~$3B; use conservative 500M
        ng_scaled        = False,
    ),
    CURVE_UST_WORMHOLE.lower(): PoolConfig(
        tokens           = {0: ("UST", 18), 1: ("3CRV", 18)},
        stablecoin_symbol = "UST",
        pool_size_usd    = 500_000_000,   # was large before Terra collapse
        ng_scaled        = False,
    ),
    CURVE_CRVUSD_USDT.lower(): PoolConfig(
        # StableSwap-ng pool: amounts in events are 18-dec internally.
        # Native decimals only matter for symbol lookup, not normalisation.
        tokens           = {0: ("crvUSD", 18), 1: ("USDT", 6)},
        stablecoin_symbol = "USDT",
        pool_size_usd    = 30_000_000,    # ~$30M TVL in June 2023
        ng_scaled        = True,
    ),
}

def _get_pool_config(contract_address: str) -> PoolConfig:
    """Return PoolConfig for *contract_address*, falling back to 3pool defaults."""
    cfg = _POOL_CONFIGS.get(contract_address.lower())
    if cfg is None:
        logger.warning(
            "No pool config for %s; using 3pool defaults.  "
            "Add an entry to _POOL_CONFIGS in curve.py for correct behaviour.",
            contract_address,
        )
        cfg = _POOL_CONFIGS[CURVE_3POOL_ADDRESS.lower()]
    return cfg


# ---------------------------------------------------------------------------
# ABI decoding helpers (no web3.py dependency)
# ---------------------------------------------------------------------------

def _decode_uint256(data_hex: str, slot: int) -> int | None:
    """Read a uint256 from ABI-encoded data at 32-byte slot `slot` (0-indexed)."""
    try:
        # Use [2:] to strip exactly the "0x" prefix — lstrip("0x") would
        # incorrectly strip all leading '0' and 'x' chars, corrupting the
        # ABI-encoded data (e.g. int128(1) starts with 62 leading zeros).
        hex_str = data_hex[2:] if data_hex.startswith("0x") else data_hex
        raw = bytes.fromhex(hex_str)
        start = slot * 32
        if len(raw) < start + 32:
            return None
        return int.from_bytes(raw[start : start + 32], "big")
    except (ValueError, TypeError):
        return None


def _decode_int128(data_hex: str, slot: int) -> int | None:
    """Read a signed int128 from ABI-encoded data at 32-byte slot `slot`."""
    try:
        hex_str = data_hex[2:] if data_hex.startswith("0x") else data_hex
        raw = bytes.fromhex(hex_str)
        start = slot * 32
        if len(raw) < start + 32:
            return None
        return int.from_bytes(raw[start : start + 32], "big", signed=True)
    except (ValueError, TypeError):
        return None


def decode_token_exchange(
    data_hex: str,
    pool_cfg: "PoolConfig | None" = None,
) -> dict[str, Any]:
    """Decode Curve TokenExchange event data field.

    ABI layout (non-indexed): int128/uint256 sold_id | uint256 tokens_sold |
                               int128/uint256 bought_id | uint256 tokens_bought

    Args:
        data_hex:  The ABI-encoded ``data`` field from the Etherscan log.
        pool_cfg:  Per-pool config controlling token map and decimal handling.
                   Falls back to the classic 3pool mapping when ``None``.
    """
    if pool_cfg is None:
        # Backward-compatible fallback — uses 3pool token map
        pool_cfg = _POOL_CONFIGS[CURVE_3POOL_ADDRESS.lower()]

    sold_id       = _decode_int128(data_hex, 0)
    tokens_sold   = _decode_uint256(data_hex, 1)
    bought_id     = _decode_int128(data_hex, 2)
    tokens_bought = _decode_uint256(data_hex, 3)
    if sold_id is None or tokens_sold is None:
        return {}

    sold_sym,  _ = pool_cfg.tokens.get(sold_id,   ("UNK", 18))
    bought_sym, _ = pool_cfg.tokens.get(bought_id, ("UNK", 18))

    return {
        "sold_id":           sold_id,
        "sold_symbol":       sold_sym,
        "tokens_sold_raw":   tokens_sold,
        "tokens_sold":       pool_cfg.normalize_amount(sold_id, tokens_sold),
        "bought_id":         bought_id,
        "bought_symbol":     bought_sym,
        "tokens_bought_raw": tokens_bought,
        "tokens_bought":     pool_cfg.normalize_amount(bought_id, tokens_bought or 0),
    }


def decode_add_liquidity(
    data_hex: str,
    pool_cfg: "PoolConfig | None" = None,
) -> dict[str, Any]:
    """Decode AddLiquidity event data: uint256[N] token_amounts | ..."""
    if pool_cfg is None:
        pool_cfg = _POOL_CONFIGS[CURVE_3POOL_ADDRESS.lower()]
    n_tokens = len(pool_cfg.tokens)
    amounts_raw = [_decode_uint256(data_hex, i) for i in range(n_tokens)]
    result: dict[str, float] = {}
    for idx, raw in enumerate(amounts_raw):
        sym = pool_cfg.tokens.get(idx, ("tok" + str(idx), 18))[0].lower()
        result[f"{sym}_in"] = pool_cfg.normalize_amount(idx, raw or 0)
    # Keep legacy keys for 3pool callers
    if "usdc_in" not in result and "usdc" not in [v[0].lower() for v in pool_cfg.tokens.values()]:
        result["usdc_in"] = 0.0
    return result


def decode_remove_liquidity(
    data_hex: str,
    pool_cfg: "PoolConfig | None" = None,
) -> dict[str, Any]:
    """Decode RemoveLiquidity event data."""
    if pool_cfg is None:
        pool_cfg = _POOL_CONFIGS[CURVE_3POOL_ADDRESS.lower()]
    n_tokens = len(pool_cfg.tokens)
    amounts_raw = [_decode_uint256(data_hex, i) for i in range(n_tokens)]
    result: dict[str, float] = {}
    for idx, raw in enumerate(amounts_raw):
        sym = pool_cfg.tokens.get(idx, ("tok" + str(idx), 18))[0].lower()
        result[f"{sym}_out"] = pool_cfg.normalize_amount(idx, raw or 0)
    if "usdc_out" not in result and "usdc" not in [v[0].lower() for v in pool_cfg.tokens.values()]:
        result["usdc_out"] = 0.0
    return result


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
    grid_seconds: int = 3600,
    save_raw: bool = False,
) -> tuple[Path | None, str]:
    """Download all Curve pool events and write a bronze pool-events parquet.

    Decodes TokenExchange events to compute a running stablecoin net-sold proxy.

    Tier assignment of the output columns:
      - ``usdc_net_sold_1h`` : **Tier A** — direct hourly sum from on-chain logs.
      - ``reserve_imbalance``, ``implied_pool_price`` : **Tier B** — derived proxy
        (hardcoded pool-size normaliser; not a true reserve-ratio).

    The *node-level* tier returned here is ``'A'`` because the primary feature
    (``usdc_net_sold_1h``) is execution-grade on-chain data.  Claims using the
    derived features must be downgraded to B at the edge-claim level.

    Returns:
        (parquet_path, 'A') on success, (None, 'fixture_non_empirical') on failure.
    """
    if not os.environ.get("ETHERSCAN_API_KEY"):
        logger.info("ETHERSCAN_API_KEY not set; skipping Curve ingest for %s", node_id)
        return None, "fixture_non_empirical"

    pool_cfg = _get_pool_config(contract_address)
    stablecoin_sym = pool_cfg.stablecoin_symbol

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
    stable_net_sold_cumsum = 0.0  # running proxy: +  = stablecoin into pool

    for log in all_logs:
        block_ts_hex = log.get("timeStamp", "0x0")
        block_ts = int(block_ts_hex, 16) if isinstance(block_ts_hex, str) else int(block_ts_hex)
        data_hex  = log.get("data", "0x")
        evt_type  = log.get("_event_type", "")

        stable_delta = 0.0
        if evt_type == "TokenExchange":
            decoded = decode_token_exchange(data_hex, pool_cfg)
            if decoded:
                if decoded.get("sold_symbol") == stablecoin_sym:
                    stable_delta = decoded.get("tokens_sold", 0.0)   # stable into pool → +
                elif decoded.get("bought_symbol") == stablecoin_sym:
                    stable_delta = -decoded.get("tokens_bought", 0.0)  # stable out → -
        elif evt_type == "AddLiquidity":
            decoded = decode_add_liquidity(data_hex, pool_cfg)
            sym_key = f"{stablecoin_sym.lower()}_in"
            stable_delta = decoded.get(sym_key, 0.0)
        elif evt_type == "RemoveLiquidity":
            decoded = decode_remove_liquidity(data_hex, pool_cfg)
            sym_key = f"{stablecoin_sym.lower()}_out"
            stable_delta = -decoded.get(sym_key, 0.0)

        stable_net_sold_cumsum += stable_delta

        rows.append({
            "block_ts":          block_ts,
            "wall_clock_utc":    datetime.fromtimestamp(block_ts, tz=timezone.utc) if block_ts else None,
            "event_type":        evt_type,
            "usdc_net_sold":     stable_delta,          # per-event stablecoin flow (legacy col name)
            "usdc_net_sold_cum": stable_net_sold_cumsum,  # running total (legacy col name)
        })

    # Build DataFrame from per-event rows
    df_raw = pl.DataFrame(rows).with_columns(
        pl.col("wall_clock_utc").cast(pl.Datetime("us", "UTC"))
    )

    # Optionally save raw per-event data (block-level resolution)
    if save_raw:
        raw_path = out_dir / f"{node_id}_pool_events_raw.parquet"
        df_raw.write_parquet(raw_path)
        logger.info("Saved %d raw events to %s", df_raw.height, raw_path.name)

    # Aggregate to grid_seconds buckets (default 3600 = 1h; use 300 for 5-min)
    grid_us = grid_seconds * 1_000_000
    feature_col = f"usdc_net_sold_{grid_seconds}s"
    df_agg = (
        df_raw.with_columns(
            ((pl.col("block_ts") // grid_seconds) * grid_us).cast(pl.Datetime("us"))
            .dt.replace_time_zone("UTC").alias("wall_clock_utc"),
        )
        .group_by("wall_clock_utc")
        .agg(
            pl.col("usdc_net_sold").sum().alias(feature_col),
            pl.col("usdc_net_sold_cum").last().alias("usdc_net_sold_cum"),
            pl.col("event_type").count().alias("n_events"),
        )
        .sort("wall_clock_utc")
        .rename({feature_col: "usdc_net_sold_1h"})  # keep canonical column name
    )

    # Compute reserve_imbalance proxy: cumulative stablecoin sold / pool size
    # Pool size is per-pool estimate; this is an approximation (Tier B derived feature).
    pool_size = pool_cfg.pool_size_usd
    df_agg = df_agg.with_columns(
        (pl.col("usdc_net_sold_cum") / pool_size).alias("reserve_imbalance"),
        (
            pl.col("usdc_net_sold_cum") / pool_size
        ).map_elements(lambda x: 1.0 / (1.0 + abs(x)) if x is not None else None,
                       return_dtype=pl.Float64)
        .alias("implied_pool_price"),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{node_id}_pool_events.parquet"
    df_agg.write_parquet(out_path)
    logger.info(
        "Wrote %d %ds-grid Curve pool state rows for %s → %s "
        "(Tier A: usdc_net_sold_1h; Tier B proxy: reserve_imbalance, implied_pool_price)",
        df_agg.height, grid_seconds, node_id, out_path.name,
    )
    return out_path, "A"


