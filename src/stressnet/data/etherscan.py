"""Etherscan API: ERC-20 token transfers, event logs, mint/burn, exchange flows.

All functions degrade gracefully when ETHERSCAN_API_KEY is absent (empty key
still works for low-rate public access but will hit rate limits faster).
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import requests

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_BASE         = "https://api.etherscan.io/v2/api"
_PAGE_SIZE    = 10_000  # max offset supported by Etherscan
_RATE_SLEEP   = 0.25    # seconds between calls (free tier: 5 calls/s)
_NULL_ADDRS   = frozenset({
    "0x0000000000000000000000000000000000000000",
    "0x000000000000000000000000000000000000dead",
})

# Well-known exchange hot-wallet addresses for common venues
# (from public labelling sources; non-exhaustive, best-effort Tier B)
# Includes older addresses (2019-2021) and newer rotation wallets active
# during 2022-2023 stress events.  Source: Etherscan labels + Arkham/Nansen
# public datasets.
KNOWN_EXCHANGE_ADDRESSES: dict[str, str] = {
    # Binance — older hot wallets
    "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be": "Binance",
    "0xd551234ae421e3bcba99a0da6d736074f22192ff": "Binance",
    "0x564286362092d8e7936f0549571a803b203aaced": "Binance",
    "0x0681d8db095565fe8a346fa0277bffde9c0edbbf": "Binance",
    "0xfe9e8709d3215310075d67e3ed32a380ccf451c8": "Binance",
    "0xbe0eb53f46cd790cd13851d5eff43d12404d33e8": "Binance_cold",
    # Binance — 2022-2023 active hot wallets (Binance 14/15/16/17)
    "0x28c6c06298d514db089934071355e5743bf21d60": "Binance",
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549": "Binance",
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d": "Binance",
    "0x56eddb7aa87536c09ccc2793473599fd21a8b17f": "Binance",
    # Coinbase
    "0x71660c4005ba85c37ccec55d0c4493e66fe775d3": "Coinbase",
    "0xa090e606e30bd747d4e6245a1517ebe430f0057e": "Coinbase",
    "0x503828976d22510aad0201ac7ec88293211d23da": "Coinbase",
    "0xb739d0895772dbb71a89a3754a160269068f0d45": "Coinbase",
    # Kraken
    "0x2910543af39aba0cd09dbb2d50200b3e800a63d2": "Kraken",
    "0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0": "Kraken",
    "0xf66852bc122fd40bfecc63cd48217e88bda12109": "Kraken",
    "0xe853c56864a2ebe4576a807d26fdc4a0ada51919": "Kraken",
    # FTX (relevant for ftx_2022 event)
    "0x2faf487a4414fe77e2327f0bf4ae2a264a776ad2": "FTX",
    "0xc098b2a3aa256d2140208c3de6543aaef5cd3a94": "FTX",
    # OKX / OKEx
    "0x6cc5f688a315f3dc28a7781717a9a798a59fda7b": "OKX",
    "0x236f9f97e0e62388479bf9e5ba4889e46b0273c3": "OKX",
}


# ---------------------------------------------------------------------------
# Core HTTP helper
# ---------------------------------------------------------------------------

def _get(params: dict[str, Any]) -> dict[str, Any]:
    params.setdefault("chainid", 1)
    params.setdefault("apikey", os.environ.get("ETHERSCAN_API_KEY", ""))
    try:
        resp = requests.get(_BASE, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Etherscan request failed: %s", exc)
        return {"status": "0", "result": []}
    data = resp.json()
    if data.get("status") == "0" and data.get("message") not in (
        "No transactions found", "No records found"
    ):
        logger.warning("Etherscan: %s — %s", data.get("message"), data.get("result"))
    return data


# ---------------------------------------------------------------------------
# Single-page fetches (existing API)
# ---------------------------------------------------------------------------

def get_token_transfers(
    contract_address: str,
    start_block: int,
    end_block: int,
    address: str | None = None,
    offset: int = _PAGE_SIZE,
    page: int = 1,
) -> list[dict[str, Any]]:
    """Fetch one page of ERC-20 token transfer events."""
    params: dict[str, Any] = {
        "module": "account",
        "action": "tokentx",
        "contractaddress": contract_address,
        "startblock": start_block,
        "endblock": end_block,
        "offset": offset,
        "page": page,
        "sort": "asc",
    }
    if address:
        params["address"] = address
    return _get(params).get("result", []) or []


def get_logs(
    contract_address: str,
    topic0: str,
    from_block: int,
    to_block: int,
    offset: int = 1_000,
    page: int = 1,
) -> list[dict[str, Any]]:
    """Fetch one page of event logs filtered by contract and topic0."""
    params = {
        "module": "logs",
        "action": "getLogs",
        "address": contract_address,
        "topic0": topic0,
        "fromBlock": from_block,
        "toBlock": to_block,
        "offset": offset,
        "page": page,
    }
    return _get(params).get("result", []) or []


def get_block_number_by_timestamp(ts_unix: int, closest: str = "before") -> int:
    """Return the block number closest to a Unix timestamp."""
    params = {
        "module": "block",
        "action": "getblocknobytime",
        "timestamp": ts_unix,
        "closest": closest,
    }
    result = _get(params).get("result", "0")
    try:
        return int(result)
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Paginated fetch helpers (new)
# ---------------------------------------------------------------------------

# Etherscan v2 constraint: PageNo × Offset ≤ 10 000.
# Use offset=1 000 → max 10 pages before we must split the block range.
_SAFE_OFFSET   = 1_000
_MAX_PAGES     = 10
_MAX_SPLIT_DEPTH = 14   # 2^14 = 16 384 sub-ranges; stops infinite recursion


def get_all_token_transfers(
    contract_address: str,
    start_block: int,
    end_block: int,
    address: str | None = None,
    max_results: int = 20_000,
    _depth: int = 0,
) -> list[dict[str, Any]]:
    """Paginated ERC-20 token transfers with block-range splitting and a results cap.

    Etherscan limits PageNo × Offset ≤ 10 000.  When a block range still
    has results after _MAX_PAGES pages the range is split recursively.
    ``max_results`` caps the total rows returned to prevent runaway recursion
    on high-volume contracts like USDC / USDT (millions of txs per week).

    For exchange-flow aggregation 20 000 rows per address is far more than
    enough signal for 1-hour bucket statistics.
    """
    if _depth > _MAX_SPLIT_DEPTH or start_block > end_block:
        return []

    all_rows: list[dict[str, Any]] = []
    for page in range(1, _MAX_PAGES + 1):
        rows = get_token_transfers(
            contract_address, start_block, end_block,
            address=address, offset=_SAFE_OFFSET, page=page,
        )
        all_rows.extend(rows)
        if len(all_rows) >= max_results:
            logger.debug(
                "Transfer fetch capped at %d results (depth=%d, addr=%s)",
                max_results, _depth, (address or "all")[:10],
            )
            return all_rows[:max_results]
        if len(rows) < _SAFE_OFFSET:
            # Partial page → we have every transfer in this sub-range.
            return all_rows
        time.sleep(_RATE_SLEEP)

    # Exhausted all safe pages without a partial result → split the range.
    mid = (start_block + end_block) // 2
    left = get_all_token_transfers(
        contract_address, start_block, mid,
        address=address, max_results=max_results, _depth=_depth + 1,
    )
    remaining = max_results - len(left)
    if remaining <= 0:
        return left
    right = get_all_token_transfers(
        contract_address, mid + 1, end_block,
        address=address, max_results=remaining, _depth=_depth + 1,
    )
    return left + right


def get_all_logs(
    contract_address: str,
    topic0: str,
    from_block: int,
    to_block: int,
    max_results: int = 200_000,
    _depth: int = 0,
) -> list[dict[str, Any]]:
    """Fully paginated event logs with block-range splitting.

    Mirrors ``get_all_token_transfers``: uses offset=1_000 to stay within
    Etherscan's PageNo×Offset≤10000 constraint, then recursively halves the
    block range when a full 10-page batch arrives.  ``max_results`` prevents
    runaway recursion on contracts with millions of events.

    Default max_results=200_000 is generous for DEX pools (Curve 3pool had
    ~100k events in March 2023).
    """
    if _depth > _MAX_SPLIT_DEPTH or from_block > to_block:
        return []

    _LOG_OFFSET = 1_000
    all_rows: list[dict[str, Any]] = []
    for page in range(1, _MAX_PAGES + 1):
        rows = get_logs(contract_address, topic0, from_block, to_block,
                        offset=_LOG_OFFSET, page=page)
        all_rows.extend(rows)
        if len(all_rows) >= max_results:
            logger.debug(
                "Log fetch capped at %d results (depth=%d, contract=%s)",
                max_results, _depth, contract_address[:10],
            )
            return all_rows[:max_results]
        if len(rows) < _LOG_OFFSET:
            return all_rows
        time.sleep(_RATE_SLEEP)

    # Exhausted all safe pages without partial result → split block range.
    mid = (from_block + to_block) // 2
    left = get_all_logs(contract_address, topic0, from_block, mid,
                        max_results=max_results, _depth=_depth + 1)
    remaining = max_results - len(left)
    if remaining <= 0:
        return left
    right = get_all_logs(contract_address, topic0, mid + 1, to_block,
                         max_results=remaining, _depth=_depth + 1)
    return left + right


# ---------------------------------------------------------------------------
# Ingestion helpers (new — write to bronze parquet)
# ---------------------------------------------------------------------------

def _transfers_to_df(transfers: list[dict[str, Any]]) -> pl.DataFrame:
    """Convert Etherscan token transfer dicts to a typed DataFrame.

    Normalises token amounts using the `tokenDecimal` field.
    """
    if not transfers:
        return pl.DataFrame(schema={
            "block_ts": pl.Int64,
            "wall_clock_utc": pl.Datetime("us", "UTC"),
            "block_number": pl.Int64,
            "tx_hash": pl.Utf8,
            "from_address": pl.Utf8,
            "to_address": pl.Utf8,
            "value_raw": pl.Float64,
            "decimals": pl.Int64,
            "value_norm": pl.Float64,
        })

    rows = []
    for t in transfers:
        try:
            decimals = int(t.get("tokenDecimal", 18))
            value_raw = float(t.get("value", 0))
            rows.append({
                "block_ts":     int(t.get("timeStamp", 0)),
                "block_number": int(t.get("blockNumber", 0)),
                "tx_hash":      t.get("hash", ""),
                "from_address": t.get("from", "").lower(),
                "to_address":   t.get("to", "").lower(),
                "value_raw":    value_raw,
                "decimals":     decimals,
                "value_norm":   value_raw / (10 ** decimals),
            })
        except (ValueError, TypeError):
            continue

    df = pl.DataFrame(rows)
    return df.with_columns(
        (pl.col("block_ts") * 1_000_000).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
        .alias("wall_clock_utc")
    ).sort("wall_clock_utc")


def ingest_mint_burn(
    token_contract: str,
    start_block: int,
    end_block: int,
    out_dir: Path,
    event_id: str,
    node_id: str,
) -> tuple[Path | None, str]:
    """Download mint/burn transfers (to/from null address) for a token contract.

    Mint  = Transfer FROM null address (0x000…0 → any)
    Burn  = Transfer TO   null address (any → 0x000…0 or 0x000…dead)

    Returns:
        (parquet_path, 'A') on success — on-chain events are Tier A.
    """
    if not os.environ.get("ETHERSCAN_API_KEY"):
        logger.info("ETHERSCAN_API_KEY not set; skipping mint/burn ingest for %s", node_id)
        return None, "fixture_non_empirical"

    logger.info("Fetching mint/burn transfers for %s blocks %d–%d", token_contract[:10], start_block, end_block)
    transfers = get_all_token_transfers(token_contract, start_block, end_block)

    if not transfers:
        logger.warning("No transfers found for %s", token_contract[:10])
        return None, "fixture_non_empirical"

    df = _transfers_to_df(transfers)
    # Keep only mint and burn rows
    null_list = list(_NULL_ADDRS)
    df = df.filter(
        pl.col("from_address").is_in(null_list) | pl.col("to_address").is_in(null_list)
    ).with_columns(
        pl.col("from_address").is_in(null_list).alias("mint_flag"),
        pl.col("to_address").is_in(null_list).alias("burn_flag"),
    )

    if df.height == 0:
        logger.warning("No mint/burn rows found for %s", token_contract[:10])
        return None, "fixture_non_empirical"

    # Aggregate to 1-hour windows to get mint_burn_net_1h
    df_agg = (
        df.with_columns(
            ((pl.col("block_ts") // 3600) * 3_600_000_000).cast(pl.Datetime("us"))
            .dt.replace_time_zone("UTC").alias("wall_clock_utc"),
        )
        .group_by("wall_clock_utc")
        .agg(
            (pl.col("value_norm") * pl.col("mint_flag").cast(pl.Float64)).sum().alias("mint_usd"),
            (pl.col("value_norm") * pl.col("burn_flag").cast(pl.Float64)).sum().alias("burn_usd"),
        )
        .with_columns(
            (pl.col("mint_usd") - pl.col("burn_usd")).alias("mint_burn_net_1h"),
        )
        .sort("wall_clock_utc")
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{node_id}_flows.parquet"
    df_agg.write_parquet(out_path)
    logger.info("Wrote %d mint/burn rows for %s → %s (Tier A)", df_agg.height, node_id, out_path.name)
    return out_path, "A"


def ingest_exchange_flows(
    token_contract: str,
    start_block: int,
    end_block: int,
    out_dir: Path,
    event_id: str,
    node_id: str,
    exchange_addresses: dict[str, str] | None = None,
) -> tuple[Path | None, str]:
    """Download exchange inflow/outflow for a token, aggregated to 1-hour windows.

    Uses KNOWN_EXCHANGE_ADDRESSES by default. Supply custom mapping for specificity.

    Returns:
        (parquet_path, 'B') — aggregate flows are Tier B.
    """
    if not os.environ.get("ETHERSCAN_API_KEY"):
        logger.info("ETHERSCAN_API_KEY not set; skipping exchange flow ingest for %s", node_id)
        return None, "fixture_non_empirical"

    if exchange_addresses is None:
        exchange_addresses = KNOWN_EXCHANGE_ADDRESSES

    # Query each known exchange address individually to stay within Etherscan's
    # PageNo × Offset ≤ 10 000 constraint.  Querying the whole contract at once
    # returns millions of transfers for USDC/USDT and immediately hits the limit.
    logger.info("Fetching exchange flow transfers for %s (%d addresses)",
                token_contract[:10], len(exchange_addresses))
    all_transfers: list[dict[str, Any]] = []
    for exch_addr in exchange_addresses:
        rows = get_all_token_transfers(
            token_contract, start_block, end_block, address=exch_addr.lower()
        )
        all_transfers.extend(rows)
        time.sleep(_RATE_SLEEP)

    transfers = all_transfers
    if not transfers:
        return None, "fixture_non_empirical"

    df = _transfers_to_df(transfers)
    # Deduplicate: per-address queries can return the same tx twice when both
    # endpoints are known exchange addresses (exchange-to-exchange transfers).
    if "tx_hash" in df.columns:
        df = df.unique(subset=["tx_hash"], keep="first")
    exch_addrs_lower = {a.lower() for a in exchange_addresses}

    # Tag inflows (to exchange) and outflows (from exchange)
    df = df.with_columns(
        pl.col("to_address").is_in(list(exch_addrs_lower)).alias("is_inflow"),
        pl.col("from_address").is_in(list(exch_addrs_lower)).alias("is_outflow"),
    ).filter(
        pl.col("is_inflow") | pl.col("is_outflow")
    )

    if df.height == 0:
        logger.warning("No exchange-labelled flows found for %s", token_contract[:10])
        return None, "fixture_non_empirical"

    df_agg = (
        df.with_columns(
            ((pl.col("block_ts") // 3600) * 3_600_000_000).cast(pl.Datetime("us"))
            .dt.replace_time_zone("UTC").alias("wall_clock_utc"),
        )
        .group_by("wall_clock_utc")
        .agg(
            (pl.col("value_norm") * pl.col("is_inflow").cast(pl.Float64)).sum().alias("exchange_inflow_1h"),
            (pl.col("value_norm") * pl.col("is_outflow").cast(pl.Float64)).sum().alias("exchange_outflow_1h"),
        )
        .with_columns(
            (pl.col("exchange_inflow_1h") - pl.col("exchange_outflow_1h")).alias("exchange_netflow_1h"),
        )
        .sort("wall_clock_utc")
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{node_id}_flows.parquet"
    df_agg.write_parquet(out_path)
    logger.info("Wrote %d flow rows for %s → %s (Tier B)", df_agg.height, node_id, out_path.name)
    return out_path, "B"
