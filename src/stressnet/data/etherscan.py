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

_BASE         = "https://api.etherscan.io/api"
_PAGE_SIZE    = 10_000  # max offset supported by Etherscan
_RATE_SLEEP   = 0.25    # seconds between calls (free tier: 5 calls/s)
_NULL_ADDRS   = frozenset({
    "0x0000000000000000000000000000000000000000",
    "0x000000000000000000000000000000000000dead",
})

# Well-known exchange hot-wallet addresses for common venues
# (from public labelling sources; non-exhaustive, best-effort Tier B)
KNOWN_EXCHANGE_ADDRESSES: dict[str, str] = {
    "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be": "Binance",
    "0xd551234ae421e3bcba99a0da6d736074f22192ff": "Binance",
    "0x564286362092d8e7936f0549571a803b203aaced": "Binance",
    "0x0681d8db095565fe8a346fa0277bffde9c0edbbf": "Binance",
    "0xfe9e8709d3215310075d67e3ed32a380ccf451c8": "Binance",
    "0x71660c4005ba85c37ccec55d0c4493e66fe775d3": "Coinbase",
    "0xa090e606e30bd747d4e6245a1517ebe430f0057e": "Coinbase",
    "0x503828976d22510aad0201ac7ec88293211d23da": "Coinbase",
    "0x2910543af39aba0cd09dbb2d50200b3e800a63d2": "Kraken",
    "0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0": "Kraken",
    "0xf66852bc122fd40bfecc63cd48217e88bda12109": "Kraken",
    "0x2faf487a4414fe77e2327f0bf4ae2a264a776ad2": "FTX",
    "0xc098b2a3aa256d2140208c3de6543aaef5cd3a94": "FTX",
    "0xbe0eb53f46cd790cd13851d5eff43d12404d33e8": "Binance_cold",
}


# ---------------------------------------------------------------------------
# Core HTTP helper
# ---------------------------------------------------------------------------

def _get(params: dict[str, Any]) -> dict[str, Any]:
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

def get_all_token_transfers(
    contract_address: str,
    start_block: int,
    end_block: int,
    address: str | None = None,
) -> list[dict[str, Any]]:
    """Fully paginated ERC-20 token transfers for a block range.

    Iterates pages until Etherscan returns fewer rows than _PAGE_SIZE.
    """
    all_rows: list[dict[str, Any]] = []
    page = 1
    while True:
        rows = get_token_transfers(
            contract_address, start_block, end_block,
            address=address, offset=_PAGE_SIZE, page=page,
        )
        all_rows.extend(rows)
        if len(rows) < _PAGE_SIZE:
            break
        page += 1
        time.sleep(_RATE_SLEEP)
    return all_rows


def get_all_logs(
    contract_address: str,
    topic0: str,
    from_block: int,
    to_block: int,
) -> list[dict[str, Any]]:
    """Fully paginated event logs for a block range."""
    all_rows: list[dict[str, Any]] = []
    page = 1
    while True:
        rows = get_logs(contract_address, topic0, from_block, to_block,
                        offset=1_000, page=page)
        all_rows.extend(rows)
        if len(rows) < 1_000:
            break
        page += 1
        time.sleep(_RATE_SLEEP)
    return all_rows


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

    logger.info("Fetching exchange flow transfers for %s", token_contract[:10])
    transfers = get_all_token_transfers(token_contract, start_block, end_block)
    if not transfers:
        return None, "fixture_non_empirical"

    df = _transfers_to_df(transfers)
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
