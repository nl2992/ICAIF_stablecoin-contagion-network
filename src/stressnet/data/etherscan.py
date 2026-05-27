"""Etherscan API: ERC-20 token transfers, event logs, L2 deposit/withdrawal."""

from __future__ import annotations

import os
from typing import Any

import requests

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_BASE = "https://api.etherscan.io/api"


def _get(params: dict[str, Any]) -> dict[str, Any]:
    params.setdefault("apikey", os.environ.get("ETHERSCAN_API_KEY", ""))
    resp = requests.get(_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") == "0" and data.get("message") not in ("No transactions found",):
        logger.warning("Etherscan error: %s — %s", data.get("message"), data.get("result"))
    return data


def get_token_transfers(
    contract_address: str,
    start_block: int,
    end_block: int,
    address: str | None = None,
    offset: int = 10_000,
    page: int = 1,
) -> list[dict[str, Any]]:
    """Fetch ERC-20 token transfer events for a contract."""
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
    """Fetch event logs filtered by contract address and topic0."""
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
    return int(result)
