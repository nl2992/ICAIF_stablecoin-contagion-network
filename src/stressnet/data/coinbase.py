"""Coinbase Advanced Trade data ingestion: WebSocket Level2 and REST candles."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_REST_BASE = "https://api.coinbase.com/api/v3/brokerage"
_WS_URL = "wss://advanced-trade-ws.coinbase.com"


def ws_subscribe_level2(product_ids: list[str]) -> dict[str, Any]:
    """Return a level2 subscription message for the Coinbase Advanced Trade WS.

    Caller is responsible for the WebSocket connection; this returns the
    JSON subscription dict to send after connecting.
    """
    return {
        "type": "subscribe",
        "product_ids": product_ids,
        "channel": "level2",
    }


def fetch_candles(
    product_id: str,
    start: datetime,
    end: datetime,
    granularity: str = "ONE_MINUTE",
) -> list[dict[str, Any]]:
    """Fetch OHLCV candles from Coinbase Advanced Trade REST API.

    Args:
        product_id: e.g. 'USDC-USD'
        start: Start datetime (UTC).
        end: End datetime (UTC).
        granularity: Candle size string; default 'ONE_MINUTE'.

    Returns:
        List of candle dicts with keys: start, low, high, open, close, volume.
    """
    url = f"{_REST_BASE}/market/products/{product_id}/candles"
    params = {
        "start": int(start.timestamp()),
        "end": int(end.timestamp()),
        "granularity": granularity,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("candles", [])


def fetch_best_bid_ask(product_id: str) -> dict[str, Any]:
    """Fetch current best bid/ask from the Coinbase REST book endpoint."""
    url = f"{_REST_BASE}/best_bid_ask"
    resp = requests.get(url, params={"product_ids": product_id}, timeout=10)
    resp.raise_for_status()
    return resp.json()
