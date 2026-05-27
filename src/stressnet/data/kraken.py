"""Kraken WebSocket book stream ingestion."""

from __future__ import annotations

from typing import Any


def ws_subscribe_book(pairs: list[str], depth: int = 10) -> dict[str, Any]:
    """Return a Kraken WS v2 book subscription message.

    See: https://docs.kraken.com/api/docs/websocket-v2/book
    """
    return {
        "method": "subscribe",
        "params": {
            "channel": "book",
            "symbol": pairs,
            "depth": depth,
            "snapshot": True,
        },
    }
