"""Coinbase Exchange public REST data ingestion.

Uses the unauthenticated Coinbase Exchange (formerly Coinbase Pro) REST API for
historical OHLCV candles. No API key required.

Tier B: 1-minute OHLCV candles.  Depth/imbalance are not computable from this feed.

Endpoint: https://api.exchange.coinbase.com/products/{product_id}/candles
Params: start (ISO8601), end (ISO8601), granularity (seconds: 60, 300, 900, …)
Returns: [[time_unix, low, high, open, close, volume], ...] descending.
Max 300 candles per request; chunk into 300-candle windows.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl
import requests

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_EXCHANGE_REST_BASE = "https://api.exchange.coinbase.com"
_MAX_CANDLES = 300    # API hard cap per request
_SLEEP_BETWEEN = 0.4  # seconds — stay under 10 req/s public limit


# ---------------------------------------------------------------------------
# REST fetch helpers
# ---------------------------------------------------------------------------

def fetch_exchange_candles(
    product_id: str,
    start: datetime,
    end: datetime,
    granularity: int = 60,
) -> list[list]:
    """Fetch OHLCV candles from the public Coinbase Exchange REST API.

    Automatically chunks the request into windows of ≤ MAX_CANDLES each.
    Returns all candles sorted ascending by time.

    Args:
        product_id: e.g. 'USDC-USD'
        start: Inclusive start (UTC).
        end: Inclusive end (UTC).
        granularity: Candle size in seconds (60, 300, 900, 3600, 21600, 86400).
    """
    url = f"{_EXCHANGE_REST_BASE}/products/{product_id}/candles"
    all_candles: list[list] = []

    chunk_secs = _MAX_CANDLES * granularity
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(seconds=chunk_secs), end)
        params = {
            "start": chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end":   chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "granularity": granularity,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            candles = resp.json()
            if isinstance(candles, list):
                all_candles.extend(candles)
            elif isinstance(candles, dict) and "message" in candles:
                logger.warning("Coinbase API error for %s: %s", product_id, candles["message"])
        except requests.RequestException as exc:
            logger.warning("Coinbase candle fetch failed %s %s: %s", product_id, chunk_start.date(), exc)
        chunk_start = chunk_end + timedelta(seconds=granularity)
        time.sleep(_SLEEP_BETWEEN)

    # Sort ascending (API returns descending) and deduplicate
    all_candles.sort(key=lambda x: x[0])
    seen: set = set()
    deduped = []
    for c in all_candles:
        if c[0] not in seen:
            seen.add(c[0])
            deduped.append(c)
    return deduped


# ---------------------------------------------------------------------------
# Range ingestion (main entry point for script 01)
# ---------------------------------------------------------------------------

def ingest_coinbase_range(
    product_id: str,
    start_date: date,
    end_date: date,
    out_dir: Path,
    granularity: int = 60,
) -> tuple[Path | None, str]:
    """Fetch and save Coinbase OHLCV data for a date range.

    Returns:
        (parquet_path, 'B') on success, or (None, 'fixture_non_empirical') if
        no data could be fetched.
    """
    start_utc = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_utc   = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=timezone.utc)

    candles = fetch_exchange_candles(product_id, start_utc, end_utc, granularity)
    if not candles:
        logger.warning("No Coinbase candles returned for %s %s–%s", product_id, start_date, end_date)
        return None, "fixture_non_empirical"

    # Build DataFrame: [[time_unix, low, high, open, close, volume], ...]
    # Coinbase candle timestamps are Unix seconds; multiply by 1e6 for Polars us-precision
    df = pl.DataFrame(
        {
            "wall_clock_utc": (
                pl.Series([c[0] for c in candles], dtype=pl.Int64) * 1_000_000
            ).cast(pl.Datetime("us", "UTC")),
            "low":    pl.Series([float(c[1]) for c in candles]),
            "high":   pl.Series([float(c[2]) for c in candles]),
            "open":   pl.Series([float(c[3]) for c in candles]),
            "close":  pl.Series([float(c[4]) for c in candles]),
            "volume": pl.Series([float(c[5]) for c in candles]),
        }
    ).sort("wall_clock_utc")

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_pid = product_id.replace("-", "_")
    out_path = out_dir / f"{safe_pid}_candles.parquet"
    df.write_parquet(out_path)
    logger.info(
        "Wrote %d Coinbase candles for %s → %s (Tier B)",
        df.height, product_id, out_path.name,
    )
    return out_path, "B"


# ---------------------------------------------------------------------------
# Legacy compatibility
# ---------------------------------------------------------------------------

def ws_subscribe_level2(product_ids: list[str]) -> dict[str, Any]:
    """Level2 subscription message for Coinbase Advanced Trade WebSocket."""
    return {"type": "subscribe", "product_ids": product_ids, "channel": "level2"}


def fetch_best_bid_ask(product_id: str) -> dict[str, Any]:
    """Fetch current best bid/ask via the public Coinbase Exchange book endpoint."""
    url = f"{_EXCHANGE_REST_BASE}/products/{product_id}/book"
    resp = requests.get(url, params={"level": 1}, timeout=10)
    resp.raise_for_status()
    return resp.json()
