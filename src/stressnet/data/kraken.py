"""Kraken data ingestion: public REST OHLC and WebSocket book stream.

Uses the unauthenticated Kraken public REST API for historical OHLCV data.
No API key required.

Tier B: 1-minute OHLCV candles.  Depth/imbalance not computable from this feed.

Endpoint: GET https://api.kraken.com/0/public/OHLC
Params: pair (e.g. 'USDCUSD'), interval (minutes), since (Unix timestamp)
Max 720 rows per response; paginate by advancing 'since' past the last row.

Note: Kraken pair names do not use slashes in the API (e.g. 'USDT/USD' → 'USDTUSD').
The function strips slashes automatically.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import requests

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_REST_BASE   = "https://api.kraken.com/0/public"
_SLEEP       = 1.2   # seconds between requests (Kraken rate limit ~1 req/s for public)
_MAX_ITER    = 500   # safety cap on pagination loops


# ---------------------------------------------------------------------------
# Single-page fetch
# ---------------------------------------------------------------------------

def fetch_ohlc(
    pair: str,
    interval: int = 1,
    since: int | None = None,
) -> tuple[list[list], int | None]:
    """Fetch one page of OHLCV data from Kraken.

    Args:
        pair: e.g. 'USDCUSD' (no slashes).
        interval: Candle size in minutes.
        since: Unix timestamp; returns candles *after* this time.

    Returns:
        (rows, last_ts) where rows is a list of
        [time, open, high, low, close, vwap, volume, count] and
        last_ts is Kraken's suggested next `since` value (or None on error).
    """
    url = f"{_REST_BASE}/OHLC"
    params: dict[str, Any] = {"pair": pair, "interval": interval}
    if since is not None:
        params["since"] = since

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.warning("Kraken OHLC request failed (%s): %s", pair, exc)
        return [], None

    if data.get("error"):
        logger.warning("Kraken API error (%s): %s", pair, data["error"])
        return [], None

    result = data.get("result", {})
    last_ts = result.get("last")
    pair_keys = [k for k in result if k != "last"]
    if not pair_keys:
        return [], last_ts
    return result[pair_keys[0]], last_ts


# ---------------------------------------------------------------------------
# Range ingestion (main entry point for script 01)
# ---------------------------------------------------------------------------

def ingest_kraken_ohlc(
    pair: str,
    start_date: date,
    end_date: date,
    out_dir: Path,
    interval: int = 1,
) -> tuple[Path | None, str]:
    """Fetch and save Kraken OHLC data for a date range.

    Paginates automatically (Kraken returns ≤720 rows = 12 h at 1m; a 12-day
    window needs ~24 requests).

    Returns:
        (parquet_path, 'B') on success, or (None, 'fixture_non_empirical').
    """
    # Strip slashes — Kraken API uses 'USDCUSD' not 'USDC/USD'
    pair_clean = pair.replace("/", "")

    start_ts = int(datetime(
        start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc
    ).timestamp())
    end_ts = int(datetime(
        end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=timezone.utc
    ).timestamp())

    all_rows: list[list] = []
    since: int | None = start_ts

    for _ in range(_MAX_ITER):
        rows, last_ts = fetch_ohlc(pair_clean, interval=interval, since=since)
        if not rows:
            break

        # Each row: [time, open, high, low, close, vwap, volume, count]
        # Convert string fields to numeric for consistent typing
        rows_typed = [
            [int(r[0]), float(r[1]), float(r[2]), float(r[3]),
             float(r[4]), float(r[5]), float(r[6]), int(r[7])]
            for r in rows
        ]

        in_window = [r for r in rows_typed if r[0] <= end_ts]
        all_rows.extend(in_window)

        if not in_window or rows_typed[-1][0] >= end_ts:
            break
        # Advance since to avoid re-fetching the same candle
        since = rows_typed[-1][0]
        time.sleep(_SLEEP)

    if not all_rows:
        logger.warning("No Kraken OHLC data for %s %s–%s", pair_clean, start_date, end_date)
        return None, "fixture_non_empirical"

    # Deduplicate and sort
    seen: set[int] = set()
    unique = []
    for r in all_rows:
        if r[0] not in seen:
            seen.add(r[0])
            unique.append(r)
    unique.sort(key=lambda x: x[0])

    df = pl.DataFrame(
        {
            "wall_clock_utc": pl.Series(
                [r[0] for r in unique], dtype=pl.Int64
            ).cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
            "open":    pl.Series([r[1] for r in unique]),
            "high":    pl.Series([r[2] for r in unique]),
            "low":     pl.Series([r[3] for r in unique]),
            "close":   pl.Series([r[4] for r in unique]),
            "vwap":    pl.Series([r[5] for r in unique]),
            "volume":  pl.Series([r[6] for r in unique]),
            "n_trades": pl.Series([r[7] for r in unique], dtype=pl.Int64),
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pair_clean}_ohlc.parquet"
    df.write_parquet(out_path)
    logger.info(
        "Wrote %d Kraken OHLC rows for %s → %s (Tier B)",
        df.height, pair_clean, out_path.name,
    )
    return out_path, "B"


# ---------------------------------------------------------------------------
# Legacy / WebSocket helper
# ---------------------------------------------------------------------------

def ws_subscribe_book(pairs: list[str], depth: int = 10) -> dict[str, Any]:
    """Return a Kraken WS v2 book subscription message."""
    return {
        "method": "subscribe",
        "params": {"channel": "book", "symbol": pairs, "depth": depth, "snapshot": True},
    }
