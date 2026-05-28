"""Binance data ingestion: Vision archive downloads and WebSocket depth streams.

Data-type priority for CEX node reconstruction (highest → lowest quality):
1. bookTicker  — best bid/ask at ~ms frequency (Tier A: real-time BBO)
2. klines/1m   — 1-minute OHLCV candles (Tier B: aggregate, spread proxy only)

Both are freely available at data.binance.vision with no API key.
The depth 20-level snapshot type ('depth') is also available on Vision but
produces very large files; use bookTicker + klines for now.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from typing import Any

import polars as pl
import requests

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_VISION_DAILY   = "https://data.binance.vision/data/spot/daily"
_VISION_MONTHLY = "https://data.binance.vision/data/spot/monthly"
_VISION_BASE    = _VISION_DAILY  # kept for external callers

# Column schemas — Binance Vision CSVs have no header row
_KLINE_COLS = [
    "open_time_ms", "open", "high", "low", "close", "volume",
    "close_time_ms", "quote_volume", "n_trades",
    "taker_buy_base", "taker_buy_quote", "_ignore",
]
_BOOK_TICKER_COLS = [
    "update_id", "best_bid_price", "best_bid_qty",
    "best_ask_price", "best_ask_qty", "transaction_time_ms", "event_time_ms",
]


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------

def vision_url(symbol: str, data_type: str, day: date, monthly: bool = False) -> str:
    """Return the Binance Vision URL for a symbol/type/day combination.

    Args:
        symbol: e.g. 'USDCUSDT'
        data_type: 'bookTicker', 'klines/1m', 'aggTrades', 'depth', etc.
        day: The date (or any day in the target month for monthly=True).
        monthly: If True, produce a monthly URL (YYYY-MM) instead of daily.

    klines have a subdirectory: 'klines/1m' path=klines subdir=1m.
    All other types: {data_type}/{symbol}/{symbol}-{data_type}-{date}.zip
    """
    base = _VISION_MONTHLY if monthly else _VISION_DAILY
    date_str = day.strftime("%Y-%m") if monthly else day.strftime("%Y-%m-%d")
    if "/" in data_type:
        path, subdir = data_type.split("/", 1)
        return f"{base}/{path}/{symbol}/{subdir}/{symbol}-{subdir}-{date_str}.zip"
    return f"{base}/{data_type}/{symbol}/{symbol}-{data_type}-{date_str}.zip"


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def download_vision_zip(url: str, dest_dir: Path, *, overwrite: bool = False) -> Path | None:
    """Download a Binance Vision zip and extract the inner CSV.

    Returns the CSV path, or None if the file is unavailable (404 or network error).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1].replace(".zip", ".csv")
    dest_path = dest_dir / filename

    if dest_path.exists() and not overwrite:
        logger.debug("Cache hit: %s", filename)
        return dest_path

    try:
        resp = requests.get(url, timeout=120)
    except requests.RequestException as exc:
        logger.warning("Download failed (%s): %s", url, exc)
        return None

    if resp.status_code == 404:
        logger.debug("Not found: %s", url)
        return None
    resp.raise_for_status()

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
            if not names:
                return None
            zf.extract(names[0], dest_dir)
            extracted = dest_dir / names[0]
            if extracted != dest_path:
                extracted.rename(dest_path)
    except zipfile.BadZipFile as exc:
        logger.warning("Bad zip for %s: %s", url, exc)
        return None

    logger.info("Downloaded %s (%d KB)", dest_path.name, len(resp.content) // 1024)
    return dest_path


# ---------------------------------------------------------------------------
# CSV parsers
# ---------------------------------------------------------------------------

def parse_klines_csv(path: Path) -> pl.DataFrame:
    """Parse a Binance Vision klines/1m CSV.

    Returns: wall_clock_utc, open, high, low, close, volume, n_trades.
    """
    df = pl.read_csv(
        path,
        has_header=False,
        new_columns=_KLINE_COLS,
        schema_overrides={
            "open_time_ms": pl.Int64,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "close_time_ms": pl.Int64,
            "quote_volume": pl.Float64,
            "n_trades": pl.Int64,
            "taker_buy_base": pl.Float64,
            "taker_buy_quote": pl.Float64,
            "_ignore": pl.Utf8,
        },
        ignore_errors=True,
    )
    return (
        df.with_columns(
            (pl.col("open_time_ms") * 1_000).cast(pl.Datetime("us", "UTC")).alias("wall_clock_utc")
        )
        .select(["wall_clock_utc", "open", "high", "low", "close", "volume", "n_trades"])
    )


def parse_book_ticker_csv(path: Path) -> pl.DataFrame:
    """Parse a Binance Vision bookTicker CSV.

    Returns: wall_clock_utc, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty.
    """
    df = pl.read_csv(
        path,
        has_header=False,
        new_columns=_BOOK_TICKER_COLS,
        schema_overrides={
            "update_id": pl.Int64,
            "best_bid_price": pl.Float64,
            "best_bid_qty": pl.Float64,
            "best_ask_price": pl.Float64,
            "best_ask_qty": pl.Float64,
            "transaction_time_ms": pl.Int64,
            "event_time_ms": pl.Int64,
        },
        ignore_errors=True,
    )
    return (
        df.with_columns(
            (pl.col("event_time_ms") * 1_000).cast(pl.Datetime("us", "UTC")).alias("wall_clock_utc")
        )
        .select(["wall_clock_utc", "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty"])
    )


# ---------------------------------------------------------------------------
# Range ingestion (main entry point for script 01)
# ---------------------------------------------------------------------------

def _parse_frame(csv_path: Path, data_type: str) -> pl.DataFrame | None:
    """Parse a downloaded CSV into a DataFrame based on data_type."""
    try:
        if data_type == "bookTicker":
            return parse_book_ticker_csv(csv_path)
        elif data_type.startswith("klines"):
            return parse_klines_csv(csv_path)
        else:
            logger.warning("Unsupported data_type '%s' for parsing", data_type)
            return None
    except Exception as exc:
        logger.warning("Parse error for %s: %s", csv_path.name, exc)
        return None


def ingest_binance_range(
    symbol: str,
    start_date: date,
    end_date: date,
    out_dir: Path,
    data_type: str = "bookTicker",
    overwrite: bool = False,
) -> tuple[Path | None, str]:
    """Download and concatenate Binance Vision data for a date range.

    Strategy:
    1. Try daily files for each day in [start_date, end_date].
    2. If no daily files found, try monthly archive(s) and filter to the range.
       This covers markets listed mid-month or pairs only archived monthly
       (e.g. USDCUSDT started trading 2023-03-11 and has only monthly archives).

    Returns:
        (parquet_path, tier_actual) where tier='A' for bookTicker,
        'B' for klines, or (None, 'fixture_non_empirical') if nothing found.
    """
    cache_daily   = out_dir / "_raw_cache" / symbol / data_type.replace("/", "_") / "daily"
    cache_monthly = out_dir / "_raw_cache" / symbol / data_type.replace("/", "_") / "monthly"
    cache_daily.mkdir(parents=True, exist_ok=True)
    cache_monthly.mkdir(parents=True, exist_ok=True)

    frames: list[pl.DataFrame] = []

    # ---- Pass 1: daily files ----
    current = start_date
    while current <= end_date:
        url = vision_url(symbol, data_type, current, monthly=False)
        csv_path = download_vision_zip(url, cache_daily, overwrite=overwrite)
        if csv_path is not None:
            frame = _parse_frame(csv_path, data_type)
            if frame is not None and frame.height > 0:
                frames.append(frame)
        current += timedelta(days=1)

    # ---- Pass 2: monthly archives (fallback when daily data missing) ----
    if not frames:
        logger.info(
            "No daily Vision data found for %s %s; trying monthly archives.", symbol, data_type
        )
        # Collect unique (year, month) pairs covering the date range
        months: set[tuple[int, int]] = set()
        current = start_date
        while current <= end_date:
            months.add((current.year, current.month))
            current += timedelta(days=1)

        for year, month in sorted(months):
            representative_day = date(year, month, 1)
            url = vision_url(symbol, data_type, representative_day, monthly=True)
            csv_path = download_vision_zip(url, cache_monthly, overwrite=overwrite)
            if csv_path is not None:
                frame = _parse_frame(csv_path, data_type)
                if frame is not None and frame.height > 0:
                    frames.append(frame)

    if not frames:
        return None, "fixture_non_empirical"

    combined = pl.concat(frames, how="diagonal").sort("wall_clock_utc")

    # Filter strictly to the requested date range
    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_dt   = datetime(end_date.year,   end_date.month,   end_date.day,   23, 59, 59, tzinfo=timezone.utc)
    combined = combined.filter(
        (pl.col("wall_clock_utc") >= start_dt) & (pl.col("wall_clock_utc") <= end_dt)
    )
    # Deduplicate by wall_clock_utc
    combined = combined.unique(subset=["wall_clock_utc"], keep="last").sort("wall_clock_utc")

    if combined.height == 0:
        logger.warning("No rows in date range %s–%s for %s %s", start_date, end_date, symbol, data_type)
        return None, "fixture_non_empirical"

    tier = "A" if data_type == "bookTicker" else "B"

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_type = data_type.replace("/", "_")
    out_path = out_dir / f"{symbol}_{safe_type}.parquet"
    combined.write_parquet(out_path)
    logger.info(
        "Wrote %d rows · %s %s · Tier %s → %s",
        combined.height, symbol, data_type, tier, out_path.name,
    )
    return out_path, tier


# ---------------------------------------------------------------------------
# Live / WebSocket helpers (kept for future use)
# ---------------------------------------------------------------------------

def fetch_book_depth_ws(
    symbol: str,
    depth_levels: int = 20,
    update_speed_ms: int = 100,
) -> dict[str, Any]:
    """Return WebSocket stream config for a Binance depth channel.

    Does not establish the connection; returns params for a WebSocket client.
    """
    stream_name = f"{symbol.lower()}@depth{depth_levels}@{update_speed_ms}ms"
    return {
        "url": f"wss://stream.binance.com:9443/ws/{stream_name}",
        "stream": stream_name,
        "symbol": symbol,
        "depth_levels": depth_levels,
        "update_speed_ms": update_speed_ms,
    }
