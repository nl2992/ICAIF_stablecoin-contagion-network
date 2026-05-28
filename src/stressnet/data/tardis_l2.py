"""Tardis vendor L2 data ingestion.

Downloads full-depth incremental order-book updates (``incremental_book_L2``)
or 25-level snapshots (``book_snapshot_25``) from the Tardis replay API and
normalises them to the bronze schema defined in ``configs/vendor_l2.yaml``.

Authentication
--------------
Set ``TARDIS_API_KEY`` in ``.env``.  Without a key the Tardis replay endpoint
returns 401; a free account can replay the last 3 days only.  Historical data
for all five events requires a paid plan or an academic licence.

Usage
-----
::

    from stressnet.data.tardis_l2 import download_l2_day, check_symbol_availability

    # Check if symbol is in the archive
    avail = check_symbol_availability("binance", "USTUSDT", "2022-05-08", "2022-05-10")

    # Download one day to bronze/
    path = download_l2_day(
        exchange="binance",
        symbol="USDCUSDT",
        data_type="incremental_book_L2",
        day="2023-03-10",
        out_dir=Path("data/bronze/vendor_l2"),
    )

See ``scripts/01b_ingest_vendor_l2.py`` for the batch ingestion entry point.
"""

from __future__ import annotations

import io
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl
import requests

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_REPLAY_BASE = "https://api.tardis.dev/v1/data-feeds"
_DATASETS_BASE = "https://api.tardis.dev/v1/datasets"
_EXCHANGE_INFO_BASE = "https://api.tardis.dev/v1/exchanges"

# Bronze column names — must match configs/vendor_l2.yaml bronze_schema
_BRONZE_COLS = [
    "wall_clock_utc",
    "exchange_ts",
    "local_ts",
    "exchange",
    "symbol",
    "update_type",
    "side",
    "price",
    "size",
    "sequence_id",
    "raw_msg_id",
    "row_position",
]

_BRONZE_SCHEMA: dict[str, type] = {
    "wall_clock_utc": pl.Float64,
    "exchange_ts":    pl.Float64,
    "local_ts":       pl.Float64,
    "exchange":       pl.Utf8,
    "symbol":         pl.Utf8,
    "update_type":    pl.Utf8,
    "side":           pl.Utf8,
    "price":          pl.Float64,
    "size":           pl.Float64,
    "sequence_id":    pl.Int64,
    "raw_msg_id":     pl.Int64,
    "row_position":   pl.Int64,
}


def _api_key() -> str:
    key = os.environ.get("TARDIS_API_KEY", "")
    if not key:
        logger.warning(
            "TARDIS_API_KEY not set. Replay requests will fail for historical data "
            "older than 3 days."
        )
    return key


def _headers() -> dict[str, str]:
    key = _api_key()
    h: dict[str, str] = {"Accept": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


# ---------------------------------------------------------------------------
# Symbol availability check
# ---------------------------------------------------------------------------

def check_symbol_availability(
    exchange: str,
    symbol: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """Check whether a symbol is available in the Tardis archive for the given window.

    Args:
        exchange: e.g. ``"binance"``
        symbol: e.g. ``"USTUSDT"``
        start_date: ISO date string ``"YYYY-MM-DD"``
        end_date: ISO date string ``"YYYY-MM-DD"``

    Returns:
        Dict with keys ``available`` (bool), ``first_available``, ``last_available``
        (ISO date strings or ``None``), and ``message`` (human-readable summary).
    """
    url = f"{_EXCHANGE_INFO_BASE}/{exchange}/datasets/options"
    try:
        resp = requests.get(url, headers=_headers(), timeout=30)
        resp.raise_for_status()
        options = resp.json()
    except requests.HTTPError as exc:
        return {"available": False, "message": str(exc)}

    # Look for the symbol in the exchange's listed options
    symbol_upper = symbol.upper()
    for item in options:
        if item.get("id", "").upper() == symbol_upper or \
                item.get("localTimestamp", "").upper() == symbol_upper:
            return {
                "available": True,
                "first_available": item.get("availableSince"),
                "last_available": item.get("availableTo"),
                "message": f"{symbol} found in Tardis {exchange} archive",
            }

    return {
        "available": False,
        "first_available": None,
        "last_available": None,
        "message": (
            f"{symbol} not found in Tardis {exchange} archive. "
            "It may be delisted or use a different symbol format."
        ),
    }


# ---------------------------------------------------------------------------
# Normalisation: Tardis CSV → bronze Polars DataFrame
# ---------------------------------------------------------------------------

def _parse_incremental_book_l2(raw: bytes, exchange: str, symbol: str) -> pl.DataFrame:
    """Parse Tardis ``incremental_book_L2`` CSV bytes to the bronze schema.

    Tardis incremental_book_L2 columns (tab-separated):
        exchange, symbol, timestamp, local_timestamp, is_snapshot,
        side, price, amount, sequence_id

    ``is_snapshot == true`` → update_type = "snapshot"
    ``is_snapshot == false`` → update_type = "delta"
    """
    text = raw.decode("utf-8", errors="replace")
    try:
        df = pl.read_csv(
            io.StringIO(text),
            separator="\t",
            infer_schema_length=1000,
            ignore_errors=True,
        )
    except Exception as exc:
        logger.warning("Failed to parse CSV for %s/%s: %s", exchange, symbol, exc)
        return _empty_bronze()

    if df.is_empty():
        return _empty_bronze()

    # Normalise column names (Tardis may vary across data types)
    col_map: dict[str, str] = {
        "timestamp":        "exchange_ts_raw",
        "local_timestamp":  "local_ts_raw",
        "is_snapshot":      "is_snapshot",
        "amount":           "size",
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename({old: new})

    rows = []
    for i, row in enumerate(df.iter_rows(named=True)):
        try:
            exch_ts_us = _parse_ts(row.get("exchange_ts_raw", row.get("exchange_ts")))
            local_ts_us = _parse_ts(row.get("local_ts_raw",   row.get("local_ts")))
            is_snap = str(row.get("is_snapshot", "false")).lower() in ("true", "1", "yes")
            rows.append({
                "wall_clock_utc": local_ts_us,
                "exchange_ts":    exch_ts_us,
                "local_ts":       local_ts_us,
                "exchange":       exchange,
                "symbol":         symbol,
                "update_type":    "snapshot" if is_snap else "delta",
                "side":           str(row.get("side", "")).lower(),
                "price":          float(row.get("price", 0.0) or 0.0),
                "size":           float(row.get("size", row.get("amount", 0.0)) or 0.0),
                "sequence_id":    int(row.get("sequence_id", -1) or -1),
                "raw_msg_id":     i,
                "row_position":   i,
            })
        except Exception:
            continue

    if not rows:
        return _empty_bronze()

    return pl.DataFrame(rows, schema=_BRONZE_SCHEMA)


def _parse_book_snapshot_25(raw: bytes, exchange: str, symbol: str) -> pl.DataFrame:
    """Parse Tardis ``book_snapshot_25`` to the bronze schema.

    Tardis book_snapshot_25 CSV columns:
        exchange, symbol, timestamp, local_timestamp, asks[0..24].price,
        asks[0..24].amount, bids[0..24].price, bids[0..24].amount
    """
    text = raw.decode("utf-8", errors="replace")
    try:
        df = pl.read_csv(io.StringIO(text), separator="\t",
                         infer_schema_length=100, ignore_errors=True)
    except Exception as exc:
        logger.warning("Failed to parse snapshot CSV: %s", exc)
        return _empty_bronze()

    if df.is_empty():
        return _empty_bronze()

    rows = []
    row_pos = 0
    for i, row in enumerate(df.iter_rows(named=True)):
        exch_ts_us = _parse_ts(row.get("timestamp"))
        local_ts_us = _parse_ts(row.get("local_timestamp", exch_ts_us))
        for side in ("bid", "ask"):
            for lvl in range(25):
                p_key = f"{side}s[{lvl}].price"
                s_key = f"{side}s[{lvl}].amount"
                if p_key not in row:
                    break
                price = row.get(p_key)
                size  = row.get(s_key, 0.0)
                if price is None:
                    break
                rows.append({
                    "wall_clock_utc": local_ts_us,
                    "exchange_ts":    exch_ts_us,
                    "local_ts":       local_ts_us,
                    "exchange":       exchange,
                    "symbol":         symbol,
                    "update_type":    "snapshot",
                    "side":           side,
                    "price":          float(price or 0.0),
                    "size":           float(size or 0.0),
                    "sequence_id":    int(i),
                    "raw_msg_id":     i,
                    "row_position":   row_pos,
                })
                row_pos += 1

    if not rows:
        return _empty_bronze()
    return pl.DataFrame(rows, schema=_BRONZE_SCHEMA)


def _parse_ts(raw: Any) -> float:
    """Parse a Tardis timestamp (ISO string or int microseconds) to float microseconds."""
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        # Tardis returns microseconds since epoch
        return float(raw)
    s = str(raw).strip()
    if not s:
        return 0.0
    try:
        # ISO 8601 with possible fractional seconds
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.timestamp() * 1_000_000
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return 0.0


def _empty_bronze() -> pl.DataFrame:
    return pl.DataFrame(schema=_BRONZE_SCHEMA)


# ---------------------------------------------------------------------------
# Download: one day at a time
# ---------------------------------------------------------------------------

def download_l2_day(
    exchange: str,
    symbol: str,
    data_type: str,
    day: str | date,
    out_dir: Path,
    *,
    overwrite: bool = False,
    retry_count: int = 3,
    retry_delay: float = 5.0,
) -> Path | None:
    """Download and normalise one day of L2 data from Tardis.

    Args:
        exchange: Exchange identifier (e.g. ``"binance"``).
        symbol: Market symbol (e.g. ``"USDCUSDT"``).
        data_type: ``"incremental_book_L2"`` or ``"book_snapshot_25"``.
        day: Date as ISO string or ``datetime.date``.
        out_dir: Directory to write the output parquet file.
        overwrite: Re-download even if the output file already exists.
        retry_count: Number of HTTP retries on transient errors.
        retry_delay: Seconds to wait between retries.

    Returns:
        Path to the written parquet file, or ``None`` if download failed.
    """
    if isinstance(day, str):
        day = date.fromisoformat(day)

    day_str = day.strftime("%Y-%m-%d")
    fname = f"{exchange}_{symbol}_{data_type}_{day_str}.parquet"
    out_path = out_dir / fname

    if out_path.exists() and not overwrite:
        logger.debug("Already exists, skipping: %s", out_path.name)
        return out_path

    out_dir.mkdir(parents=True, exist_ok=True)

    # Tardis replay API URL
    url = f"{_REPLAY_BASE}/{exchange}"
    params = {
        "filters":  f"[{{\"channel\":\"{data_type}\",\"symbols\":[\"{symbol}\"]}}]",
        "from":     f"{day_str}T00:00:00.000Z",
        "to":       f"{day_str}T23:59:59.999Z",
        "format":   "csv",
        "delimiter": "\t",
    }

    raw: bytes | None = None
    for attempt in range(retry_count):
        try:
            resp = requests.get(url, params=params, headers=_headers(),
                                timeout=120, stream=True)
            resp.raise_for_status()
            raw = resp.content
            break
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (401, 403):
                logger.error(
                    "Tardis auth error for %s/%s/%s. "
                    "Check TARDIS_API_KEY and subscription.",
                    exchange, symbol, day_str,
                )
                return None
            logger.warning(
                "HTTP error on attempt %d/%d for %s/%s/%s: %s",
                attempt + 1, retry_count, exchange, symbol, day_str, exc,
            )
        except requests.RequestException as exc:
            logger.warning(
                "Request error on attempt %d/%d: %s",
                attempt + 1, retry_count, exc,
            )
        if attempt < retry_count - 1:
            time.sleep(retry_delay)

    if raw is None:
        logger.error("All %d attempts failed for %s/%s/%s", retry_count, exchange, symbol, day_str)
        return None

    # Normalise to bronze schema
    if data_type == "incremental_book_L2":
        df = _parse_incremental_book_l2(raw, exchange, symbol)
    elif data_type == "book_snapshot_25":
        df = _parse_book_snapshot_25(raw, exchange, symbol)
    else:
        logger.error("Unsupported data_type: %s", data_type)
        return None

    if df.is_empty():
        logger.warning("No rows parsed for %s/%s on %s", exchange, symbol, day_str)
        return None

    df.write_parquet(out_path, compression="zstd")
    logger.info(
        "Wrote %d rows → %s (%.1f MB)",
        len(df),
        out_path.name,
        out_path.stat().st_size / 1e6,
    )
    return out_path


def download_l2_range(
    exchange: str,
    symbol: str,
    data_type: str,
    start: str | date,
    end: str | date,
    out_dir: Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Download a date range of L2 data, one parquet file per day.

    Args:
        start: First day inclusive (``"YYYY-MM-DD"`` or ``date``).
        end: Last day inclusive.

    Returns:
        List of paths to successfully written files (empty on complete failure).
    """
    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(end, str):
        end = date.fromisoformat(end)

    paths: list[Path] = []
    current = start
    while current <= end:
        path = download_l2_day(exchange, symbol, data_type, current, out_dir,
                               overwrite=overwrite)
        if path is not None:
            paths.append(path)
        current += timedelta(days=1)

    logger.info(
        "Downloaded %d / %d days for %s/%s",
        len(paths),
        (end - start).days + 1,
        exchange,
        symbol,
    )
    return paths


# ---------------------------------------------------------------------------
# Load bronze files into a single DataFrame
# ---------------------------------------------------------------------------

def load_bronze(
    exchange: str,
    symbol: str,
    data_type: str,
    bronze_dir: Path,
    start: str | date | None = None,
    end: str | date | None = None,
) -> pl.DataFrame:
    """Load all bronze parquet files for a given exchange/symbol/data_type.

    Args:
        start: Inclusive start date filter (optional).
        end: Inclusive end date filter (optional).

    Returns:
        Concatenated DataFrame sorted by ``exchange_ts``.
    """
    pattern = f"{exchange}_{symbol}_{data_type}_*.parquet"
    files = sorted(bronze_dir.glob(pattern))

    if not files:
        logger.warning(
            "No bronze files found matching %s in %s", pattern, bronze_dir
        )
        return _empty_bronze()

    if start is not None:
        start_d = date.fromisoformat(str(start)) if isinstance(start, str) else start
        files = [f for f in files if _date_from_fname(f) >= start_d]
    if end is not None:
        end_d = date.fromisoformat(str(end)) if isinstance(end, str) else end
        files = [f for f in files if _date_from_fname(f) <= end_d]

    if not files:
        logger.warning("No files in date range for %s/%s", exchange, symbol)
        return _empty_bronze()

    frames = []
    for f in files:
        try:
            frames.append(pl.read_parquet(f))
        except Exception as exc:
            logger.warning("Could not read %s: %s", f.name, exc)

    if not frames:
        return _empty_bronze()

    combined = pl.concat(frames, how="diagonal").sort("exchange_ts")
    # Overwrite row_position with a fresh sequential index after sort
    if "row_position" in combined.columns:
        combined = combined.drop("row_position")
    return combined.with_row_index("row_position").cast({"row_position": pl.Int64})


def _date_from_fname(path: Path) -> date:
    """Extract the date from a bronze filename like ``binance_USDCUSDT_..._2023-03-10.parquet``."""
    stem = path.stem
    # Last 10 chars before extension are YYYY-MM-DD
    date_str = stem[-10:]
    try:
        return date.fromisoformat(date_str)
    except ValueError:
        return date(1970, 1, 1)
