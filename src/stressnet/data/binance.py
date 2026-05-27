"""Binance data ingestion: Vision archive downloads and WebSocket depth streams."""

from __future__ import annotations

import io
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import requests

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_VISION_BASE = "https://data.binance.vision/data/spot/daily"


def vision_url(symbol: str, data_type: str, day: date) -> str:
    """Return the Binance Vision URL for a symbol/type/day combination.

    Args:
        symbol: e.g. 'USDCUSDT'
        data_type: e.g. 'aggTrades', 'klines/1m', 'bookDepth'
        day: Date to fetch.
    """
    day_str = day.strftime("%Y-%m-%d")
    return f"{_VISION_BASE}/{data_type}/{symbol}/{symbol}-{data_type.split('/')[-1]}-{day_str}.zip"


def download_vision_zip(url: str, dest_dir: Path, *, overwrite: bool = False) -> Path | None:
    """Download a Binance Vision zip file to dest_dir.

    Returns the path to the extracted CSV, or None if the file is unavailable.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1].replace(".zip", ".csv")
    dest_path = dest_dir / filename

    if dest_path.exists() and not overwrite:
        logger.debug("Skipping %s (already exists)", filename)
        return dest_path

    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        logger.warning("Not found: %s", url)
        return None
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        if not names:
            return None
        zf.extract(names[0], dest_dir)
        extracted = dest_dir / names[0]
        if extracted != dest_path:
            extracted.rename(dest_path)

    logger.info("Downloaded %s", dest_path.name)
    return dest_path


def fetch_book_depth_ws(
    symbol: str,
    depth_levels: int = 20,
    update_speed_ms: int = 100,
) -> dict[str, Any]:
    """Return WebSocket stream parameters for a Binance depth channel.

    Does not establish the connection; returns a config dict for use with
    a WebSocket client. Connection management is handled by the caller.
    """
    stream_name = f"{symbol.lower()}@depth{depth_levels}@{update_speed_ms}ms"
    return {
        "url": f"wss://stream.binance.com:9443/ws/{stream_name}",
        "stream": stream_name,
        "symbol": symbol,
        "depth_levels": depth_levels,
        "update_speed_ms": update_speed_ms,
    }
