"""I/O helpers: Parquet read/write, manifest writing, path resolution."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def write_parquet(df: pl.DataFrame, path: Path | str, *, mkdir: bool = True) -> None:
    """Write a Polars DataFrame to Parquet, creating parent directories if needed."""
    path = Path(path)
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    logger.info("Wrote %d rows to %s", len(df), path)


def read_parquet(path: Path | str) -> pl.DataFrame:
    """Read a Parquet file into a Polars DataFrame."""
    return pl.read_parquet(Path(path))


def write_manifest(
    path: Path | str,
    source: str,
    params: dict[str, Any],
    raw_bytes: bytes | None = None,
) -> None:
    """Write a data provenance manifest file alongside raw data.

    Args:
        path: Where to write the manifest JSON.
        source: Human-readable source name (e.g. 'coinbase_ws_level2').
        params: Query parameters or API arguments.
        raw_bytes: Optional raw response bytes for SHA-256 hashing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source": source,
        "params": params,
        "fetched_utc": datetime.now(tz=timezone.utc).isoformat(),
        "sha256": hashlib.sha256(raw_bytes).hexdigest() if raw_bytes else None,
    }
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
