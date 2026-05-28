"""SHA-256 checksum verification for Binance Vision zip downloads.

Binance Vision provides a ``CHECKSUM`` file alongside each data zip:
  https://data.binance.vision/data/spot/daily/klines/USDCUSDT/1m/
      USDCUSDT-1m-2023-03-10.zip
      USDCUSDT-1m-2023-03-10.zip.CHECKSUM

This module fetches and verifies those checksums before the downloaded zip
is consumed by the ingestion pipeline.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import requests

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Compute SHA-256 of a local file in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def fetch_checksum(checksum_url: str, timeout: int = 30) -> str:
    """Fetch the expected SHA-256 from a Binance Vision .CHECKSUM file.

    The file format is ``<sha256hex>  <filename>`` (two spaces, then filename).
    Returns the hex digest string (lower-case, 64 chars).

    Raises:
        requests.HTTPError: if the URL is not found or returns an error.
        ValueError: if the checksum file cannot be parsed.
    """
    resp = requests.get(checksum_url, timeout=timeout)
    resp.raise_for_status()
    text = resp.text.strip()
    parts = text.split()
    if len(parts) < 1 or len(parts[0]) != 64:
        raise ValueError(
            f"Cannot parse checksum from '{checksum_url}': got {text!r}"
        )
    return parts[0].lower()


def verify_zip(zip_path: Path, checksum_url: str) -> bool:
    """Verify a downloaded zip file against its Binance Vision checksum.

    Args:
        zip_path: Local path to the downloaded .zip file.
        checksum_url: URL of the corresponding .zip.CHECKSUM file on Vision.

    Returns:
        True if the digest matches, False otherwise.

    Raises:
        FileNotFoundError: if ``zip_path`` does not exist.
        requests.HTTPError / ValueError: if the remote checksum cannot be retrieved.
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    expected = fetch_checksum(checksum_url)
    actual = _sha256_file(zip_path)

    if actual == expected:
        logger.debug("Checksum OK: %s", zip_path.name)
        return True

    logger.error(
        "Checksum MISMATCH for %s: expected=%s actual=%s",
        zip_path.name,
        expected,
        actual,
    )
    return False


def checksum_url_for(data_url: str) -> str:
    """Return the corresponding .CHECKSUM URL for a Binance Vision data URL.

    Example:
        >>> checksum_url_for("https://data.binance.vision/data/spot/daily/klines/USDCUSDT/1m/USDCUSDT-1m-2023-03-10.zip")
        'https://data.binance.vision/data/spot/daily/klines/USDCUSDT/1m/USDCUSDT-1m-2023-03-10.zip.CHECKSUM'
    """
    return data_url + ".CHECKSUM"


def verify_vision_download(zip_path: Path, data_url: str) -> bool:
    """Convenience wrapper: fetch checksum URL from the data URL and verify.

    Args:
        zip_path: Local path to the downloaded file.
        data_url: The URL that was used to download ``zip_path``.

    Returns:
        True if digest matches; False if mismatch; raises on network/parse errors.
    """
    cs_url = checksum_url_for(data_url)
    return verify_zip(zip_path, cs_url)
