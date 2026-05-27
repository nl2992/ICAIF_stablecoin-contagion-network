"""Coin Metrics API: exchange flow and asset-level network metrics."""

from __future__ import annotations

import os
from datetime import date
from typing import Any

import requests

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_BASE = "https://api.coinmetrics.io/v4"


def _headers() -> dict[str, str]:
    api_key = os.environ.get("COINMETRICS_API_KEY", "")
    return {"x-cm-api-key": api_key} if api_key else {}


def get_asset_metrics(
    assets: list[str],
    metrics: list[str],
    start_time: str,
    end_time: str,
    frequency: str = "1h",
) -> list[dict[str, Any]]:
    """Fetch asset-level metrics (e.g. exchange flows) from Coin Metrics.

    Args:
        assets: e.g. ['usdc', 'usdt']
        metrics: e.g. ['FlowInExNtv', 'FlowOutExNtv', 'FlowNetInExNtv']
        start_time: ISO-8601 string, e.g. '2023-03-08T00:00:00Z'
        end_time: ISO-8601 string
        frequency: '1h', '1d', etc.

    Returns:
        List of data point dicts with keys: asset, time, and one key per metric.
    """
    url = f"{_BASE}/timeseries/asset-metrics"
    params = {
        "assets": ",".join(assets),
        "metrics": ",".join(metrics),
        "start_time": start_time,
        "end_time": end_time,
        "frequency": frequency,
        "page_size": 10_000,
    }
    results = []
    while url:
        resp = requests.get(url, params=params, headers=_headers(), timeout=60)
        resp.raise_for_status()
        data = resp.json()
        results.extend(data.get("data", []))
        url = data.get("next_page_url")
        params = {}  # pagination URL already includes params

    logger.info("Fetched %d Coin Metrics data points", len(results))
    return results
