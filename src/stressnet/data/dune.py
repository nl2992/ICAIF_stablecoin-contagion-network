"""Dune Analytics API: execute queries and fetch results."""

from __future__ import annotations

import os
import time
from typing import Any

import requests

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_BASE = "https://api.dune.com/api/v1"


def _headers() -> dict[str, str]:
    api_key = os.environ.get("DUNE_API_KEY", "")
    return {"X-Dune-API-Key": api_key}


def execute_query(query_id: int, params: dict[str, Any] | None = None) -> str:
    """Trigger a Dune query execution and return the execution_id."""
    url = f"{_BASE}/query/{query_id}/execute"
    payload: dict[str, Any] = {}
    if params:
        payload["query_parameters"] = params
    resp = requests.post(url, json=payload, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()["execution_id"]


def get_results(execution_id: str, *, poll_interval: float = 5.0) -> list[dict[str, Any]]:
    """Poll for Dune execution results; blocks until complete.

    Returns the list of row dicts from the result set.
    """
    url = f"{_BASE}/execution/{execution_id}/results"
    while True:
        resp = requests.get(url, headers=_headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()
        state = data.get("state", "QUERY_STATE_PENDING")
        if state == "QUERY_STATE_COMPLETED":
            return data.get("result", {}).get("rows", [])
        if state in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"):
            raise RuntimeError(f"Dune execution {execution_id} ended with state {state}")
        logger.debug("Waiting for execution %s (state=%s)", execution_id, state)
        time.sleep(poll_interval)


def run_query(query_id: int, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Execute a Dune query and return results synchronously."""
    execution_id = execute_query(query_id, params)
    return get_results(execution_id)
