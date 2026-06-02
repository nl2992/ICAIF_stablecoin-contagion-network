"""Generic The Graph subgraph GraphQL client.

Provides a reusable paginated query helper for any The Graph hosted
subgraph.  Other modules (uniswap.py, etc.) import from here rather than
embedding HTTP boilerplate directly.

Requires THE_GRAPH_API_KEY environment variable.  Without it, all calls
return empty lists and log a debug message — the caller is responsible for
falling back to fixtures.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_GRAPH_BASE = "https://gateway.thegraph.com/api"
_RATE_SLEEP  = 0.2   # seconds between paginated requests (polite rate limiting)


def query_subgraph(
    subgraph_id: str,
    query: str,
    variables: dict[str, Any] | None = None,
    data_key: str | None = None,
    first: int = 1_000,
    max_results: int = 100_000,
) -> list[dict[str, Any]]:
    """Execute a paginated The Graph subgraph query.

    Paginates using ``skip``/``first`` variables until a partial page is
    returned.  The caller's ``query`` must accept ``$skip: Int!`` and
    ``$first: Int!`` variables; extra variables are merged in via
    ``variables``.

    Args:
        subgraph_id:  The Graph subgraph deployment ID (alphanumeric hash).
        query:        GraphQL query string.  Must use $skip and $first.
        variables:    Additional variables for the query (merged per page).
        data_key:     Top-level key in ``data`` to extract as the result list.
                      If ``None``, the entire ``data`` dict is returned in a
                      one-element list (useful for non-paginated queries).
        first:        Page size (max 1 000 for most subgraphs).
        max_results:  Hard cap on total rows returned.

    Returns:
        List of result dicts from the subgraph.  Empty list on any error or
        when ``THE_GRAPH_API_KEY`` is absent.
    """
    api_key = os.environ.get("THE_GRAPH_API_KEY", "")
    if not api_key:
        logger.debug(
            "THE_GRAPH_API_KEY not set; skipping The Graph query for subgraph %s",
            subgraph_id[:12],
        )
        return []

    url = f"{_GRAPH_BASE}/{api_key}/subgraphs/id/{subgraph_id}"
    all_results: list[dict[str, Any]] = []
    skip = 0

    while True:
        page_vars: dict[str, Any] = {**(variables or {}), "skip": skip, "first": first}
        payload = {"query": query, "variables": page_vars}

        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("The Graph request failed (subgraph=%s): %s", subgraph_id[:12], exc)
            break

        body = resp.json()
        if "errors" in body:
            logger.warning("The Graph returned errors: %s", body["errors"][:2])
            break

        data = body.get("data", {})
        if data_key:
            page: list[dict[str, Any]] = data.get(data_key, [])
            if not isinstance(page, list):
                logger.warning(
                    "Expected list for key %r, got %s", data_key, type(page).__name__
                )
                break
        else:
            # Non-paginated single-result query
            all_results.append(data)
            break

        all_results.extend(page)
        if len(all_results) >= max_results:
            logger.debug(
                "The Graph query capped at %d results (subgraph=%s)",
                max_results, subgraph_id[:12],
            )
            all_results = all_results[:max_results]
            break
        if len(page) < first:
            # Partial page — we have all results
            break

        skip += first
        time.sleep(_RATE_SLEEP)

    logger.debug(
        "The Graph: fetched %d total rows from subgraph %s", len(all_results), subgraph_id[:12]
    )
    return all_results
