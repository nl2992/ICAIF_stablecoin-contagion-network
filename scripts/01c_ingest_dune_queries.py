"""Ingest bridge-flow and Tron USDT data from Dune Analytics.

Writes Tier-B bronze parquet files for:
  - eth_bridge_flows    (cross-chain USDC/USDT bridge transfers per hour)
  - tron_usdt_exchange_flows (Tron USDT exchange inflows/outflows per hour)

These are Tier B: Dune query results are pre-aggregated by Dune's
indexer and depend on address labelling quality.  They cannot reach
Tier A regardless of the underlying on-chain data.

Usage:
    python scripts/01c_ingest_dune_queries.py --event usdt_curve_2023
    python scripts/01c_ingest_dune_queries.py --event ftx_2022

Requires:
    DUNE_API_KEY environment variable (set in .env)

Dune query IDs:
    To set up the queries, create them on https://dune.com and paste
    the numeric IDs into _DUNE_QUERIES below.  The queries should
    accept ``start_time`` and ``end_time`` as timestamp parameters.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import polars as pl

from stressnet.config import bronze_root, load_events
from stressnet.data.dune import run_query
from stressnet.utils.logging import get_logger
from stressnet.utils.time import parse_iso_utc

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Dune query registry
# ---------------------------------------------------------------------------
# Fill in the numeric Dune query IDs after creating them on dune.com.
# Each entry: (query_id, output_node_id, description)
# Parameters passed: start_time (ISO string), end_time (ISO string)

_DUNE_QUERIES: list[tuple[int | None, str, str]] = [
    (
        None,  # TODO: replace with actual Dune query ID
        "eth_bridge_flows",
        "Ethereum cross-chain USDC/USDT bridge inflows and outflows (hourly)",
    ),
    (
        None,  # TODO: replace with actual Dune query ID
        "tron_usdt_exchange_flows",
        "Tron USDT exchange inflows and outflows (hourly)",
    ),
]

# Nodes produced by Dune queries, keyed by node_id → events covered
_NODE_EVENTS: dict[str, list[str]] = {
    "eth_bridge_flows":         ["ftx_2022", "usdc_svb_2023"],
    "tron_usdt_exchange_flows": ["usdt_curve_2023"],
}


def _event_bounds(event_id: str) -> tuple[str, str]:
    cfg = load_events()[event_id]
    start_str = f"{cfg['analysis_window_utc'][0]}T00:00:00Z"
    end_str   = f"{cfg['analysis_window_utc'][1]}T23:59:59Z"
    return start_str, end_str


def _rows_to_hourly_df(rows: list[dict], event_id: str) -> pl.DataFrame | None:
    """Convert Dune result rows to a standardised hourly flow DataFrame.

    Expected Dune columns (case-insensitive):
        hour / block_hour / period   — timestamp bucket
        inflow / net_inflow          — USD inflow
        outflow / net_outflow        — USD outflow
        netflow / net_flow           — USD net flow (inflow - outflow)
    """
    if not rows:
        return None

    df = pl.DataFrame(rows)
    # Normalise column names to lowercase
    df = df.rename({c: c.lower() for c in df.columns})

    # Detect timestamp column
    ts_col = next(
        (c for c in df.columns if c in ("hour", "block_hour", "period", "timestamp")),
        None,
    )
    if ts_col is None:
        logger.warning("No recognised timestamp column in Dune result; got: %s", df.columns)
        return None

    # Detect flow columns
    inflow_col  = next((c for c in df.columns if "inflow"  in c), None)
    outflow_col = next((c for c in df.columns if "outflow" in c), None)
    netflow_col = next((c for c in df.columns if "netflow" in c or "net_flow" in c), None)

    df = df.with_columns(
        pl.col(ts_col).cast(pl.Utf8).str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S", strict=False)
        .dt.replace_time_zone("UTC").alias("wall_clock_utc")
    )

    keep = ["wall_clock_utc"]
    renames: dict[str, str] = {}

    if inflow_col:
        keep.append(inflow_col)
        renames[inflow_col] = "exchange_inflow_1h"
    if outflow_col:
        keep.append(outflow_col)
        renames[outflow_col] = "exchange_outflow_1h"
    if netflow_col:
        keep.append(netflow_col)
        renames[netflow_col] = "exchange_netflow_1h"

    df = df.select(keep).rename(renames)

    # Derive netflow if missing
    if "exchange_netflow_1h" not in df.columns and \
       "exchange_inflow_1h" in df.columns and "exchange_outflow_1h" in df.columns:
        df = df.with_columns(
            (pl.col("exchange_inflow_1h") - pl.col("exchange_outflow_1h"))
            .alias("exchange_netflow_1h")
        )

    return df.sort("wall_clock_utc")


def ingest_dune_for_event(event_id: str, out_root: Path) -> dict[str, str]:
    """Run all applicable Dune queries for one event window.

    Returns a dict mapping node_id → tier_actual ('B' or 'fixture_non_empirical').
    """
    if not os.environ.get("DUNE_API_KEY"):
        logger.info("DUNE_API_KEY not set; all Dune nodes will remain fixture.")
        return {}

    start_str, end_str = _event_bounds(event_id)
    results: dict[str, str] = {}

    for query_id, node_id, description in _DUNE_QUERIES:
        # Skip if this node is not relevant to the requested event
        if event_id not in _NODE_EVENTS.get(node_id, []):
            logger.debug("Skipping %s for event %s (not in events list)", node_id, event_id)
            continue

        if query_id is None:
            logger.warning(
                "Dune query ID not set for %s — add it to _DUNE_QUERIES in %s",
                node_id, __file__,
            )
            results[node_id] = "fixture_non_empirical"
            continue

        logger.info("Executing Dune query %d for %s (%s)", query_id, node_id, description)
        try:
            rows = run_query(query_id, params={
                "start_time": start_str,
                "end_time":   end_str,
            })
        except Exception as exc:
            logger.warning("Dune query %d failed for %s: %s", query_id, node_id, exc)
            results[node_id] = "fixture_non_empirical"
            continue

        df = _rows_to_hourly_df(rows, event_id)
        if df is None or df.height == 0:
            logger.warning("No rows returned by Dune for %s", node_id)
            results[node_id] = "fixture_non_empirical"
            continue

        out_dir = out_root / event_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{node_id}_flows.parquet"
        df.write_parquet(out_path)
        logger.info(
            "Wrote %d hourly rows for %s → %s (Tier B)", df.height, node_id, out_path.name
        )
        results[node_id] = "B"

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Dune bridge / Tron USDT flows for one event."
    )
    parser.add_argument("--event", required=True, help="Event ID from configs/events.yaml")
    args = parser.parse_args()

    events = load_events()
    if args.event not in events:
        raise SystemExit(f"Unknown event '{args.event}'. "
                         f"Valid options: {list(events.keys())}")

    out_root = bronze_root()
    tier_map = ingest_dune_for_event(args.event, out_root)

    if tier_map:
        logger.info("Dune ingest complete for %s:", args.event)
        for node_id, tier in tier_map.items():
            logger.info("  %-35s → %s", node_id, tier)
    else:
        logger.info(
            "No Dune data ingested for %s "
            "(check DUNE_API_KEY and _DUNE_QUERIES in this script).",
            args.event,
        )


if __name__ == "__main__":
    main()
