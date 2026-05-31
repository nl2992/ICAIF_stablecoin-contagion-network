"""Ingest Uniswap v3 pool Swap events via Etherscan getLogs.

Upgrades uniswap_usdc_usdt_005 from fixture_non_empirical → Tier A by
fetching real on-chain Swap events, matching the Curve pool feature schema
(usdc_net_sold_1h) exactly so the two protocols can be compared directly.

Pool coverage
-------------
  uniswap_usdc_usdt_005  — USDC/USDT 0.05% (0x3416cF6C…)
    events: usdc_svb_2023, usdt_curve_2023

Usage
-----
    python scripts/19_ingest_uniswap_v3.py --event usdt_curve_2023
    python scripts/19_ingest_uniswap_v3.py --all-events
    python scripts/19_ingest_uniswap_v3.py --dry-run
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import polars as pl

from stressnet.config import bronze_root, load_events, results_root
from stressnet.data.etherscan import get_block_number_by_timestamp
from stressnet.data.uniswap_etherscan import (
    ingest_uniswap_pool_events,
)
from stressnet.utils.logging import get_logger
from stressnet.utils.time import parse_iso_utc

logger = get_logger(__name__)

# Uniswap pool nodes to ingest, keyed by event_id
_UNISWAP_NODES: dict[str, list[tuple[str, str]]] = {
    "usdt_curve_2023": [
        ("uniswap_usdc_usdt_005", "0x3416cF6C708Da44DB2624D63ea0AAef7113527C6"),
    ],
    "usdc_svb_2023": [
        ("uniswap_usdc_usdt_005", "0x3416cF6C708Da44DB2624D63ea0AAef7113527C6"),
    ],
    "terra_luna_2022": [
        ("uniswap_usdc_usdt_005", "0x3416cF6C708Da44DB2624D63ea0AAef7113527C6"),
    ],
    "ftx_2022": [
        ("uniswap_usdc_usdt_005", "0x3416cF6C708Da44DB2624D63ea0AAef7113527C6"),
    ],
    "busd_2023": [
        ("uniswap_usdc_usdt_005", "0x3416cF6C708Da44DB2624D63ea0AAef7113527C6"),
    ],
}


def _block_range(event_id: str) -> tuple[int, int]:
    events_cfg = load_events()
    cfg = events_cfg[event_id]
    start_str = f"{cfg['analysis_window_utc'][0]}T00:00:00Z"
    end_str   = f"{cfg['analysis_window_utc'][1]}T23:59:59Z"
    start_ts  = int(parse_iso_utc(start_str).timestamp())
    end_ts    = int(parse_iso_utc(end_str).timestamp())
    logger.info("Resolving blocks for %s…", event_id)
    start_block = get_block_number_by_timestamp(start_ts, "before")
    end_block   = get_block_number_by_timestamp(end_ts, "after")
    if start_block is None or end_block is None:
        raise RuntimeError(f"Could not resolve blocks for {event_id}")
    logger.info("Block range: %d → %d", start_block, end_block)
    return start_block, end_block


def ingest_event(event_id: str, grid_seconds: int = 3600, dry_run: bool = False) -> list[Path]:
    """Ingest all Uniswap nodes for one event. Returns written paths."""
    if not os.environ.get("ETHERSCAN_API_KEY"):
        logger.error("ETHERSCAN_API_KEY not set — cannot fetch Uniswap data.")
        return []

    nodes = _UNISWAP_NODES.get(event_id, [])
    if not nodes:
        logger.warning("No Uniswap nodes configured for %s", event_id)
        return []

    if dry_run:
        logger.info("[DRY RUN] Would ingest %d Uniswap nodes for %s", len(nodes), event_id)
        return []

    start_block, end_block = _block_range(event_id)
    out_dir = bronze_root() / event_id
    written: list[Path] = []

    for node_id, pool_address in nodes:
        logger.info("Ingesting %s (%s) for %s…", node_id, pool_address[:10], event_id)
        path, tier = ingest_uniswap_pool_events(
            pool_address=pool_address,
            start_block=start_block,
            end_block=end_block,
            out_dir=out_dir,
            event_id=event_id,
            node_id=node_id,
            grid_seconds=grid_seconds,
            save_raw=True,
        )
        if path:
            written.append(path)
            df = pl.read_parquet(path)
            logger.info(
                "  ✓ %s: %d rows, %s–%s, total_net_sold=%.1f M USDC",
                node_id, df.height,
                df["wall_clock_utc"].min(),
                df["wall_clock_utc"].max(),
                df["usdc_net_sold_1h"].sum() / 1e6,
            )
        else:
            logger.warning("  ✗ %s: ingest returned no data (API error or no events)", node_id)

    return written


def update_manifest(event_id: str, node_id: str, path: Path) -> None:
    """Update the provenance manifest to reflect tier upgrade."""
    manifest_dir = Path("data/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{event_id}_uniswap.csv"
    row = pl.DataFrame([{
        "event_id": event_id,
        "node_id": node_id,
        "tier_actual": "A",
        "source": "etherscan_getlogs",
        "path": str(path),
        "note": "Uniswap v3 Swap events via Etherscan; upgraded from fixture_non_empirical",
    }])
    if manifest_path.exists():
        existing = pl.read_csv(manifest_path)
        row = pl.concat([existing.filter(pl.col("node_id") != node_id), row])
    row.write_csv(manifest_path)
    logger.info("Updated manifest: %s", manifest_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Uniswap v3 pool events.")
    parser.add_argument("--event", default=None)
    parser.add_argument("--all-events", action="store_true")
    parser.add_argument("--grid-seconds", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.event and not args.all_events:
        raise SystemExit("Specify --event <id> or --all-events")

    events = list(_UNISWAP_NODES.keys()) if args.all_events else [args.event]

    for event_id in events:
        logger.info("=== Uniswap ingest: %s ===", event_id)
        written = ingest_event(event_id, args.grid_seconds, args.dry_run)
        for p in written:
            node_id = p.stem.replace("_pool_events", "")
            update_manifest(event_id, node_id, p)

    logger.info("Done.")


if __name__ == "__main__":
    main()
