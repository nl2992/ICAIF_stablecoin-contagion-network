"""Reconstruct Tier-A silver features from bronze L2 vendor data.

Reads: data/bronze/vendor_l2/{exchange}_{symbol}_{data_type}_{YYYY-MM-DD}.parquet
Writes:
    data/silver/l2_books/{node_id}_{event}.parquet   — silver features
    data/silver/l2_books/manifest_{node_id}_{event}.json — BookManifest diagnostics

Usage::

    python scripts/02b_reconstruct_l2_books.py --event usdc_svb_2023
    python scripts/02b_reconstruct_l2_books.py --event usdc_svb_2023 --node-id usdc_binance_l2
    python scripts/02b_reconstruct_l2_books.py --all-events
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
import yaml

from stressnet.config import bronze_root, load_events, silver_root
from stressnet.data.tardis_l2 import load_bronze
from stressnet.reconstruct.orderbook_l2 import L2BookReconstructor
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
_VENDOR_CONFIG = _ROOT / "configs" / "vendor_l2.yaml"
_REQUIRED_NODES_CONFIG = _ROOT / "configs" / "required_nodes.yaml"


def _load_vendor_config() -> dict:
    with open(_VENDOR_CONFIG) as fh:
        return yaml.safe_load(fh)


def _event_window_us(event_name: str) -> tuple[float, float]:
    """Return (start_ts_us, end_ts_us) for an event's analysis window."""
    from datetime import datetime, timezone
    events = load_events()
    ev = events[event_name]
    aw = ev.get("analysis_window_utc", [])
    if len(aw) != 2:
        return 0.0, 0.0
    fmt = "%Y-%m-%d"
    t0 = datetime.strptime(aw[0], fmt).replace(tzinfo=timezone.utc)
    t1 = datetime.strptime(aw[1], fmt).replace(tzinfo=timezone.utc)
    return t0.timestamp() * 1_000_000, t1.timestamp() * 1_000_000


def reconstruct_node(
    node_id: str,
    event_name: str,
    vcfg: dict,
    bronze_dir: Path,
    silver_dir: Path,
    *,
    grid_seconds: int = 60,
    overwrite: bool = False,
) -> bool:
    """Reconstruct silver features for one node/event pair.

    Returns True if successful, False if no bronze data found.
    """
    out_silver = silver_dir / f"{node_id}_{event_name}.parquet"
    out_manifest = silver_dir / f"manifest_{node_id}_{event_name}.json"

    if out_silver.exists() and out_manifest.exists() and not overwrite:
        logger.debug("Skipping (already exists): %s", out_silver.name)
        return True

    # Look up node config
    nmap = vcfg.get("node_symbol_map", {})
    if node_id not in nmap:
        logger.warning("Node '%s' not in vendor_l2.yaml node_symbol_map — skipping.", node_id)
        return False

    ncfg = nmap[node_id]
    exchange  = ncfg["exchange"]
    symbol    = ncfg["symbol"]
    data_type = ncfg.get("data_type", "incremental_book_L2")

    # Load bronze data for the event's analysis window
    events = load_events()
    if event_name not in events:
        logger.error("Unknown event: %s", event_name)
        return False

    ev = events[event_name]
    aw = ev.get("analysis_window_utc", [])
    if len(aw) != 2:
        logger.error("Event '%s' has no analysis_window_utc.", event_name)
        return False

    start_date, end_date = aw[0], aw[1]
    start_ts_us, end_ts_us = _event_window_us(event_name)

    logger.info(
        "Reconstructing %s for %s (%s → %s)...",
        node_id, event_name, start_date, end_date,
    )

    bronze_df = load_bronze(
        exchange, symbol, data_type,
        bronze_dir,
        start=start_date,
        end=end_date,
    )

    if bronze_df.is_empty():
        logger.warning(
            "No bronze data for %s/%s/%s in %s→%s. "
            "Run scripts/01b_ingest_vendor_l2.py first.",
            exchange, symbol, data_type, start_date, end_date,
        )
        return False

    logger.info("  Loaded %d bronze rows", len(bronze_df))

    # Reconstruct
    recon = L2BookReconstructor(notional_usd=10_000.0, depth_bps=10.0)
    recon.apply_from_df(bronze_df)

    # Silver features
    silver = recon.to_silver_df(
        start_ts_us=start_ts_us,
        end_ts_us=end_ts_us,
        grid_seconds=grid_seconds,
    )

    # Manifest
    manifest = recon.manifest(
        exchange=exchange,
        symbol=symbol,
        start_ts_us=start_ts_us,
        end_ts_us=end_ts_us,
    )

    logger.info(
        "  Silver: %d rows  coverage=%.1f%%  tier=%s  gap_rate=%.4f  resyncs=%d",
        len(silver),
        manifest.coverage_pct * 100,
        manifest.tier_actual,
        manifest.gap_rate,
        manifest.resync_count,
    )

    if manifest.tier_actual == "B":
        logger.warning(
            "  Tier downgraded to B: %s", manifest.tier_downgrade_reason
        )

    silver_dir.mkdir(parents=True, exist_ok=True)

    if not silver.is_empty():
        # Add node/event metadata
        silver = silver.with_columns(
            pl.lit(node_id).alias("node_id"),
            pl.lit(event_name).alias("event_id"),
            pl.lit(exchange).alias("exchange"),
            pl.lit(symbol).alias("symbol"),
            pl.lit(manifest.tier_actual).alias("tier_actual"),
        )
        silver.write_parquet(out_silver, compression="zstd")
        logger.info("  Wrote silver: %s (%d rows)", out_silver.name, len(silver))
    else:
        logger.warning("  No silver rows produced for %s/%s", node_id, event_name)

    # Write manifest JSON
    manifest_dict = manifest.to_dict()
    manifest_dict["node_id"] = node_id
    manifest_dict["event_id"] = event_name
    manifest_dict["grid_seconds"] = grid_seconds
    with open(out_manifest, "w") as fh:
        json.dump(manifest_dict, fh, indent=2)
    logger.info("  Wrote manifest: %s", out_manifest.name)

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct Tier-A silver features from bronze L2 data."
    )
    parser.add_argument("--event", help="Event name (or use --all-events).")
    parser.add_argument("--all-events", action="store_true")
    parser.add_argument("--node-id", help="Reconstruct a single node only.")
    parser.add_argument("--grid-seconds", type=int, default=60,
                        help="Resampling grid in seconds (default 60).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-reconstruct even if output files exist.")
    parser.add_argument("--bronze-dir", help="Override bronze directory.")
    parser.add_argument("--silver-dir", help="Override silver output directory.")
    args = parser.parse_args()

    if not args.event and not args.all_events:
        parser.error("Specify --event <name> or --all-events.")

    vcfg = _load_vendor_config()
    events_cfg = load_events()

    event_names: list[str] = []
    if args.all_events:
        event_names = list(events_cfg.keys())
    else:
        event_names = [args.event]

    bronze_dir = Path(args.bronze_dir) if args.bronze_dir else bronze_root() / "vendor_l2"
    silver_dir = Path(args.silver_dir) if args.silver_dir else silver_root() / "l2_books"

    # Determine nodes to process
    nmap = vcfg.get("node_symbol_map", {})
    node_ids = [args.node_id] if args.node_id else list(nmap.keys())

    n_success = 0
    n_skip    = 0
    n_fail    = 0

    for event_name in event_names:
        if event_name not in events_cfg:
            logger.warning("Unknown event '%s' — skipping.", event_name)
            continue

        for node_id in node_ids:
            # Check if this node covers this event
            ncfg = nmap.get(node_id, {})
            node_events = ncfg.get("events", None)
            # If events key missing from config, attempt anyway; let bronze check handle it
            ok = reconstruct_node(
                node_id=node_id,
                event_name=event_name,
                vcfg=vcfg,
                bronze_dir=bronze_dir,
                silver_dir=silver_dir,
                grid_seconds=args.grid_seconds,
                overwrite=args.overwrite,
            )
            if ok:
                n_success += 1
            else:
                n_skip += 1

    logger.info(
        "Reconstruction complete: %d success, %d skipped/failed",
        n_success,
        n_skip,
    )

    # Print manifest summary
    manifests = sorted(silver_dir.glob("manifest_*.json"))
    if manifests:
        print(f"\n=== BookManifest summary ({len(manifests)} nodes) ===")
        print(f"{'Node':35s} {'Event':20s} {'Tier':5s} {'Cov%':6s} {'GapRate':8s} {'Resyncs':7s}")
        print("-" * 90)
        for mf in manifests:
            with open(mf) as fh:
                m = json.load(fh)
            print(
                f"{m.get('node_id','?'):35s} {m.get('event_id','?'):20s} "
                f"{m.get('tier_actual','?'):5s} {m.get('coverage_pct',0)*100:5.1f}% "
                f"{m.get('gap_rate',0):8.4f} {m.get('resync_count',0):7d}"
            )


if __name__ == "__main__":
    main()
