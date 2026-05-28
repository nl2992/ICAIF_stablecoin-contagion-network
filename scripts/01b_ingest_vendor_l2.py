"""Ingest Tier-A CEX L2 data from vendor (Tardis / Kaiko).

Downloads incremental_book_L2 or book_snapshot_25 for one node/symbol/date range
and writes normalised bronze parquet files.

Writes:
    data/bronze/vendor_l2/{exchange}_{symbol}_{data_type}_{YYYY-MM-DD}.parquet

Usage example::

    python scripts/01b_ingest_vendor_l2.py \\
        --exchange binance \\
        --symbol USDCUSDT \\
        --data-type incremental_book_L2 \\
        --start 2023-03-08 --end 2023-03-14 \\
        --node-id usdc_binance_l2

    python scripts/01b_ingest_vendor_l2.py \\
        --from-config configs/vendor_l2.yaml \\
        --node-id usdc_binance_l2 \\
        --event usdc_svb_2023
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

from stressnet.config import bronze_root, load_events
from stressnet.data.tardis_l2 import (
    check_symbol_availability,
    download_l2_range,
    load_bronze,
)
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
_VENDOR_CONFIG = _ROOT / "configs" / "vendor_l2.yaml"


def _load_vendor_config() -> dict:
    with open(_VENDOR_CONFIG) as fh:
        return yaml.safe_load(fh)


def _resolve_vendor(cfg: dict) -> str:
    """Return effective vendor: env DATA_VENDOR → config default → 'tardis'."""
    return os.environ.get("DATA_VENDOR", cfg.get("default_vendor", "tardis")).lower()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Tier-A CEX L2 data from Tardis or Kaiko."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exchange", help="Exchange name (e.g. binance).")
    group.add_argument("--from-config", action="store_true",
                       help="Resolve exchange/symbol from configs/vendor_l2.yaml using --node-id.")

    parser.add_argument("--symbol", help="Market symbol (e.g. USDCUSDT).")
    parser.add_argument("--data-type", default="incremental_book_L2",
                        choices=["incremental_book_L2", "book_snapshot_25", "trades", "book_ticker"],
                        help="Tardis data channel (default: incremental_book_L2).")
    parser.add_argument("--start", help="Start date YYYY-MM-DD.")
    parser.add_argument("--end",   help="End date YYYY-MM-DD (inclusive).")
    parser.add_argument("--event", help="Event name to auto-derive start/end dates.")
    parser.add_argument("--node-id", help="Node ID (used with --from-config or for labelling).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-download even if bronze file already exists.")
    parser.add_argument("--check-availability", action="store_true",
                        help="Only check Tardis archive availability; do not download.")
    parser.add_argument("--out-dir", help="Override output directory.")
    args = parser.parse_args()

    vcfg = _load_vendor_config()
    vendor = _resolve_vendor(vcfg)
    if vendor != "tardis":
        logger.warning(
            "DATA_VENDOR=%s — only Tardis is fully implemented. "
            "Kaiko support is a stub.",
            vendor,
        )

    # Resolve exchange/symbol/data_type from config if requested
    exchange = args.exchange
    symbol = args.symbol
    data_type = args.data_type

    if args.from_config:
        if not args.node_id:
            parser.error("--from-config requires --node-id")
        nmap = vcfg.get("node_symbol_map", {})
        if args.node_id not in nmap:
            parser.error(
                f"Node '{args.node_id}' not found in configs/vendor_l2.yaml "
                f"node_symbol_map. Available: {list(nmap)}"
            )
        node_cfg = nmap[args.node_id]
        exchange = node_cfg["exchange"]
        symbol = node_cfg["symbol"]
        data_type = node_cfg.get("data_type", data_type)
        if note := node_cfg.get("notes"):
            logger.info("Node note: %s", note)

    if not exchange or not symbol:
        parser.error("Specify --exchange and --symbol (or --from-config with --node-id).")

    # Resolve start/end
    start = args.start
    end   = args.end
    if args.event and (not start or not end):
        events = load_events()
        if args.event not in events:
            raise SystemExit(f"Unknown event '{args.event}'. Known: {list(events)}")
        ev = events[args.event]
        aw = ev.get("analysis_window_utc", [])
        if len(aw) == 2:
            start = start or aw[0]
            end   = end   or aw[1]
        else:
            raise SystemExit(
                f"Event '{args.event}' has no analysis_window_utc in events.yaml. "
                "Provide --start and --end explicitly."
            )

    if not start or not end:
        parser.error("Provide --start and --end, or --event with analysis_window_utc set.")

    # Availability check
    if args.check_availability:
        result = check_symbol_availability(exchange, symbol, start, end)
        print(result)
        sys.exit(0 if result.get("available") else 1)

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = bronze_root() / "vendor_l2"

    logger.info(
        "Ingesting %s/%s [%s] %s → %s → %s",
        exchange, symbol, data_type, start, end, out_dir,
    )

    paths = download_l2_range(
        exchange=exchange,
        symbol=symbol,
        data_type=data_type,
        start=start,
        end=end,
        out_dir=out_dir,
        overwrite=args.overwrite,
    )

    if not paths:
        raise SystemExit(
            f"No files downloaded for {exchange}/{symbol}/{data_type} "
            f"{start}→{end}.  Check TARDIS_API_KEY and subscription."
        )

    # Summary
    total_rows = 0
    total_bytes = 0
    for p in paths:
        try:
            df = load_bronze(exchange, symbol, data_type,
                             out_dir, start=p.stem[-10:], end=p.stem[-10:])
            total_rows += len(df)
        except Exception:
            pass
        total_bytes += p.stat().st_size

    logger.info(
        "Ingestion complete: %d files, ~%d rows, %.1f MB",
        len(paths),
        total_rows,
        total_bytes / 1e6,
    )
    print(f"\nDownloaded {len(paths)} file(s) to {out_dir}")
    for p in paths:
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
