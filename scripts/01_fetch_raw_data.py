"""Fetch raw data for an event from all configured sources.

Writes raw files to data/raw/ and manifests to data/manifests/.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from stressnet.config import load_events, manifests_root
from stressnet.utils.logging import get_logger
from stressnet.utils.time import parse_iso_utc

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch raw data for one event.")
    parser.add_argument("--event", required=True, help="Event ID from configs/events.yaml")
    parser.add_argument("--nodes", nargs="+", default=None, help="Subset of node IDs to fetch")
    parser.add_argument("--dry-run", action="store_true", help="Print planned fetches without executing")
    args = parser.parse_args()

    events = load_events()
    if args.event not in events:
        raise SystemExit(f"Unknown event '{args.event}'. Available: {list(events.keys())}")

    cfg = events[args.event]
    start_str = cfg["analysis_window_utc"][0] + "T00:00:00Z"
    end_str = cfg["analysis_window_utc"][1] + "T23:59:59Z"
    start_utc = parse_iso_utc(start_str)
    end_utc = parse_iso_utc(end_str)

    logger.info("Event: %s | Window: %s → %s", args.event, start_str, end_str)

    if args.dry_run:
        logger.info("[DRY RUN] Would fetch data for %s (%s → %s)", args.event, start_utc, end_utc)
        return

    # TODO: implement per-source fetch calls using:
    #   stressnet.data.binance.download_vision_zip(...)
    #   stressnet.data.coinbase.fetch_candles(...)
    #   stressnet.data.kraken.ws_subscribe_book(...)
    #   stressnet.data.etherscan.get_token_transfers(...)
    #   stressnet.data.curve.fetch_3pool_events(...)
    #   stressnet.data.uniswap.fetch_pool_swaps(...)
    #   stressnet.data.coinmetrics.get_asset_metrics(...)
    logger.warning("fetch implementation pending — add source-specific fetch calls here.")


if __name__ == "__main__":
    main()
