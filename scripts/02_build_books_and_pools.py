"""Reconstruct order books and DEX pool states from raw bronze data.

Reads data/raw/ and writes reconstructed Silver artefacts to data/silver/.
"""

import argparse
from pathlib import Path

from stressnet.config import load_events, silver_root
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct books and pools from raw data.")
    parser.add_argument("--event", required=True, help="Event ID from configs/events.yaml")
    parser.add_argument("--nodes", nargs="+", default=None)
    args = parser.parse_args()

    events = load_events()
    if args.event not in events:
        raise SystemExit(f"Unknown event '{args.event}'")

    out_dir = silver_root() / args.event
    out_dir.mkdir(parents=True, exist_ok=True)

    # TODO: implement for each market node:
    #   1. Load raw Bronze Binance/Coinbase/Kraken depth messages
    #   2. Feed through OrderBook.apply_snapshot + apply_update
    #   3. Sample book state at regular intervals (1s, 5s, 60s)
    #   4. Save to data/silver/{event}/{node_id}_books.parquet

    # TODO: implement for each pool node:
    #   1. Load raw on-chain Curve/Uniswap events from Bronze
    #   2. Reconstruct pool state sequence (reserves, price, imbalance)
    #   3. Save to data/silver/{event}/{node_id}_pool_states.parquet

    logger.warning("Book and pool reconstruction not yet implemented for event %s", args.event)
    logger.info("See stressnet.reconstruct.orderbook and stressnet.reconstruct.dex_pool")


if __name__ == "__main__":
    main()
