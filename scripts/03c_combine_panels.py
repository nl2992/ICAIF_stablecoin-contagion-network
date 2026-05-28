"""Combine per-event gold panels into a single all-events parquet.

Reads: data/gold/dataset_contagion_features_{event}.parquet  (for each event)
Writes:
    data/gold/dataset_contagion_features_all.parquet
"""

import argparse

import polars as pl

from stressnet.config import gold_root, load_events
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_EVENTS = [
    "usdc_svb_2023",
    "terra_luna_2022",
    "usdt_curve_2023",
    "ftx_2022",
    "busd_2023",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine per-event gold panels.")
    parser.add_argument(
        "--events",
        nargs="+",
        default=_DEFAULT_EVENTS,
        help="Event IDs to combine (default: all 5 events).",
    )
    parser.add_argument(
        "--out", default="dataset_contagion_features_all.parquet",
        help="Output filename in data/gold/.",
    )
    args = parser.parse_args()

    gold = gold_root()
    frames = []
    for event in args.events:
        path = gold / f"dataset_contagion_features_{event}.parquet"
        if not path.exists():
            logger.warning("Panel not found for %s — skipping: %s", event, path)
            continue
        df = pl.read_parquet(path)
        logger.info("Loaded %s: %d rows × %d cols", event, df.height, df.width)
        frames.append(df)

    if not frames:
        raise SystemExit("No panels found. Run script 03 for each event first.")

    combined = pl.concat(frames, how="diagonal")
    out_path = gold / args.out
    combined.write_parquet(out_path)
    logger.info("Combined panel: %d rows × %d cols → %s", combined.height, combined.width, out_path)

    # Quick provenance summary
    summary = (
        combined
        .group_by(["event_id", "tier_actual"])
        .agg(pl.len().alias("n_rows"))
        .sort(["event_id", "tier_actual"])
    )
    print("\n=== Combined panel provenance ===")
    print(summary)


if __name__ == "__main__":
    main()
