"""Attach provenance tiers to result edges and gate paper claims."""

from __future__ import annotations

import argparse

import polars as pl

from stressnet.config import load_events, results_root
from stressnet.evaluation.claim_gate import annotate_table, event_tables, load_tier_map, paper_tables
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate result claims by node provenance tiers.")
    parser.add_argument("--event", default=None, help="Annotate one event's result tables.")
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Check all paper-relevant result tables instead of one event.",
    )
    parser.add_argument(
        "--require-real",
        action="store_true",
        help="Exit nonzero if any processed edge table uses non-claimable endpoint tiers.",
    )
    args = parser.parse_args()

    if not args.event and not args.paper:
        raise SystemExit("Provide --event or --paper.")

    tables_dir = results_root() / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    events = list(load_events().keys()) if args.paper else [args.event]
    tier_map = load_tier_map([event for event in events if event], tables_dir)

    if args.paper:
        tables = paper_tables(tables_dir)
        default_event = None
        summary_name = "table_claim_gate_paper.csv"
    else:
        tables = event_tables(args.event, tables_dir)
        default_event = args.event
        summary_name = f"table_claim_gate_{args.event}.csv"

    if not tables:
        message = "No result tables found to gate."
        if args.paper or args.require_real:
            raise SystemExit(message)
        logger.info(message)
        return

    summaries = []
    blocked_total = 0
    for path in tables:
        annotated, summary = annotate_table(path, tier_map, default_event)
        summaries.append(summary)
        if annotated is not None:
            annotated.write_csv(path)
            blocked_total += int(summary["blocked_rows"])
            logger.info(
                "Annotated %s: %d blocked rows / %d rows",
                path.name,
                summary["blocked_rows"],
                summary["rows"],
            )

    summary_df = pl.DataFrame(summaries)
    summary_path = tables_dir / summary_name
    summary_df.write_csv(summary_path)
    logger.info("Wrote %s", summary_path)

    if args.require_real and blocked_total > 0:
        raise SystemExit(f"Claim gate failed: {blocked_total} edge rows are not paper-claimable.")

    print(summary_df)


if __name__ == "__main__":
    main()
