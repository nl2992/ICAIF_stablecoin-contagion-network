"""Attach provenance tiers to result edges and gate paper claims.

Two output paths:
  results/tables/        — annotated tables (all edges, with claim_allowed column)
  results/paper/tables/  — filtered tables (claim_allowed == True only)

Usage:
  python scripts/00c_claim_gate.py --event usdc_svb_2023
  python scripts/00c_claim_gate.py --all-events
  python scripts/00c_claim_gate.py --all-events --strict   # nonzero exit if any fixture
"""

from __future__ import annotations

import argparse

import polars as pl

from stressnet.config import load_events, results_root
from stressnet.evaluation.claim_gate import (
    FIXTURE,
    annotate_table,
    event_tables,
    load_tier_map,
    paper_tables,
)
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def _write_paper_table(annotated: pl.DataFrame, source_path, paper_dir) -> None:
    """Write the claim_allowed subset to results/paper/tables/."""
    paper_dir.mkdir(parents=True, exist_ok=True)
    paper_path = paper_dir / source_path.name
    claimable = annotated.filter(pl.col("claim_allowed"))
    claimable.write_csv(paper_path)
    return paper_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate result claims by node provenance tiers.")
    parser.add_argument("--event", default=None, help="Annotate one event's result tables.")
    parser.add_argument(
        "--all-events",
        action="store_true",
        help="Annotate all events and write combined paper tables.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if any paper table contains fixture-derived edges.",
    )
    # Legacy aliases kept for compatibility
    parser.add_argument("--paper", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--require-real", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    do_all = args.all_events or args.paper
    if not args.event and not do_all:
        raise SystemExit("Provide --event EVENT or --all-events.")

    tables_dir  = results_root() / "tables"
    paper_dir   = results_root() / "paper" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    all_events = list(load_events().keys())
    events_to_load = all_events if do_all else [args.event]
    tier_map = load_tier_map(events_to_load, tables_dir)

    if do_all:
        tables = paper_tables(tables_dir)
        default_event = None
        summary_name = "table_claim_gate_all_events.csv"
    else:
        tables = event_tables(args.event, tables_dir)
        default_event = args.event
        summary_name = f"table_claim_gate_{args.event}.csv"

    if not tables:
        logger.warning("No result tables found to gate.")
        return

    summaries = []
    blocked_total = 0
    fixture_rows_total = 0

    for path in tables:
        annotated, summary = annotate_table(path, tier_map, default_event)
        summaries.append(summary)

        if annotated is not None:
            # Overwrite source table with annotated version (adds claim columns)
            annotated.write_csv(path)
            # Write claim_allowed-only subset to paper dir
            _write_paper_table(annotated, path, paper_dir)

            blocked_total += int(summary["blocked_rows"])
            fixture_rows_total += int(summary["fixture_or_missing_rows"])
            logger.info(
                "%-50s  claimable=%d / %d rows  (fixture=%d)",
                path.name,
                summary["claimable_rows"],
                summary["rows"],
                summary["fixture_or_missing_rows"],
            )

    summary_df = pl.DataFrame(summaries)
    summary_path = tables_dir / summary_name
    summary_df.write_csv(summary_path)
    # Also write summary to paper dir
    (paper_dir / summary_name).parent.mkdir(parents=True, exist_ok=True)
    summary_df.write_csv(paper_dir / summary_name)
    logger.info("Wrote %s", summary_path)

    # Print real-node-only result summary
    edge_summaries = summary_df.filter(pl.col("status") == "annotated")
    if edge_summaries.height > 0:
        print("\n=== Claim gate summary ===")
        print(edge_summaries.select(["table", "rows", "claimable_rows", "fixture_or_missing_rows"]))

    total_claimable = int(summary_df["claimable_rows"].sum())
    total_rows = int(summary_df.filter(pl.col("status") == "annotated")["rows"].sum())
    logger.info(
        "Total: %d / %d edge rows are paper-claimable (%.0f%%)",
        total_claimable,
        total_rows,
        100 * total_claimable / max(total_rows, 1),
    )

    strict_mode = args.strict or args.require_real
    if strict_mode and fixture_rows_total > 0:
        raise SystemExit(
            f"--strict: {fixture_rows_total} edge rows involve fixture or missing provenance. "
            f"Paper outputs blocked."
        )


if __name__ == "__main__":
    main()
