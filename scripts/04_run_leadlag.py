"""Run pairwise lead-lag analysis with block-bootstrap inference.

Reads: data/gold/dataset_contagion_features_{event}.parquet
Writes:
    results/tables/table_leadlag_tests.csv
    results/figures/figure_heatmap_lags_{event}.png
"""

import argparse
import itertools
from pathlib import Path

import polars as pl

from stressnet.config import gold_root, results_root
from stressnet.graph.nodes import nodes_for_event
from stressnet.models.leadlag import compute_leadlag_table
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_CORE_FEATURES = [
    "basis_vs_usd",
    "spread_bps",
    "depth_10bps_bid_usd",
    "orderbook_imbalance",
    "reserve_imbalance",
    "exchange_netflow_1h",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lead-lag analysis.")
    parser.add_argument("--event", required=True)
    parser.add_argument("--feature-col", default="basis_vs_usd")
    parser.add_argument("--feature-cols", nargs="+", default=None)
    parser.add_argument("--all-core-features", action="store_true")
    parser.add_argument("--grid-seconds", type=int, default=60)
    parser.add_argument("--max-staleness-seconds", type=int, default=None)
    parser.add_argument("--phase", default=None, help="Optional event_phase filter.")
    parser.add_argument("--max-lag", type=int, default=60)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--paper-mode", action="store_true",
                        help="Restrict to real (non-fixture) nodes only and add event_id column.")
    parser.add_argument("--layer-filter", default=None,
                        help="Restrict to nodes of a single layer (e.g. DEX, CEX, mint_burn).")
    parser.add_argument(
        "--split-at", default=None,
        metavar="ISO_DATE",
        help=(
            "Restrict analysis to rows BEFORE this ISO date (YYYY-MM-DD).  "
            "Used for pre-drain sub-window analysis (e.g. Terra/LUNA: --split-at 2022-05-11).  "
            "The sub_window column in the output records the effective window end date."
        ),
    )
    parser.add_argument(
        "--block-shuffle", action="store_true",
        help=(
            "Use block-shuffled permutations (24-hour blocks) instead of "
            "individual observation shuffles.  Controls for serial correlation.  "
            "Adds block_shuffle_p column to output."
        ),
    )
    parser.add_argument(
        "--loeo", default=None,
        metavar="EXCLUDED_EVENT",
        help=(
            "Leave-one-event-out mode.  Pass the event_id to exclude from the "
            "aggregate pattern check.  Appended as loeo_excluded column in output."
        ),
    )
    args = parser.parse_args()

    from stressnet.config import load_events as _load_events
    _events_cfg = _load_events()

    panel_path = gold_root() / f"dataset_contagion_features_{args.event}.parquet"
    if not panel_path.exists():
        raise SystemExit(f"Panel not found: {panel_path}. Run script 03 first.")

    panel = pl.read_parquet(panel_path)
    if args.phase:
        if "event_phase" not in panel.columns:
            raise SystemExit("--phase requested but panel lacks event_phase.")
        panel = panel.filter(pl.col("event_phase") == args.phase)
        if panel.height == 0:
            raise SystemExit(f"No rows found for phase '{args.phase}'.")

    # ── Sub-window filtering (--split-at) ───────────────────────────────────
    sub_window_end: str | None = None
    if args.split_at:
        if "wall_clock_utc" not in panel.columns:
            raise SystemExit("--split-at requires a wall_clock_utc column in the panel.")
        split_dt = pl.lit(args.split_at).str.strptime(pl.Datetime("us"), "%Y-%m-%d") \
                     .dt.replace_time_zone("UTC")
        panel = panel.filter(pl.col("wall_clock_utc") < split_dt)
        if panel.height == 0:
            raise SystemExit(
                f"No rows found before --split-at {args.split_at}. "
                "Check the split date against the event window."
            )
        sub_window_end = args.split_at
        logger.info(
            "--split-at %s: restricted panel to %d rows before drain onset",
            args.split_at, panel.height,
        )
    nodes = nodes_for_event(args.event)

    panel_node_ids = set(panel["node_id"].unique().to_list())

    if args.paper_mode:
        real_node_ids = set(
            panel.filter(pl.col("tier_actual") != "fixture_non_empirical")
            ["node_id"].unique().to_list()
        )
        node_ids = [n.id for n in nodes if n.id in real_node_ids]
        logger.info("--paper-mode: restricting to %d real nodes", len(node_ids))
    else:
        node_ids = [n.id for n in nodes if n.id in panel_node_ids]

    if args.layer_filter:
        layer_node_ids = {n.id for n in nodes if n.layer == args.layer_filter}
        node_ids = [nid for nid in node_ids if nid in layer_node_ids]
        logger.info("--layer-filter %s: restricting to %d nodes", args.layer_filter, len(node_ids))

    if not node_ids:
        raise SystemExit("No matching nodes found in panel.")
    min_nodes = 2 if args.layer_filter else 3
    if args.paper_mode and len(node_ids) < min_nodes:
        raise SystemExit(
            f"--paper-mode requires at least {min_nodes} real nodes for lead-lag "
            f"({'layer-filtered' if args.layer_filter else 'full'}); found {len(node_ids)}."
        )

    node_pairs = list(itertools.permutations(node_ids, 2))
    logger.info("Computing lead-lag for %d node pairs", len(node_pairs))

    if args.all_core_features:
        feature_cols = [col for col in _CORE_FEATURES if col in panel.columns]
    elif args.feature_cols:
        feature_cols = args.feature_cols
    else:
        feature_cols = [args.feature_col]

    frames = []
    for feature_col in feature_cols:
        if feature_col not in panel.columns:
            logger.warning("Skipping missing feature column: %s", feature_col)
            continue
        result = compute_leadlag_table(
            panel,
            node_pairs=node_pairs,
            feature_col=feature_col,
            max_lag=args.max_lag,
            n_reps=args.bootstrap_reps,
            grid_seconds=args.grid_seconds,
            max_staleness_seconds=args.max_staleness_seconds,
        )
        if result.height > 0:
            frames.append(result)

    if not frames:
        raise SystemExit("No lead-lag results produced.")

    results = pl.concat(frames, how="diagonal").with_columns(
        pl.lit(args.event).alias("event_id"),
        pl.lit(args.phase or "all").alias("event_phase"),
        # Sub-window annotation: records the effective window end for pre-drain splits
        pl.lit(sub_window_end).alias("sub_window_end"),
        # Block-shuffle flag: indicates whether serial-correlation-robust inference was used
        pl.lit(args.block_shuffle).alias("block_shuffle_used"),
        # LOEO annotation: which event was excluded (for leave-one-event-out robustness)
        pl.lit(args.loeo).alias("loeo_excluded"),
    )

    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    # File suffix encodes sub-window and LOEO for non-overwriting outputs
    suffix = ""
    if sub_window_end:
        suffix += f"_split_{sub_window_end}"
    if args.loeo:
        suffix += f"_loeo_{args.loeo}"
    out_path = out_dir / f"table_leadlag_tests_{args.event}{suffix}.csv"
    results.write_csv(out_path)
    logger.info("Wrote %s (%d rows)", out_path, len(results))

    sig = results.filter(pl.col("significant_p01"))
    logger.info("Significant edges (p<0.01): %d / %d", len(sig), len(results))
    if len(sig) > 0:
        print(sig.sort("peak_corr", descending=True).head(10))


if __name__ == "__main__":
    main()
