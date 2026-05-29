"""Event-arrival analysis for sparse flow features (mint/burn, settlement).

For nodes whose flow features are very sparse (a handful of non-zero rows per
event window), continuous lead-lag has low power.  This script runs a
conditional response estimator: for each arrival in the source node's sparse
feature, measure the mean response in adjacent target nodes over a post-arrival
window vs. a pre-arrival baseline.

P-values are estimated by a permutation test (shuffle arrival times within
the analysis span).

Input:
    data/gold/dataset_contagion_features_{event}.parquet

Output:
    results/tables/table_sparse_events_{event}.csv

Example usage (AMM-only on-chain analysis — the paper's Tier-A narrative):

    python scripts/06b_run_sparse_flow_event_study.py \\
        --event usdc_svb_2023 \\
        --source-node usdc_mint_burn \\
        --source-feature mint_burn_net_1h \\
        --target-nodes curve_3pool usdc_binance usdc_coinbase \\
        --target-feature usdc_net_sold_1h \\
        --post-hours 3 \\
        --baseline-hours 12 \\
        --n-permutations 1000
"""

from __future__ import annotations

import argparse

import polars as pl

from stressnet.config import gold_root, results_root
from stressnet.graph.nodes import nodes_for_event
from stressnet.models.sparse_events import compute_sparse_response_table
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sparse-flow event-arrival response study."
    )
    parser.add_argument("--event", required=True, help="Event id (e.g. usdc_svb_2023).")
    parser.add_argument(
        "--source-node",
        required=True,
        help="Node whose feature produces sparse arrivals (e.g. usdc_mint_burn).",
    )
    parser.add_argument(
        "--source-feature",
        default="mint_burn_net_1h",
        help="Feature column to detect arrivals in (default: mint_burn_net_1h).",
    )
    parser.add_argument(
        "--target-nodes",
        nargs="+",
        default=None,
        help="Target node ids to measure response for.  If not set, all non-source nodes.",
    )
    parser.add_argument(
        "--target-feature",
        default="usdc_net_sold_1h",
        help="Feature to measure on target nodes (default: usdc_net_sold_1h).",
    )
    parser.add_argument(
        "--post-hours",
        type=float,
        default=3.0,
        help="Response window in hours after each arrival (default: 3).",
    )
    parser.add_argument(
        "--baseline-hours",
        type=float,
        default=12.0,
        help="Baseline window in hours before each arrival (default: 12).",
    )
    parser.add_argument(
        "--min-abs-source",
        type=float,
        default=0.0,
        help="Min |source_feature| to count as an arrival (default: 0 = any non-zero).",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=500,
        help="Number of permutation replicates for p-value (default: 500).",
    )
    parser.add_argument("--phase", default=None, help="Restrict to one event_phase.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42).")
    parser.add_argument(
        "--paper-mode",
        action="store_true",
        help="Restrict target nodes to real (non-fixture) nodes only.",
    )
    args = parser.parse_args()

    panel_path = gold_root() / f"dataset_contagion_features_{args.event}.parquet"
    if not panel_path.exists():
        raise SystemExit(f"Panel not found: {panel_path}. Run script 03 first.")

    panel = pl.read_parquet(panel_path)
    all_nodes = nodes_for_event(args.event)
    panel_node_ids = set(panel["node_id"].unique().to_list())

    # Determine target nodes
    if args.target_nodes:
        target_nodes = [n for n in args.target_nodes if n in panel_node_ids]
    else:
        target_nodes = [
            n.id for n in all_nodes
            if n.id != args.source_node and n.id in panel_node_ids
        ]

    if args.paper_mode:
        real_node_ids = set(
            panel.filter(pl.col("tier_actual") != "fixture_non_empirical")
            ["node_id"].unique().to_list()
        )
        target_nodes = [n for n in target_nodes if n in real_node_ids]
        logger.info("--paper-mode: restricting to %d real target nodes", len(target_nodes))

    if not target_nodes:
        raise SystemExit("No target nodes found in panel.")

    # Check source node is in panel
    if args.source_node not in panel_node_ids:
        raise SystemExit(
            f"Source node '{args.source_node}' not found in panel. "
            f"Available nodes: {sorted(panel_node_ids)}"
        )

    logger.info(
        "Sparse event study: source=%s  feature=%s  targets=%d  post=%.1fh  baseline=%.1fh",
        args.source_node, args.source_feature, len(target_nodes),
        args.post_hours, args.baseline_hours,
    )

    result = compute_sparse_response_table(
        panel,
        source_node_id=args.source_node,
        source_feature=args.source_feature,
        target_node_ids=target_nodes,
        target_feature_col=args.target_feature,
        post_seconds=args.post_hours * 3600,
        baseline_seconds=args.baseline_hours * 3600,
        min_abs_source=args.min_abs_source,
        n_permutations=args.n_permutations,
        phase=args.phase,
        seed=args.seed,
    )

    if result.height == 0:
        raise SystemExit("No results produced (check source node has non-zero arrivals).")

    result = result.with_columns(
        pl.lit(args.event).alias("event_id"),
        pl.lit(args.phase or "all").alias("event_phase"),
    )

    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"table_sparse_events_{args.event}.csv"
    result.write_csv(out_path)
    logger.info("Wrote %s (%d rows)", out_path, result.height)

    sig = result.filter(pl.col("significant_p05"))
    logger.info(
        "Significant responses (p<0.05): %d / %d",
        sig.height, result.height,
    )
    if sig.height > 0:
        print(
            sig.sort("p_value")
            .select(["source_node_id", "target_node_id", "feature_col",
                     "n_events", "mean_diff", "pct_change", "p_value"])
        )


if __name__ == "__main__":
    main()
