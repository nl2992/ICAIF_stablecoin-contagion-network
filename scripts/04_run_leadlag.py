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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lead-lag analysis.")
    parser.add_argument("--event", required=True)
    parser.add_argument("--feature-col", default="basis_vs_usd")
    parser.add_argument("--max-lag", type=int, default=60)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--paper-mode", action="store_true",
                        help="Restrict to real (non-fixture) nodes only and add event_id column.")
    args = parser.parse_args()

    panel_path = gold_root() / f"dataset_contagion_features_{args.event}.parquet"
    if not panel_path.exists():
        raise SystemExit(f"Panel not found: {panel_path}. Run script 03 first.")

    panel = pl.read_parquet(panel_path)
    nodes = nodes_for_event(args.event)

    if args.paper_mode:
        real_node_ids = (
            panel.filter(pl.col("tier_actual") != "fixture_non_empirical")
            ["node_id"].unique().to_list()
        )
        node_ids = [n.id for n in nodes if n.id in real_node_ids]
        logger.info("--paper-mode: restricting to %d real nodes", len(node_ids))
    else:
        node_ids = [n.id for n in nodes if n.id in panel["node_id"].unique().to_list()]

    if not node_ids:
        raise SystemExit("No matching nodes found in panel.")

    node_pairs = list(itertools.permutations(node_ids, 2))
    logger.info("Computing lead-lag for %d node pairs", len(node_pairs))

    results = compute_leadlag_table(
        panel,
        node_pairs=node_pairs,
        feature_col=args.feature_col,
        max_lag=args.max_lag,
        n_reps=args.bootstrap_reps,
    )
    results = results.with_columns(pl.lit(args.event).alias("event_id"))

    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"table_leadlag_tests_{args.event}.csv"
    results.write_csv(out_path)
    logger.info("Wrote %s (%d rows)", out_path, len(results))

    sig = results.filter(pl.col("significant_p01"))
    logger.info("Significant edges (p<0.01): %d / %d", len(sig), len(results))
    if len(sig) > 0:
        print(sig.sort("peak_corr", descending=True).head(10))


if __name__ == "__main__":
    main()
