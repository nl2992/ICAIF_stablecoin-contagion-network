"""Estimate pairwise transfer entropy for non-linear directional information flow.

Writes:
    results/tables/table_transfer_entropy_{event}.csv
"""

import argparse
import itertools

import polars as pl

from stressnet.config import gold_root, results_root
from stressnet.graph.nodes import nodes_for_event
from stressnet.models.transfer_entropy import compute_te_table
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate transfer entropy.")
    parser.add_argument("--event", required=True)
    parser.add_argument("--feature-col", default="basis_vs_usd")
    parser.add_argument("--history-length", type=int, default=10)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--n-shuffles", type=int, default=200)
    parser.add_argument("--block-size", type=int, default=60,
                        help="Block size for block-shuffle null (default 60 = 1h of 1-min data).")
    parser.add_argument("--paper-mode", action="store_true",
                        help="Restrict to real (non-fixture) nodes only.")
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

    node_pairs = list(itertools.permutations(node_ids, 2))

    logger.info("Estimating TE for %d node pairs", len(node_pairs))
    results = compute_te_table(
        panel,
        node_pairs=node_pairs,
        feature_col=args.feature_col,
        history_length=args.history_length,
        n_bins=args.n_bins,
        n_shuffles=args.n_shuffles,
        block_size=args.block_size,
    )
    results = results.with_columns(pl.lit(args.event).alias("event_id"))

    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"table_transfer_entropy_{args.event}.csv"
    results.write_csv(out_path)
    logger.info("Wrote %s (%d pairs)", out_path, len(results))

    sig_p05 = results.filter(pl.col("significant_p05"))
    sig_fdr  = results.filter(pl.col("significant_fdr")) if "significant_fdr" in results.columns else sig_p05
    sig_blk  = results.filter(pl.col("significant_block_fdr")) if "significant_block_fdr" in results.columns else pl.DataFrame()
    logger.info("Significant TE pairs  p<0.05=%d  FDR=%d  block-FDR=%d  / %d",
                len(sig_p05), len(sig_fdr), len(sig_blk), len(results))
    if len(sig_blk) > 0:
        print(sig_blk.head(10))
    elif len(sig_fdr) > 0:
        print(sig_fdr.head(10))


if __name__ == "__main__":
    main()
