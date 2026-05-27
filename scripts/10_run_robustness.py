"""Run robustness checks: alternative grids, subsamples, placebo events.

Writes:
    results/tables/table_robustness_{event}.csv
"""

import argparse

import polars as pl

from stressnet.config import gold_root, results_root
from stressnet.evaluation.robustness import (
    run_grid_robustness,
    subsample_cex_only,
    subsample_without_dominant,
)
from stressnet.models.leadlag import compute_leadlag_table
from stressnet.graph.nodes import nodes_for_event
from stressnet.utils.logging import get_logger
import itertools

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robustness checks.")
    parser.add_argument("--event", required=True)
    parser.add_argument("--feature-col", default="basis_vs_usd")
    args = parser.parse_args()

    panel_path = gold_root() / f"dataset_contagion_features_{args.event}.parquet"
    if not panel_path.exists():
        raise SystemExit(f"Panel not found: {panel_path}. Run script 03 first.")

    panel = pl.read_parquet(panel_path)
    nodes = nodes_for_event(args.event)
    node_ids = [n.id for n in nodes if n.id in panel["node_id"].unique().to_list()]
    node_pairs = list(itertools.permutations(node_ids, 2))

    def run_leadlag(p: pl.DataFrame, _grid: int) -> pl.DataFrame:
        return compute_leadlag_table(p, node_pairs=node_pairs, feature_col=args.feature_col)

    checks = {}

    # 1. Alternative grids
    logger.info("Running grid robustness checks...")
    grid_results = run_grid_robustness(panel, run_leadlag, grids=[1, 5, 60])
    for grid, df in grid_results.items():
        df = df.with_columns([pl.lit(f"grid_{grid}s").alias("check"), pl.lit(args.event).alias("event_id")])
        checks[f"grid_{grid}s"] = df

    # 2. CEX-only subsample
    logger.info("Running CEX-only subsample...")
    cex_panel = subsample_cex_only(panel)
    cex_results = run_leadlag(cex_panel, 60)
    cex_results = cex_results.with_columns([pl.lit("cex_only").alias("check"), pl.lit(args.event).alias("event_id")])
    checks["cex_only"] = cex_results

    # 3. Without Binance
    logger.info("Running without Binance (dominant venue removal)...")
    no_binance = subsample_without_dominant(panel, dominant_node="usdt_binance")
    nb_results = run_leadlag(no_binance, 60)
    nb_results = nb_results.with_columns([pl.lit("no_binance").alias("check"), pl.lit(args.event).alias("event_id")])
    checks["no_binance"] = nb_results

    all_results = pl.concat(list(checks.values()), how="diagonal")
    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"table_robustness_{args.event}.csv"
    all_results.write_csv(out_path)
    logger.info("Wrote robustness table: %d rows", len(all_results))


if __name__ == "__main__":
    main()
