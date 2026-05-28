"""Run Hayashi-Yoshida asynchronous correlation robustness."""

from __future__ import annotations

import argparse
import itertools

import polars as pl

from stressnet.config import gold_root, results_root
from stressnet.graph.nodes import nodes_for_event
from stressnet.models.hayashi_yoshida import compute_hayashi_yoshida_table
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hayashi-Yoshida robustness.")
    parser.add_argument("--event", required=True)
    parser.add_argument("--feature-col", default="basis_vs_usd")
    parser.add_argument("--paper-mode", action="store_true")
    parser.add_argument("--phase", default=None)
    args = parser.parse_args()

    panel_path = gold_root() / f"dataset_contagion_features_{args.event}.parquet"
    if not panel_path.exists():
        raise SystemExit(f"Panel not found: {panel_path}. Run script 03 first.")
    panel = pl.read_parquet(panel_path)
    if args.phase:
        if "event_phase" not in panel.columns:
            raise SystemExit("--phase requested but panel lacks event_phase.")
        panel = panel.filter(pl.col("event_phase") == args.phase)

    nodes = nodes_for_event(args.event)
    if args.paper_mode:
        real_node_ids = (
            panel.filter(pl.col("tier_actual") != "fixture_non_empirical")
            ["node_id"].unique().to_list()
        )
        node_ids = [n.id for n in nodes if n.id in real_node_ids]
    else:
        node_ids = [n.id for n in nodes if n.id in panel["node_id"].unique().to_list()]

    if len(node_ids) < 2:
        raise SystemExit("Need at least 2 nodes for Hayashi-Yoshida robustness.")

    table = compute_hayashi_yoshida_table(
        panel,
        list(itertools.permutations(node_ids, 2)),
        feature_col=args.feature_col,
    ).with_columns(
        pl.lit(args.event).alias("event_id"),
        pl.lit(args.phase or "all").alias("event_phase"),
    )

    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"table_hayashi_yoshida_{args.event}.csv"
    table.write_csv(out_path)
    logger.info("Wrote %s (%d rows)", out_path, table.height)
    print(table.sort("hy_corr", descending=True).head(10))


if __name__ == "__main__":
    main()
