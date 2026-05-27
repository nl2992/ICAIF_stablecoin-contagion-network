"""Estimate VAR/Granger causality and FEVD spillover table.

Writes:
    results/tables/table_var_spillovers_{event}.csv
"""

import argparse

import numpy as np
import polars as pl

from stressnet.config import gold_root, results_root
from stressnet.graph.nodes import nodes_for_event
from stressnet.models.var_granger import fit_var, granger_causality_table, fevd_spillover_table
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VAR/Granger and FEVD spillovers.")
    parser.add_argument("--event", required=True)
    parser.add_argument("--feature-col", default="basis_vs_usd")
    parser.add_argument("--max-lags", type=int, default=10)
    parser.add_argument("--fevd-horizon", type=int, default=10)
    args = parser.parse_args()

    panel_path = gold_root() / f"dataset_contagion_features_{args.event}.parquet"
    if not panel_path.exists():
        raise SystemExit(f"Panel not found: {panel_path}. Run script 03 first.")

    panel = pl.read_parquet(panel_path)
    nodes = nodes_for_event(args.event)
    node_ids = [n.id for n in nodes if n.id in panel["node_id"].unique().to_list()]

    if len(node_ids) < 2:
        raise SystemExit("Need at least 2 nodes for VAR.")

    ts_col = "event_time_seconds"
    if ts_col not in panel.columns:
        raise SystemExit(f"Column '{ts_col}' not found in panel.")

    # Build (T, N) matrix: align all nodes on event_time grid
    pivot = (
        panel
        .filter(pl.col("node_id").is_in(node_ids))
        .select(["node_id", ts_col, args.feature_col])
        .pivot(values=args.feature_col, index=ts_col, on="node_id")
        .sort(ts_col)
        .drop_nulls()
    )

    node_cols = [c for c in pivot.columns if c != ts_col and c in node_ids]
    data = pivot.select(node_cols).to_numpy()

    if data.shape[0] < 50:
        raise SystemExit(f"Too few observations after alignment: {data.shape[0]}")

    logger.info("Fitting VAR on %d observations × %d nodes", *data.shape)
    var_fit = fit_var(data, node_names=node_cols, max_lags=args.max_lags)

    granger = granger_causality_table(var_fit)
    fevd = fevd_spillover_table(var_fit, horizon=args.fevd_horizon)

    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    granger.write_csv(out_dir / f"table_granger_{args.event}.csv")
    fevd.write_csv(out_dir / f"table_var_spillovers_{args.event}.csv")
    logger.info("Wrote VAR results to %s", out_dir)

    sig = granger.filter(pl.col("significant"))
    logger.info("Significant Granger edges: %d / %d", len(sig), len(granger))
    if len(sig) > 0:
        print(sig.sort("f_stat", descending=True).head(10))


if __name__ == "__main__":
    main()
