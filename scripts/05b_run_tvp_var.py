"""Run Time-Varying Parameter VAR for one event.

Writes:
    results/tables/table_tvp_var_spillovers_{event}.csv   – per-window FEVD shares
    results/tables/table_tvp_var_summary_{event}.csv      – mean/max summary per edge
"""

import argparse
import itertools

import numpy as np
import polars as pl

from stressnet.config import gold_root, load_events, results_root
from stressnet.graph.nodes import nodes_for_event
from stressnet.models.tvp_var import run_tvp_var, tvp_var_summary
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TVP-VAR for one event.")
    parser.add_argument("--event", required=True)
    parser.add_argument("--feature-col", default="basis_vs_usd")
    parser.add_argument(
        "--window-type",
        default="rolling",
        choices=["rolling", "forgetting_factor", "kalman"],
        help="TVP-VAR estimation mode.",
    )
    parser.add_argument("--window-size", type=int, default=3600,
                        help="Rows per rolling window (default: 3600).")
    parser.add_argument("--step-size", type=int, default=300,
                        help="Step between windows in rows (default: 300).")
    parser.add_argument("--forgetting-factor", type=float, default=0.99)
    parser.add_argument("--max-lags", type=int, default=5)
    parser.add_argument("--fevd-horizon", type=int, default=10)
    args = parser.parse_args()

    events = load_events()
    if args.event not in events:
        raise SystemExit(f"Unknown event '{args.event}'")

    panel_path = gold_root() / f"dataset_contagion_features_{args.event}.parquet"
    if not panel_path.exists():
        raise SystemExit(f"Panel not found: {panel_path}. Run script 03 first.")

    panel = pl.read_parquet(panel_path)
    nodes = nodes_for_event(args.event)
    node_ids = [n.id for n in nodes if n.id in panel["node_id"].unique().to_list()]

    if not node_ids:
        raise SystemExit("No configured nodes found in panel.")

    feature_col = args.feature_col
    ts_col = "event_time_seconds"

    # Build (T, N) matrix from pivot
    pivot = (
        panel.filter(pl.col("node_id").is_in(node_ids))
        .select([ts_col, "node_id", feature_col])
        .pivot(values=feature_col, index=ts_col, on="node_id")
        .sort(ts_col)
    )

    available_nodes = [n for n in node_ids if n in pivot.columns]
    if len(available_nodes) < 2:
        raise SystemExit("Need at least 2 nodes with non-null data for TVP-VAR.")

    timestamps = pivot[ts_col].to_numpy().astype(float)
    data = pivot.select(available_nodes).to_numpy().astype(float)

    logger.info(
        "TVP-VAR (%s): %d nodes × %d timesteps",
        args.window_type, len(available_nodes), len(timestamps),
    )

    tvp_df = run_tvp_var(
        data=data,
        node_names=available_nodes,
        timestamps=timestamps,
        window_type=args.window_type,
        window_size=args.window_size,
        step_size=args.step_size,
        forgetting_factor=args.forgetting_factor,
        max_lags=args.max_lags,
        fevd_horizon=args.fevd_horizon,
    )

    if tvp_df.is_empty():
        logger.warning("TVP-VAR produced no results. Try a smaller --window-size.")
        return

    tvp_df = tvp_df.with_columns(pl.lit(args.event).alias("event_id"))
    summary_df = tvp_var_summary(tvp_df).with_columns(pl.lit(args.event).alias("event_id"))

    tables_dir = results_root() / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    per_window_path = tables_dir / f"table_tvp_var_spillovers_{args.event}.csv"
    summary_path = tables_dir / f"table_tvp_var_summary_{args.event}.csv"

    tvp_df.write_csv(per_window_path)
    summary_df.write_csv(summary_path)

    logger.info("Wrote %s (%d rows)", per_window_path.name, tvp_df.height)
    logger.info("Wrote %s (%d rows)", summary_path.name, summary_df.height)
    print(summary_df.head(20))


if __name__ == "__main__":
    main()
