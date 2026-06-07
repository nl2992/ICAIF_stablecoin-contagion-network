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
    parser.add_argument(
        "--feature-col", default="basis_vs_usd",
        help=(
            "Feature column to use for TVP-VAR.  Defaults to 'basis_vs_usd'.  "
            "For DEX-layer paper-mode runs use 'usdc_net_sold_1h'.  "
            "When 'auto' is passed, the script selects the first column in "
            "[usdc_net_sold_1h, basis_vs_usd, exchange_netflow_1h] that has "
            "non-null data for at least half the selected nodes."
        ),
    )
    parser.add_argument(
        "--layer-filter", default=None,
        help="Restrict TVP-VAR to nodes of a single layer (e.g. DEX, CEX).",
    )
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
    parser.add_argument("--paper-mode", action="store_true",
                        help="Restrict to real (non-fixture) nodes only.")
    args = parser.parse_args()

    events = load_events()
    if args.event not in events:
        raise SystemExit(f"Unknown event '{args.event}'")

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

    # Optional layer filter (e.g. --layer-filter DEX for AMM-only TVP-VAR)
    if args.layer_filter:
        layer_node_ids = {n.id for n in nodes if n.layer == args.layer_filter}
        node_ids = [nid for nid in node_ids if nid in layer_node_ids]
        logger.info("--layer-filter %s: %d nodes remaining", args.layer_filter, len(node_ids))

    feature_col = args.feature_col

    # Auto-select feature column: pick the first column that has non-null data
    # for at least half the selected nodes.  Resolves the common failure mode
    # where basis_vs_usd (the default) is null for DEX nodes but
    # usdc_net_sold_1h is not.
    _FEATURE_PRIORITY = ["usdc_net_sold_1h", "basis_vs_usd", "exchange_netflow_1h",
                         "mint_burn_net_1h", "reserve_imbalance"]
    if feature_col == "auto":
        node_panel = panel.filter(pl.col("node_id").is_in(node_ids))
        for cand in _FEATURE_PRIORITY:
            if cand not in node_panel.columns:
                continue
            n_with_data = (
                node_panel.group_by("node_id")
                .agg((pl.col(cand).drop_nulls().len() > 10).alias("has_data"))
                .filter(pl.col("has_data"))
                .height
            )
            if n_with_data >= max(1, len(node_ids) // 2):
                feature_col = cand
                logger.info("--feature-col auto: selected '%s' (%d/%d nodes have data)",
                            cand, n_with_data, len(node_ids))
                break
        if feature_col == "auto":
            logger.warning("--feature-col auto: no suitable column found; falling back to basis_vs_usd")
            feature_col = "basis_vs_usd"
    elif feature_col == "basis_vs_usd":
        # Proactive check: if basis_vs_usd is missing/null for most selected nodes
        # AND usdc_net_sold_1h is available, switch automatically and warn.
        node_panel = panel.filter(pl.col("node_id").is_in(node_ids))
        if "basis_vs_usd" in node_panel.columns:
            n_basis = (
                node_panel.group_by("node_id")
                .agg((pl.col("basis_vs_usd").drop_nulls().len() > 10).alias("has_data"))
                .filter(pl.col("has_data")).height
            )
        else:
            n_basis = 0
        if n_basis < max(1, len(node_ids) // 2) and "usdc_net_sold_1h" in node_panel.columns:
            logger.warning(
                "basis_vs_usd has data for only %d/%d selected nodes; "
                "auto-switching to usdc_net_sold_1h for TVP-VAR (use --feature-col to override).",
                n_basis, len(node_ids),
            )
            feature_col = "usdc_net_sold_1h"

    if len(node_ids) < 2:
        logger.warning(
            "Fewer than 2 nodes for event '%s'%s — skipping TVP-VAR.",
            args.event,
            " in paper-mode" if args.paper_mode else "",
        )
        return

    ts_col = "event_time_seconds"

    # Build (T, N) matrix from pivot — hourly grid to align mixed-resolution series
    _HOUR = 3600
    sub = (
        panel
        .filter(pl.col("node_id").is_in(node_ids))
        .filter(pl.col("tier_actual") != "fixture_non_empirical")
        .select(["node_id", ts_col, feature_col])
        .with_columns(
            ((pl.col(ts_col) // _HOUR) * _HOUR).alias("hour_bucket")
        )
        .group_by(["node_id", "hour_bucket"])
        .agg(pl.col(feature_col).mean().alias(feature_col))
    )

    pivot = (
        sub
        .pivot(values=feature_col, index="hour_bucket", on="node_id")
        .sort("hour_bucket")
    )

    available_nodes = [n for n in node_ids if n in pivot.columns]
    if len(available_nodes) < 2:
        logger.warning(
            "Need at least 2 nodes with data after resampling for event '%s' — skipping.",
            args.event,
        )
        return

    # Drop nodes with fewer than MIN_ROWS of data
    _MIN_ROWS = 48  # at least 48 hourly obs
    available_nodes = [
        c for c in available_nodes
        if pivot[c].drop_nulls().len() >= _MIN_ROWS
    ]
    if len(available_nodes) < 2:
        logger.warning(
            "Fewer than 2 nodes with >= %d hourly observations for event '%s' — skipping.",
            _MIN_ROWS, args.event,
        )
        return

    # Restrict to rows where ALL selected nodes have data (overlap period)
    overlap = pivot.select(["hour_bucket"] + available_nodes).drop_nulls()
    if overlap.height < _MIN_ROWS:
        logger.warning(
            "Overlap period only %d hours for event '%s' — need >= %d. Skipping TVP-VAR.",
            overlap.height, args.event, _MIN_ROWS,
        )
        return

    # Auto-scale window size if larger than overlap period
    effective_window = min(args.window_size, overlap.height // 2)
    effective_step   = min(args.step_size, effective_window // 4)
    if effective_window != args.window_size:
        logger.info(
            "Auto-scaled window %d→%d, step %d→%d to fit %d-row overlap.",
            args.window_size, effective_window, args.step_size, effective_step, overlap.height,
        )

    timestamps = overlap["hour_bucket"].to_numpy().astype(float)
    data = overlap.select(available_nodes).to_numpy().astype(float)

    logger.info(
        "TVP-VAR (%s): %d nodes × %d overlap timesteps (window=%d, step=%d)",
        args.window_type, len(available_nodes), len(timestamps),
        effective_window, effective_step,
    )

    tvp_df = run_tvp_var(
        data=data,
        node_names=available_nodes,
        timestamps=timestamps,
        window_type=args.window_type,
        window_size=effective_window,
        step_size=effective_step,
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

    spillover_df = pl.read_csv(per_window_path)
    if "method" in spillover_df.columns:
        fallback_rows = spillover_df.filter(pl.col("method") == "var_coeff_fallback").height
        if fallback_rows > 0:
            logger.warning(
                "%d / %d VAR spillover rows use coefficient fallback (FEVD failed). "
                "These rows are labelled 'var_coeff_fallback' in the method column.",
                fallback_rows, spillover_df.height,
            )

    print(summary_df.head(20))


if __name__ == "__main__":
    main()
