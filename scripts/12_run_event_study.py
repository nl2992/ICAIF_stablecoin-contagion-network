"""Event-study layer: abnormal basis and cumulative abnormal basis per node.

Reads: data/gold/dataset_contagion_features_{event}.parquet
Writes:
    results/tables/table_event_study_timeseries_{event}.csv   – AB/CAB per timestep
    results/tables/table_event_study_summary_{event}.csv      – per-node summary

Notes
-----
``event_time_seconds`` in the panel is seconds relative to the shock onset
(T=0 = shock_onset_utc).  Phase boundaries are therefore at fixed offsets:
  pre:       t < 0
  onset:     0 <= t < 6h
  panic:     6h <= t < 72h
  recovery:  72h <= t < 168h
  post:      t >= 168h
"""

import argparse

import numpy as np
import polars as pl

from stressnet.config import gold_root, load_events, results_root
from stressnet.graph.nodes import nodes_for_event
from stressnet.models.event_study import compute_event_study_table
from stressnet.models.leadlag import fdr_correct
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run event-study AB/CAB analysis.")
    parser.add_argument("--event", required=True)
    parser.add_argument("--feature-col", default="basis_vs_usd")
    parser.add_argument("--n-bootstrap", type=int, default=500,
                        help="Block-bootstrap reps for CAB p-value (default 500).")
    parser.add_argument("--block-size", type=int, default=60,
                        help="Block size for bootstrap null (default 60).")
    parser.add_argument("--paper-mode", action="store_true",
                        help="Restrict to real (non-fixture) nodes only.")
    args = parser.parse_args()

    events = load_events()
    if args.event not in events:
        raise SystemExit(f"Unknown event '{args.event}'. Known: {list(events)}")

    ev_cfg = events[args.event]
    shock_onset_str = ev_cfg.get("shock_onset_utc", "(unknown)")

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
        logger.warning("No nodes found for event '%s'. Skipping.", args.event)
        return

    logger.info(
        "Event study for '%s': shock_onset=%s  n_nodes=%d",
        args.event, shock_onset_str, len(node_ids),
    )

    # event_time_seconds in panel is already relative to shock onset (T=0)
    ts_df, sum_df = compute_event_study_table(
        panel=panel,
        node_ids=node_ids,
        feature_col=args.feature_col,
        n_bootstrap=args.n_bootstrap,
        block_size=args.block_size,
    )

    if sum_df.is_empty():
        logger.warning("Event study produced no results for '%s'.", args.event)
        return

    # BH-FDR correction across nodes (#30: multiple-testing discipline)
    if "p_value" in sum_df.columns and sum_df.height > 1:
        p_arr = sum_df["p_value"].fill_nan(1.0).fill_null(1.0).to_numpy()
        reject_fdr, adj_p_fdr = fdr_correct(p_arr, alpha=0.05)
        p_bonf = np.minimum(p_arr * len(p_arr), 1.0)
        sum_df = sum_df.with_columns([
            pl.Series("p_value_fdr",        adj_p_fdr.tolist()),
            pl.Series("significant_fdr",    reject_fdr.tolist()),
            pl.Series("p_bonferroni",       p_bonf.tolist()),
            pl.Series("significant_bonferroni", (p_bonf < 0.05).tolist()),
        ])

    # Attach event_id
    ts_df  = ts_df.with_columns(pl.lit(args.event).alias("event_id"))
    sum_df = sum_df.with_columns(pl.lit(args.event).alias("event_id"))

    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_path  = out_dir / f"table_event_study_timeseries_{args.event}.csv"
    sum_path = out_dir / f"table_event_study_summary_{args.event}.csv"

    ts_df.write_csv(ts_path)
    sum_df.write_csv(sum_path)

    logger.info("Wrote %s (%d rows)", ts_path.name,  ts_df.height)
    logger.info("Wrote %s (%d rows)", sum_path.name, sum_df.height)

    sig_p05  = sum_df.filter(pl.col("significant_p05"))
    sig_fdr  = sum_df.filter(pl.col("significant_fdr"))  if "significant_fdr"  in sum_df.columns else sig_p05
    sig_bonf = sum_df.filter(pl.col("significant_bonferroni")) if "significant_bonferroni" in sum_df.columns else pl.DataFrame()
    logger.info(
        "Significant AB (p<0.05): %d / %d nodes  FDR: %d  Bonferroni: %d",
        len(sig_p05), len(sum_df), len(sig_fdr), len(sig_bonf),
    )

    print("\n=== Event-study summary ===")
    display_cols = ["node_id", "has_baseline", "estimation_mean", "estimation_std",
                    "cab_event", "p_value", "significant_p05", "n_pre_obs", "transmission_rank"]
    if "significant_fdr" in sum_df.columns:
        display_cols.insert(display_cols.index("significant_p05") + 1, "significant_fdr")
    print(sum_df.sort("transmission_rank").select(display_cols))


if __name__ == "__main__":
    main()
