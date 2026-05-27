"""Train and evaluate prediction models for downstream stress.

Runs non-graph baselines and (if available) the temporal GNN.
Writes:
    results/tables/table_prediction_metrics_{event}.csv
    results/figures/figure_auc_by_event.png
"""

import argparse

import numpy as np
import polars as pl

from stressnet.config import gold_root, results_root
from stressnet.graph.nodes import nodes_for_event
from stressnet.models.baselines import prepare_Xy, run_baselines
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_LABEL_COL = "label_downstream_gt10bps_1m"

_FEATURE_COLS = [
    "mid_price", "spread_bps", "depth_10bps_bid_usd", "depth_10bps_ask_usd",
    "orderbook_imbalance", "basis_vs_usd",
    "reserve_imbalance", "implied_pool_price", "pool_slippage_10k",
    "exchange_netflow_1h", "mint_burn_net_1h", "gas_base_fee_gwei",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prediction models.")
    parser.add_argument("--event", required=True)
    parser.add_argument("--label-col", default=_LABEL_COL)
    parser.add_argument("--test-fraction", type=float, default=0.3,
                        help="Last fraction of event time used as test set.")
    args = parser.parse_args()

    panel_path = gold_root() / f"dataset_contagion_features_{args.event}.parquet"
    if not panel_path.exists():
        raise SystemExit(f"Panel not found: {panel_path}. Run script 03 first.")

    panel = pl.read_parquet(panel_path)

    if args.label_col not in panel.columns:
        raise SystemExit(
            f"Label column '{args.label_col}' not in panel. "
            f"Available labels: {[c for c in panel.columns if c.startswith('label_')]}"
        )

    feature_cols = [c for c in _FEATURE_COLS if c in panel.columns]
    logger.info("Using %d feature columns", len(feature_cols))

    # Event-time split (no random split)
    ts_col = "event_time_seconds"
    if ts_col not in panel.columns:
        raise SystemExit(f"Column '{ts_col}' not found.")

    t_max = panel[ts_col].max()
    t_min = panel[ts_col].min()
    split_point = t_min + (t_max - t_min) * (1 - args.test_fraction)

    train_panel = panel.filter(pl.col(ts_col) < split_point)
    test_panel = panel.filter(pl.col(ts_col) >= split_point)

    X_train, y_train = prepare_Xy(train_panel, feature_cols, args.label_col)
    X_test, y_test = prepare_Xy(test_panel, feature_cols, args.label_col)

    logger.info("Train: %d rows | Test: %d rows | Prevalence: %.3f",
                len(y_train), len(y_test), y_test.mean())

    if len(np.unique(y_test)) < 2:
        logger.warning("Test set has only one class; skipping metrics.")
        return

    results = run_baselines(X_train, y_train, X_test, y_test)
    results = results.with_columns(pl.lit(args.event).alias("event_id"))

    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"table_prediction_metrics_{args.event}.csv"
    results.write_csv(out_path)
    logger.info("Wrote %s", out_path)
    print(results)


if __name__ == "__main__":
    main()
