"""Train and evaluate prediction models for downstream stress.

Runs non-graph baselines and (if available) the temporal GNN.
Writes:
    results/tables/table_prediction_metrics_{event}.csv
    results/figures/figure_auc_by_event.png

Modes:
    Default  -- within-event temporal split (last --test-fraction of event time).
    --loeo   -- leave-one-event-out: train on all other event panels, test on --event.

Ablation:
    --ablation -- additionally run a no-graph-features ablation experiment.
"""

import argparse

import numpy as np
import polars as pl

from stressnet.config import gold_root, load_events, results_root
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

# Features that encode cross-node graph/flow signals; used in ablation
_GRAPH_FEATURE_COLS = [
    "exchange_netflow_1h", "reserve_imbalance", "implied_pool_price",
    "pool_slippage_10k", "mint_burn_net_1h",
]


def load_all_panels(events_list: list[str], exclude_event: str) -> pl.DataFrame:
    """Load and concatenate gold panels for all events except *exclude_event*.

    Rows flagged as fixture_non_empirical are dropped.  Missing panels are
    skipped with a warning (graceful degradation).

    Returns a concatenated DataFrame (diagonal concat handles schema drift).
    Raises SystemExit if no panels could be loaded.
    """
    frames: list[pl.DataFrame] = []
    for ev in events_list:
        if ev == exclude_event:
            continue
        p = gold_root() / f"dataset_contagion_features_{ev}.parquet"
        if not p.exists():
            logger.warning("Missing panel for train event %s; skipping.", ev)
            continue
        df = pl.read_parquet(p)
        if "tier_actual" in df.columns:
            df = df.filter(pl.col("tier_actual") != "fixture_non_empirical")
        frames.append(df)

    if not frames:
        raise SystemExit("No training panels found for LOEO mode.")
    return pl.concat(frames, how="diagonal")


def _drop_fixture_rows(df: pl.DataFrame) -> pl.DataFrame:
    if "tier_actual" not in df.columns:
        return df
    return df.filter(pl.col("tier_actual") != "fixture_non_empirical")


def _assert_label_has_signal(y_train: np.ndarray, y_test: np.ndarray, split_name: str) -> None:
    if len(np.unique(y_train)) < 2:
        raise SystemExit(f"{split_name}: training labels contain only one class.")
    if len(np.unique(y_test)) < 2:
        raise SystemExit(f"{split_name}: test labels contain only one class.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prediction models.")
    parser.add_argument("--event", required=True)
    parser.add_argument("--label-col", default=_LABEL_COL)
    parser.add_argument("--test-fraction", type=float, default=0.3,
                        help="Last fraction of event time used as test set (default mode only).")
    parser.add_argument("--loeo", action="store_true",
                        help="Leave-one-event-out: train on all other events, test on --event.")
    parser.add_argument("--ablation", action="store_true",
                        help="Run no-network ablation experiments (no graph features).")
    args = parser.parse_args()

    # Always load the target event panel
    panel_path = gold_root() / f"dataset_contagion_features_{args.event}.parquet"
    if not panel_path.exists():
        raise SystemExit(f"Panel not found: {panel_path}. Run script 03 first.")

    panel = _drop_fixture_rows(pl.read_parquet(panel_path))

    if args.label_col not in panel.columns:
        raise SystemExit(
            f"Label column '{args.label_col}' not in panel. "
            f"Available labels: {[c for c in panel.columns if c.startswith('label_')]}"
        )

    feature_cols = [c for c in _FEATURE_COLS if c in panel.columns]
    logger.info("Using %d feature columns", len(feature_cols))

    ts_col = "event_time_seconds"
    if ts_col not in panel.columns:
        raise SystemExit(f"Column '{ts_col}' not found.")

    # -------------------------------------------------------------------------
    # Build train / test splits
    # -------------------------------------------------------------------------
    if args.loeo:
        all_event_ids = list(load_events().keys())
        train_events = [e for e in all_event_ids if e != args.event]

        train_frames: list[pl.DataFrame] = []
        for ev in train_events:
            p = gold_root() / f"dataset_contagion_features_{ev}.parquet"
            if p.exists():
                df = _drop_fixture_rows(pl.read_parquet(p))
                train_frames.append(df)
            else:
                logger.warning("Missing panel for train event %s; skipping.", ev)

        if not train_frames:
            raise SystemExit("No training panels found for LOEO mode.")

        train_panel = pl.concat(train_frames, how="diagonal")
        test_panel = panel  # full test-event panel; no time split in LOEO mode
        feature_cols = [
            c for c in feature_cols
            if c in train_panel.columns and c in test_panel.columns
        ]
        split_type = "leave_one_event_out"
        heldout_event = args.event

        logger.info(
            "LOEO mode: training on %d events (%d rows), testing on %s (%d rows)",
            len(train_frames), train_panel.height, args.event, test_panel.height,
        )
    else:
        # Default: within-event temporal split
        t_max = panel[ts_col].max()
        t_min = panel[ts_col].min()
        split_point = t_min + (t_max - t_min) * (1 - args.test_fraction)

        train_panel = panel.filter(pl.col(ts_col) < split_point)
        test_panel = panel.filter(pl.col(ts_col) >= split_point)
        split_type = "within_event_time"
        heldout_event = None

    if not feature_cols:
        raise SystemExit("No feature columns are shared by train/test panels.")

    X_train, y_train = prepare_Xy(train_panel, feature_cols, args.label_col)
    X_test, y_test = prepare_Xy(test_panel, feature_cols, args.label_col)

    logger.info("Train: %d rows | Test: %d rows | Prevalence: %.3f",
                len(y_train), len(y_test), y_test.mean())

    _assert_label_has_signal(y_train, y_test, split_type)

    # -------------------------------------------------------------------------
    # Baselines
    # -------------------------------------------------------------------------
    results = run_baselines(X_train, y_train, X_test, y_test)
    results = results.with_columns([
        pl.lit(args.event).alias("event_id"),
        pl.lit(split_type).alias("split_type"),
        pl.lit(heldout_event).alias("heldout_event"),
        pl.lit("full").alias("ablation"),
        pl.lit(args.label_col).alias("label_col"),
        pl.lit(int(y_train.sum())).alias("train_positive"),
        pl.lit(int(y_test.sum())).alias("test_positive"),
        pl.lit(float(y_train.mean())).alias("train_prevalence"),
        pl.lit(float(y_test.mean())).alias("test_prevalence"),
    ])

    # -------------------------------------------------------------------------
    # Ablation: no-graph-features
    # -------------------------------------------------------------------------
    if args.ablation:
        graph_features = _GRAPH_FEATURE_COLS
        non_graph_cols = [c for c in feature_cols if c not in graph_features]
        if len(non_graph_cols) >= 2:
            X_train_ng, y_train_ng = prepare_Xy(train_panel, non_graph_cols, args.label_col)
            X_test_ng, y_test_ng = prepare_Xy(test_panel, non_graph_cols, args.label_col)
            ablation_results = run_baselines(X_train_ng, y_train_ng, X_test_ng, y_test_ng)
            ablation_results = ablation_results.with_columns([
                pl.lit(args.event).alias("event_id"),
                pl.lit(split_type).alias("split_type"),
                pl.lit(heldout_event).alias("heldout_event"),
                pl.lit("no_graph_features").alias("ablation"),
                pl.lit(args.label_col).alias("label_col"),
                pl.lit(int(y_train_ng.sum())).alias("train_positive"),
                pl.lit(int(y_test_ng.sum())).alias("test_positive"),
                pl.lit(float(y_train_ng.mean())).alias("train_prevalence"),
                pl.lit(float(y_test_ng.mean())).alias("test_prevalence"),
            ])
            results = pl.concat([results, ablation_results], how="diagonal")
            logger.info(
                "Ablation (no_graph_features): dropped %d graph features, kept %d",
                len(graph_features), len(non_graph_cols),
            )
        else:
            logger.warning(
                "Skipping no_graph_features ablation: only %d non-graph columns available.",
                len(non_graph_cols),
            )

    # -------------------------------------------------------------------------
    # Write results
    # -------------------------------------------------------------------------
    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"table_prediction_metrics_{args.event}.csv"
    results.write_csv(out_path)
    logger.info("Wrote %s", out_path)
    print(results)


if __name__ == "__main__":
    main()
