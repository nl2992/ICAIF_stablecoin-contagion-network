"""Build a prediction-ready dataset from the gold panel with event-time splits.

Writes:
    data/gold/dataset_prediction_{event}.parquet   – feature matrix with split tags
    results/tables/table_prediction_split_{event}.csv – split stats

The event-time split (no random shuffling) divides the panel at the
train_fraction boundary of event_time_seconds, preserving temporal order.
"""

import argparse
from pathlib import Path

import polars as pl

from stressnet.config import gold_root, results_root
from stressnet.graph.nodes import nodes_for_event
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_FEATURE_COLS = [
    "mid_price",
    "spread_bps",
    "depth_10bps_bid_usd",
    "depth_10bps_ask_usd",
    "orderbook_imbalance",
    "executable_price_10k_buy",
    "executable_price_10k_sell",
    "basis_vs_usd",
    "reserve_imbalance",
    "implied_pool_price",
    "pool_slippage_10k",
    "exchange_inflow_1h",
    "exchange_outflow_1h",
    "exchange_netflow_1h",
    "mint_burn_net_1h",
    "gas_base_fee_gwei",
]

_LABEL_COLS = [
    "label_basis_gt10bps",
    "label_basis_gt50bps",
    "label_downstream_gt10bps_1m",
    "label_downstream_gt50bps_5m",
]


def _lagged_features(df: pl.DataFrame, feature_cols: list[str], lags: list[int]) -> pl.DataFrame:
    """Add lagged versions of each feature column.

    Lags are computed per (event_id, node_id) group, sorted by event_time_seconds.
    This ensures no cross-node or cross-event contamination.
    """
    result = df.sort(["event_id", "node_id", "event_time_seconds"])
    for col in feature_cols:
        if col not in result.columns:
            continue
        for lag in lags:
            alias = f"{col}_lag{lag}"
            result = result.with_columns(
                pl.col(col)
                .shift(lag)
                .over(["event_id", "node_id"])
                .alias(alias)
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build prediction dataset with event-time train/test split."
    )
    parser.add_argument("--event", required=True)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Fraction of event_time_seconds used for training (default: 0.7).",
    )
    parser.add_argument(
        "--lags",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="Lag steps to add as features (default: 1 5 10).",
    )
    parser.add_argument(
        "--primary-label",
        default="label_downstream_gt10bps_1m",
        help="Primary label column to report split prevalence for.",
    )
    args = parser.parse_args()

    panel_path = gold_root() / f"dataset_contagion_features_{args.event}.parquet"
    if not panel_path.exists():
        raise SystemExit(f"Panel not found: {panel_path}. Run script 03 first.")

    panel = pl.read_parquet(panel_path)
    ts_col = "event_time_seconds"

    feature_cols = [c for c in _FEATURE_COLS if c in panel.columns]
    label_cols = [c for c in _LABEL_COLS if c in panel.columns]

    if not feature_cols:
        raise SystemExit("No feature columns found in panel.")
    if not label_cols:
        raise SystemExit("No label columns found in panel. Run script 03 first.")

    logger.info(
        "Building prediction dataset: %d features, %d labels, %d rows",
        len(feature_cols), len(label_cols), panel.height,
    )

    # Add lagged features (per node, no lookahead)
    panel = _lagged_features(panel, feature_cols, lags=args.lags)

    # Event-time split
    t_min = float(panel[ts_col].min())
    t_max = float(panel[ts_col].max())
    split_point = t_min + (t_max - t_min) * args.train_fraction

    panel = panel.with_columns(
        pl.when(pl.col(ts_col) < split_point)
        .then(pl.lit("train"))
        .otherwise(pl.lit("test"))
        .alias("split")
    )

    # Drop rows that are NaN in all feature columns (lag padding rows)
    lag_cols = [c for c in panel.columns if "_lag" in c]
    keep_cols = feature_cols + lag_cols + label_cols + [ts_col, "node_id", "event_id", "split",
                                                         "layer", "tier_actual"]
    keep_cols = [c for c in keep_cols if c in panel.columns]
    panel = panel.select(keep_cols)

    # Write
    out_path = gold_root() / f"dataset_prediction_{args.event}.parquet"
    gold_root().mkdir(parents=True, exist_ok=True)
    panel.write_parquet(out_path)
    logger.info("Wrote prediction dataset: %s (%d rows × %d cols)",
                out_path.name, panel.height, panel.width)

    # Split stats
    stats_rows = []
    for split in ("train", "test"):
        sub = panel.filter(pl.col("split") == split)
        row: dict = {
            "event_id": args.event,
            "split": split,
            "n_rows": sub.height,
            "n_nodes": sub["node_id"].n_unique() if "node_id" in sub.columns else 0,
        }
        if args.primary_label in sub.columns:
            labels = sub[args.primary_label]
            row["label_prevalence"] = float(labels.mean()) if labels.len() > 0 else float("nan")
        stats_rows.append(row)

    stats_df = pl.DataFrame(stats_rows)
    tables_dir = results_root() / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    stats_path = tables_dir / f"table_prediction_split_{args.event}.csv"
    stats_df.write_csv(stats_path)
    logger.info("Wrote split stats: %s", stats_path.name)
    print(stats_df)


if __name__ == "__main__":
    main()
