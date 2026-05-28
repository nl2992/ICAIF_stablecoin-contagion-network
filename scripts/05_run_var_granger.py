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

    # Resample to hourly buckets so all nodes (1-min real + hourly fixture)
    # land on a common grid.  Fixture-only nodes with all-null features are
    # dropped automatically when we filter to columns with sufficient coverage.
    _HOUR = 3600  # seconds
    sub = (
        panel
        .filter(pl.col("node_id").is_in(node_ids))
        .select(["node_id", ts_col, args.feature_col])
        .with_columns(
            ((pl.col(ts_col) / _HOUR).floor().cast(pl.Int64) * _HOUR).alias("hour_bucket")
        )
        .group_by(["node_id", "hour_bucket"])
        .agg(pl.col(args.feature_col).mean().alias(args.feature_col))
        .sort("hour_bucket")
    )

    pivot = (
        sub
        .pivot(values=args.feature_col, index="hour_bucket", on="node_id")
        .sort("hour_bucket")
    )

    # Keep only nodes with enough non-null coverage (≥50% of rows)
    n_rows = pivot.height
    node_cols = [
        c for c in pivot.columns
        if c != "hour_bucket"
        and c in node_ids
        and pivot[c].null_count() < n_rows * 0.5
    ]

    if len(node_cols) < 2:
        raise SystemExit(
            f"Fewer than 2 nodes have sufficient '{args.feature_col}' coverage "
            f"after hourly resampling.  Try a different --feature-col."
        )

    # Forward-fill sparse columns then drop any remaining incomplete rows
    pivot = (
        pivot
        .select(["hour_bucket"] + node_cols)
        .with_columns([pl.col(c).forward_fill() for c in node_cols])
        .drop_nulls()
    )

    data_mat = pivot.select(node_cols).to_numpy()

    # Remove near-perfectly correlated duplicates (|r| > 0.9999) to avoid
    # singular covariance matrix in the VAR. Keep the first occurrence.
    corr = np.corrcoef(data_mat.T)
    keep_mask = np.ones(len(node_cols), dtype=bool)
    for i in range(len(node_cols)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(node_cols)):
            if keep_mask[j] and abs(corr[i, j]) > 0.9999:
                logger.warning(
                    "Dropping near-duplicate node '%s' (|r|=%.6f with '%s')",
                    node_cols[j], corr[i, j], node_cols[i],
                )
                keep_mask[j] = False

    node_cols = [c for c, k in zip(node_cols, keep_mask) if k]
    data = data_mat[:, keep_mask]

    if len(node_cols) < 2:
        raise SystemExit("After deduplication fewer than 2 non-collinear nodes remain.")

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
