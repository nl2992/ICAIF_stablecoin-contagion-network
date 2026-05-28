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
    parser.add_argument("--bucket-seconds", type=int, default=3600)
    parser.add_argument("--min-observations", type=int, default=50)
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

    # Resample to a configurable bucket so all nodes land on a common grid.
    # Exclude fixture nodes: synthetic data is
    # meaningless for Granger causality and inflates node count with
    # near-perfectly correlated artificial series.
    if args.bucket_seconds <= 0:
        raise SystemExit("--bucket-seconds must be positive.")
    sub = (
        panel
        .filter(pl.col("node_id").is_in(node_ids))
        .filter(pl.col("tier_actual") != "fixture_non_empirical")
        .select(["node_id", ts_col, args.feature_col])
        .with_columns(
            ((pl.col(ts_col) / args.bucket_seconds).floor().cast(pl.Int64) * args.bucket_seconds)
            .alias("time_bucket")
        )
        .group_by(["node_id", "time_bucket"])
        .agg(pl.col(args.feature_col).mean().alias(args.feature_col))
        .sort("time_bucket")
    )

    pivot = (
        sub
        .pivot(values=args.feature_col, index="time_bucket", on="node_id")
        .sort("time_bucket")
    )

    # Keep only nodes with at least 10% non-null coverage OR ≥ 48 hours of data.
    # A low threshold lets in short-window nodes (e.g. UST delisted mid-crisis)
    # while still excluding all-fixture null columns.
    n_rows = pivot.height
    _MIN_ROWS = 48  # at least 48 h of real data
    node_cols = [
        c for c in pivot.columns
        if c != "hour_bucket"
        and c in node_ids
        and (pivot[c].drop_nulls().len() >= _MIN_ROWS
             or pivot[c].null_count() < n_rows * 0.9)
    ]

    if len(node_cols) < 2:
        logger.warning(
            "Fewer than 2 real nodes have sufficient '%s' coverage for event '%s'. "
            "VAR/Granger skipped (need ≥2 non-fixture nodes). "
            "Try --feature-col with another column, or add real data for this event.",
            args.feature_col, args.event,
        )
        return

    # Drop rows where ANY retained node is null — this naturally clips to the
    # overlapping active trading window (important for delisted tokens).
    # Do NOT forward-fill: propagating a stale price for a delisted asset
    # would contaminate the VAR.
    pivot = pivot.select(["time_bucket"] + node_cols).drop_nulls()

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
        logger.warning("After deduplication fewer than 2 non-collinear nodes remain for '%s'. VAR skipped.", args.event)
        return

    if data.shape[0] < args.min_observations:
        logger.warning(
            "Too few observations (%d) after alignment for '%s'. VAR skipped.",
            data.shape[0],
            args.event,
        )
        return

    logger.info("Fitting VAR on %d observations × %d nodes", *data.shape)
    var_fit = fit_var(data, node_names=node_cols, max_lags=args.max_lags)

    granger = granger_causality_table(var_fit).with_columns(
        pl.lit(args.feature_col).alias("feature_col"),
        pl.lit(args.bucket_seconds).alias("bucket_seconds"),
    )
    fevd = fevd_spillover_table(var_fit, horizon=args.fevd_horizon).with_columns(
        pl.lit(args.feature_col).alias("feature_col"),
        pl.lit(args.bucket_seconds).alias("bucket_seconds"),
    )

    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    granger.write_csv(out_dir / f"table_granger_{args.event}.csv")
    fevd.write_csv(out_dir / f"table_var_spillovers_{args.event}.csv")
    logger.info("Wrote VAR results to %s", out_dir)

    spillover_df = pl.read_csv(out_dir / f"table_var_spillovers_{args.event}.csv")
    if "method" in spillover_df.columns:
        fallback_rows = spillover_df.filter(pl.col("method") == "var_coeff_fallback").height
        if fallback_rows > 0:
            logger.warning(
                "%d / %d VAR spillover rows use coefficient fallback (FEVD failed). "
                "These rows are labelled 'var_coeff_fallback' in the method column.",
                fallback_rows, spillover_df.height,
            )

    # Report with multiple-testing correction
    sig_p05 = granger.filter(pl.col("significant_p05")) if "significant_p05" in granger.columns else granger.filter(pl.col("significant"))
    sig_fdr  = granger.filter(pl.col("significant_fdr"))      if "significant_fdr"      in granger.columns else sig_p05
    sig_bonf = granger.filter(pl.col("significant_bonferroni")) if "significant_bonferroni" in granger.columns else pl.DataFrame()
    logger.info(
        "Significant Granger edges  p<0.05=%d  FDR=%d  Bonferroni=%d  / %d",
        len(sig_p05), len(sig_fdr), len(sig_bonf), len(granger),
    )
    if len(sig_fdr) > 0:
        print(sig_fdr.sort("f_stat", descending=True).head(10))
    elif len(sig_p05) > 0:
        print(sig_p05.sort("f_stat", descending=True).head(10))


if __name__ == "__main__":
    main()
