"""Run robustness checks for one event.

Checks performed:
  1. Alternative sampling grids (1s / 5s / 60s)
  2. CEX-only node subsample (exclude DEX and flow nodes)
  3. Without dominant venue (Binance removal)
  4. Bootstrap block-size sensitivity (50 / 300 / 1800 rows)
  5. Alternative basis thresholds (10bps / 25bps / 50bps)
  6. Event-phase analysis (pre / peak / post shock)
  7. Placebo window check (same calendar window, prior year)

Writes:
    results/tables/table_robustness_{event}.csv   – all checks stacked
"""

from __future__ import annotations

import argparse
import itertools

import polars as pl

from stressnet.config import gold_root, load_events, results_root
from stressnet.evaluation.placebo import build_placebo_windows, tag_placebo_rows
from stressnet.evaluation.robustness import (
    run_grid_robustness,
    subsample_cex_only,
    subsample_without_dominant,
)
from stressnet.models.leadlag import compute_leadlag_table
from stressnet.graph.nodes import nodes_for_event
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def _run_leadlag(
    panel: pl.DataFrame,
    node_pairs: list[tuple[str, str]],
    feature_col: str,
    block_size: int = 300,
) -> pl.DataFrame:
    return compute_leadlag_table(
        panel,
        node_pairs=node_pairs,
        feature_col=feature_col,
        block_size=block_size,
        n_reps=200,  # reduced for speed in robustness sweeps
    )


def _phase_panels(panel: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Split panel into pre / peak / post phases by event_time_seconds terciles."""
    ts = panel["event_time_seconds"]
    t_min, t_max = float(ts.min()), float(ts.max())
    span = t_max - t_min
    cut1 = t_min + span / 3
    cut2 = t_min + 2 * span / 3
    return {
        "phase_pre": panel.filter(pl.col("event_time_seconds") < cut1),
        "phase_peak": panel.filter(
            (pl.col("event_time_seconds") >= cut1)
            & (pl.col("event_time_seconds") < cut2)
        ),
        "phase_post": panel.filter(pl.col("event_time_seconds") >= cut2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robustness checks.")
    parser.add_argument("--event", required=True)
    parser.add_argument("--feature-col", default="basis_vs_usd")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["grids", "node_subset", "block_size", "threshold", "phase", "placebo"],
        help="Checks to skip (useful for quick runs).",
    )
    args = parser.parse_args()

    panel_path = gold_root() / f"dataset_contagion_features_{args.event}.parquet"
    if not panel_path.exists():
        raise SystemExit(f"Panel not found: {panel_path}. Run script 03 first.")

    panel = pl.read_parquet(panel_path)
    nodes = nodes_for_event(args.event)
    node_ids = [n.id for n in nodes if n.id in panel["node_id"].unique().to_list()]
    node_pairs = list(itertools.permutations(node_ids, 2))
    feature_col = args.feature_col

    def _tagged(df: pl.DataFrame, check: str) -> pl.DataFrame:
        return df.with_columns(
            pl.lit(check).alias("check"),
            pl.lit(args.event).alias("event_id"),
        )

    all_checks: list[pl.DataFrame] = []
    skip = set(args.skip)

    # --- 1. Baseline (60s grid) ---
    logger.info("Baseline check (60s grid)...")
    df = _run_leadlag(panel, node_pairs, feature_col)
    all_checks.append(_tagged(df, "baseline_60s"))

    # --- 2. Alternative sampling grids ---
    if "grids" not in skip:
        logger.info("Grid robustness checks (1s / 5s / 60s)...")
        grid_results = run_grid_robustness(
            panel,
            lambda p, g: _run_leadlag(p, node_pairs, feature_col),
            grids=[1, 5, 60],
        )
        for grid, df in grid_results.items():
            all_checks.append(_tagged(df, f"grid_{grid}s"))

    # --- 3. Node subsamples ---
    if "node_subset" not in skip:
        logger.info("CEX-only subsample...")
        cex_panel = subsample_cex_only(panel)
        if cex_panel.height > 0:
            cex_pairs = [
                p for p in node_pairs
                if p[0] in cex_panel["node_id"].unique().to_list()
                and p[1] in cex_panel["node_id"].unique().to_list()
            ]
            if cex_pairs:
                all_checks.append(_tagged(_run_leadlag(cex_panel, cex_pairs, feature_col), "cex_only"))

        logger.info("Without Binance...")
        no_binance = subsample_without_dominant(panel, dominant_node="usdt_binance")
        nb_pairs = [
            p for p in node_pairs
            if p[0] in no_binance["node_id"].unique().to_list()
            and p[1] in no_binance["node_id"].unique().to_list()
        ]
        if nb_pairs:
            all_checks.append(_tagged(_run_leadlag(no_binance, nb_pairs, feature_col), "no_binance"))

    # --- 4. Bootstrap block-size sensitivity ---
    if "block_size" not in skip:
        for bs in (50, 300, 1800):
            logger.info("Block size = %d...", bs)
            df = _run_leadlag(panel, node_pairs, feature_col, block_size=bs)
            all_checks.append(_tagged(df, f"block_{bs}"))

    # --- 5. Alternative basis thresholds ---
    if "threshold" not in skip and "basis_vs_usd" in panel.columns:
        for thresh_bps in (10, 25, 50):
            logger.info("Threshold = %d bps...", thresh_bps)
            threshold_val = thresh_bps / 10_000
            sub = panel.with_columns(
                pl.when(pl.col("basis_vs_usd").abs() > threshold_val)
                .then(pl.col("basis_vs_usd"))
                .otherwise(0.0)
                .alias("basis_vs_usd")
            )
            all_checks.append(_tagged(_run_leadlag(sub, node_pairs, feature_col), f"thresh_{thresh_bps}bps"))

    # --- 6. Event-phase analysis ---
    if "phase" not in skip:
        logger.info("Event-phase analysis (pre/peak/post)...")
        for phase_name, phase_panel in _phase_panels(panel).items():
            if phase_panel.height < 20:
                continue
            ph_pairs = [
                p for p in node_pairs
                if p[0] in phase_panel["node_id"].unique().to_list()
                and p[1] in phase_panel["node_id"].unique().to_list()
            ]
            if ph_pairs:
                all_checks.append(_tagged(_run_leadlag(phase_panel, ph_pairs, feature_col), phase_name))

    # --- 7. Placebo windows ---
    if "placebo" not in skip:
        logger.info("Placebo window check...")
        try:
            events_cfg = load_events()
            placebo_windows = build_placebo_windows(events_cfg.get(args.event, {}))
            if placebo_windows:
                placebo_panel = tag_placebo_rows(panel, placebo_windows)
                placebo_sub = placebo_panel.filter(pl.col("is_placebo") == True)
                if placebo_sub.height > 20:
                    pl_pairs = [
                        p for p in node_pairs
                        if p[0] in placebo_sub["node_id"].unique().to_list()
                        and p[1] in placebo_sub["node_id"].unique().to_list()
                    ]
                    if pl_pairs:
                        all_checks.append(_tagged(_run_leadlag(placebo_sub, pl_pairs, feature_col), "placebo"))
        except Exception as exc:
            logger.warning("Placebo check skipped: %s", exc)

    # --- Write ---
    if not all_checks:
        logger.warning("No robustness checks produced output.")
        return

    combined = pl.concat(all_checks, how="diagonal")
    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"table_robustness_{args.event}.csv"
    combined.write_csv(out_path)
    logger.info("Wrote robustness table: %s (%d rows)", out_path.name, combined.height)

    # Summary: count of significant edges per check
    if "significant_p01" in combined.columns:
        summary = (
            combined.group_by("check")
            .agg(
                pl.col("significant_p01").sum().alias("n_significant"),
                pl.col("significant_p01").count().alias("n_total"),
            )
            .sort("check")
        )
        print(summary)


if __name__ == "__main__":
    main()
