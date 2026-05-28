"""Estimate multivariate Hawkes excitation matrix.

Writes:
    results/tables/table_hawkes_params_{event}.csv
"""

import argparse

import polars as pl

from stressnet.config import gold_root, results_root
from stressnet.graph.nodes import nodes_for_event
from stressnet.models.hawkes import define_stress_events, fit_hawkes, hawkes_bootstrap_ci, hawkes_results_table
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate Hawkes excitation.")
    parser.add_argument("--event", required=True)
    parser.add_argument("--threshold-bps", type=float, default=10.0)
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument("--n-bootstraps", type=int, default=100,
                        help="Number of bootstrap replicates for branching-ratio CIs (default: 100).")
    args = parser.parse_args()

    panel_path = gold_root() / f"dataset_contagion_features_{args.event}.parquet"
    if not panel_path.exists():
        raise SystemExit(f"Panel not found: {panel_path}. Run script 03 first.")

    panel = pl.read_parquet(panel_path)
    nodes = nodes_for_event(args.event)
    node_ids = [n.id for n in nodes if n.id in panel["node_id"].unique().to_list()]

    logger.info("Extracting stress events (threshold=%.0fbps) for %d nodes", args.threshold_bps, len(node_ids))
    events = define_stress_events(panel, node_ids, threshold_bps=args.threshold_bps)

    n_total = sum(len(v) for v in events.values())
    logger.info("Total stress event arrivals: %d", n_total)
    if n_total < 10:
        logger.warning("Very few events; Hawkes estimates will be unreliable.")

    fit = fit_hawkes(events, decay=args.decay)
    ci = hawkes_bootstrap_ci(events, fit, n_bootstraps=args.n_bootstraps)
    table = hawkes_results_table(fit, ci=ci)

    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"table_hawkes_params_{args.event}.csv"
    table.write_csv(out_path)
    logger.info("Wrote %s (%d edges)", out_path, len(table))

    contagious = table.filter(pl.col("contagious"))
    logger.info("Contagious edges (n_ij > 0.1): %d / %d", len(contagious), len(table))
    if "ci_excludes_zero" in table.columns:
        ci_confirmed = table.filter(pl.col("ci_excludes_zero") == True)
        logger.info(
            "CI-confirmed contagious edges (ci_lower > 0): %d / %d",
            len(ci_confirmed), len(table),
        )
    if len(contagious) > 0:
        print(contagious.sort("branching_ratio_ij", descending=True).head(10))


if __name__ == "__main__":
    main()
