"""Create descriptive event-time maps from the gold panel."""

from __future__ import annotations

import argparse

import polars as pl

from stressnet.config import gold_root, results_root
from stressnet.plotting.event_maps import plot_event_time_map
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create event-time descriptive maps.")
    parser.add_argument("--event", required=True)
    args = parser.parse_args()

    panel_path = gold_root() / f"dataset_contagion_features_{args.event}.parquet"
    if not panel_path.exists():
        raise SystemExit(f"Panel not found: {panel_path}. Run script 03 first.")

    panel = pl.read_parquet(panel_path)
    out_dir = results_root() / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = [
        ("basis_vs_usd", f"figure_basis_by_node_{args.event}.png", "Basis by node"),
        ("spread_bps", f"figure_liquidity_by_node_{args.event}.png", "CEX spread by node"),
        ("exchange_netflow_1h", f"figure_flows_{args.event}.png", "On-chain exchange netflows"),
    ]
    for feature, name, title in figures:
        if feature not in panel.columns or panel[feature].null_count() == panel.height:
            logger.warning("Skipping %s; no non-null values.", feature)
            continue
        plot_event_time_map(
            panel.filter(pl.col(feature).is_not_null()),
            feature_col=feature,
            output_path=out_dir / name,
            title=f"{title}: {args.event}",
        )

    plot_event_time_map(
        panel.filter(pl.col("basis_vs_usd").is_not_null()),
        feature_col="basis_vs_usd",
        output_path=out_dir / f"figure_event_time_map_{args.event}.png",
        title=f"Event-time map: {args.event}",
    )


if __name__ == "__main__":
    main()
