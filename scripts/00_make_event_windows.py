"""Generate event window tables and node coverage audit.

Outputs:
    results/tables/table_event_windows.csv
    results/tables/table_node_coverage.csv
    results/figures/figure_heatmap_coverage.png
"""

import argparse
from pathlib import Path

import polars as pl

from stressnet.config import load_events, load_nodes, results_root
from stressnet.graph.nodes import load_all_nodes
from stressnet.plotting.coverage import plot_coverage_heatmap
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def build_event_windows_table(events: dict) -> pl.DataFrame:
    rows = []
    for event_id, cfg in events.items():
        rows.append({
            "event_id": event_id,
            "name": cfg.get("name"),
            "mechanism": cfg.get("mechanism"),
            "core_start": cfg.get("core_window_utc", [None, None])[0],
            "core_end": cfg.get("core_window_utc", [None, None])[1],
            "analysis_start": cfg.get("analysis_window_utc", [None, None])[0],
            "analysis_end": cfg.get("analysis_window_utc", [None, None])[1],
            "shock_onset_utc": cfg.get("shock_onset_utc"),
            "data_quality": cfg.get("data_quality"),
            "primary": cfg.get("primary", False),
        })
    return pl.DataFrame(rows)


def build_node_coverage_table(events: dict) -> pl.DataFrame:
    nodes = load_all_nodes()
    rows = []
    for node in nodes:
        for event_id in events.keys():
            covered = event_id in node.events_covered
            rows.append({
                "node_id": node.id,
                "layer": node.layer,
                "asset": node.asset,
                "venue": node.venue,
                "event_id": event_id,
                "covered": covered,
                "tier_nominal": node.tier if covered else "missing",
                "tier_actual": node.tier if covered else "missing",
            })
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate event windows and coverage audit.")
    parser.add_argument("--coverage-audit", action="store_true",
                        help="Also attempt to verify data availability from local files.")
    args = parser.parse_args()

    events = load_events()
    tables_dir = results_root() / "tables"
    figures_dir = results_root() / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Event windows table
    event_table = build_event_windows_table(events)
    event_table.write_csv(tables_dir / "table_event_windows.csv")
    logger.info("Wrote table_event_windows.csv (%d rows)", len(event_table))

    # Node coverage table
    coverage = build_node_coverage_table(events)
    coverage.write_csv(tables_dir / "table_node_coverage.csv")
    logger.info("Wrote table_node_coverage.csv (%d rows)", len(coverage))

    # Coverage heatmap
    covered_only = coverage.filter(pl.col("covered"))
    plot_coverage_heatmap(
        covered_only,
        output_path=figures_dir / "figure_heatmap_coverage.png",
    )


if __name__ == "__main__":
    main()
