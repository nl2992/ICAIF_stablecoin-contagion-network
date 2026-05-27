"""Audit per-event manifest provenance and coverage."""

from __future__ import annotations

import argparse

import polars as pl

from stressnet.config import manifests_root, results_root
from stressnet.graph.nodes import nodes_for_event
from stressnet.plotting.coverage import plot_coverage_heatmap
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit data provenance for one event.")
    parser.add_argument("--event", required=True)
    args = parser.parse_args()

    manifest_path = manifests_root() / f"manifest_{args.event}.csv"
    if manifest_path.exists():
        manifest = pl.read_csv(manifest_path)
    else:
        manifest = pl.DataFrame(schema={"event_id": pl.String, "node_id": pl.String})

    rows = []
    for node in nodes_for_event(args.event):
        node_rows = manifest.filter(pl.col("node_id") == node.id)
        if node_rows.height == 0:
            rows.append(
                {
                    "event_id": args.event,
                    "node_id": node.id,
                    "layer": node.layer,
                    "tier_nominal": node.tier,
                    "tier_actual": "missing",
                    "rows_available": 0,
                    "artefact_count": 0,
                    "stages": "",
                    "sources": "",
                    "missing": True,
                }
            )
            continue

        stages = (
            node_rows["file_stage"].drop_nulls().unique().sort().to_list()
            if "file_stage" in node_rows.columns
            else []
        )
        sources = node_rows["source_name"].drop_nulls().unique().sort().to_list()
        rows.append(
            {
                "event_id": args.event,
                "node_id": node.id,
                "layer": node.layer,
                "tier_nominal": node.tier,
                "tier_actual": node_rows["source_tier_actual"].drop_nulls()[-1],
                "rows_available": int(node_rows["row_count"].fill_null(0).sum()),
                "artefact_count": node_rows["file_path"].n_unique(),
                "stages": ";".join(stages),
                "sources": ";".join(sources),
                "missing": False,
            }
        )

    coverage = pl.DataFrame(rows).sort(["layer", "node_id"])
    tables_dir = results_root() / "tables"
    figures_dir = results_root() / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    table_path = tables_dir / f"table_node_coverage_{args.event}.csv"
    figure_path = figures_dir / f"figure_heatmap_coverage_{args.event}.png"
    coverage.write_csv(table_path)
    plot_coverage_heatmap(
        coverage.rename({"tier_nominal": "source_tier_nominal"}),
        output_path=figure_path,
        title=f"Data Coverage: {args.event}",
    )
    logger.info("Wrote %s and %s", table_path, figure_path)


if __name__ == "__main__":
    main()
