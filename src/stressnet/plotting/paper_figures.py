"""Generate all paper-ready figures and tables."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from stressnet.config import results_root
from stressnet.plotting.coverage import plot_coverage_heatmap
from stressnet.plotting.event_maps import plot_event_time_map
from stressnet.plotting.networks import plot_contagion_map
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def make_all_figures(
    panel: pl.DataFrame,
    coverage_df: pl.DataFrame,
    network_data: dict,
    event_id: str,
    output_dir: Path | None = None,
) -> None:
    """Generate all paper figures for a given event.

    Args:
        panel: Gold-layer feature panel.
        coverage_df: Node × event coverage table.
        network_data: Dict with 'graph' and 'node_roles' keys.
        event_id: Event identifier string.
        output_dir: Directory to write figures; defaults to results/figures/.
    """
    if output_dir is None:
        output_dir = results_root() / "figures"

    # Coverage heatmap
    plot_coverage_heatmap(
        coverage_df,
        output_path=output_dir / "figure_heatmap_coverage.png",
        title="Data Coverage by Node and Event",
    )

    # Event-time map
    plot_event_time_map(
        panel.filter(pl.col("event_id") == event_id),
        feature_col="basis_vs_usd",
        output_path=output_dir / f"figure_event_time_map_{event_id}.png",
        title=f"Event-time basis map: {event_id}",
    )

    # Contagion network
    if "graph" in network_data and network_data["graph"] is not None:
        plot_contagion_map(
            network_data["graph"],
            node_roles=network_data.get("node_roles"),
            output_path=output_dir / f"figure_contagion_map_{event_id}.png",
            title=f"Contagion network: {event_id}",
        )

    logger.info("All figures written to %s", output_dir)
