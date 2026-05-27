"""Coverage heatmap: data availability by node and event."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def plot_coverage_heatmap(
    coverage_df: pl.DataFrame,
    node_col: str = "node_id",
    event_col: str = "event_id",
    tier_col: str = "tier_actual",
    output_path: Path | None = None,
    title: str = "Data Coverage by Node and Event",
) -> None:
    """Draw a heatmap showing data tier for each (node, event) combination.

    Tier colour mapping:
        A → green
        B → yellow
        C → orange
        missing → lightgrey
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available; cannot plot coverage.")
        return

    tier_colours = {"A": "#2ecc71", "B": "#f39c12", "C": "#e67e22", "missing": "#ecf0f1"}

    nodes = coverage_df[node_col].unique().sort().to_list()
    events = coverage_df[event_col].unique().sort().to_list()

    tier_lookup = {
        (row[node_col], row[event_col]): row[tier_col]
        for row in coverage_df.iter_rows(named=True)
    }

    fig, ax = plt.subplots(figsize=(len(events) * 1.5 + 2, len(nodes) * 0.5 + 2))

    for i, node in enumerate(nodes):
        for j, event in enumerate(events):
            tier = tier_lookup.get((node, event), "missing")
            colour = tier_colours.get(tier, "#ecf0f1")
            rect = plt.Rectangle([j, i], 1, 1, color=colour)
            ax.add_patch(rect)
            ax.text(j + 0.5, i + 0.5, tier, ha="center", va="center",
                    fontsize=9, fontweight="bold")

    ax.set_xlim(0, len(events))
    ax.set_ylim(0, len(nodes))
    ax.set_xticks([j + 0.5 for j in range(len(events))])
    ax.set_xticklabels(events, rotation=30, ha="right", fontsize=9)
    ax.set_yticks([i + 0.5 for i in range(len(nodes))])
    ax.set_yticklabels(nodes, fontsize=9)
    ax.set_title(title, fontsize=12)

    patches = [mpatches.Patch(color=c, label=f"Tier {t}") for t, c in tier_colours.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=8)

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved coverage heatmap → %s", output_path)
    else:
        plt.show()
    plt.close()
