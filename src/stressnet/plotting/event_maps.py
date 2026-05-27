"""Event-time stress maps: basis, spread, depth, imbalance by node."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def plot_event_time_map(
    panel: pl.DataFrame,
    feature_col: str = "basis_vs_usd",
    node_col: str = "node_id",
    ts_col: str = "event_time_seconds",
    output_path: Path | None = None,
    title: str | None = None,
) -> None:
    """Plot time series of feature_col for all nodes on a single event-time axis."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available.")
        return

    nodes = panel[node_col].unique().sort().to_list()
    fig, ax = plt.subplots(figsize=(14, 6))

    for node in nodes:
        node_df = panel.filter(pl.col(node_col) == node).sort(ts_col)
        ts = node_df[ts_col].to_numpy()
        vals = node_df[feature_col].to_numpy()
        ax.plot(ts / 3600, vals * 10_000, label=node, linewidth=1.2)

    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="T=0 (shock onset)")
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Event time (hours)", fontsize=11)
    ax.set_ylabel(f"{feature_col} (bps)", fontsize=11)
    ax.set_title(title or f"Event-time map: {feature_col}", fontsize=12)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved event map → %s", output_path)
    else:
        plt.show()
    plt.close()
