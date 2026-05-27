"""Contagion network visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def plot_contagion_map(
    G: nx.DiGraph,
    node_roles: dict[str, str] | None = None,
    output_path: Path | None = None,
    title: str = "Stablecoin Contagion Network",
) -> None:
    """Draw a directed contagion network with node-role colour coding.

    Node colours:
        originator → red
        amplifier  → orange
        sink       → blue
        mixed      → grey
        isolated   → lightgrey

    Edge thickness is proportional to edge weight.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; cannot plot network.")
        return

    colour_map = {
        "originator": "#e74c3c",
        "amplifier": "#e67e22",
        "sink": "#2980b9",
        "mixed": "#95a5a6",
        "isolated": "#bdc3c7",
    }

    if node_roles is None:
        node_roles = {n: "mixed" for n in G.nodes}

    colours = [colour_map.get(node_roles.get(n, "mixed"), "#95a5a6") for n in G.nodes]
    weights = [G[u][v].get("weight", 0.5) for u, v in G.edges]
    edge_widths = [max(0.5, w * 5) for w in weights]

    pos = nx.spring_layout(G, seed=42, k=2.0)

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_color=colours, node_size=800, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, arrowsize=20,
                           edge_color="#555555", ax=ax, connectionstyle="arc3,rad=0.1")

    ax.set_title(title, fontsize=13)
    ax.axis("off")

    # Legend
    for role, colour in colour_map.items():
        ax.scatter([], [], c=colour, label=role, s=100)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved contagion map → %s", output_path)
    else:
        plt.show()
    plt.close()
