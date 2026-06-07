"""Compute temporal network centrality from claim-gated edge tables.

Reads:
    results/paper/tables/table_aa_paper_claimable_edges.csv  (A/A edges)
    results/paper/tables/table_provenance_claimable_edges.csv (A/B edges)

Writes:
    results/paper/tables/table_centrality_by_event.csv

Computes four centrality measures using NetworkX on the directed weighted
graph built from claim-gated edges:

    in_degree    — receiver strength (weighted in-degree)
    out_degree   — transmitter strength (weighted out-degree)
    betweenness  — amplifier score (node bridges flow between clusters)
    eigenvector  — systemic importance (high-centrality neighbours)

Also generates a heatmap figure at:
    results/paper/figures/fig_centrality_heatmap.pdf

Usage:
    python scripts/08b_run_centrality.py [--event EVENT]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from stressnet.config import results_root
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_TABLES = results_root() / "paper" / "tables"
_FIGURES = results_root() / "paper" / "figures"

_AA_PATH = _TABLES / "table_aa_paper_claimable_edges.csv"
_AB_PATH = _TABLES / "table_provenance_claimable_edges.csv"


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        logger.warning("Edge table not found: %s", path)
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _role_label(in_d: float, out_d: float, between: float) -> str:
    """Classify node role from centrality scores."""
    if out_d > in_d and out_d > 0.3:
        return "transmitter"
    if in_d > out_d and in_d > 0.3:
        return "receiver"
    if between > 0.2:
        return "amplifier"
    return "peripheral"


def compute_centrality(event_filter: str | None = None) -> list[dict[str, Any]]:
    """Build directed graph from claim-gated edges and compute centralities."""
    try:
        import networkx as nx
    except ImportError:
        logger.error(
            "networkx is required for centrality computation.  "
            "Install with: pip install networkx"
        )
        return []

    aa_rows = _load_csv(_AA_PATH)
    ab_rows = _load_csv(_AB_PATH)
    all_edges = aa_rows + ab_rows

    if not all_edges:
        logger.warning("No edge rows found in claim tables.")
        return []

    # Collect unique events
    events = sorted({r.get("event_id", "") for r in all_edges if r.get("event_id")})
    if event_filter:
        events = [e for e in events if e == event_filter]

    output_rows: list[dict[str, Any]] = []

    for event_id in events:
        evt_edges = [r for r in all_edges if r.get("event_id") == event_id]
        if not evt_edges:
            continue

        G = nx.DiGraph()
        for row in evt_edges:
            src   = row.get("source_node", row.get("node_i", ""))
            tgt   = row.get("target_node", row.get("node_j", ""))
            # Use absolute peak correlation as edge weight (proxy for strength)
            try:
                weight = abs(float(row.get("peak_corr", row.get("weight", 1.0)) or 1.0))
            except (ValueError, TypeError):
                weight = 1.0
            if src and tgt:
                G.add_edge(src, tgt, weight=weight)

        if G.number_of_nodes() < 2:
            logger.info("Event %s: fewer than 2 nodes in graph; skipping.", event_id)
            continue

        # Compute centrality measures
        in_deg   = nx.in_degree_centrality(G)
        out_deg  = nx.out_degree_centrality(G)
        between  = nx.betweenness_centrality(G, weight="weight", normalized=True)
        try:
            eigvec = nx.eigenvector_centrality(G, weight="weight", max_iter=500)
        except nx.PowerIterationFailedConvergence:
            logger.warning("Eigenvector centrality did not converge for %s; using zeros.", event_id)
            eigvec = {n: 0.0 for n in G.nodes}

        for node in G.nodes:
            i_d = in_deg.get(node, 0.0)
            o_d = out_deg.get(node, 0.0)
            b   = between.get(node, 0.0)
            output_rows.append({
                "event_id":           event_id,
                "node":               node,
                "in_degree":          round(i_d,  4),
                "out_degree":         round(o_d,  4),
                "betweenness":        round(b,    4),
                "eigenvector":        round(eigvec.get(node, 0.0), 4),
                "role_label":         _role_label(i_d, o_d, b),
            })

        logger.info(
            "Event %s: %d nodes, %d edges in claim graph",
            event_id, G.number_of_nodes(), G.number_of_edges(),
        )

    return output_rows


def make_heatmap(rows: list[dict[str, Any]]) -> None:
    """Generate centrality heatmap saved to results/paper/figures/."""
    if not rows:
        logger.info("No centrality rows; skipping heatmap.")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import polars as pl
    except ImportError:
        logger.warning("matplotlib/polars not available; skipping heatmap.")
        return

    df = pl.DataFrame(rows)
    events = df["event_id"].unique().to_list()

    fig, axes = plt.subplots(1, len(events), figsize=(4 * len(events), 6), squeeze=False)
    metrics = ["in_degree", "out_degree", "betweenness", "eigenvector"]

    for ax, event_id in zip(axes[0], events):
        evt_df = df.filter(pl.col("event_id") == event_id).sort("node")
        nodes   = evt_df["node"].to_list()
        values  = [[evt_df[m].to_list()[i] for m in metrics] for i in range(len(nodes))]

        im = ax.imshow(values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_title(event_id.replace("_", " "), fontsize=9)
        ax.set_yticks(range(len(nodes)))
        ax.set_yticklabels(nodes, fontsize=7)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=7)

    plt.suptitle("Network Centrality by Event (claim-gated edges)", fontsize=11)
    plt.colorbar(im, ax=axes[0], orientation="vertical", fraction=0.05, label="score (0–1)")
    plt.tight_layout()

    _FIGURES.mkdir(parents=True, exist_ok=True)
    out_path = _FIGURES / "fig_centrality_heatmap.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logger.info("Wrote centrality heatmap → %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute temporal network centrality from claim-gated edges."
    )
    parser.add_argument("--event", default=None, help="Filter to one event ID.")
    args = parser.parse_args()

    rows = compute_centrality(event_filter=args.event)
    if not rows:
        logger.warning("No centrality results produced.")
        return

    _TABLES.mkdir(parents=True, exist_ok=True)
    out_path = _TABLES / "table_centrality_by_event.csv"
    fieldnames = ["event_id", "node", "in_degree", "out_degree",
                  "betweenness", "eigenvector", "role_label"]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows → %s", len(rows), out_path)

    make_heatmap(rows)


if __name__ == "__main__":
    main()
