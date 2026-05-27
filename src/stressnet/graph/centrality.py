"""Network centrality measures and node-role classification."""

from __future__ import annotations

import networkx as nx
import polars as pl


def compute_centrality(G: nx.DiGraph) -> pl.DataFrame:
    """Compute a suite of centrality measures for all nodes in G.

    Returns a DataFrame with one row per node and columns:
        node_id, out_degree_w, in_degree_w, eigenvector, betweenness, role
    """
    if len(G) == 0:
        return pl.DataFrame(schema={
            "node_id": pl.Utf8,
            "out_degree_w": pl.Float64,
            "in_degree_w": pl.Float64,
            "eigenvector": pl.Float64,
            "betweenness": pl.Float64,
            "role": pl.Utf8,
        })

    out_deg = dict(G.out_degree(weight="weight"))
    in_deg = dict(G.in_degree(weight="weight"))

    try:
        eig = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eig = {n: 0.0 for n in G.nodes}

    between = nx.betweenness_centrality(G, weight="weight", normalized=True)

    rows = []
    for node in G.nodes:
        od = float(out_deg.get(node, 0))
        id_ = float(in_deg.get(node, 0))
        role = _classify_role(od, id_, float(between.get(node, 0)))
        rows.append({
            "node_id": node,
            "out_degree_w": od,
            "in_degree_w": id_,
            "eigenvector": float(eig.get(node, 0)),
            "betweenness": float(between.get(node, 0)),
            "role": role,
        })

    return pl.DataFrame(rows).sort("out_degree_w", descending=True)


def _classify_role(out_deg: float, in_deg: float, betweenness: float) -> str:
    """Classify a node as originator, amplifier, or sink."""
    total = out_deg + in_deg
    if total == 0:
        return "isolated"
    out_share = out_deg / total
    if out_share > 0.7:
        return "originator"
    if out_share < 0.3:
        return "sink"
    if betweenness > 0.1:
        return "amplifier"
    return "mixed"


def detect_communities(G: nx.DiGraph) -> dict[str, int]:
    """Detect communities using Louvain on the undirected projection.

    Returns a dict mapping node_id → community index.
    """
    try:
        import community as community_louvain
        G_undirected = G.to_undirected()
        partition = community_louvain.best_partition(G_undirected, weight="weight")
        return partition
    except ImportError:
        # Fall back to greedy modularity communities
        G_undirected = G.to_undirected()
        communities = nx.community.greedy_modularity_communities(G_undirected, weight="weight")
        return {node: i for i, community in enumerate(communities) for node in community}
