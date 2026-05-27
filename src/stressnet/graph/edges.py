"""Edge construction for the contagion network from estimated directed weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import polars as pl


@dataclass
class Edge:
    """A directed edge in the contagion network."""

    source: str
    target: str
    weight: float
    method: str          # 'leadlag' | 'var_granger' | 'hawkes' | 'transfer_entropy'
    p_value: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    event_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def build_networkx_graph(
    edges: list[Edge],
    weight_threshold: float = 0.0,
    p_threshold: float = 0.05,
) -> nx.DiGraph:
    """Build a directed NetworkX graph from a list of Edge objects.

    Args:
        edges: List of directed Edge objects with weight and optional p_value.
        weight_threshold: Minimum edge weight to include.
        p_threshold: Maximum p-value to include (ignored if p_value is None).

    Returns:
        Directed weighted NetworkX graph.
    """
    G = nx.DiGraph()
    for edge in edges:
        if edge.weight < weight_threshold:
            continue
        if edge.p_value is not None and edge.p_value > p_threshold:
            continue
        G.add_edge(
            edge.source,
            edge.target,
            weight=edge.weight,
            method=edge.method,
            p_value=edge.p_value,
            event_id=edge.event_id,
        )
    return G


def edges_from_table(
    df: pl.DataFrame,
    source_col: str = "node_i",
    target_col: str = "node_j",
    weight_col: str = "weight",
    method_col: str = "method",
    p_col: str | None = "p_value",
    event_id: str | None = None,
) -> list[Edge]:
    """Convert a Polars DataFrame of edge estimates to a list of Edge objects."""
    edges = []
    for row in df.iter_rows(named=True):
        p = row.get(p_col) if p_col and p_col in df.columns else None
        edges.append(Edge(
            source=row[source_col],
            target=row[target_col],
            weight=float(row[weight_col]),
            method=row.get(method_col, "unknown"),
            p_value=float(p) if p is not None else None,
            event_id=event_id,
        ))
    return edges
