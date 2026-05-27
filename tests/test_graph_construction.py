"""Tests for graph node loading and network construction."""

import pytest
from stressnet.graph.nodes import load_all_nodes, nodes_for_event
from stressnet.graph.edges import build_networkx_graph, edges_from_table, Edge
from stressnet.graph.centrality import compute_centrality

import polars as pl


def test_all_nodes_load():
    nodes = load_all_nodes()
    assert len(nodes) > 0


def test_nodes_have_ids():
    nodes = load_all_nodes()
    for node in nodes:
        assert node.id, f"Node missing id: {node}"


def test_nodes_for_event_usdc_svb():
    nodes = nodes_for_event("usdc_svb_2023")
    assert len(nodes) >= 4  # at least CEX nodes + Curve 3pool


def test_build_graph_from_edges():
    edges = [
        Edge("usdc_coinbase", "curve_3pool", weight=0.5, method="leadlag"),
        Edge("curve_3pool", "usdt_binance", weight=0.3, method="hawkes"),
    ]
    G = build_networkx_graph(edges)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2


def test_build_graph_weight_threshold():
    edges = [
        Edge("a", "b", weight=0.5, method="leadlag"),
        Edge("b", "c", weight=0.01, method="leadlag"),
    ]
    G = build_networkx_graph(edges, weight_threshold=0.1)
    assert G.number_of_edges() == 1


def test_centrality_empty_graph():
    import networkx as nx
    result = compute_centrality(nx.DiGraph())
    assert result.height == 0


def test_centrality_simple_graph():
    import networkx as nx
    G = nx.DiGraph()
    G.add_edge("A", "B", weight=0.8)
    G.add_edge("A", "C", weight=0.6)
    G.add_edge("B", "C", weight=0.3)
    result = compute_centrality(G)
    assert result.height == 3
    assert "originator" in result["role"].to_list() or "mixed" in result["role"].to_list()


def test_edges_from_table():
    df = pl.DataFrame({
        "node_i": ["A", "B"],
        "node_j": ["B", "C"],
        "branching_ratio_ij": [0.5, 0.2],
        "method": ["hawkes", "hawkes"],
    })
    edges = edges_from_table(df, weight_col="branching_ratio_ij")
    assert len(edges) == 2
    assert edges[0].source == "A"
    assert edges[0].weight == pytest.approx(0.5)
