"""Temporal edge-stream dataset for TGN/DySAT training.

Builds a sequence of (src_node, dst_node, timestamp, edge_features, node_features, label)
tuples suitable for temporal GNN training.

This is a scaffold — it defines the interface and data contract, but the
actual edge-stream construction requires the TE/lead-lag result tables
as the edge index source.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from stressnet.config import gold_root, results_root
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TemporalEdge:
    """One directed temporal interaction edge."""
    src: int          # node index
    dst: int          # node index
    timestamp: float  # event_time_seconds
    edge_features: list[float]
    label: int        # 0/1 downstream stress


def build_temporal_dataset(
    event_id: str,
    feature_cols: list[str] | None = None,
    label_col: str = "label_downstream_gt10bps_1m",
    edge_source: str = "te",  # "te" or "leadlag"
) -> tuple[list[TemporalEdge], dict[str, int], list[str]]:
    """Build temporal edge stream from panel + TE/lead-lag edge tables.

    Returns:
        edges: list of TemporalEdge sorted by timestamp
        node_to_idx: mapping from node_id to integer index
        feature_names: list of feature column names used
    """
    panel_path = gold_root() / f"dataset_contagion_features_{event_id}.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel not found: {panel_path}")

    panel = pl.read_parquet(panel_path)

    # Load TE edge table to determine directed graph structure
    te_path = results_root() / "tables" / f"table_transfer_entropy_{event_id}.csv"
    if not te_path.exists():
        raise FileNotFoundError(
            f"TE table not found: {te_path}. Run script 07 first to define edges."
        )

    te_edges = pl.read_csv(te_path)
    if "claim_allowed" in te_edges.columns:
        te_edges = te_edges.filter(pl.col("claim_allowed"))

    # Build node index
    all_nodes = sorted(set(
        te_edges["node_i"].to_list() + te_edges["node_j"].to_list()
    ))
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}

    # Default feature columns
    if feature_cols is None:
        feature_cols = [
            c for c in [
                "basis_vs_usd", "spread_bps", "orderbook_imbalance",
                "reserve_imbalance", "exchange_netflow_1h",
            ]
            if c in panel.columns
        ]

    edges: list[TemporalEdge] = []
    for te_row in te_edges.iter_rows(named=True):
        src_id = te_row["node_i"]
        dst_id = te_row["node_j"]
        if src_id not in node_to_idx or dst_id not in node_to_idx:
            continue
        src_idx = node_to_idx[src_id]
        dst_idx = node_to_idx[dst_id]

        # One temporal edge per timestep where both nodes have data
        src_panel = panel.filter(pl.col("node_id") == src_id).sort("event_time_seconds")
        for row in src_panel.iter_rows(named=True):
            ts = row.get("event_time_seconds")
            if ts is None:
                continue
            feat = [float(row.get(c) or 0.0) for c in feature_cols]
            label = int(row.get(label_col) or 0)
            edges.append(TemporalEdge(src_idx, dst_idx, ts, feat, label))

    edges.sort(key=lambda e: e.timestamp)
    logger.info(
        "Built temporal dataset for %s: %d edges, %d nodes, %d features",
        event_id, len(edges), len(node_to_idx), len(feature_cols),
    )
    return edges, node_to_idx, feature_cols
