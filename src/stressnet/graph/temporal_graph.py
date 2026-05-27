"""Temporal graph snapshot construction from event-time feature panels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import polars as pl


@dataclass
class TemporalSnapshot:
    """Graph snapshot at a single event-time window."""

    t_start: float        # event_time_seconds lower bound
    t_end: float          # event_time_seconds upper bound
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    node_features: dict[str, dict[str, float]] = field(default_factory=dict)


def build_temporal_snapshots(
    panel: pl.DataFrame,
    edge_estimates: pl.DataFrame,
    window_size_seconds: float = 60.0,
    step_size_seconds: float = 60.0,
    ts_col: str = "event_time_seconds",
) -> list[TemporalSnapshot]:
    """Build a sequence of temporal graph snapshots from a feature panel.

    Each snapshot covers a window of event time. Node features are the mean
    values of all feature columns within that window.

    Args:
        panel: Gold-layer feature panel with node_id and event_time_seconds.
        edge_estimates: DataFrame with source, target, weight columns (pre-estimated edges).
        window_size_seconds: Duration of each snapshot window.
        step_size_seconds: Step between windows.

    Returns:
        List of TemporalSnapshot objects ordered by t_start.
    """
    if ts_col not in panel.columns:
        return []

    t_min = panel[ts_col].min()
    t_max = panel[ts_col].max()
    if t_min is None or t_max is None:
        return []

    feature_cols = [
        c for c in panel.columns
        if c not in {"event_id", "node_id", "wall_clock_utc", ts_col, "tier_nominal", "tier_actual"}
        and panel[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
    ]

    snapshots = []
    t = float(t_min)
    while t < float(t_max):
        t_end = t + window_size_seconds
        window_df = panel.filter(
            (pl.col(ts_col) >= t) & (pl.col(ts_col) < t_end)
        )

        if window_df.height == 0:
            t += step_size_seconds
            continue

        # Mean node features within window
        node_feats = {}
        for row in (
            window_df
            .group_by("node_id")
            .agg([pl.col(c).mean() for c in feature_cols if c in window_df.columns])
            .iter_rows(named=True)
        ):
            node_id = row.pop("node_id")
            node_feats[node_id] = {k: v for k, v in row.items() if v is not None}

        # Build graph for this window
        G = nx.DiGraph()
        G.add_nodes_from(node_feats.keys())
        for edge_row in edge_estimates.iter_rows(named=True):
            src = edge_row.get("source") or edge_row.get("node_i")
            tgt = edge_row.get("target") or edge_row.get("node_j")
            w = float(edge_row.get("weight", 0))
            if src in G and tgt in G:
                G.add_edge(src, tgt, weight=w)

        snapshot = TemporalSnapshot(t_start=t, t_end=t_end, graph=G, node_features=node_feats)
        snapshots.append(snapshot)
        t += step_size_seconds

    return snapshots
