"""Standardized edge schema for the multi-layer contagion network.

All estimation methods (lead-lag, VAR/FEVD, Hawkes, TE, TVP-VAR) emit
ContagionEdge records. Downstream code (build_networkx_graph, plot_contagion_map,
ensemble network) consumes these records uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import polars as pl

# Valid method identifiers produced by the modelling layer
METHOD_NAMES: frozenset[str] = frozenset(
    {"leadlag", "var", "hawkes", "te", "tvp_var", "tvp_var_rolling",
     "tvp_var_ff", "ensemble", "var_abscoef_fallback", "fevd", "unknown"}
)

# Tier ordering: lower index = stronger claim
_TIER_ORDER: dict[str, int] = {
    "A": 0,
    "B": 1,
    "C": 2,
    "fixture_non_empirical": 3,
}


@dataclass
class ContagionEdge:
    """A single directed contagion edge source → target."""

    source: str
    target: str
    weight: float
    method: str
    event_id: str
    p_value: float = 1.0
    tier_source: str = "C"
    tier_target: str = "C"
    lag_seconds: Optional[float] = None
    window_center: Optional[float] = None
    branching_ratio: Optional[float] = None
    fevd_share: Optional[float] = None
    te_value: Optional[float] = None

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise ValueError(f"Self-loop edge not allowed: {self.source}")

    @property
    def tier_edge(self) -> str:
        """Edge provenance tier is the weaker of its two endpoint tiers."""
        src_rank = _TIER_ORDER.get(self.tier_source, 3)
        tgt_rank = _TIER_ORDER.get(self.tier_target, 3)
        worst = max(src_rank, tgt_rank)
        for tier, rank in _TIER_ORDER.items():
            if rank == worst:
                return tier
        return "C"

    @property
    def is_tier_a(self) -> bool:
        return self.tier_edge == "A"

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def edge_table_to_records(
    df: pl.DataFrame,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str = "weight",
    method_col: str = "method",
    p_col: Optional[str] = "p_value",
    event_col: str = "event_id",
) -> list[ContagionEdge]:
    """Convert a Polars DataFrame of edges to ContagionEdge records.

    Handles the many column-name conventions produced by different scripts
    (causing_node/caused_node from VAR; node_i/node_j from lead-lag; etc.).
    """
    records: list[ContagionEdge] = []
    for row in df.iter_rows(named=True):
        src = str(
            row.get(source_col)
            or row.get("causing_node")
            or row.get("node_i")
            or ""
        )
        tgt = str(
            row.get(target_col)
            or row.get("caused_node")
            or row.get("node_j")
            or ""
        )
        if not src or not tgt or src == tgt:
            continue
        try:
            records.append(ContagionEdge(
                source=src,
                target=tgt,
                weight=float(row.get(weight_col) or 0.0),
                method=str(row.get(method_col) or "unknown"),
                event_id=str(row.get(event_col) or ""),
                p_value=float(row.get(p_col) or 1.0) if p_col else 1.0,
            ))
        except (ValueError, TypeError):
            continue
    return records


def records_to_edge_table(records: list[ContagionEdge]) -> pl.DataFrame:
    """Convert ContagionEdge records to a Polars DataFrame."""
    _SCHEMA = {
        "source": pl.String,
        "target": pl.String,
        "weight": pl.Float64,
        "method": pl.String,
        "event_id": pl.String,
        "p_value": pl.Float64,
        "tier_edge": pl.String,
        "lag_seconds": pl.Float64,
        "window_center": pl.Float64,
        "branching_ratio": pl.Float64,
        "fevd_share": pl.Float64,
        "te_value": pl.Float64,
    }
    if not records:
        return pl.DataFrame(schema=_SCHEMA)

    rows = [
        {
            "source": e.source,
            "target": e.target,
            "weight": e.weight,
            "method": e.method,
            "event_id": e.event_id,
            "p_value": e.p_value,
            "tier_edge": e.tier_edge,
            "lag_seconds": e.lag_seconds,
            "window_center": e.window_center,
            "branching_ratio": e.branching_ratio,
            "fevd_share": e.fevd_share,
            "te_value": e.te_value,
        }
        for e in records
    ]
    return pl.DataFrame(rows)


def ensemble_edges(
    edge_tables: list[pl.DataFrame],
    weight_cols: dict[str, str] | None = None,
    p_threshold: float = 0.05,
) -> pl.DataFrame:
    """Merge edges from multiple methods via vote-weighted ensemble.

    For each (source, target) pair, collect all significant edges across
    methods and aggregate: mean weight, vote count, min p-value.

    Args:
        edge_tables: List of per-method edge DataFrames (any column convention).
        weight_cols: Optional mapping from method name to column carrying the
            edge weight. Falls back to "weight" → "fevd_share" → "te_i_to_j"
            → "branching_ratio_ij" → "peak_corr" in that order.
        p_threshold: Significance threshold applied per-method before merging.

    Returns:
        DataFrame with columns: source, target, weight_mean, vote_count,
        p_value_min, methods, method=ensemble.
    """
    all_rows: list[dict] = []
    for df in edge_tables:
        if df.is_empty():
            continue
        src_col = next((c for c in ("source", "causing_node", "node_i") if c in df.columns), None)
        tgt_col = next((c for c in ("target", "caused_node", "node_j") if c in df.columns), None)
        if src_col is None or tgt_col is None:
            continue
        w_col = next(
            (c for c in ("weight", "fevd_share", "te_i_to_j", "branching_ratio_ij", "peak_corr")
             if c in df.columns),
            None,
        )
        p_col = "p_value" if "p_value" in df.columns else None
        method_col = "method" if "method" in df.columns else None

        for row in df.iter_rows(named=True):
            p = float(row.get(p_col) or 1.0) if p_col else 0.0
            if p_col and p >= p_threshold:
                continue
            src = str(row.get(src_col) or "")
            tgt = str(row.get(tgt_col) or "")
            if not src or not tgt or src == tgt:
                continue
            all_rows.append({
                "source": src,
                "target": tgt,
                "weight": float(row.get(w_col) or 0.0) if w_col else 0.0,
                "p_value": p,
                "method": str(row.get(method_col) or "unknown") if method_col else "unknown",
            })

    if not all_rows:
        return pl.DataFrame(schema={
            "source": pl.String, "target": pl.String,
            "weight_mean": pl.Float64, "vote_count": pl.UInt32,
            "p_value_min": pl.Float64, "methods": pl.String,
        })

    raw = pl.DataFrame(all_rows)
    return (
        raw.group_by(["source", "target"])
        .agg([
            pl.col("weight").mean().alias("weight_mean"),
            pl.col("weight").count().cast(pl.UInt32).alias("vote_count"),
            pl.col("p_value").min().alias("p_value_min"),
            pl.col("method").unique().sort().str.join(",").alias("methods"),
        ])
        .with_columns(pl.lit("ensemble").alias("method"))
        .sort("vote_count", descending=True)
    )
