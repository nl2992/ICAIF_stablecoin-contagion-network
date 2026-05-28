"""Minimum empirical coverage checks before model estimation."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

FIXTURE = "fixture_non_empirical"
MISSING = "missing"
REAL_DISALLOWED = {FIXTURE, MISSING, None}


@dataclass(frozen=True)
class CoverageGateResult:
    """Summary of whether an event panel is model-ready."""

    event_id: str
    n_nodes_total: int
    n_nodes_real: int
    n_nodes_fixture: int
    n_var_eligible_nodes: int
    real_layers: tuple[str, ...]
    passes: bool
    reason: str


def _real_panel(panel: pl.DataFrame) -> pl.DataFrame:
    if "tier_actual" not in panel.columns:
        return panel.head(0)
    return panel.filter(~pl.col("tier_actual").is_in(list(REAL_DISALLOWED)))


def _eligible_basis_nodes(panel: pl.DataFrame, min_rows: int) -> set[str]:
    if "basis_vs_usd" not in panel.columns:
        return set()
    eligible = (
        panel.filter(pl.col("basis_vs_usd").is_not_null())
        .group_by("node_id")
        .agg(pl.len().alias("n_rows"))
        .filter(pl.col("n_rows") >= min_rows)
    )
    return set(eligible["node_id"].to_list())


def check_empirical_coverage(
    panel: pl.DataFrame,
    *,
    event_id: str,
    min_real_nodes: int = 3,
    min_var_nodes: int = 2,
    min_rows_per_node: int = 20,
    required_layers: tuple[str, ...] = (),
) -> CoverageGateResult:
    """Validate that a gold panel has enough real coverage for modelling.

    The gate is intentionally model-aware:
    - lead-lag / transfer entropy need at least ``min_real_nodes`` real nodes;
    - VAR needs at least ``min_var_nodes`` real nodes with non-null basis rows;
    - optional layer requirements prevent a "multi-layer" paper run from
      silently becoming CEX-only.
    """
    if "node_id" not in panel.columns:
        return CoverageGateResult(event_id, 0, 0, 0, 0, tuple(), False, "panel lacks node_id")

    n_nodes_total = panel["node_id"].n_unique()
    real = _real_panel(panel)
    n_nodes_real = real["node_id"].n_unique() if real.height else 0
    n_nodes_fixture = 0
    if "tier_actual" in panel.columns:
        n_nodes_fixture = panel.filter(pl.col("tier_actual") == FIXTURE)["node_id"].n_unique()

    real_layers: tuple[str, ...] = tuple()
    if "layer" in real.columns and real.height:
        real_layers = tuple(sorted(str(layer) for layer in real["layer"].drop_nulls().unique()))

    var_eligible_nodes = _eligible_basis_nodes(real, min_rows=min_rows_per_node)
    n_var_eligible_nodes = len(var_eligible_nodes)

    failures = []
    if n_nodes_real < min_real_nodes:
        failures.append(f"{n_nodes_real} real nodes < required {min_real_nodes}")
    if n_var_eligible_nodes < min_var_nodes:
        failures.append(
            f"{n_var_eligible_nodes} VAR-eligible real nodes < required {min_var_nodes}"
        )
    missing_layers = [layer for layer in required_layers if layer not in real_layers]
    if missing_layers:
        failures.append(f"missing required real layers: {', '.join(missing_layers)}")

    return CoverageGateResult(
        event_id=event_id,
        n_nodes_total=n_nodes_total,
        n_nodes_real=n_nodes_real,
        n_nodes_fixture=n_nodes_fixture,
        n_var_eligible_nodes=n_var_eligible_nodes,
        real_layers=real_layers,
        passes=not failures,
        reason="ok" if not failures else "; ".join(failures),
    )
