"""Provenance-based claim gating for directed result edges."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from stressnet.config import load_events, manifests_root
from stressnet.graph.nodes import nodes_for_event

FIXTURE = "fixture_non_empirical"
MISSING = "missing"
TIER_ORDER = {"A": 0, "B": 1, "C": 2, FIXTURE: 3, MISSING: 4}
EDGE_COLUMN_CANDIDATES = [
    ("node_i", "node_j"),
    ("causing_node", "caused_node"),
    ("source", "target"),
    ("source_node", "target_node"),
]
_CLAIM_SENTENCES = {
    "A_A_directional_microstructure": "We find directional microstructure transmission from {i} to {j}.",
    "A_B_suggestive_directional":     "We find suggestive timing evidence of stress propagation from {i} to {j}.",
    "B_B_context_only":               "We document contextual co-movement between {i} and {j} (Tier B proxy data).",
    "fixture_disallowed":             "Not claimable: fixture data.",
    "C_taxonomy_only":                "Not claimable: taxonomy context only.",
    "diagnostic_only":                "Not claimable: diagnostic fallback output.",
}

_CLAIM_LANGUAGE = {
    "A_A_directional_microstructure": "directional",
    "A_B_suggestive_directional":     "suggestive",
    "B_B_context_only":               "context-only",
    "fixture_disallowed":             "not claimable (fixture)",
    "C_taxonomy_only":                "not claimable (taxonomy)",
    "diagnostic_only":                "not claimable (diagnostic)",
}

RESULT_TABLE_PREFIXES = (
    "table_leadlag_tests",
    "table_transfer_entropy",
    "table_hayashi_yoshida",
    "table_granger",
    "table_var_spillovers",
    "table_tvp_var_summary",
    "table_tvp_var_spillovers",
    "table_tvp_var_edges",
    "table_hawkes_params",
    "table_edges",
)


@dataclass(frozen=True)
class ClaimDecision:
    """Claim metadata for one directed edge."""

    tier_i_actual: str
    tier_j_actual: str
    edge_tier_actual: str
    uses_fixture: bool
    claim_allowed: bool
    claim_level: str
    claim_reason: str
    claim_sentence: str


def tier_rank(tier: str | None) -> int:
    """Return weaker-tier rank; higher numbers mean weaker provenance."""
    return TIER_ORDER.get(str(tier) if tier is not None else MISSING, TIER_ORDER[MISSING])


def weaker_tier(tier_i: str | None, tier_j: str | None) -> str:
    """Return the weaker of two endpoint tiers."""
    left = str(tier_i) if tier_i is not None else MISSING
    right = str(tier_j) if tier_j is not None else MISSING
    return left if tier_rank(left) >= tier_rank(right) else right


def decide_claim(tier_i: str | None, tier_j: str | None) -> ClaimDecision:
    """Classify what kind of paper claim an edge can support."""
    left = str(tier_i) if tier_i is not None else MISSING
    right = str(tier_j) if tier_j is not None else MISSING
    edge_tier = weaker_tier(left, right)
    uses_fixture = left == FIXTURE or right == FIXTURE

    if uses_fixture:
        _level = "fixture_disallowed"
        return ClaimDecision(
            left,
            right,
            edge_tier,
            uses_fixture=True,
            claim_allowed=False,
            claim_level=_level,
            claim_reason="At least one endpoint is deterministic fixture data.",
            claim_sentence=_CLAIM_SENTENCES[_level],
        )
    if left == MISSING or right == MISSING:
        _level = "C_taxonomy_only"
        return ClaimDecision(
            left,
            right,
            edge_tier,
            uses_fixture=False,
            claim_allowed=False,
            claim_level=_level,
            claim_reason="At least one endpoint is missing from provenance coverage.",
            claim_sentence=_CLAIM_SENTENCES[_level],
        )
    if left == "C" or right == "C":
        _level = "C_taxonomy_only"
        return ClaimDecision(
            left,
            right,
            edge_tier,
            uses_fixture=False,
            claim_allowed=False,
            claim_level=_level,
            claim_reason="Tier C endpoint supports taxonomy or context only.",
            claim_sentence=_CLAIM_SENTENCES[_level],
        )
    if left == "A" and right == "A":
        _level = "A_A_directional_microstructure"
        return ClaimDecision(
            left,
            right,
            edge_tier,
            uses_fixture=False,
            claim_allowed=True,
            claim_level=_level,
            claim_reason="Both endpoints have Tier A provenance.",
            claim_sentence=_CLAIM_SENTENCES[_level],
        )
    if {left, right} == {"A", "B"}:
        _level = "A_B_suggestive_directional"
        return ClaimDecision(
            left,
            right,
            edge_tier,
            uses_fixture=False,
            claim_allowed=True,
            claim_level=_level,
            claim_reason="Edge is capped by the Tier B endpoint.",
            claim_sentence=_CLAIM_SENTENCES[_level],
        )
    if left == "B" and right == "B":
        _level = "B_B_context_only"
        return ClaimDecision(
            left,
            right,
            edge_tier,
            uses_fixture=False,
            claim_allowed=True,
            claim_level=_level,
            claim_reason="Both endpoints are Tier B, so use contextual language.",
            claim_sentence=_CLAIM_SENTENCES[_level],
        )
    _level = "C_taxonomy_only"
    return ClaimDecision(
        left,
        right,
        edge_tier,
        uses_fixture=False,
        claim_allowed=False,
        claim_level=_level,
        claim_reason="Endpoint provenance tier is not paper-claimable.",
        claim_sentence=_CLAIM_SENTENCES[_level],
    )


def infer_event_from_name(path: Path) -> str | None:
    """Infer event id from a result-table file name."""
    event_ids = sorted(load_events().keys(), key=len, reverse=True)
    for event_id in event_ids:
        if path.stem.endswith(f"_{event_id}") or f"_{event_id}_" in path.stem:
            return event_id
    return None


def load_tiers_from_coverage(tables_dir: Path) -> dict[str, dict[str, str]]:
    """Load event/node tiers from the consolidated coverage table, if present."""
    tiers: dict[str, dict[str, str]] = {}
    coverage_path = tables_dir / "table_node_coverage.csv"
    if not coverage_path.exists():
        return tiers

    coverage = pl.read_csv(coverage_path)
    tier_col = "source_tier_actual" if "source_tier_actual" in coverage.columns else "tier_actual"
    if not {"event_id", "node_id", tier_col}.issubset(coverage.columns):
        return tiers

    coverage_pct_col = "coverage_pct" if "coverage_pct" in coverage.columns else None
    for row in coverage.iter_rows(named=True):
        event_id = row.get("event_id")
        node_id = row.get("node_id")
        if not event_id or not node_id or node_id == "__event_panel__":
            continue
        tier = str(row.get(tier_col) or MISSING)
        # Apply coverage-percentage downgrade: < 50% coverage → downgrade A→B
        if coverage_pct_col:
            pct = row.get(coverage_pct_col)
            if pct is not None and float(pct) < 50.0 and tier == "A":
                tier = "B"  # downgrade due to < 50% coverage
        tiers.setdefault(str(event_id), {})[str(node_id)] = tier
    return tiers


def load_tiers_from_manifest(event_id: str) -> dict[str, str]:
    """Load latest per-node tiers from the event manifest."""
    manifest_path = manifests_root() / f"manifest_{event_id}.csv"
    configured = {node.id for node in nodes_for_event(event_id)}
    if not manifest_path.exists():
        return {}

    tiers = {node_id: MISSING for node_id in configured}
    manifest = pl.read_csv(manifest_path)
    if "node_id" not in manifest.columns or "source_tier_actual" not in manifest.columns:
        return tiers
    manifest = manifest.filter(pl.col("node_id") != "__event_panel__")

    if "file_stage" in manifest.columns:
        silver = manifest.filter(pl.col("file_stage") == "silver")
        if silver.height > 0:
            manifest = silver

    for row in manifest.iter_rows(named=True):
        node_id = row.get("node_id")
        if node_id in configured:
            tiers[str(node_id)] = str(row.get("source_tier_actual") or MISSING)
    return tiers


def load_tier_map(events: list[str], tables_dir: Path) -> dict[str, dict[str, str]]:
    """Load coverage tiers and let manifests override stale coverage summaries."""
    tier_map = load_tiers_from_coverage(tables_dir)
    for event_id in events:
        manifest_tiers = load_tiers_from_manifest(event_id)
        if event_id not in tier_map:
            tier_map[event_id] = manifest_tiers
        else:
            tier_map[event_id] = {**tier_map[event_id], **manifest_tiers}
    return tier_map


def edge_columns(df: pl.DataFrame) -> tuple[str, str] | None:
    """Return source/target columns for a supported edge table."""
    for source_col, target_col in EDGE_COLUMN_CANDIDATES:
        if source_col in df.columns and target_col in df.columns:
            return source_col, target_col
    return None


def is_result_table(path: Path) -> bool:
    """Return whether a table should be considered claim-bearing."""
    if path.suffix != ".csv":
        return False
    return path.name.startswith(RESULT_TABLE_PREFIXES)


def event_tables(event_id: str, tables_dir: Path) -> list[Path]:
    """Return claim-bearing result tables for one event."""
    pattern = re.compile(rf"^({'|'.join(RESULT_TABLE_PREFIXES)}).*_{re.escape(event_id)}\.csv$")
    return sorted(path for path in tables_dir.glob("*.csv") if pattern.match(path.name))


def paper_tables(tables_dir: Path) -> list[Path]:
    """Return all claim-bearing result tables in a tables directory."""
    return sorted(path for path in tables_dir.glob("*.csv") if is_result_table(path))


def row_event(row: dict[str, object], path: Path, default_event: str | None) -> str | None:
    """Resolve the event id for one row."""
    value = row.get("event_id")
    if value:
        return str(value)
    return default_event or infer_event_from_name(path)


def annotate_edge_table(
    df: pl.DataFrame,
    tier_map: dict[str, dict[str, str]],
    *,
    source_col: str,
    target_col: str,
    table_path: Path,
    default_event: str | None = None,
) -> pl.DataFrame:
    """Return an edge table annotated with provenance claim metadata."""
    annotated = []
    for row in df.iter_rows(named=True):
        event_id = row_event(row, table_path, default_event)
        event_tiers = tier_map.get(event_id or "", {})
        source = str(row.get(source_col))
        target = str(row.get(target_col))
        decision = decide_claim(event_tiers.get(source, MISSING), event_tiers.get(target, MISSING))
        claim_allowed = decision.claim_allowed
        claim_level = decision.claim_level
        claim_reason = decision.claim_reason
        claim_sentence = decision.claim_sentence

        if str(row.get("method", "")) in {"var_coeff_fallback", "var_abscoef_fallback"}:
            claim_allowed = False
            claim_level = "diagnostic_only"
            claim_reason = "VAR spillover used coefficient fallback after FEVD failed."
            claim_sentence = _CLAIM_SENTENCES[claim_level]

        # Note: __event_panel__ tier is intentionally excluded from edge annotation;
        # only individual node tiers matter.
        row["tier_i_actual"] = decision.tier_i_actual
        row["tier_j_actual"] = decision.tier_j_actual
        row["edge_tier_actual"] = decision.edge_tier_actual
        row["uses_fixture"] = decision.uses_fixture
        row["claim_allowed"] = claim_allowed
        row["claim_level"] = claim_level
        row["claim_reason"] = claim_reason
        row["claim_language"] = _CLAIM_LANGUAGE.get(claim_level, "unknown")
        row["claim_sentence"] = (
            claim_sentence
            .replace("{i}", str(row.get(source_col, "node_i")))
            .replace("{j}", str(row.get(target_col, "node_j")))
        )
        annotated.append(row)
    return pl.DataFrame(annotated)


def annotate_table(
    path: Path,
    tier_map: dict[str, dict[str, str]],
    default_event: str | None,
) -> tuple[pl.DataFrame | None, dict[str, object]]:
    """Annotate one CSV result table, returning the table and audit summary."""
    df = pl.read_csv(path)
    cols = edge_columns(df)
    summary = {
        "table": path.name,
        "event_id": default_event or "multi",
        "status": "skipped_no_edge_columns",
        "rows": df.height,
        "blocked_rows": 0,
        "fixture_or_missing_rows": 0,
        "claimable_rows": 0,
    }
    if cols is None or df.height == 0:
        return None, summary

    source_col, target_col = cols
    out = annotate_edge_table(
        df,
        tier_map,
        source_col=source_col,
        target_col=target_col,
        table_path=path,
        default_event=default_event,
    )
    blocked_rows = out.filter(~pl.col("claim_allowed")).height
    fixture_or_missing_rows = out.filter(
        pl.col("edge_tier_actual").is_in([FIXTURE, MISSING])
    ).height
    summary.update(
        {
            "status": "annotated",
            "rows": out.height,
            "blocked_rows": blocked_rows,
            "fixture_or_missing_rows": fixture_or_missing_rows,
            "claimable_rows": out.height - blocked_rows,
        }
    )
    return out, summary
