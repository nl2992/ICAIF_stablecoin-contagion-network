"""Provenance-based claim gating for directed result edges.

Implements a three-gate pipeline:
  1. Provenance gate  — data tier, fixture check, feature-level tier cap.
  2. Statistical gate — significance columns present and passing threshold.
  3. Paper gate       — both gates must pass for a publishable directional claim.

Claim taxonomy
--------------
A/A edges are split by *layer combination* so the claim language matches the
actual evidence type:

  A_A_dex_flow              Both endpoints are Tier-A DEX (Curve AMM) nodes.
  A_A_onchain_settlement    Both are Tier-A on-chain settlement/flow nodes.
  A_A_cex_microstructure    Both are Tier-A CEX nodes with L2 microstructure.
  A_A_high_provenance       A/A pair with other layer combination.
  A_B_suggestive            Mixed A/B edge — lower-tier endpoint caps claim.
  B_B_context_only          Both Tier B — contextual co-movement only.
  fixture_disallowed        Fixture endpoint — not claimable.
  C_taxonomy_only           Tier-C or missing endpoint.
  diagnostic_only           Fallback / diagnostic output.

Feature-level tiers
-------------------
configs/feature_tiers.yaml defines per-feature tiers (A or B).
An edge is capped at min(tier_i, tier_j, feature_tier).
A feature marked tier A but only available without L2 data is effectively B
(controlled by the ``effective_tier_without_l2`` key in the config).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import polars as pl

from stressnet.config import load_events, manifests_root
from stressnet.graph.nodes import nodes_for_event

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXTURE = "fixture_non_empirical"
MISSING = "missing"
TIER_ORDER = {"A": 0, "B": 1, "C": 2, FIXTURE: 3, MISSING: 4}

# Features that require a full L2 book.  If the node has no vendor L2 key set,
# the effective tier for these features is B, not A.
_L2_ONLY_FEATURES = frozenset({
    "depth_10bps_bid_usd",
    "depth_10bps_ask_usd",
    "orderbook_imbalance",
    "executable_price_10k_buy",
    "executable_price_10k_sell",
})

# Layer groups used for claim taxonomy
_DEX_LAYERS          = frozenset({"DEX"})
_SETTLEMENT_LAYERS   = frozenset({"mint_burn", "onchain_flow", "bridge_flow"})
_CEX_LAYERS          = frozenset({"CEX"})
_MICROSTRUCTURE_FEATURES = frozenset({
    "depth_10bps_bid_usd",
    "depth_10bps_ask_usd",
    "orderbook_imbalance",
    "executable_price_10k_buy",
    "executable_price_10k_sell",
    "spread_bps",
})

EDGE_COLUMN_CANDIDATES = [
    ("node_i", "node_j"),
    ("causing_node", "caused_node"),
    ("source", "target"),
    ("source_node", "target_node"),
    ("source_node_id", "target_node_id"),  # sparse-flow event study
]

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
    "table_amm_leadlag",
    "table_sparse_events",
)

# ---------------------------------------------------------------------------
# Claim taxonomy sentences and language tags
# ---------------------------------------------------------------------------

_CLAIM_SENTENCES: dict[str, str] = {
    # A/A claims — layer-differentiated
    "A_A_dex_flow": (
        "We find high-provenance on-chain AMM-flow evidence of stress propagation "
        "from {i} to {j}."
    ),
    "A_A_onchain_settlement": (
        "We find high-provenance on-chain settlement-flow evidence linking "
        "{i} to {j}."
    ),
    # Used when the settlement edge is provenance-valid but statistically underpowered
    "A_A_onchain_settlement_underpowered": (
        "We document a high-provenance sparse settlement-flow response candidate "
        "from {i} to {j}, but it is not statistically supported under the "
        "event-arrival test."
    ),
    "A_A_cex_microstructure": (
        "We find directional CEX microstructure transmission from {i} to {j}."
    ),
    "A_A_high_provenance": (
        "We find high-provenance directional stress-propagation evidence "
        "from {i} to {j}."
    ),
    # Mixed / lower tiers
    "A_B_suggestive_directional": (
        "We find suggestive timing evidence of stress propagation from {i} to {j}."
    ),
    "B_B_context_only": (
        "We document contextual co-movement between {i} and {j} "
        "(Tier B proxy data)."
    ),
    # Blocked claims
    "fixture_disallowed": "Not claimable: fixture data.",
    "C_taxonomy_only":    "Not claimable: taxonomy context only.",
    "diagnostic_only":    "Not claimable: diagnostic fallback output.",
}

_CLAIM_LANGUAGE: dict[str, str] = {
    "A_A_dex_flow":               "A/A directional (on-chain AMM flow)",
    "A_A_onchain_settlement":     "A/A directional (on-chain settlement)",
    "A_A_cex_microstructure":     "A/A directional (CEX microstructure)",
    "A_A_high_provenance":        "A/A directional (high provenance)",
    "A_B_suggestive_directional": "A/B suggestive",
    "B_B_context_only":           "B/B context-only",
    "fixture_disallowed":         "not claimable (fixture)",
    "C_taxonomy_only":            "not claimable (taxonomy)",
    "diagnostic_only":            "not claimable (diagnostic)",
}

# Backward-compatible alias so old callers that check for the previous label
# still work.
_CLAIM_LANGUAGE["A_A_directional_microstructure"] = _CLAIM_LANGUAGE["A_A_dex_flow"]


# ---------------------------------------------------------------------------
# ClaimDecision dataclass
# ---------------------------------------------------------------------------

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
    # Feature-level cap metadata (empty string = not applicable)
    feature_col: str = ""
    feature_tier: str = ""


# ---------------------------------------------------------------------------
# Feature-tier loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_feature_tiers() -> dict[str, dict[str, Any]]:
    """Load configs/feature_tiers.yaml and return the ``features`` dict."""
    import yaml
    from stressnet.config import _CONFIGS_DIR
    path = _CONFIGS_DIR / "feature_tiers.yaml"
    if not path.exists():
        return {}
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    return raw.get("features", {})


def feature_tier(feature_col: str, *, has_l2: bool = False) -> str:
    """Return the effective provenance tier for *feature_col*.

    Args:
        feature_col: Column name in the gold panel (e.g. ``"usdc_net_sold_1h"``).
        has_l2: Whether a real full-L2 order book is available for this node.
                Defaults to False.  L2-only features are demoted to B without it.
    """
    tiers = _load_feature_tiers()
    entry = tiers.get(feature_col)
    if entry is None:
        return "B"  # conservative default for unknown features
    tier_val = str(entry.get("tier", "B"))
    if tier_val == "A" and feature_col in _L2_ONLY_FEATURES and not has_l2:
        return str(entry.get("effective_tier_without_l2", "B"))
    return tier_val


# ---------------------------------------------------------------------------
# Tier arithmetic helpers
# ---------------------------------------------------------------------------

def tier_rank(tier: str | None) -> int:
    """Return weaker-tier rank; higher numbers mean weaker provenance."""
    return TIER_ORDER.get(str(tier) if tier is not None else MISSING, TIER_ORDER[MISSING])


def weaker_tier(tier_i: str | None, tier_j: str | None) -> str:
    """Return the weaker of two endpoint tiers."""
    left  = str(tier_i) if tier_i is not None else MISSING
    right = str(tier_j) if tier_j is not None else MISSING
    return left if tier_rank(left) >= tier_rank(right) else right


def effective_edge_tier(
    tier_i: str | None,
    tier_j: str | None,
    feature_col: str = "",
    *,
    has_l2: bool = False,
) -> str:
    """Edge tier capped by min(tier_i, tier_j, feature_tier)."""
    node_edge = weaker_tier(tier_i, tier_j)
    if feature_col:
        ft = feature_tier(feature_col, has_l2=has_l2)
        return node_edge if tier_rank(node_edge) >= tier_rank(ft) else ft
    return node_edge


# ---------------------------------------------------------------------------
# Core claim decision
# ---------------------------------------------------------------------------

def decide_claim(
    tier_i: str | None,
    tier_j: str | None,
    layer_i: str = "",
    layer_j: str = "",
    feature_col: str = "",
    *,
    has_l2: bool = False,
) -> ClaimDecision:
    """Classify what kind of paper claim an edge can support.

    Args:
        tier_i, tier_j:   Node-level provenance tiers.
        layer_i, layer_j: Node layer labels (CEX | DEX | mint_burn | onchain_flow | …).
        feature_col:      The feature column driving this edge (for tier cap).
        has_l2:           Whether a real L2 book backs L2-only features.
    """
    left  = str(tier_i) if tier_i is not None else MISSING
    right = str(tier_j) if tier_j is not None else MISSING
    ft    = feature_tier(feature_col, has_l2=has_l2) if feature_col else ""
    edge_tier = effective_edge_tier(left, right, feature_col, has_l2=has_l2)
    uses_fixture = left == FIXTURE or right == FIXTURE

    def _make(level: str, allowed: bool, reason: str) -> ClaimDecision:
        return ClaimDecision(
            tier_i_actual=left,
            tier_j_actual=right,
            edge_tier_actual=edge_tier,
            uses_fixture=uses_fixture,
            claim_allowed=allowed,
            claim_level=level,
            claim_reason=reason,
            claim_sentence=_CLAIM_SENTENCES.get(level, ""),
            feature_col=feature_col,
            feature_tier=ft,
        )

    # ── Blocked states ────────────────────────────────────────────────────────
    if uses_fixture:
        return _make(
            "fixture_disallowed", False,
            "At least one endpoint is deterministic fixture data.",
        )
    if left == MISSING or right == MISSING:
        return _make(
            "C_taxonomy_only", False,
            "At least one endpoint is missing from provenance coverage.",
        )
    if left == "C" or right == "C":
        return _make(
            "C_taxonomy_only", False,
            "Tier C endpoint supports taxonomy or context only.",
        )

    # ── Feature-tier cap: if the feature is Tier B, highest claim is A/B ─────
    feature_cap_applies = (
        feature_col
        and ft == "B"
        and left == "A"
        and right == "A"
    )

    # ── A/A claims — layer-differentiated taxonomy ────────────────────────────
    if left == "A" and right == "A" and not feature_cap_applies:
        layers = {layer_i, layer_j} - {""}
        if layer_i in _DEX_LAYERS and layer_j in _DEX_LAYERS:
            return _make(
                "A_A_dex_flow", True,
                "Both endpoints are Tier-A DEX (AMM) nodes.",
            )
        if layers & _SETTLEMENT_LAYERS:
            return _make(
                "A_A_onchain_settlement", True,
                "At least one endpoint is a Tier-A on-chain settlement/flow node.",
            )
        if (
            layer_i in _CEX_LAYERS
            and layer_j in _CEX_LAYERS
            and feature_col in _MICROSTRUCTURE_FEATURES
        ):
            return _make(
                "A_A_cex_microstructure", True,
                "Both endpoints are Tier-A CEX nodes with microstructure features.",
            )
        return _make(
            "A_A_high_provenance", True,
            "Both endpoints have Tier A provenance.",
        )

    # ── Feature cap downgrades A/A to A/B ─────────────────────────────────────
    if feature_cap_applies:
        return _make(
            "A_B_suggestive_directional", True,
            (
                f"Node tiers are A/A but feature '{feature_col}' is Tier B "
                f"(derived proxy), so edge is capped to A/B."
            ),
        )

    # ── Mixed A/B ─────────────────────────────────────────────────────────────
    if {left, right} == {"A", "B"}:
        return _make(
            "A_B_suggestive_directional", True,
            "Edge is capped by the Tier B endpoint.",
        )

    # ── B/B ───────────────────────────────────────────────────────────────────
    if left == "B" and right == "B":
        return _make(
            "B_B_context_only", True,
            "Both endpoints are Tier B; use contextual language.",
        )

    return _make(
        "C_taxonomy_only", False,
        "Endpoint provenance tier is not paper-claimable.",
    )


# ---------------------------------------------------------------------------
# Statistical support helpers (TODO 3)
# ---------------------------------------------------------------------------

# Mapping from result-table column → (threshold, direction)
# direction: "lt" = lower is more significant, "gt" = higher is more significant
_SIGNIFICANCE_COLS: list[tuple[str, float, str]] = [
    ("significant_block_fdr", 0.5, "gt"),   # bool column: True if significant
    ("significant_fdr",       0.5, "gt"),
    ("significant_bonferroni",0.5, "gt"),
    ("significant_p01",       0.5, "gt"),
    ("significant_p05",       0.5, "gt"),
    ("granger_pval",          0.05, "lt"),
    ("p_value",               0.05, "lt"),
    ("TE_p",                  0.05, "lt"),
    ("hawkes_ci_lower",       0.0,  "gt"),   # branching ratio CI excludes zero
]

_CLAIM_STRENGTH_ORDER = ["descriptive", "suggestive", "statistically_supported", "robust"]


def row_has_statistical_support(row: dict[str, Any]) -> bool:
    """Return True if *row* passes at least one significance criterion."""
    for col, threshold, direction in _SIGNIFICANCE_COLS:
        val = row.get(col)
        if val is None:
            continue
        try:
            fval = float(val)
        except (ValueError, TypeError):
            # Treat string 'True'/'False'
            if str(val).lower() in ("true", "1", "yes"):
                fval = 1.0
            elif str(val).lower() in ("false", "0", "no"):
                fval = 0.0
            else:
                continue
        if direction == "gt" and fval > threshold:
            return True
        if direction == "lt" and fval < threshold:
            return True
    return False


def claim_strength(provenance_ok: bool, stat_ok: bool, claim_level: str) -> str:
    """Return a ranked descriptive claim strength label."""
    if not provenance_ok:
        return "descriptive"
    if not stat_ok:
        return "suggestive" if claim_level.startswith("A_") else "descriptive"
    if claim_level in ("A_A_dex_flow", "A_A_onchain_settlement",
                       "A_A_cex_microstructure", "A_A_high_provenance"):
        return "robust"
    return "statistically_supported"


# ---------------------------------------------------------------------------
# Manifest / coverage tier + layer loading
# ---------------------------------------------------------------------------

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
        if coverage_pct_col:
            pct = row.get(coverage_pct_col)
            if pct is not None and float(pct) < 50.0 and tier == "A":
                tier = "B"
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


def load_layers_for_event(event_id: str) -> dict[str, str]:
    """Return {node_id: layer} for all nodes configured for *event_id*."""
    return {node.id: node.layer for node in nodes_for_event(event_id)}


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


def load_layer_map(events: list[str]) -> dict[str, dict[str, str]]:
    """Return {event_id: {node_id: layer}} for all requested events."""
    return {e: load_layers_for_event(e) for e in events}


# ---------------------------------------------------------------------------
# Edge table helpers
# ---------------------------------------------------------------------------

def edge_columns(df: pl.DataFrame) -> tuple[str, str] | None:
    """Return source/target column names for a supported edge table."""
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
    pattern = re.compile(
        rf"^({'|'.join(RESULT_TABLE_PREFIXES)}).*_{re.escape(event_id)}\.csv$"
    )
    return sorted(p for p in tables_dir.glob("*.csv") if pattern.match(p.name))


def paper_tables(tables_dir: Path) -> list[Path]:
    """Return all claim-bearing result tables in a tables directory."""
    return sorted(p for p in tables_dir.glob("*.csv") if is_result_table(p))


def row_event(
    row: dict[str, object], path: Path, default_event: str | None
) -> str | None:
    """Resolve the event id for one row."""
    value = row.get("event_id")
    if value:
        return str(value)
    return default_event or infer_event_from_name(path)


# ---------------------------------------------------------------------------
# Table annotation (TODO 3: add statistical support columns)
# ---------------------------------------------------------------------------

def annotate_edge_table(
    df: pl.DataFrame,
    tier_map: dict[str, dict[str, str]],
    *,
    source_col: str,
    target_col: str,
    table_path: Path,
    default_event: str | None = None,
    layer_map: dict[str, dict[str, str]] | None = None,
) -> pl.DataFrame:
    """Return an edge table annotated with provenance + statistical claim metadata.

    New columns added:
      tier_i_actual, tier_j_actual, edge_tier_actual  — provenance tiers
      uses_fixture                                     — fixture contamination flag
      claim_allowed (=provenance_claim_allowed)        — passes provenance gate
      claim_level                                      — taxonomy label
      claim_language, claim_reason, claim_sentence     — human-readable text
      feature_tier                                     — feature-level cap tier
      provenance_claim_allowed                         — explicit provenance gate bool
      statistical_claim_allowed                        — passes significance test
      paper_claim_allowed                              — both gates pass
      claim_strength                                   — descriptive/suggestive/
                                                         statistically_supported/robust
    """
    annotated = []
    for row in df.iter_rows(named=True):
        event_id = row_event(row, table_path, default_event)
        event_tiers  = tier_map.get(event_id or "", {})
        event_layers = (layer_map or {}).get(event_id or "", {})

        source      = str(row.get(source_col, ""))
        target      = str(row.get(target_col, ""))
        feature_col = str(row.get("feature_col", row.get("feature", "")))

        decision = decide_claim(
            event_tiers.get(source, MISSING),
            event_tiers.get(target, MISSING),
            layer_i=event_layers.get(source, ""),
            layer_j=event_layers.get(target, ""),
            feature_col=feature_col,
        )

        # VAR coefficient fallback: always diagnostic only
        if str(row.get("method", "")) in {"var_coeff_fallback", "var_abscoef_fallback"}:
            decision = ClaimDecision(
                tier_i_actual=decision.tier_i_actual,
                tier_j_actual=decision.tier_j_actual,
                edge_tier_actual=decision.edge_tier_actual,
                uses_fixture=decision.uses_fixture,
                claim_allowed=False,
                claim_level="diagnostic_only",
                claim_reason="VAR spillover used coefficient fallback after FEVD failed.",
                claim_sentence=_CLAIM_SENTENCES["diagnostic_only"],
                feature_col=feature_col,
                feature_tier=decision.feature_tier,
            )

        prov_ok = decision.claim_allowed
        stat_ok = row_has_statistical_support(row)
        paper_ok = prov_ok and stat_ok
        strength = claim_strength(prov_ok, stat_ok, decision.claim_level)

        # For sparse settlement-flow rows that are provenance-valid but
        # not statistically supported, use the underpowered sentence so
        # the claim language is explicit about the limitation.
        is_sparse_table = "sparse_events" in str(table_path.name) if table_path else False
        claim_sentence_template = decision.claim_sentence
        if (
            decision.claim_level == "A_A_onchain_settlement"
            and prov_ok
            and not stat_ok
            and is_sparse_table
        ):
            claim_sentence_template = _CLAIM_SENTENCES["A_A_onchain_settlement_underpowered"]

        row["tier_i_actual"]            = decision.tier_i_actual
        row["tier_j_actual"]            = decision.tier_j_actual
        row["edge_tier_actual"]         = decision.edge_tier_actual
        row["uses_fixture"]             = decision.uses_fixture
        row["feature_tier"]             = decision.feature_tier or "B"
        # Provenance gate
        row["provenance_claim_allowed"] = prov_ok
        row["claim_allowed"]            = prov_ok   # backward-compat alias
        row["claim_level"]              = decision.claim_level
        row["claim_language"]           = _CLAIM_LANGUAGE.get(decision.claim_level, "unknown")
        row["claim_reason"]             = decision.claim_reason
        row["claim_sentence"]           = (
            claim_sentence_template
            .replace("{i}", str(row.get(source_col, "node_i")))
            .replace("{j}", str(row.get(target_col, "node_j")))
        )
        # Statistical + paper gates
        row["statistical_claim_allowed"] = stat_ok
        row["paper_claim_allowed"]        = paper_ok
        row["claim_strength"]             = strength

        annotated.append(row)
    return pl.DataFrame(annotated)


def annotate_table(
    path: Path,
    tier_map: dict[str, dict[str, str]],
    default_event: str | None,
    layer_map: dict[str, dict[str, str]] | None = None,
) -> tuple[pl.DataFrame | None, dict[str, object]]:
    """Annotate one CSV result table, returning the table and audit summary."""
    df = pl.read_csv(path)
    cols = edge_columns(df)
    summary: dict[str, object] = {
        "table": path.name,
        "event_id": default_event or "multi",
        "status": "skipped_no_edge_columns",
        "rows": df.height,
        "blocked_rows": 0,
        "fixture_or_missing_rows": 0,
        "claimable_rows": 0,
        "paper_claimable_rows": 0,
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
        layer_map=layer_map,
    )
    blocked_rows = out.filter(~pl.col("claim_allowed")).height
    fixture_or_missing_rows = out.filter(
        pl.col("edge_tier_actual").is_in([FIXTURE, MISSING])
    ).height
    paper_rows = out.filter(pl.col("paper_claim_allowed")).height if "paper_claim_allowed" in out.columns else 0
    summary.update({
        "status": "annotated",
        "rows": out.height,
        "blocked_rows": blocked_rows,
        "fixture_or_missing_rows": fixture_or_missing_rows,
        "claimable_rows": out.height - blocked_rows,
        "paper_claimable_rows": paper_rows,
    })
    return out, summary
