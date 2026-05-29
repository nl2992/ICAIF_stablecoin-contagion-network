"""Acceptance tests for the paper package.

These tests verify that the claim-gated output tables, figures, and narrative
files are internally consistent with the paper's headline result:

    During the USDT/Curve 2023 event, curve_3pool ↔ curve_crvusd_usdt exhibit
    statistically supported bidirectional AMM-flow linkage using Tier-A
    usdc_net_sold_1h data (both directions, Bonferroni p ≤ 0.014, claim_strength=robust).

All other events must NOT appear as A/A paper-claimable.

Run:
    python -m pytest tests/test_paper_package.py -v
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

REPO_ROOT       = Path(__file__).resolve().parents[1]
TABLE_DIR       = REPO_ROOT / "results" / "paper" / "tables"
RAW_TBL         = REPO_ROOT / "results" / "tables"
FIG_DIR         = REPO_ROOT / "results" / "paper" / "figures"
COLUMBIA_FIG_DIR = REPO_ROOT / "results" / "paper" / "figures_columbia"
README          = REPO_ROOT / "README.md"
PAPER_DIR       = REPO_ROOT / "paper"

EXPECTED_FIGURES = [
    "figure_01_multilayer_architecture.png",
    "figure_02_claim_gate_pipeline.png",
    "figure_03_claim_audit_by_event.png",
    "figure_04_usdt_curve_amm_flow_timeline.png",
    "figure_05_usdt_curve_leadlag_profile.png",
    "figure_06_aa_paper_claimable_network.png",
    "figure_07_aa_provenance_vs_paper_claimable.png",
    "figure_08_terra_amm_flow_candidate.png",
    "figure_09_usdc_svb_sparse_settlement_response.png",
    "figure_10_feature_tier_matrix.png",
    "figure_11_node_provenance_coverage.png",
    "figure_12_full_paper_claimable_network.png",
]

COLUMBIA_EXPECTED_FILES = [
    "01_architecture_columbia.png",
    "02_claim_gate_columbia.png",
    "03_claim_audit_columbia.png",
    "04_usdt_curve_timeline_columbia.png",
    "05_usdt_curve_leadlag_columbia.png",
    "06_aa_network_columbia.png",
    "07_cross_event_evidence_map_columbia.png",
    "08_full_paper_network_columbia.png",
    "A01_leadlag_heatmap_columbia.png",
    "A02_transfer_entropy_heatmap_columbia.png",
    "A03_terra_negative_result_columbia.png",
    "A04_usdc_svb_sparse_response_columbia.png",
    "A05_feature_tier_matrix_columbia.png",
    "A06_node_provenance_heatmap_columbia.png",
    "A07_data_lineage_sankey_columbia.png",
    "A08_non_claims_map_columbia.png",
    "A09_method_comparison_columbia.png",
    "A10_paper_claim_waterfall_columbia.png",
]

README_BANNED_PHRASES = [
    "headline microstructure claims",
    "A/A edges are confirmed",
]

PAPER_MARKDOWN_BANNED_PHRASES = [
    "proves contagion",
    "causal contagion",
    "directional microstructure transmission",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_paper_table(name: str) -> pl.DataFrame | None:
    path = TABLE_DIR / name
    if not path.exists():
        return None
    return pl.read_csv(path)


def _bool_col(df: pl.DataFrame, col: str) -> pl.Series:
    """Cast a claim-gate boolean column to Polars Boolean."""
    return (
        df[col].cast(pl.String).str.to_lowercase().is_in(["true", "1", "yes"])
        if col in df.columns
        else pl.Series([False] * df.height)
    )


# ─────────────────────────────────────────────────────────────────────────────
# A. table_aa_paper_claimable_edges.csv: only A/A DEX-flow USDT/Curve rows
# ─────────────────────────────────────────────────────────────────────────────

def test_aa_paper_claimable_exists():
    df = _read_paper_table("table_aa_paper_claimable_edges.csv")
    assert df is not None, "table_aa_paper_claimable_edges.csv not found"
    assert df.height > 0, "table_aa_paper_claimable_edges.csv is empty"


def test_aa_paper_claimable_only_usdt_curve():
    df = _read_paper_table("table_aa_paper_claimable_edges.csv")
    if df is None:
        pytest.skip("table not found — run paper_gate first")
    if "event_id" in df.columns:
        events = df["event_id"].cast(pl.String).unique().to_list()
        non_usdt = [e for e in events if e and e != "usdt_curve_2023"]
        assert not non_usdt, (
            f"Unexpected non-USDT events in A/A paper-claimable table: {non_usdt}. "
            "Terra, USDC/SVB, FTX, BUSD should not appear here."
        )


def test_aa_paper_claimable_correct_feature():
    df = _read_paper_table("table_aa_paper_claimable_edges.csv")
    if df is None:
        pytest.skip("table not found")
    if "feature_col" in df.columns:
        bad = df.filter(pl.col("feature_col").cast(pl.String) != "usdc_net_sold_1h")
        assert bad.height == 0, (
            f"{bad.height} rows with feature != usdc_net_sold_1h in headline table"
        )


def test_aa_paper_claimable_correct_claim_level():
    df = _read_paper_table("table_aa_paper_claimable_edges.csv")
    if df is None:
        pytest.skip("table not found")
    if "claim_level" in df.columns:
        bad = df.filter(pl.col("claim_level").cast(pl.String) != "A_A_dex_flow")
        assert bad.height == 0, (
            f"{bad.height} rows with claim_level != A_A_dex_flow"
        )


def test_aa_paper_claimable_correct_strength():
    df = _read_paper_table("table_aa_paper_claimable_edges.csv")
    if df is None:
        pytest.skip("table not found")
    if "claim_strength" in df.columns:
        bad = df.filter(pl.col("claim_strength").cast(pl.String) != "robust")
        assert bad.height == 0, (
            f"{bad.height} rows with claim_strength != robust"
        )


def test_aa_paper_claimable_tier_a():
    df = _read_paper_table("table_aa_paper_claimable_edges.csv")
    if df is None:
        pytest.skip("table not found")
    for tier_col in ("tier_i_actual", "tier_j_actual", "feature_tier"):
        if tier_col in df.columns:
            bad = df.filter(pl.col(tier_col).cast(pl.String) != "A")
            assert bad.height == 0, (
                f"{bad.height} rows where {tier_col} != A in headline table"
            )


# ─────────────────────────────────────────────────────────────────────────────
# B. No self-loops in A/A paper-claimable table
# ─────────────────────────────────────────────────────────────────────────────

def test_no_self_loops_in_aa_paper_claimable():
    df = _read_paper_table("table_aa_paper_claimable_edges.csv")
    if df is None:
        pytest.skip("table not found")
    for col_i, col_j in [
        ("node_i", "node_j"),
        ("causing_node", "caused_node"),
        ("source_node_id", "target_node_id"),
    ]:
        if col_i in df.columns and col_j in df.columns:
            loops = df.filter(
                pl.col(col_i).cast(pl.String) == pl.col(col_j).cast(pl.String)
            )
            assert loops.height == 0, (
                f"{loops.height} self-loops detected via {col_i}/{col_j}"
            )


def test_no_self_loops_in_aa_provenance_valid():
    df = _read_paper_table("table_aa_provenance_valid_edges.csv")
    if df is None:
        pytest.skip("table not found")
    for col_i, col_j in [
        ("node_i", "node_j"),
        ("causing_node", "caused_node"),
        ("source_node_id", "target_node_id"),
    ]:
        if col_i in df.columns and col_j in df.columns:
            loops = df.filter(
                pl.col(col_i).cast(pl.String) == pl.col(col_j).cast(pl.String)
            )
            assert loops.height == 0, (
                f"{loops.height} self-loops in table_aa_provenance_valid_edges.csv "
                f"via {col_i}/{col_j}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# C. table_claim_audit_summary.csv structure
# ─────────────────────────────────────────────────────────────────────────────

def test_claim_audit_exists_and_has_columns():
    df = _read_paper_table("table_claim_audit_summary.csv")
    assert df is not None, "table_claim_audit_summary.csv not found"
    required_cols = [
        "event_id", "n_total_edges", "n_AA_provenance",
        "n_AA_paper_claimable", "n_AB_paper_claimable", "n_BB_context",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column {col!r} in audit summary"


def test_claim_audit_five_events():
    df = _read_paper_table("table_claim_audit_summary.csv")
    if df is None:
        pytest.skip("table not found")
    events = df["event_id"].cast(pl.String).to_list()
    assert len(events) == 5, f"Expected 5 events in audit, got {len(events)}: {events}"


# ─────────────────────────────────────────────────────────────────────────────
# D. USDT/Curve 2023 has n_AA_paper_claimable >= 2
# ─────────────────────────────────────────────────────────────────────────────

def test_usdt_curve_has_aa_paper_claimable():
    df = _read_paper_table("table_claim_audit_summary.csv")
    if df is None:
        pytest.skip("table not found")
    row = df.filter(pl.col("event_id").cast(pl.String) == "usdt_curve_2023")
    assert row.height == 1, "usdt_curve_2023 not found in audit summary"
    if "n_AA_paper_claimable" in df.columns:
        n = int(row["n_AA_paper_claimable"].fill_null(0)[0])
        assert n >= 2, (
            f"Expected n_AA_paper_claimable >= 2 for usdt_curve_2023, got {n}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# E. Terra/LUNA has provenance candidates but zero paper-claimable A/A
# ─────────────────────────────────────────────────────────────────────────────

def test_terra_has_aa_provenance_but_not_paper():
    df = _read_paper_table("table_claim_audit_summary.csv")
    if df is None:
        pytest.skip("table not found")
    row = df.filter(pl.col("event_id").cast(pl.String) == "terra_luna_2022")
    if row.height == 0:
        pytest.skip("terra_luna_2022 not in audit")
    if "n_AA_provenance" in df.columns:
        n_prov = int(row["n_AA_provenance"].fill_null(0)[0])
        assert n_prov >= 2, (
            f"Expected n_AA_provenance >= 2 for terra_luna_2022, got {n_prov}"
        )
    if "n_AA_paper_claimable" in df.columns:
        n_paper = int(row["n_AA_paper_claimable"].fill_null(0)[0])
        assert n_paper == 0, (
            f"Expected n_AA_paper_claimable == 0 for terra_luna_2022, got {n_paper}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# F. Sparse-flow table is annotated but not paper-claimable
# ─────────────────────────────────────────────────────────────────────────────

def test_sparse_flow_annotated_not_paper_claimable():
    # Check both locations
    path = RAW_TBL / "table_sparse_events_usdc_svb_2023.csv"
    alt  = TABLE_DIR / "table_sparse_events_usdc_svb_2023.csv"
    actual_path = path if path.exists() else (alt if alt.exists() else None)

    if actual_path is None:
        pytest.skip("sparse-flow table not found — run make sparse_flow first")

    df = pl.read_csv(actual_path)
    assert "paper_claim_allowed" in df.columns, (
        "sparse table missing paper_claim_allowed — run 00c_claim_gate.py first"
    )
    paper_rows = df.filter(
        _bool_col(df, "paper_claim_allowed")
    )
    assert paper_rows.height == 0, (
        f"Expected 0 paper-claimable rows in sparse table, got {paper_rows.height}"
    )


def test_sparse_flow_has_provenance_annotation():
    path = RAW_TBL / "table_sparse_events_usdc_svb_2023.csv"
    alt  = TABLE_DIR / "table_sparse_events_usdc_svb_2023.csv"
    actual_path = path if path.exists() else (alt if alt.exists() else None)
    if actual_path is None:
        pytest.skip("sparse-flow table not found")

    df = pl.read_csv(actual_path)
    assert "provenance_claim_allowed" in df.columns, (
        "sparse table missing provenance_claim_allowed"
    )
    assert "claim_strength" in df.columns, (
        "sparse table missing claim_strength"
    )


# ─────────────────────────────────────────────────────────────────────────────
# G. README does not contain banned overclaim phrases
# ─────────────────────────────────────────────────────────────────────────────

def test_readme_no_banned_phrases():
    if not README.exists():
        pytest.skip("README.md not found")
    text = README.read_text()
    for phrase in README_BANNED_PHRASES:
        assert phrase not in text, (
            f"README contains banned overclaim phrase: {phrase!r}"
        )


def test_readme_has_required_phrases():
    if not README.exists():
        pytest.skip("README.md not found")
    text = README.read_text()
    required = [
        "Provenance-valid ≠ paper-claimable",
        "paper_claim_allowed == True",
        "historical full-depth CEX order books are not freely available",
    ]
    for phrase in required:
        assert phrase in text, (
            f"README missing required phrase: {phrase!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# H. All 12 expected figure files exist
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fname", EXPECTED_FIGURES)
def test_figure_exists(fname: str):
    p = FIG_DIR / fname
    assert p.exists(), (
        f"Missing paper figure: {fname}\n"
        f"Run: python scripts/13_make_paper_figures.py"
    )


# ─────────────────────────────────────────────────────────────────────────────
# I. No fixture leakage in paper outputs
# ─────────────────────────────────────────────────────────────────────────────

def test_no_fixture_rows_in_paper_tables():
    leaked = []
    for path in sorted(TABLE_DIR.glob("*.csv")):
        try:
            df = pl.read_csv(path)
        except Exception:
            continue
        if "uses_fixture" not in df.columns or "paper_claim_allowed" not in df.columns:
            continue
        leak = df.filter(
            _bool_col(df, "uses_fixture") & _bool_col(df, "paper_claim_allowed")
        )
        if leak.height > 0:
            leaked.append(f"{path.name}: {leak.height} rows")
    assert not leaked, (
        "Fixture rows found in paper-claimable tables:\n  " + "\n  ".join(leaked)
    )


# ─────────────────────────────────────────────────────────────────────────────
# J. figure_captions.md exists and has all 12 captions
# ─────────────────────────────────────────────────────────────────────────────

def test_figure_captions_complete():
    cap_path = PAPER_DIR / "figure_captions.md"
    assert cap_path.exists(), "paper/figure_captions.md not found"
    text = cap_path.read_text()
    for i in range(1, 13):
        assert f"Figure {i}" in text, f"Figure {i} caption missing from figure_captions.md"


# ─────────────────────────────────────────────────────────────────────────────
# K. All 18 Columbia figure-pack files exist
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fname", COLUMBIA_EXPECTED_FILES)
def test_columbia_figure_exists(fname: str):
    p = COLUMBIA_FIG_DIR / fname
    assert p.exists(), (
        f"Missing Columbia figure: {fname}\n"
        f"Run: python scripts/15_make_columbia_paper_pack.py"
    )


def test_columbia_figures_directory_exists():
    assert COLUMBIA_FIG_DIR.exists(), (
        f"Columbia figures directory not found: {COLUMBIA_FIG_DIR}\n"
        "Run: python scripts/15_make_columbia_paper_pack.py"
    )


def test_columbia_figures_count():
    if not COLUMBIA_FIG_DIR.exists():
        pytest.skip("Columbia figures directory not found")
    found = sorted(COLUMBIA_FIG_DIR.glob("*.png"))
    assert len(found) == len(COLUMBIA_EXPECTED_FILES), (
        f"Expected {len(COLUMBIA_EXPECTED_FILES)} Columbia figures, "
        f"found {len(found)}: {[f.name for f in found]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# L. Paper markdown must not contain banned overclaim phrases
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("phrase", PAPER_MARKDOWN_BANNED_PHRASES)
def test_paper_main_md_no_banned_phrases(phrase: str):
    md_path = PAPER_DIR / "main.md"
    if not md_path.exists():
        pytest.skip("paper/main.md not found")
    text = md_path.read_text().lower()
    assert phrase.lower() not in text, (
        f"paper/main.md contains banned overclaim phrase: {phrase!r}"
    )


@pytest.mark.parametrize("phrase", PAPER_MARKDOWN_BANNED_PHRASES)
def test_paper_readme_package_no_banned_phrases(phrase: str):
    md_path = PAPER_DIR / "README_paper_package.md"
    if not md_path.exists():
        pytest.skip("paper/README_paper_package.md not found")
    text = md_path.read_text().lower()
    assert phrase.lower() not in text, (
        f"paper/README_paper_package.md contains banned overclaim phrase: {phrase!r}"
    )
