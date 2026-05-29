"""
scripts/14_validate_paper_package.py
=====================================
Read-only validation of the complete paper package.

Runs checks A–K and exits nonzero if any check fails.

Usage:
    python scripts/14_validate_paper_package.py
    python scripts/14_validate_paper_package.py --verbose

Exit codes:
    0  — all checks PASS
    1  — one or more checks FAIL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

REPO_ROOT  = Path(__file__).resolve().parents[1]
TABLE_DIR  = REPO_ROOT / "results" / "paper" / "tables"
RAW_TBL    = REPO_ROOT / "results" / "tables"
FIG_DIR    = REPO_ROOT / "results" / "paper" / "figures"
README     = REPO_ROOT / "README.md"
DATA_INV   = REPO_ROOT / "DATA_INVENTORY.md"
PAPER_DIR  = REPO_ROOT / "paper"

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

COLUMBIA_FIG_DIR = REPO_ROOT / "results" / "paper" / "figures_columbia"

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

README_REQUIRED_PHRASES = [
    "Provenance-Aware Stablecoin Stress Propagation Networks",
    "Provenance-valid ≠ paper-claimable",
    "paper_claim_allowed == True",
    "historical full-depth CEX order books are not freely available",
    "does not claim",
]

README_BANNED_PHRASES = [
    "headline microstructure claims",
    "A/A edges are confirmed",
]

# Phrases that must never appear in any paper markdown (main.md, README_paper_package.md)
PAPER_MARKDOWN_BANNED_PHRASES = [
    "proves contagion",
    "causal contagion",
    "directional microstructure transmission",
]

_AA_LEVELS = frozenset({
    "A_A_dex_flow",
    "A_A_onchain_settlement",
    "A_A_cex_microstructure",
    "A_A_high_provenance",
})

# ─────────────────────────────────────────────────────────────────────────────
# Result accumulator
# ─────────────────────────────────────────────────────────────────────────────

class _Check:
    def __init__(self, name: str):
        self.name    = name
        self.passed  = True
        self.details: list[str] = []

    def fail(self, msg: str) -> None:
        self.passed = False
        self.details.append(f"  ✗ {msg}")

    def ok(self, msg: str) -> None:
        self.details.append(f"  ✓ {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────

def check_A(verbose: bool) -> _Check:
    """A. table_claim_gate_all_events.csv exists and has required columns."""
    c = _Check("A – claim gate summary columns")
    path = TABLE_DIR / "table_claim_gate_all_events.csv"
    if not path.exists():
        c.fail(f"{path.name} not found")
        return c
    df = pl.read_csv(path)
    for col in ("paper_claimable_rows", "claimable_rows", "fixture_or_missing_rows"):
        if col in df.columns:
            c.ok(f"column {col!r} present")
        else:
            c.fail(f"column {col!r} missing from {path.name}")
    return c


def check_B(verbose: bool) -> _Check:
    """B. No fixture rows survive in paper-claimable tables."""
    c = _Check("B – no fixture leakage in paper outputs")
    for path in sorted(TABLE_DIR.glob("*.csv")):
        try:
            df = pl.read_csv(path)
        except Exception:
            continue
        # Only check tables that have at least one relevant column
        has_fixture  = "uses_fixture" in df.columns
        has_paper    = "paper_claim_allowed" in df.columns
        has_tier_col = "edge_tier_actual" in df.columns
        if not (has_fixture or has_paper or has_tier_col):
            continue

        if has_fixture and has_paper:
            leak = df.filter(
                (pl.col("uses_fixture").cast(pl.String).str.to_lowercase().is_in(["true","1","yes"]))
                & (pl.col("paper_claim_allowed").cast(pl.String).str.to_lowercase().is_in(["true","1","yes"]))
            )
            if leak.height > 0:
                c.fail(f"{path.name}: {leak.height} fixture+paper_claim rows leaked")
            elif verbose:
                c.ok(f"{path.name}: no fixture leakage")
        if has_tier_col and has_paper:
            leak2 = df.filter(
                (pl.col("edge_tier_actual").cast(pl.String).is_in(["fixture_non_empirical","missing"]))
                & (pl.col("paper_claim_allowed").cast(pl.String).str.to_lowercase().is_in(["true","1","yes"]))
            )
            if leak2.height > 0:
                c.fail(f"{path.name}: {leak2.height} fixture-tier paper_claim rows")
    if c.passed:
        c.ok("no fixture leakage in any paper table")
    return c


def check_C(verbose: bool) -> _Check:
    """C. table_aa_paper_claimable_edges.csv contains only the headline A/A rows."""
    c = _Check("C – headline A/A paper-claimable rows")
    path = TABLE_DIR / "table_aa_paper_claimable_edges.csv"
    if not path.exists():
        c.fail(f"{path.name} not found")
        return c
    df = pl.read_csv(path)
    if df.height == 0:
        c.fail("table is empty — expected headline rows")
        return c

    # Must contain usdt_curve_2023 A/A DEX-flow rows
    if "event_id" in df.columns:
        events = df["event_id"].cast(pl.String).unique().to_list()
        if "usdt_curve_2023" not in events:
            c.fail("no usdt_curve_2023 rows in paper-claimable table")
        else:
            c.ok("usdt_curve_2023 present")
        non_usdt = [e for e in events if e and e != "usdt_curve_2023"]
        if non_usdt:
            c.fail(f"unexpected non-USDT events in headline table: {non_usdt}")
        else:
            c.ok("no non-USDT events in headline A/A table")

    # Mandatory field values
    req_checks = [
        ("claim_level",          "A_A_dex_flow"),
        ("feature_col",          "usdc_net_sold_1h"),
        ("paper_claim_allowed",  None),  # just check col exists
        ("claim_strength",       "robust"),
    ]
    for col, expected in req_checks:
        if col not in df.columns:
            c.fail(f"missing column {col!r}")
            continue
        if expected is not None:
            bad = df.filter(pl.col(col).cast(pl.String) != expected)
            if bad.height > 0:
                c.fail(f"{bad.height} rows where {col} != {expected!r}")
            else:
                c.ok(f"all rows have {col}={expected!r}")

    # Tier A on both endpoints
    for tier_col in ("tier_i_actual", "tier_j_actual", "feature_tier"):
        if tier_col in df.columns:
            bad = df.filter(pl.col(tier_col).cast(pl.String) != "A")
            if bad.height > 0:
                c.fail(f"{bad.height} rows where {tier_col} != A")
            else:
                c.ok(f"all rows have {tier_col}=A")

    n_aa = df.height
    c.ok(f"{n_aa} A/A paper-claimable headline rows")
    return c


def check_D(verbose: bool) -> _Check:
    """D. A/A provenance-valid table must not contain self-loops."""
    c = _Check("D – no self-loops in A/A provenance-valid table")
    path = TABLE_DIR / "table_aa_provenance_valid_edges.csv"
    if not path.exists():
        c.fail(f"{path.name} not found")
        return c
    df = pl.read_csv(path)
    self_loops = 0
    for col_i, col_j in [
        ("node_i", "node_j"),
        ("causing_node", "caused_node"),
        ("source_node_id", "target_node_id"),
        ("source", "target"),
        ("source_node", "target_node"),
    ]:
        if col_i in df.columns and col_j in df.columns:
            loops = df.filter(pl.col(col_i).cast(pl.String) == pl.col(col_j).cast(pl.String))
            self_loops += loops.height
            if loops.height > 0:
                c.fail(f"{loops.height} self-loops found via {col_i}/{col_j}")
    if self_loops == 0:
        c.ok("no self-loops detected")
    return c


def check_E(verbose: bool) -> _Check:
    """E. Sparse-flow table is claim-gated and not paper-claimable."""
    c = _Check("E – sparse-flow table annotated (not paper-claimable)")
    path = RAW_TBL / "table_sparse_events_usdc_svb_2023.csv"
    if not path.exists():
        # also check paper dir
        alt = TABLE_DIR / "table_sparse_events_usdc_svb_2023.csv"
        if alt.exists():
            path = alt
        else:
            c.fail("table_sparse_events_usdc_svb_2023.csv not found")
            return c

    df = pl.read_csv(path)
    required_cols = [
        "provenance_claim_allowed",
        "statistical_claim_allowed",
        "paper_claim_allowed",
        "claim_strength",
    ]
    for col in required_cols:
        if col in df.columns:
            c.ok(f"sparse table has {col!r}")
        else:
            c.fail(f"sparse table missing {col!r}")

    if "paper_claim_allowed" in df.columns:
        paper_rows = df.filter(
            pl.col("paper_claim_allowed").cast(pl.String).str.to_lowercase().is_in(["true","1","yes"])
        )
        if paper_rows.height > 0:
            c.fail(f"unexpected: {paper_rows.height} paper_claim_allowed rows in sparse table")
        else:
            c.ok("sparse rows are not paper-claimable (expected)")

    if "provenance_claim_allowed" in df.columns:
        prov_rows = df.filter(
            pl.col("provenance_claim_allowed").cast(pl.String).str.to_lowercase().is_in(["true","1","yes"])
        )
        if prov_rows.height > 0:
            c.ok(f"{prov_rows.height} provenance-valid rows in sparse table (expected)")
    return c


def check_F(verbose: bool) -> _Check:
    """F. README required phrases present; banned phrases absent."""
    c = _Check("F – README narrative accuracy")
    if not README.exists():
        c.fail("README.md not found")
        return c
    text = README.read_text()
    for phrase in README_REQUIRED_PHRASES:
        if phrase in text:
            c.ok(f"README contains required: {phrase[:60]!r}")
        else:
            c.fail(f"README missing required phrase: {phrase!r}")
    for phrase in README_BANNED_PHRASES:
        if phrase in text:
            c.fail(f"README contains banned phrase: {phrase!r}")
        else:
            c.ok(f"README does not contain banned: {phrase!r}")
    return c


def check_G(verbose: bool) -> _Check:
    """G. DATA_INVENTORY distinguishes claim tiers correctly."""
    c = _Check("G – DATA_INVENTORY narrative accuracy")
    if not DATA_INV.exists():
        c.fail("DATA_INVENTORY.md not found")
        return c
    text = DATA_INV.read_text()
    required = [
        "provenance-valid",
        "paper-claimable",
        "sparse",
        "no free historical",
    ]
    banned = [
        "microstructure unlocked",
        "CEX microstructure confirmed",
    ]
    for phrase in required:
        if phrase.lower() in text.lower():
            c.ok(f"DATA_INVENTORY contains: {phrase!r}")
        else:
            c.fail(f"DATA_INVENTORY missing: {phrase!r}")
    for phrase in banned:
        if phrase.lower() in text.lower():
            c.fail(f"DATA_INVENTORY contains banned phrase: {phrase!r}")
    return c


def check_H(verbose: bool) -> _Check:
    """H. All 12 expected figure files exist."""
    c = _Check("H – all 12 paper figures present")
    for fname in EXPECTED_FIGURES:
        p = FIG_DIR / fname
        if p.exists():
            c.ok(f"  {fname}")
        else:
            c.fail(f"missing figure: {fname}")
    return c


def check_I(verbose: bool) -> _Check:
    """I. figure_captions.md exists and has non-empty captions for all 12 figures."""
    c = _Check("I – figure_captions.md complete")
    cap_path = PAPER_DIR / "figure_captions.md"
    if not cap_path.exists():
        c.fail("paper/figure_captions.md not found")
        return c
    text = cap_path.read_text()
    for i in range(1, 13):
        marker = f"Figure {i}"
        if marker in text:
            c.ok(f"{marker} caption present")
        else:
            c.fail(f"{marker} caption missing from figure_captions.md")
    return c


def check_I_paper_md(verbose: bool) -> _Check:
    """I-ext. Paper markdown files must not contain banned overclaim phrases."""
    c = _Check("I-ext – paper markdown banned phrases absent")
    candidates = [
        PAPER_DIR / "main.md",
        PAPER_DIR / "README_paper_package.md",
    ]
    for md_path in candidates:
        if not md_path.exists():
            continue
        text = md_path.read_text()
        for phrase in PAPER_MARKDOWN_BANNED_PHRASES:
            if phrase.lower() in text.lower():
                c.fail(f"{md_path.name}: banned phrase found: {phrase!r}")
            elif verbose:
                c.ok(f"{md_path.name}: does not contain banned phrase: {phrase!r}")
    if c.passed:
        c.ok("no banned overclaim phrases in paper markdown files")
    return c


def check_K(verbose: bool) -> _Check:
    """K. All 18 Columbia figure-pack files exist."""
    c = _Check("K – all 18 Columbia figures present")
    if not COLUMBIA_FIG_DIR.exists():
        c.fail(f"Columbia figures directory not found: {COLUMBIA_FIG_DIR}")
        c.fail("Run: python scripts/15_make_columbia_paper_pack.py")
        return c
    for fname in COLUMBIA_EXPECTED_FILES:
        p = COLUMBIA_FIG_DIR / fname
        if p.exists():
            if verbose:
                c.ok(f"  {fname}")
        else:
            c.fail(f"missing Columbia figure: {fname}")
    if c.passed:
        c.ok(f"all {len(COLUMBIA_EXPECTED_FILES)} Columbia figures present")
    return c


def check_J(verbose: bool) -> _Check:
    """J. Compact summary report of key counts."""
    c = _Check("J – key counts summary")
    audit_path = TABLE_DIR / "table_claim_audit_summary.csv"
    aa_path    = TABLE_DIR / "table_aa_paper_claimable_edges.csv"

    n_aa_paper = 0
    n_ab_paper = 0
    n_bb       = 0
    n_paper_total = 0
    n_fixture_leaked = 0

    if audit_path.exists():
        df = pl.read_csv(audit_path).filter(
            pl.col("event_id").cast(pl.String).is_in(
                ["usdc_svb_2023","terra_luna_2022","usdt_curve_2023","ftx_2022","busd_2023"]
            )
        )
        if "n_AA_paper_claimable" in df.columns:
            n_aa_paper = int(df["n_AA_paper_claimable"].fill_null(0).sum())
        if "n_AB_paper_claimable" in df.columns:
            n_ab_paper = int(df["n_AB_paper_claimable"].fill_null(0).sum())
        if "n_BB_context" in df.columns:
            n_bb = int(df["n_BB_context"].fill_null(0).sum())
        if "n_paper_claimable" in df.columns:
            n_paper_total = int(df["n_paper_claimable"].fill_null(0).sum())

    # fixture leak count
    for path in sorted(TABLE_DIR.glob("*.csv")):
        try:
            df2 = pl.read_csv(path)
        except Exception:
            continue
        if "uses_fixture" in df2.columns and "paper_claim_allowed" in df2.columns:
            n_fixture_leaked += df2.filter(
                (pl.col("uses_fixture").cast(pl.String).str.to_lowercase().is_in(["true","1","yes"]))
                & (pl.col("paper_claim_allowed").cast(pl.String).str.to_lowercase().is_in(["true","1","yes"]))
            ).height

    all_figs_ok = all((FIG_DIR / f).exists() for f in EXPECTED_FIGURES)

    c.ok(f"total paper-claimable edges:    {n_paper_total}")
    c.ok(f"A/A paper-claimable edges:      {n_aa_paper}")
    c.ok(f"A/B paper-claimable edges:      {n_ab_paper}")
    c.ok(f"B/B context edges:              {n_bb}")
    c.ok(f"fixture rows in paper outputs:  {n_fixture_leaked}")
    c.ok(f"all 12 figures present:         {all_figs_ok}")

    if n_fixture_leaked > 0:
        c.fail(f"{n_fixture_leaked} fixture rows leaked into paper outputs")
    if n_aa_paper == 0:
        c.fail("no A/A paper-claimable edges — expected 2 USDT/Curve rows")
    if not all_figs_ok:
        c.fail("one or more figures missing")
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the paper package.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print all passing checks, not just failures.")
    args = parser.parse_args()

    checks = [
        check_A, check_B, check_C, check_D,
        check_E, check_F, check_G, check_H,
        check_I, check_I_paper_md, check_J, check_K,
    ]

    results: list[_Check] = []
    for fn in checks:
        r = fn(args.verbose)
        results.append(r)

    print()
    print("=" * 62)
    print("  PAPER PACKAGE VALIDATION REPORT")
    print("=" * 62)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"\n[{status}] Check {r.name}")
        if not r.passed or args.verbose:
            for line in r.details:
                print(line)

    n_fail = sum(1 for r in results if not r.passed)
    n_pass = len(results) - n_fail
    print()
    print("=" * 62)
    if n_fail == 0:
        print(f"  RESULT: PASS  ({n_pass}/{len(results)} checks passed)")
    else:
        print(f"  RESULT: FAIL  ({n_fail} of {len(results)} checks failed)")
    print("=" * 62)
    print()

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
