"""Tests for strict paper-gate enforcement in 99_make_paper_outputs.py.

Covers:
- A/A  → claim_allowed True
- A/B  → claim_allowed True, claim_level suggestive
- B/B  → claim_allowed True, claim_level context_only
- fixture/A → claim_allowed False
- missing/A → claim_allowed False
- strict gate fails when fixture rows remain
- 99_make_paper_outputs --strict never writes fixture rows
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

from stressnet.evaluation.claim_gate import FIXTURE, decide_claim


# ---------------------------------------------------------------------------
# Unit-level claim decision tests (comprehensive)
# ---------------------------------------------------------------------------

def test_AA_is_claimable_and_directional():
    # No layer info → high-provenance fallback
    d = decide_claim("A", "A")
    assert d.claim_allowed is True
    assert d.claim_level == "A_A_high_provenance"
    assert d.uses_fixture is False

    # DEX/DEX with Tier-A feature → AMM-flow claim
    d_dex = decide_claim("A", "A", "DEX", "DEX", feature_col="usdc_net_sold_1h")
    assert d_dex.claim_level == "A_A_dex_flow"
    assert d_dex.claim_allowed is True


def test_AB_is_claimable_and_suggestive():
    d = decide_claim("A", "B")
    assert d.claim_allowed is True
    assert "suggestive" in d.claim_level
    assert d.uses_fixture is False


def test_BA_is_same_as_AB():
    d = decide_claim("B", "A")
    assert d.claim_allowed is True
    assert "suggestive" in d.claim_level


def test_BB_is_claimable_but_context_only():
    d = decide_claim("B", "B")
    assert d.claim_allowed is True
    assert "context_only" in d.claim_level
    assert d.uses_fixture is False


def test_fixture_A_is_blocked():
    d = decide_claim(FIXTURE, "A")
    assert d.claim_allowed is False
    assert d.uses_fixture is True
    assert d.claim_level == "fixture_disallowed"


def test_A_fixture_is_blocked():
    d = decide_claim("A", FIXTURE)
    assert d.claim_allowed is False
    assert d.uses_fixture is True


def test_missing_A_is_blocked():
    d = decide_claim(None, "A")
    assert d.claim_allowed is False
    assert d.uses_fixture is False
    assert "missing" in d.edge_tier_actual or d.claim_level in ("C_taxonomy_only",)


def test_C_taxonomy_only():
    d = decide_claim("C", "B")
    assert d.claim_allowed is False
    assert d.claim_level == "C_taxonomy_only"


# ---------------------------------------------------------------------------
# Integration: _enforce_clean in 99_make_paper_outputs
# ---------------------------------------------------------------------------

# Import the helper directly from the script
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from importlib import import_module as _imp

def _get_enforce_clean():
    """Import _enforce_clean without executing main()."""
    import importlib.util, types
    spec = importlib.util.spec_from_file_location(
        "paper_outputs",
        Path(__file__).parent.parent / "scripts" / "99_make_paper_outputs.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._enforce_clean


def test_enforce_clean_filters_non_claimable():
    enforce = _get_enforce_clean()
    df = pl.DataFrame({
        "node_i": ["a", "b"],
        "node_j": ["c", "d"],
        "claim_allowed": [True, False],
        "uses_fixture": [False, False],
    })
    out = enforce(df, "test_table", strict=False)
    assert out.height == 1
    assert out["node_i"][0] == "a"


def test_enforce_clean_strict_raises_on_fixture():
    enforce = _get_enforce_clean()
    df = pl.DataFrame({
        "node_i": ["a"],
        "node_j": ["b"],
        "claim_allowed": [True],
        "uses_fixture": [True],          # ← fixture leaked through
        "edge_tier_actual": ["B"],
    })
    with pytest.raises(SystemExit):
        enforce(df, "test_table", strict=True)


def test_enforce_clean_strict_raises_on_fixture_tier():
    enforce = _get_enforce_clean()
    df = pl.DataFrame({
        "node_i": ["a"],
        "claim_allowed": [True],
        "uses_fixture": [False],
        "edge_tier_actual": [FIXTURE],   # ← fixture tier
    })
    with pytest.raises(SystemExit):
        enforce(df, "test_table", strict=True)


def test_enforce_clean_strict_passes_clean_df():
    enforce = _get_enforce_clean()
    df = pl.DataFrame({
        "node_i": ["a"],
        "claim_allowed": [True],
        "uses_fixture": [False],
        "edge_tier_actual": ["B"],
    })
    out = enforce(df, "test_table", strict=True)
    assert out.height == 1


# ---------------------------------------------------------------------------
# Integration: consolidate_table writes no fixture rows in strict mode
# ---------------------------------------------------------------------------

def test_consolidate_table_strict_writes_no_fixture(tmp_path: Path):
    """consolidate_table in strict mode must not include fixture-derived rows."""
    from importlib import import_module as _imod
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "paper_outputs",
        Path(__file__).parent.parent / "scripts" / "99_make_paper_outputs.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Write a fake per-event table with one clean and one fixture row
    per_event = tmp_path / "table_leadlag_tests_evt.csv"
    pl.DataFrame({
        "event_id":        ["evt", "evt"],
        "node_i":          ["real_a", "real_a"],
        "node_j":          ["real_b", "fixture_c"],
        "p_value":         [0.001, 0.001],
        "claim_allowed":   [True, False],
        "uses_fixture":    [False, True],
        "edge_tier_actual": ["B", FIXTURE],
    }).write_csv(per_event)

    out = mod.consolidate_table(
        "table_leadlag_tests",
        "table_leadlag_tests_{event}.csv",
        tables_dir=tmp_path,
        out_dir=tmp_path,
        events=["evt"],
        strict=True,
    )
    assert out is not None
    assert out.height == 1
    assert out["claim_allowed"][0] is True

    written = pl.read_csv(tmp_path / "table_leadlag_tests.csv")
    assert written.height == 1
    assert FIXTURE not in written["edge_tier_actual"].to_list()


# ---------------------------------------------------------------------------
# Makefile regression tests
# ---------------------------------------------------------------------------

def test_paper_gate_uses_strict():
    """Makefile paper_gate target must invoke 99_make_paper_outputs.py with --strict."""
    makefile = Path(__file__).parent.parent / "Makefile"
    assert "99_make_paper_outputs.py --strict" in makefile.read_text()


def test_all_routes_to_empirical_all():
    """Makefile 'all' target must route to empirical_all, not demo_all or fixture pipeline."""
    makefile = Path(__file__).parent.parent / "Makefile"
    text = makefile.read_text()
    assert "all: empirical_all" in text
