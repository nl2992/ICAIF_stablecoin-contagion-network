from pathlib import Path

import polars as pl

from stressnet.evaluation.claim_gate import (
    FIXTURE,
    annotate_edge_table,
    decide_claim,
    weaker_tier,
)


def test_weaker_tier_caps_edge_by_lower_provenance() -> None:
    assert weaker_tier("A", "A") == "A"
    assert weaker_tier("A", "B") == "B"
    assert weaker_tier("B", "C") == "C"
    assert weaker_tier("A", FIXTURE) == FIXTURE


def test_claim_decision_blocks_fixture_endpoint() -> None:
    decision = decide_claim("A", FIXTURE)

    assert decision.uses_fixture is True
    assert decision.claim_allowed is False
    assert decision.edge_tier_actual == FIXTURE
    assert decision.claim_level == "fixture_disallowed"


def test_claim_decision_assigns_expected_claim_levels() -> None:
    # No layer → falls through to high-provenance fallback
    assert decide_claim("A", "A").claim_level == "A_A_high_provenance"
    # DEX/DEX layer → AMM-flow claim
    assert decide_claim("A", "A", "DEX", "DEX").claim_level == "A_A_dex_flow"
    assert decide_claim("A", "B").claim_level == "A_B_suggestive_directional"
    assert decide_claim("B", "B").claim_level == "B_B_context_only"
    assert decide_claim("C", "A").claim_allowed is False
    assert decide_claim("C", "A").claim_level == "C_taxonomy_only"


def test_claim_decision_settlement_layers() -> None:
    d = decide_claim("A", "A", "mint_burn", "DEX")
    assert d.claim_level == "A_A_onchain_settlement"
    assert d.claim_allowed is True

    d = decide_claim("A", "A", "onchain_flow", "CEX")
    assert d.claim_level == "A_A_onchain_settlement"


def test_feature_cap_demotes_aa_to_ab() -> None:
    # reserve_imbalance is Tier B (derived proxy) — caps A/A to A/B
    d = decide_claim("A", "A", "DEX", "DEX", feature_col="reserve_imbalance")
    assert d.claim_level == "A_B_suggestive_directional"
    assert d.claim_allowed is True
    assert d.feature_tier == "B"

    # usdc_net_sold_1h is Tier A — A/A stays A/A
    d = decide_claim("A", "A", "DEX", "DEX", feature_col="usdc_net_sold_1h")
    assert d.claim_level == "A_A_dex_flow"
    assert d.feature_tier == "A"


def test_annotate_edge_table_adds_claim_metadata(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "event_id": ["event_a", "event_a"],
            "node_i": ["real_a", "real_a"],
            "node_j": ["real_b", "fixture_c"],
            "p_value": [0.01, 0.01],
        }
    )
    tier_map = {"event_a": {"real_a": "A", "real_b": "B", "fixture_c": FIXTURE}}

    out = annotate_edge_table(
        df,
        tier_map,
        source_col="node_i",
        target_col="node_j",
        table_path=tmp_path / "table_leadlag_tests_event_a.csv",
    )

    assert {
        "tier_i_actual",
        "tier_j_actual",
        "edge_tier_actual",
        "uses_fixture",
        "claim_allowed",
        "claim_level",
        "claim_reason",
    }.issubset(out.columns)
    assert out["claim_allowed"].to_list() == [True, False]
    assert out["edge_tier_actual"].to_list() == ["B", FIXTURE]
    assert out["claim_level"].to_list() == [
        "A_B_suggestive_directional",
        "fixture_disallowed",
    ]


def test_var_fallback_rows_are_diagnostic_only(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "event_id": ["event_a"],
            "causing_node": ["real_a"],
            "caused_node": ["real_b"],
            "method": ["var_coeff_fallback"],
        }
    )
    tier_map = {"event_a": {"real_a": "A", "real_b": "A"}}

    out = annotate_edge_table(
        df,
        tier_map,
        source_col="causing_node",
        target_col="caused_node",
        table_path=tmp_path / "table_var_spillovers_event_a.csv",
    )

    assert out["claim_allowed"].to_list() == [False]
    assert out["claim_level"].to_list() == ["diagnostic_only"]
