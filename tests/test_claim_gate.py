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
    assert decide_claim("A", "A").claim_level == "A_A_directional_microstructure"
    assert decide_claim("A", "B").claim_level == "A_B_suggestive_directional"
    assert decide_claim("B", "B").claim_level == "B_B_context_only"
    assert decide_claim("C", "A").claim_allowed is False
    assert decide_claim("C", "A").claim_level == "C_taxonomy_only"


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
