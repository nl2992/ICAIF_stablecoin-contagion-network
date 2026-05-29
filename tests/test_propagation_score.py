"""Tests for propagation-intensity scoring."""

from __future__ import annotations

from stressnet.evaluation.propagation_score import (
    PropagationScoreInputs,
    fixture_penalty,
    paper_claim_tier,
    placebo_gap,
    propagation_intensity_score,
)


def test_placebo_gap_is_positive_part_only():
    assert placebo_gap(0.5, 0.2) == 0.3
    assert placebo_gap(0.2, 0.5) == 0.0
    assert placebo_gap(None, 0.5) == 0.0


def test_fixture_penalty_is_proportional():
    assert fixture_penalty(0, 10) == 0.0
    assert fixture_penalty(5, 10) == 2.5
    assert fixture_penalty(1, 0) == 0.0


def test_propagation_score_rewards_aa_edges_and_penalizes_fixtures():
    clean = propagation_intensity_score(
        PropagationScoreInputs(
            aa_paper_edges=2,
            ab_paper_edges=1,
            bb_context_edges=0,
            mean_abs_effect=0.4,
            true_sig_rate=0.5,
            placebo_sig_rate=0.1,
            fixture_nodes=0,
            total_nodes=5,
        )
    )
    dirty = propagation_intensity_score(
        PropagationScoreInputs(
            aa_paper_edges=2,
            ab_paper_edges=1,
            bb_context_edges=0,
            mean_abs_effect=0.4,
            true_sig_rate=0.5,
            placebo_sig_rate=0.1,
            fixture_nodes=3,
            total_nodes=5,
        )
    )

    assert clean > dirty
    assert clean == 36.0


def test_paper_claim_tier_prefers_highest_evidence_level():
    assert paper_claim_tier({"n_AA_paper_claimable": 1}) == "paper_claimable_AA"
    assert paper_claim_tier({"n_AA_paper_claimable": 0, "n_AB_paper_claimable": 2}) == "suggestive_AB"
    assert paper_claim_tier({"n_BB_context": 3}) == "context_only"
    assert paper_claim_tier({}) == "insufficient_supported_edges"
