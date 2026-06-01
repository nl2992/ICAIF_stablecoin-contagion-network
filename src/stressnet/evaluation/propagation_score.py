"""Composite propagation-intensity scores for event-level comparison.

The score is intentionally simple and audit-friendly. It rewards
paper-claimable high-provenance edges, includes effect-size strength when
available, and penalizes placebo-like behavior. It is not a structural
causal estimand; it is a reusable benchmark summary statistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PropagationScoreInputs:
    """Inputs needed to score one event."""

    aa_paper_edges: int = 0
    ab_paper_edges: int = 0
    bb_context_edges: int = 0
    mean_abs_effect: float = 0.0
    true_sig_rate: float | None = None
    placebo_sig_rate: float | None = None
    fixture_nodes: int = 0
    total_nodes: int = 0


def placebo_gap(true_sig_rate: float | None, placebo_sig_rate: float | None) -> float:
    """Return positive true-minus-placebo significance-rate gap."""
    if true_sig_rate is None or placebo_sig_rate is None:
        return 0.0
    return max(0.0, float(true_sig_rate) - float(placebo_sig_rate))


def fixture_penalty(fixture_nodes: int, total_nodes: int) -> float:
    """Return a proportional fixture penalty in score points."""
    if total_nodes <= 0:
        return 0.0
    return 5.0 * max(0.0, min(1.0, fixture_nodes / total_nodes))


def propagation_intensity_score(inputs: PropagationScoreInputs) -> float:
    """Compute an event-level propagation-intensity score.

    Weights:
      * A/A paper-claimable edge: 15 points
      * A/B paper-claimable edge: 2 points
      * B/B contextual edge: 0.25 points
      * Mean absolute effect size: up to 5 scaled points
      * Placebo gap: up to 5 points
      * Fixture penalty: up to -5 points
    """
    effect_component = 5.0 * max(0.0, min(1.0, inputs.mean_abs_effect))
    placebo_component = 5.0 * placebo_gap(inputs.true_sig_rate, inputs.placebo_sig_rate)
    score = (
        15.0 * inputs.aa_paper_edges
        + 2.0 * inputs.ab_paper_edges
        + 0.25 * inputs.bb_context_edges
        + effect_component
        + placebo_component
        - fixture_penalty(inputs.fixture_nodes, inputs.total_nodes)
    )
    return round(max(0.0, score), 6)


def paper_claim_tier(row: dict[str, Any]) -> str:
    """Assign a plain-language event evidence tier from score ingredients."""
    if int(row.get("n_AA_paper_claimable") or 0) > 0:
        return "paper_claimable_AA"
    if int(row.get("n_AB_paper_claimable") or 0) > 0:
        return "suggestive_AB"
    if int(row.get("n_BB_context") or 0) > 0:
        return "context_only"
    return "insufficient_supported_edges"
