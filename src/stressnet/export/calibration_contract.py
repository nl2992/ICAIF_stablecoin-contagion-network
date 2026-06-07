"""
calibration_contract.py
------------------------
The cross-repo artifact. Turns this repo's gated edge tables + metric estimates
into a SINGLE versioned contract that `stablecoin-abm` and
`stablecoin-contagion-gnn` consume as calibration / validation targets.

Why this exists
===============
Both downstream repos currently say "calibrate to the empirical half-lives and
propagation rho-hat from the contagion network." But under Option A this repo
robustly establishes ONE edge; everything else is suggestive or context-only.
If the downstream repos calibrate against weak targets as if they were robust,
the whole program's causal claims rest on sand.

The contract fixes this by tagging EVERY exported target with its
`claim_strength` and a machine-readable `tier`, and by defining a strict
consumption rule:

    PRIMARY targets   : claim_strength == "robust"        -> calibrate to these
    SECONDARY targets : "statistically_supported"|"suggestive" -> report-only,
                         caveated, never the basis of a headline causal claim
    CONTEXT targets   : "descriptive"|"context_only"      -> excluded from
                         calibration entirely; available for narrative only

Downstream repos MUST:
  - assert the contract `schema_version` they were built against,
  - calibrate primary parameters ONLY to PRIMARY targets,
  - print the claim_strength of every target they consume in their paper.

This makes the program's honesty machine-enforced rather than prose-promised.

Output
======
    calibration_contract.json   versioned, hashed, with provenance per target
    calibration_targets.csv     flat table for quick inspection / paper appendix
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import pandas as pd


SCHEMA_VERSION = "calib-contract/1.0.0"


# --------------------------------------------------------------------------- #
# Consumption tiers derived from claim_strength                               #
# --------------------------------------------------------------------------- #
class ConsumptionTier(str, Enum):
    PRIMARY = "primary"        # calibrate to these
    SECONDARY = "secondary"    # report-only, caveated
    CONTEXT = "context"        # narrative only, never calibrate


# Map the repo's claim_strength taxonomy onto consumption tiers.
_STRENGTH_TO_TIER = {
    "robust": ConsumptionTier.PRIMARY,
    "statistically_supported": ConsumptionTier.SECONDARY,
    "suggestive": ConsumptionTier.SECONDARY,
    "descriptive": ConsumptionTier.CONTEXT,
    "context_only": ConsumptionTier.CONTEXT,
}


def strength_to_tier(claim_strength: str) -> ConsumptionTier:
    if claim_strength not in _STRENGTH_TO_TIER:
        raise ValueError(
            f"unknown claim_strength '{claim_strength}'. "
            f"expected one of {sorted(_STRENGTH_TO_TIER)}"
        )
    return _STRENGTH_TO_TIER[claim_strength]


# --------------------------------------------------------------------------- #
# Target schema                                                               #
# --------------------------------------------------------------------------- #
class TargetKind(str, Enum):
    PROPAGATION_EDGE = "propagation_edge"   # directed/bidirectional lead-lag edge (rho-hat)
    OU_HALF_LIFE = "ou_half_life"           # peg-recovery half-life
    CONTAGION_MAGNITUDE = "contagion_magnitude"  # peak cross-venue depeg spread
    OTHER = "other"


@dataclass
class CalibrationTarget:
    target_id: str                 # stable unique key, e.g. "edge:usdt_curve_2023:3pool->crvusd"
    kind: str                      # TargetKind value
    event: str
    description: str
    value: float                   # the point estimate the downstream repo matches
    ci_low: float | None           # lower CI bound (None if not estimable)
    ci_high: float | None
    units: str                     # "hours", "bps", "dimensionless", ...
    claim_strength: str            # robust|statistically_supported|suggestive|descriptive|context_only
    consumption_tier: str          # filled from claim_strength
    edge_tier: str | None          # A_A_dex_flow | A_B_* | ... (None for non-edge targets)
    method: str                    # leadlag|transfer_entropy|granger|hawkes|sparse_flow|ou_fit
    p_value: float | None
    q_value_global: float | None   # global-FDR q from the attrition funnel
    n_events_support: int | None
    provenance_note: str           # short human-readable provenance summary

    def __post_init__(self) -> None:
        # enforce the consumption tier is consistent with claim_strength
        derived = strength_to_tier(self.claim_strength).value
        if self.consumption_tier != derived:
            self.consumption_tier = derived  # claim_strength is source of truth


# --------------------------------------------------------------------------- #
# Build targets from the repo's gated edge table + metric estimates           #
# --------------------------------------------------------------------------- #
def edge_targets_from_table(
    edges: pd.DataFrame,
    *,
    rho_col: str = "rho_hat",        # ADAPT: column holding the propagation estimate
    only_provenance_valid: bool = True,
) -> list[CalibrationTarget]:
    """Build propagation-edge targets from the gated edge table.

    Every provenance-valid edge becomes a target, tagged with its claim_strength
    so downstream repos can filter by consumption tier. Non-provenance-valid
    edges are excluded by default (they can't be calibration targets at all).
    """
    targets: list[CalibrationTarget] = []
    df = edges
    if only_provenance_valid and "provenance_claim_allowed" in df.columns:
        df = df[df["provenance_claim_allowed"].astype(bool)]

    for _, r in df.iterrows():
        strength = str(r.get("claim_strength", "context_only"))
        tid = f"edge:{r['event']}:{r.get('src','?')}->{r.get('dst','?')}"
        targets.append(CalibrationTarget(
            target_id=tid,
            kind=TargetKind.PROPAGATION_EDGE.value,
            event=str(r["event"]),
            description=f"propagation {r.get('src','?')} -> {r.get('dst','?')}",
            value=float(r[rho_col]) if rho_col in r and pd.notna(r[rho_col]) else float("nan"),
            ci_low=_get_opt(r, "rho_ci_low"),
            ci_high=_get_opt(r, "rho_ci_high"),
            units="dimensionless",
            claim_strength=strength,
            consumption_tier=strength_to_tier(strength).value,
            edge_tier=str(r.get("edge_tier")) if pd.notna(r.get("edge_tier")) else None,
            method=str(r.get("method", "leadlag")),
            p_value=_get_opt(r, "p_value"),
            q_value_global=_get_opt(r, "q_value_global"),
            n_events_support=_get_int_opt(r, "n_events_support"),
            provenance_note=str(r.get("provenance_note", "")),
        ))
    return targets


def metric_target(
    *, target_id: str, kind: TargetKind, event: str, description: str,
    value: float, units: str, claim_strength: str, method: str,
    ci: tuple[float, float] | None = None, p_value: float | None = None,
    n_events_support: int | None = None, provenance_note: str = "",
) -> CalibrationTarget:
    """Convenience builder for OU half-life / contagion-magnitude targets."""
    return CalibrationTarget(
        target_id=target_id, kind=kind.value, event=event, description=description,
        value=value, ci_low=(ci[0] if ci else None), ci_high=(ci[1] if ci else None),
        units=units, claim_strength=claim_strength,
        consumption_tier=strength_to_tier(claim_strength).value,
        edge_tier=None, method=method, p_value=p_value, q_value_global=None,
        n_events_support=n_events_support, provenance_note=provenance_note,
    )


# --------------------------------------------------------------------------- #
# Assemble + serialize the contract                                           #
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationContract:
    schema_version: str
    generated_utc: str
    source_repo: str
    source_git_sha: str
    targets: list[dict]
    consumption_rule: dict
    content_hash: str = ""

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    def to_csv(self, path: str) -> None:
        pd.DataFrame(self.targets).to_csv(path, index=False)


def build_contract(
    targets: list[CalibrationTarget],
    *,
    source_repo: str = "stablecoin-contagion-network",
    source_git_sha: str = "UNKNOWN",
) -> CalibrationContract:
    """Assemble the versioned, hashed contract from a list of targets."""
    target_dicts = [asdict(t) for t in targets]

    consumption_rule = {
        "PRIMARY": "claim_strength == 'robust'  -> calibrate primary parameters to these",
        "SECONDARY": "'statistically_supported' | 'suggestive'  -> report-only, caveated, "
                     "never the basis of a headline causal claim",
        "CONTEXT": "'descriptive' | 'context_only'  -> narrative only, excluded from calibration",
        "consumer_obligations": [
            "assert schema_version matches the version you built against",
            "calibrate primary parameters ONLY to PRIMARY targets",
            "print claim_strength of every consumed target in your paper",
            "if a PRIMARY target set is empty, you may not make a calibrated causal claim",
        ],
    }

    # content hash over the targets + rule (stable, sorted) for reproducibility stamping
    payload = json.dumps(
        {"targets": target_dicts, "rule": consumption_rule},
        sort_keys=True, default=str,
    ).encode()
    content_hash = hashlib.sha256(payload).hexdigest()[:16]

    return CalibrationContract(
        schema_version=SCHEMA_VERSION,
        generated_utc=datetime.now(timezone.utc).isoformat(),
        source_repo=source_repo,
        source_git_sha=source_git_sha,
        targets=target_dicts,
        consumption_rule=consumption_rule,
        content_hash=content_hash,
    )


def summarize(contract: CalibrationContract) -> pd.DataFrame:
    """Quick tier-count summary -- the line the downstream papers must quote."""
    df = pd.DataFrame(contract.targets)
    if df.empty:
        return df
    return (df.groupby("consumption_tier")
              .size()
              .reindex([t.value for t in ConsumptionTier], fill_value=0)
              .rename("n_targets")
              .reset_index())


# --------------------------------------------------------------------------- #
# Consumer-side helper (ship a copy to the downstream repos)                   #
# --------------------------------------------------------------------------- #
def load_primary_targets(contract_path: str, *, expected_schema: str = SCHEMA_VERSION):
    """Downstream repos call THIS. Asserts schema, returns only PRIMARY targets.

    Raises if the contract has zero PRIMARY targets -- which, under Option A,
    forces the downstream repo to acknowledge it cannot make a calibrated causal
    claim rather than silently calibrating to weak targets.
    """
    with open(contract_path) as f:
        c = json.load(f)
    if c["schema_version"] != expected_schema:
        raise ValueError(
            f"contract schema {c['schema_version']} != expected {expected_schema}; "
            "rebuild downstream calibration against the current contract."
        )
    primary = [t for t in c["targets"] if t["consumption_tier"] == "primary"]
    if not primary:
        raise RuntimeError(
            "calibration contract has ZERO primary (robust) targets. "
            "A calibrated causal claim is not permitted; report secondary "
            "targets as suggestive only."
        )
    return primary, c["content_hash"], c["schema_version"]


# --------------------------------------------------------------------------- #
# small helpers                                                               #
# --------------------------------------------------------------------------- #
def _get_opt(r: pd.Series, k: str) -> float | None:
    v = r.get(k)
    return float(v) if (v is not None and pd.notna(v)) else None


def _get_int_opt(r: pd.Series, k: str) -> int | None:
    v = r.get(k)
    return int(v) if (v is not None and pd.notna(v)) else None


# --------------------------------------------------------------------------- #
# Self-test                                                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # the one robust edge + a couple of weaker targets, mirroring real status
    edges = pd.DataFrame([
        dict(event="usdt_curve_2023", src="curve_3pool", dst="curve_crvusd_usdt",
             rho_hat=0.41, rho_ci_low=0.28, rho_ci_high=0.55,
             provenance_claim_allowed=True, claim_strength="robust",
             edge_tier="A_A_dex_flow", method="leadlag",
             p_value=0.0007, q_value_global=0.011, n_events_support=8,
             provenance_note="A/A Curve TokenExchange, Bonferroni + global FDR"),
        dict(event="terra_2022", src="curve_3pool", dst="curve_ust_wormhole",
             rho_hat=0.22, rho_ci_low=-0.05, rho_ci_high=0.49,
             provenance_claim_allowed=True, claim_strength="suggestive",
             edge_tier="A_A_dex_flow", method="leadlag",
             p_value=0.18, q_value_global=0.42, n_events_support=7,
             provenance_note="A/A provenance but not sig at hourly grid"),
        dict(event="usdc_svb_2023", src="usdc_mint_burn", dst="curve_3pool",
             rho_hat=float("nan"), provenance_claim_allowed=True,
             claim_strength="suggestive", edge_tier="A_A_onchain_settlement",
             method="sparse_flow", p_value=float("nan"), q_value_global=float("nan"),
             n_events_support=4, provenance_note="sparse; underpowered at 4 events"),
    ])

    targets = edge_targets_from_table(edges)
    # add an OU half-life target (suggestive) and a contagion-magnitude (robust on the one event)
    targets.append(metric_target(
        target_id="ou:usdt_curve_2023:3pool", kind=TargetKind.OU_HALF_LIFE,
        event="usdt_curve_2023", description="peg-recovery half-life, 3pool",
        value=6.4, units="hours", claim_strength="suggestive", method="ou_fit",
        ci=(3.1, 11.8), provenance_note="OU fit on hourly on-chain price proxy (Tier-B feature)"))
    targets.append(metric_target(
        target_id="mag:usdt_curve_2023", kind=TargetKind.CONTAGION_MAGNITUDE,
        event="usdt_curve_2023", description="peak cross-pool depeg spread",
        value=37.0, units="bps", claim_strength="robust", method="event_study",
        ci=(29.0, 45.0), n_events_support=8,
        provenance_note="A/A Tier-A flow + on-chain price"))

    contract = build_contract(targets, source_git_sha="abc1234")
    contract.to_json("/mnt/user-data/outputs/stressnet/export/calibration_contract.json")
    contract.to_csv("/mnt/user-data/outputs/stressnet/export/calibration_targets.csv")

    print("=== CONTRACT SUMMARY ===")
    print(summarize(contract).to_string(index=False))
    print(f"\nschema: {contract.schema_version}   hash: {contract.content_hash}")

    print("\n=== consumer-side load_primary_targets() ===")
    primary, h, ver = load_primary_targets(
        "/mnt/user-data/outputs/stressnet/export/calibration_contract.json")
    print(f"{len(primary)} PRIMARY target(s) available for calibration:")
    for t in primary:
        print(f"  - {t['target_id']}  ({t['kind']}, {t['claim_strength']}, "
              f"value={t['value']} {t['units']})")
    print("\nDownstream repos calibrate causal parameters to ONLY these; "
          "everything else is reported as suggestive.")
