"""
attrition_funnel.py
-------------------
The empirical spine of the gating-as-contribution paper (Option A).

Turns the per-edge claim-gate tables into the headline artifact: how many
candidate edges enter, how many survive each gate, and -- crucially -- WHY the
rest die. The funnel IS the thesis ("most contagion claims don't survive a
joint provenance+significance gate"), so this module is the most load-bearing
figure-and-table generator in the repo.

Three things it does that the existing per-event audit does not:

1. POOLED global multiple-comparison correction.
   Within-event Bonferroni is not enough: you test many pairs across five
   events, so the surviving edge must be shown to survive correction over the
   ENTIRE candidate set, or a reviewer reads it as the winner of many
   unreported tests. This module applies a global FDR (Benjamini-Hochberg)
   across all candidate p-values and reports both the local and global verdict.

2. FAILURE-REASON decomposition.
   Every edge that doesn't reach paper-claimable is attributed to the FIRST
   gate it failed: fixture-blocked -> tier-capped -> failed-significance ->
   underpowered. This is what makes the funnel a diagnosis rather than a blank.

3. Per-event AND pooled funnels in one pass, emitted as both a tidy table
   (for the paper) and a plot-ready structure (for the stacked-bar figure).

Input contract
==============
A long-format edge table (one row per candidate edge, per event) carrying at
least the columns below. These mirror the repo's existing claim-gate output;
adapt names at the `# ADAPT:` marks if yours differ.

    event                     str
    src, dst                  str          (node ids)
    provenance_claim_allowed  bool
    statistical_claim_allowed bool
    paper_claim_allowed       bool
    claim_strength            str          descriptive|suggestive|statistically_supported|robust
    is_fixture                bool         (fixture/diagnostic row)
    edge_tier                 str          A_A_* | A_B_* | B_B_*   (capped tier)
    p_value                   float|nan    (method p before global correction; nan if untested)
    n_events_support          int          (for the underpowered check on sparse edges)

Outputs
=======
    funnel_table.csv          tidy per-event + pooled stage counts
    failure_reasons.csv       per-event + pooled counts by first-failed gate
    funnel_global_corrected.csv  edges with global q-values + global verdict
    (plot helpers return matplotlib-ready arrays; figure script calls them)

No plotting dependency at import time; the figure helper imports matplotlib
lazily so the table path stays light for CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Gate stages and failure reasons                                             #
# --------------------------------------------------------------------------- #
class Stage(str, Enum):
    CANDIDATE = "candidate"
    PROVENANCE_VALID = "provenance_valid"
    STATISTICAL_VALID = "statistical_valid"
    PAPER_CLAIMABLE = "paper_claimable"


class FailureReason(str, Enum):
    FIXTURE_BLOCKED = "fixture_blocked"        # synthetic/diagnostic row, never claimable
    TIER_CAPPED = "tier_capped"                # provenance gate: capped below A/A by tier
    FAILED_SIGNIFICANCE = "failed_significance"  # statistical gate: p/q above threshold
    UNDERPOWERED = "underpowered"              # too few events to resolve; not a true null
    SURVIVED = "survived"                      # reached paper_claimable


# Minimum events to call a non-significant edge "tested" rather than "underpowered".
# ADAPT: set from your power analysis; the repo flags USDC/SVB at 4 events as sparse.
UNDERPOWERED_EVENT_THRESHOLD = 5


@dataclass
class FunnelConfig:
    fdr_q: float = 0.05                  # global Benjamini-Hochberg level
    underpowered_n: int = UNDERPOWERED_EVENT_THRESHOLD
    require_global_significance: bool = True  # paper-claim requires passing GLOBAL correction


# --------------------------------------------------------------------------- #
# Global multiple-comparison correction across the full candidate set         #
# --------------------------------------------------------------------------- #
def global_bh(df: pd.DataFrame, *, q: float) -> pd.DataFrame:
    """Add `q_value_global` and `global_sig` over ALL testable candidate edges.

    Only rows with a non-NaN p_value enter the correction (untested edges -- e.g.
    fixture or tier-capped-before-testing -- are excluded from the family, which
    is the correct denominator). Returns a copy.
    """
    out = df.copy()
    out["q_value_global"] = np.nan
    out["global_sig"] = False

    testable = out["p_value"].notna()
    pvals = out.loc[testable, "p_value"].to_numpy(dtype=float)
    m = pvals.size
    if m == 0:
        return out

    order = np.argsort(pvals)
    ranked = pvals[order]
    # BH q: p_(k) * m / k, enforced monotone from the top
    q_raw = ranked * m / (np.arange(1, m + 1))
    q_mono = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_mono = np.clip(q_mono, 0, 1)

    q_out = np.empty(m)
    q_out[order] = q_mono
    idx = out.index[testable]
    out.loc[idx, "q_value_global"] = q_out
    out.loc[idx, "global_sig"] = q_out <= q
    return out


# --------------------------------------------------------------------------- #
# Failure-reason attribution (first gate failed)                              #
# --------------------------------------------------------------------------- #
def attribute_failure(row: pd.Series, cfg: FunnelConfig) -> str:
    """Assign each edge to the FIRST gate it failed (or SURVIVED)."""
    if bool(row.get("is_fixture", False)):
        return FailureReason.FIXTURE_BLOCKED.value

    # provenance gate
    if not bool(row.get("provenance_claim_allowed", False)):
        return FailureReason.TIER_CAPPED.value

    # statistical gate -- distinguish "tested and failed" from "couldn't test (sparse)"
    stat_ok = bool(row.get("statistical_claim_allowed", False))
    if cfg.require_global_significance:
        stat_ok = stat_ok and bool(row.get("global_sig", False))

    if not stat_ok:
        p = row.get("p_value", np.nan)
        n_sup = row.get("n_events_support", np.nan)
        # underpowered: never got a usable p AND too few events to resolve
        if (pd.isna(p)) or (not pd.isna(n_sup) and n_sup < cfg.underpowered_n):
            return FailureReason.UNDERPOWERED.value
        return FailureReason.FAILED_SIGNIFICANCE.value

    return FailureReason.SURVIVED.value


# --------------------------------------------------------------------------- #
# Stage counts                                                                #
# --------------------------------------------------------------------------- #
def _stage_counts(g: pd.DataFrame, cfg: FunnelConfig) -> dict[str, int]:
    n_candidate = len(g)
    non_fixture = g[~g.get("is_fixture", pd.Series(False, index=g.index)).astype(bool)]
    n_prov = int(non_fixture["provenance_claim_allowed"].astype(bool).sum())

    prov_ok = non_fixture[non_fixture["provenance_claim_allowed"].astype(bool)]
    n_stat = int(prov_ok["statistical_claim_allowed"].astype(bool).sum())

    # paper-claimable, honoring the global-correction requirement
    paper_mask = prov_ok["statistical_claim_allowed"].astype(bool)
    if cfg.require_global_significance and "global_sig" in prov_ok.columns:
        paper_mask &= prov_ok["global_sig"].astype(bool)
    n_paper = int(paper_mask.sum())

    return {
        Stage.CANDIDATE.value: n_candidate,
        Stage.PROVENANCE_VALID.value: n_prov,
        Stage.STATISTICAL_VALID.value: n_stat,
        Stage.PAPER_CLAIMABLE.value: n_paper,
    }


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def build_funnel(
    edges: pd.DataFrame,
    cfg: FunnelConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute the full attrition funnel.

    Returns a dict of DataFrames:
      'funnel'           per-event + POOLED stage counts (tidy, plot-ready)
      'failure_reasons'  per-event + POOLED counts by first-failed gate
      'corrected'        every edge with global q-value + verdict columns
    """
    cfg = cfg or FunnelConfig()
    required = {
        "event", "provenance_claim_allowed", "statistical_claim_allowed",
        "paper_claim_allowed",
    }
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"edge table missing required columns: {sorted(missing)}")

    # 1. global correction across the whole candidate family
    corrected = global_bh(edges, q=cfg.fdr_q)

    # 2. attribute failure reason per edge
    corrected["failure_reason"] = corrected.apply(
        lambda r: attribute_failure(r, cfg), axis=1
    )

    # 3. stage counts per event + pooled
    funnel_rows: list[dict] = []
    for ev, g in corrected.groupby("event"):
        row = {"event": ev, **_stage_counts(g, cfg)}
        funnel_rows.append(row)
    pooled = {"event": "ALL_POOLED", **_stage_counts(corrected, cfg)}
    funnel_rows.append(pooled)
    funnel = pd.DataFrame(funnel_rows)

    # add attrition rates for the abstract headline number
    funnel["pct_surviving"] = (
        funnel[Stage.PAPER_CLAIMABLE.value] / funnel[Stage.CANDIDATE.value]
    ).where(funnel[Stage.CANDIDATE.value] > 0, 0.0)

    # 4. failure-reason breakdown per event + pooled
    fr_rows: list[dict] = []
    reasons = [r.value for r in FailureReason]
    for ev, g in corrected.groupby("event"):
        counts = g["failure_reason"].value_counts().to_dict()
        fr_rows.append({"event": ev, **{r: int(counts.get(r, 0)) for r in reasons}})
    pooled_counts = corrected["failure_reason"].value_counts().to_dict()
    fr_rows.append({"event": "ALL_POOLED",
                    **{r: int(pooled_counts.get(r, 0)) for r in reasons}})
    failure_reasons = pd.DataFrame(fr_rows)

    return {"funnel": funnel,
            "failure_reasons": failure_reasons,
            "corrected": corrected}


# --------------------------------------------------------------------------- #
# Figure helpers (matplotlib imported lazily)                                 #
# --------------------------------------------------------------------------- #
def plot_funnel(funnel: pd.DataFrame, failure_reasons: pd.DataFrame,
                outpath: str) -> None:
    """Two-panel headline figure: stage funnel + stacked failure-reason bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pooled = funnel[funnel["event"] == "ALL_POOLED"].iloc[0]
    stages = [Stage.CANDIDATE.value, Stage.PROVENANCE_VALID.value,
              Stage.STATISTICAL_VALID.value, Stage.PAPER_CLAIMABLE.value]
    vals = [int(pooled[s]) for s in stages]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # panel 1: pooled funnel
    ax1.barh(range(len(stages))[::-1], vals)
    ax1.set_yticks(range(len(stages))[::-1])
    ax1.set_yticklabels([s.replace("_", " ") for s in stages])
    ax1.set_xlabel("number of edges")
    ax1.set_title("Candidate-edge attrition (pooled, all events)")
    for i, v in zip(range(len(stages))[::-1], vals):
        ax1.text(v, i, f" {v}", va="center")

    # panel 2: per-event stacked failure reasons
    per_event = failure_reasons[failure_reasons["event"] != "ALL_POOLED"]
    reasons = [FailureReason.SURVIVED.value,
               FailureReason.FAILED_SIGNIFICANCE.value,
               FailureReason.UNDERPOWERED.value,
               FailureReason.TIER_CAPPED.value,
               FailureReason.FIXTURE_BLOCKED.value]
    bottom = np.zeros(len(per_event))
    x = np.arange(len(per_event))
    for r in reasons:
        if r not in per_event.columns:
            continue
        h = per_event[r].to_numpy()
        ax2.bar(x, h, bottom=bottom, label=r.replace("_", " "))
        bottom += h
    ax2.set_xticks(x)
    ax2.set_xticklabels(per_event["event"], rotation=30, ha="right")
    ax2.set_ylabel("number of edges")
    ax2.set_title("Why edges fail, by event")
    ax2.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Self-test on a synthetic edge table that mimics the repo's real status      #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Mimic the README's reported state: 5 events, mostly provenance-valid but
    # only ONE globally-significant A/A edge (USDT/Curve 2023).
    rng = np.random.default_rng(0)
    rows = []

    def mk(event, src, dst, *, fixture=False, prov=True, tier="A_A_dex_flow",
           p=np.nan, n_sup=8, stat=False):
        rows.append(dict(
            event=event, src=src, dst=dst, is_fixture=fixture,
            provenance_claim_allowed=prov, statistical_claim_allowed=stat,
            paper_claim_allowed=False, claim_strength="descriptive",
            edge_tier=tier, p_value=p, n_events_support=n_sup,
        ))

    # USDT/Curve 2023 -- the one robust edge (tiny p, will pass global FDR)
    mk("usdt_curve_2023", "curve_3pool", "curve_crvusd_usdt",
       p=0.0007, stat=True, n_sup=8)
    # plus some non-significant candidates in the same event
    for k in range(6):
        mk("usdt_curve_2023", "curve_3pool", f"ctx_{k}", p=rng.uniform(0.2, 0.9))

    # Terra/LUNA -- provenance-valid but not sig at hourly grid
    mk("terra_2022", "curve_3pool", "curve_ust_wormhole", p=0.18, n_sup=7)
    for k in range(5):
        mk("terra_2022", "curve_3pool", f"ctx_{k}", p=rng.uniform(0.15, 0.95))

    # USDC/SVB -- sparse / underpowered settlement edge (no usable p)
    mk("usdc_svb_2023", "usdc_mint_burn", "curve_3pool", p=np.nan, n_sup=4)
    for k in range(4):
        mk("usdc_svb_2023", "curve_3pool", f"ctx_{k}", p=rng.uniform(0.3, 0.9))

    # FTX & BUSD -- A/B context, capped at provenance (tier_capped)
    for k in range(5):
        mk("ftx_2022", "curve_3pool", f"binance_ctx_{k}", prov=False,
           tier="A_B_suggestive_directional", p=rng.uniform(0.05, 0.6))
    for k in range(5):
        mk("busd_2023", "curve_3pool", f"binance_ctx_{k}", prov=False,
           tier="A_B_suggestive_directional", p=rng.uniform(0.05, 0.6))

    # a couple of fixtures that MUST be blocked
    mk("usdt_curve_2023", "fixture_a", "fixture_b", fixture=True, p=0.0001, stat=True)

    edges = pd.DataFrame(rows)
    out = build_funnel(edges, FunnelConfig(fdr_q=0.05, require_global_significance=True))

    print("=== FUNNEL (stage counts) ===")
    print(out["funnel"].to_string(index=False))
    print("\n=== FAILURE REASONS ===")
    print(out["failure_reasons"].to_string(index=False))

    pooled = out["funnel"].iloc[-1]
    print(f"\nHEADLINE: {pooled['candidate']} candidate edges -> "
          f"{pooled['paper_claimable']} paper-claimable "
          f"({pooled['pct_surviving']*100:.1f}% survive both gates).")

    # confirm the fixture never reached paper-claimable
    fx = out["corrected"]
    leaked = fx[(fx["is_fixture"]) & (fx["failure_reason"] == "survived")]
    assert leaked.empty, "FIXTURE LEAK -- a fixture row reached paper-claimable!"
    print("fixture-leak check: PASS (no fixture survived the gate)")
