# Funnel + Calibration Contract — integration guide

Two modules, both serving the Option-A thesis ("most contagion claims don't
survive a joint provenance+significance gate; here's the one that does, and
here's why the rest don't").

- `evaluation/attrition_funnel.py` — the paper's empirical spine.
- `export/calibration_contract.py` — the cross-repo handshake.

---

## 1. Attrition funnel — the headline artifact

The funnel turns your per-edge claim-gate table into the paper's central
figure and number: **N candidate edges → 1 paper-claimable**, decomposed by
*why* each edge died.

What it adds over the existing per-event audit:

1. **Pooled global FDR.** Within-event Bonferroni isn't enough — you test many
   pairs across five events. `global_bh()` corrects across the *entire* candidate
   family, so the surviving edge is demonstrably not the winner of many
   unreported tests. This is the single line that disarms the obvious reviewer
   attack. Paper-claimable requires passing the **global** correction when
   `require_global_significance=True`.
2. **Failure-reason attribution.** Every non-surviving edge is assigned to the
   *first* gate it failed: `fixture_blocked → tier_capped → failed_significance
   → underpowered`. The stacked-bar panel makes the funnel a diagnosis, not a
   blank table.
3. **Underpowered ≠ null.** Sparse edges (e.g. USDC/SVB at 4 events) are tagged
   `underpowered`, not `failed_significance`, so empty cells read as "couldn't
   resolve at this N" rather than "no contagion."

### Wiring

```python
import pandas as pd
from stressnet.evaluation.attrition_funnel import build_funnel, plot_funnel, FunnelConfig

edges = pd.read_csv("results/tables/table_claim_gate_all_events.csv")  # ADAPT path
out = build_funnel(edges, FunnelConfig(fdr_q=0.05, require_global_significance=True))

out["funnel"].to_csv("results/paper/tables/table_attrition_funnel.csv", index=False)
out["failure_reasons"].to_csv("results/paper/tables/table_failure_reasons.csv", index=False)
out["corrected"].to_csv("results/tables/table_edges_global_corrected.csv", index=False)
plot_funnel(out["funnel"], out["failure_reasons"],
            "results/paper/figures/fig_attrition_funnel.png")
```

Add this to `make paper_gate` *after* `00c_claim_gate.py` and before
`13_make_paper_figures.py`. The `pct_surviving` of the `ALL_POOLED` row is your
abstract's headline number.

### Acceptance tests to add

- Funnel monotone: candidate ≥ provenance_valid ≥ statistical_valid ≥ paper_claimable.
- Fixture-leak guard: no `is_fixture` row ever has `failure_reason == "survived"`.
- Exactly the USDT/Curve 2023 A/A edge reaches `paper_claimable` in the pooled funnel.
- Global FDR strictly tightens vs. within-event: paper-claimable count under
  global correction ≤ count under local-only.

### `# ADAPT` marks

- `UNDERPOWERED_EVENT_THRESHOLD` — set from your power analysis (README flags 4
  events as sparse, so 5 is the natural cut).
- Input column names — the module expects the repo's existing `*_claim_allowed`,
  `claim_strength`, `is_fixture`, `edge_tier`, `p_value`, `n_events_support`.

---

## 2. Calibration contract — the cross-repo handshake

This is the artifact `stablecoin-abm` and `stablecoin-contagion-gnn` consume so
their calibration is honest about what this repo actually establishes.

The rule, machine-enforced:

| Consumption tier | From claim_strength | Downstream use |
|---|---|---|
| **PRIMARY** | `robust` | calibrate causal parameters to these |
| **SECONDARY** | `statistically_supported`, `suggestive` | report-only, caveated, never a headline causal basis |
| **CONTEXT** | `descriptive`, `context_only` | narrative only, excluded from calibration |

### Producer side (this repo)

```python
from stressnet.export.calibration_contract import (
    edge_targets_from_table, metric_target, build_contract, TargetKind, summarize)

edges = pd.read_csv("results/tables/table_edges_global_corrected.csv")  # funnel output
targets = edge_targets_from_table(edges)            # propagation-edge targets
targets += [ ... metric_target(...) for OU half-lives, magnitudes ... ]

contract = build_contract(targets, source_git_sha=GIT_SHA)
contract.to_json("results/paper/export/calibration_contract.json")
contract.to_csv("results/paper/export/calibration_targets.csv")
print(summarize(contract))   # the tier-count line both downstream papers quote
```

Add a CI check that the contract regenerates deterministically (same
`content_hash`) given the same inputs, and stamp the hash into the paper appendix.

### Consumer side (ABM and GNN repos)

Ship a copy of `load_primary_targets` to each downstream repo. They call:

```python
from calibration_contract import load_primary_targets
primary, content_hash, schema = load_primary_targets(
    "calibration_contract.json", expected_schema="calib-contract/1.0.0")
# calibrate ONLY to `primary`; raises if schema drifts or zero robust targets
```

The function **raises if there are zero PRIMARY targets** — so under Option A,
if only suggestive targets exist for a given parameter, the downstream repo is
*forced* to acknowledge it cannot make a calibrated causal claim rather than
quietly fitting to weak evidence. That is the honesty guarantee, enforced in
code rather than promised in prose.

### Consumer obligations (write into both downstream papers)

1. Assert `schema_version` matches the version built against.
2. Calibrate primary parameters ONLY to PRIMARY targets.
3. Print the `claim_strength` of every consumed target.
4. If the PRIMARY set for a parameter is empty, state plainly that no calibrated
   causal claim is made for it.

---

## 3. How this serves the three-way triangulation

The contract's one robust propagation edge —
`curve_3pool ↔ curve_crvusd_usdt` (USDT/Curve 2023) — is the anchor all three
repos meet at:

- **This repo** says it is the single globally-FDR-significant A/A edge.
- **The GNN** should be checked for whether it ranks `curve_crvusd_usdt` a hub.
- **The ABM** should be checked for whether intervening on it causally reduces
  contagion.

Three independent methods converging on one node turns the "only one edge"
result from a weakness into the program's linchpin. The funnel quantifies the
discipline; the contract makes the downstream calibration honest; the
triangulation makes the single edge a strength.

---

## 4. Lineage naming — fix before publishing

The GNN repo cites "IAQF 2026 codebase" for the same features the ABM repo
attributes to `stablecoin-contagion-network`. Unify to one canonical source
name across all three READMEs and all three papers, or the provenance chain a
reviewer traces across the trio looks inconsistent.
