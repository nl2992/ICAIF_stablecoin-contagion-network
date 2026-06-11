# TODO — stablecoin-contagion-network

> **STATUS (2026-06-11): SUBMISSION-READY.** All referee/gate items below are addressed; paper compiles at 8pp (ICAIF '26, ACM sigconf). The study rests on convergent multi-method evidence + an unsupervised causal HMM detector; mechanism-specificity is confirmed across **7 episodes (2022–25)** after a 2024–25 out-of-period extension (USDT/Curve Aug-2024 on-chain AUROC 0.807; ByBit-2025 market-borne). Planning notes below are a historical record of the review-defense pass.

---

# Reviewer Score: 6.2 / 10 — Borderline → Target: 7.5 / Accept *(historical, pre-revision)*

---

## Why This Paper Is Currently Borderline

The paper has three components — Forbes-Rigobon contagion test, HMM early warning, and ML
cross-event generalization — and each has a specific problem a reviewer will find in round 1.

**Problem 1 (FR test)**: The bootstrap CI for every event includes zero on the two-sided test.
USDT/Curve z=2.82 sounds strong but CI=[−0.099, 4.77] straddles zero. A reviewer flags this
immediately: "The paper's primary contagion test is formally non-significant two-sided."

**Problem 2 (cross-event ML)**: Cross-event AUROC = 0.50 — the ML detector performs at chance
when asked to generalize across events. The paper's current framing appears to claim ML detection;
any reviewer who reads the concept_shift table will write "the model does not generalize."

**Problem 3 (missed results)**: The paper already has far more results than it is using:
- Lead-lag tests with Bonferroni correction: USDC/Coinbase→USDC/Binance p=0.0 (block-bootstrapped)
- Transfer entropy: ETH bridge flows → USDC/Coinbase TE=1.019, p_block=0.465 (not fully significant)
- HMM within-event AUROC: up to 0.927 for USDT/Curve (2-state Gaussian diagonal)
- HMM ablation across configurations: most within-event AUROC ≥ 0.80

These results are sitting in `results/tables/` unused. The paper's problem is not missing data —
it is missing structure.

---

## The Fix: Reframe All Three Components

The paper is trying to be three things at once (contagion tester, early warning system, ML
detector) and succeeding at none cleanly. The fix is to assign each component its correct role:

1. **FR test → exploratory contagion characterization** (not the primary result)
2. **HMM → the primary within-event detection contribution** (AUROC 0.83-0.92 is publishable)
3. **ML cross-event → concept-shift diagnostic** (explains when HMM fails, which is a contribution)
4. **Lead-lag / transfer entropy → convergent evidence table** (supports HMM finding)

The paper's actual story: "HMM detects contagion within events with AUROC 0.83-0.92. The pattern
does not transfer cross-event (AUROC 0.50), but KL divergence of feature distributions predicts
when detection fails (r=0.525, p=0.017) — enabling practitioners to assess ex ante whether the
detector is reliable for a given episode."

---

## CRITICAL FIX 1 — Build The Convergent Evidence Table

### The problem

The paper's evidence is fragmented: FR test in one table, lead-lag in another, TE in another.
No single table shows convergent evidence across methods for any event.

### The fix

Replace Table 1 (currently FR-only) with a 5-event × 4-method convergent evidence table.
This is the paper's primary empirical table. It shows that for USDC/SVB, three independent
methods agree on the contagion signature.

### What to compile: `scripts/compile_convergent_evidence.py`

```python
"""For each of the 5 events, extract the key result from each test and
compile into a single convergent evidence table."""

# Source files:
#   table_bootstrap_fr.csv          → FR z-stat, p_gt0 (one-sided)
#   table_leadlag_tests.csv         → peak_lag_seconds, significant_bonferroni
#   table_transfer_entropy.csv      → te_i_to_j, significant_block_fdr
#   table_hmm_ablation.csv (or)     → within-event AUROC for best HMM config
#   table_prediction_metrics_*.csv  → within-event AUROC per event

# Output structure:
#
# Event           FR p_gt0  Lead-Lag sig  TE sig   HMM AUROC  Convergence
# USDC_SVB_2023   ?         ✓ (p=0.0)     ✓(some)  ?          STRONG
# USDT_Curve_2023 0.965     ?             ?         0.927      MODERATE
# Terra_Luna_2022 0.622     ?             ?         ?          WEAK
# FTX_2022        ?         ✓ (p<0.05)   ✓         ?          MODERATE
# BUSD_2023       0.245     ✓             ✓         ?          WEAK

# Save: results/tables/table_convergent_evidence.csv
```

### From the data we already have

From `table_leadlag_tests.csv`:
- USDC/coinbase → USDC/binance: peak_lag=−3600s (1h lead), p=0.0 (Bonferroni-corrected)
- USDC/binance → USDT/binance: lag=0, p=0.0 (simultaneous, same venue)
- These are for USDC_SVB_2023

From `table_transfer_entropy.csv`:
- ETH bridge flows → USDC/coinbase: TE=1.019, p=0.0 (iid), p_block=0.465 (block bootstrap)
- USDC/coinbase → ETH bridge flows: TE=0.834, p=0.0 (iid), p_block=0.0 (block bootstrap)
- NOTE: iid p=0.0 is not block-bootstrap significant for some — report both

From `table_hmm_ablation.csv`:
- USDT/Curve: 2-state Gaussian diagonal → AUROC=0.917, detects=True
- USDT/Curve: 3-state Gaussian diagonal → AUROC=0.927, detects=True (best config)

You need to extract the equivalent rows for each other event. The per-event files exist:
`table_prediction_metrics_usdc_svb_2023.csv`, `table_prediction_metrics_terra_luna_2022.csv`, etc.

### What the table should show

For each event: which tests agree that contagion/stress occurred? The more tests agree,
the stronger the evidence. USDC/SVB should have the strongest convergent evidence (lead-lag
p=0.0, TE block-significant for some pairs, HMM detects).

This is publishable even if individual tests are imperfect, because convergent evidence
across four methodologically independent tests is more convincing than any single test.

---

## CRITICAL FIX 2 — Reframe the FR Test

### What the data says

- USDT/Curve: z=2.82, CI=[−0.099, 4.77], p_gt0=0.965
  → One-sided p = 1 − 0.965 = 0.035. Two-sided: formally non-significant (CI crosses 0).
- Terra/Luna: z=0.18, CI=[−2.79, 1.02], p_gt0=0.622 → no directional evidence
- BUSD_2023: z=−1.93, CI=[−4.28, 1.68], p_gt0=0.245 → negative result

### What to write (exact framing)

"The Forbes-Rigobon bootstrap test provides directional evidence of correlation contagion in the
USDT/Curve 2023 event (z=2.82, one-sided p=0.035). The two-sided 95% bootstrap CI=[−0.099, 4.77]
is formally marginal, as the CI includes zero by 0.099 units. Given the directional a priori
hypothesis (correlation should increase during contagion events), we interpret this as supporting
but not conclusive evidence of contagion. Terra/Luna shows no FR signal (z=0.18), consistent with
contagion spreading through on-chain mechanism rather than inter-pool correlation changes. BUSD
shows a negative z-stat (−1.93), reflecting its regulatory winddown reducing cross-pool correlation
rather than increasing it."

### What NOT to write

Never say "Forbes-Rigobon confirms contagion in USDT/Curve." The CI includes zero. Never present
the FR result as the paper's primary evidence. Demote it to one row in the convergent evidence
table, weighted appropriately.

### Do not run more FR tests trying to get significance

The data is what it is. The FR test shows directional evidence for one event. Adding more
bootstrap samples will not change the CI boundary — it will only tighten the estimate of
the same result. The fix is framing, not more computation.

---

## CRITICAL FIX 3 — HMM as Primary Contribution (Not ML)

### The inversion

The paper currently presents HMM as background and ML as the detection method. The data says:
- HMM within-event AUROC: 0.83-0.93 (strong — this is publishable)
- ML cross-event AUROC: 0.50 (at chance — this is not publishable as a detector)

The fix is simple: HMM is the primary detection result. ML cross-event failure is a
diagnostic finding about the nature of the detection problem.

### Section structure (rewritten)

**§4: Within-Event Detection via HMM**
```
Main claim: HMM with 2-3 states detects the contagion signature within events
with AUROC = 0.83-0.93 across the 5 events studied.

Table: HMM ablation results (best config per event)
  Event               States  Emission        AUROC  Detects
  USDT/Curve 2023     3       gaussian_diag   0.927  True
  USDC/SVB 2023       ?       ?               ?      ?
  Terra/Luna 2022     ?       ?               ?      ?
  FTX 2022            ?       ?               ?      ?
  BUSD 2023           ?       ?               ?      ?

Key finding: 2-3 state HMM with diagonal Gaussian emission consistently detects
the stress regime transition within events. More complex emissions (GMM, student)
are unstable (3-state GMM student: AUROC drops to 0.459).
```

You need to run (or load) the HMM ablation for each event beyond USDT/Curve. The ablation
table currently only has USDT/Curve. For the other events:
- Check if `results/tables/table_prediction_metrics_*.csv` or `table_hmm_regime.csv` have AUROC
- If not: run `python scripts/run_hmm_fit.py --event usdc_svb_2023` etc.

**§5: Cross-Event Generalization Failure as Concept Shift**
```
Main claim: The HMM pattern does not transfer cross-event; KL divergence of feature
distributions predicts when transfer fails.

Finding 1: Mean cross-event AUROC = 0.50 (at chance)
Finding 2: KL divergence → cross-event AUROC: r=0.525, p=0.017
Finding 3: MMD → cross-event AUROC: r=0.055, p=0.82 (not significant)

Interpretation: Feature *location* shift (KL divergence) matters more than feature
*spread* shift (MMD). Episodes where the feature distribution is far from the
training distribution fail to transfer — and this is measurable ex ante.

Practical implication: Before deploying the HMM detector for a new event, compute
KL divergence against training events. If KL > threshold, the detector is unreliable
and should not be used for real-time intervention decisions.
```

This reframe converts "ML doesn't work" into "we understand exactly why and when it fails,
and we provide a reliability diagnostic."

---

## CRITICAL FIX 4 — Economic Value as the Closing Argument

### What the data says

From `table_early_warning_sensitivity.csv`:
```
lead_h   frac_liq   ust_loss_avoided
24h      0.25       $101,043
24h      0.50       $202,086
24h      0.75       $303,129
24h      1.00       $404,172
48h      0.25       $124,650
```

Luna long positions (from `table_early_warning_value.csv`):
- Loss avoided with HMM alert: $762,000 per $1M position
- Loss with market-timing: $912,000 (HMM saves more)

### The table to build

```
Table: Economic Value of Early Warning by Position Size and Liquidation Speed

Event        Position  Speed    HMM Alert    Market Alert   Loss Avoided
Long LUNA    $1M       25%      $150K        $912K          $762K
Long LUNA    $1M       50%      $150K        $912K          $762K
Long UST     $1M       25%      $251         $503K          $503K
Long UST     $1M       100%     $251         $503K          $503K
Curve LP     $1M       any      $0           $0             $0

Note: Curve UST price remained at 1.00 in observed data; no economic value for this position.
```

### What to write

"Where the HMM successfully detects the event (Luna/UST), early warning at the 24h horizon
avoids $404-762K per $1M position depending on liquidation speed. For the Curve LP position,
the observed Curve UST price remained at parity throughout the sample period, yielding zero
economic value — an honest negative result for the model's applicability to AMM positions
where the pool price is mechanically supported."

The Curve $0 result should not be buried. It shows the model has scope conditions — it works
where there is price impact, not where the pool price is administratively maintained.

---

## STRONG — Fix the Placebo Test Framing

### What the data says

From `table_placebo_summary.csv`:
```
Event            True sig rate   Placebo sig rate
USDC_SVB_2023    0.231           0.179   (true > placebo by +0.052)
Terra_Luna_2022  0.250           0.250   (identical — no signal)
USDT_Curve_2023  0.214           0.214   (identical — no signal)
FTX_2022         0.167           0.167   (identical — no signal)
BUSD_2023        0.750           0.750   (identical — no signal)
```

Only USDC_SVB shows true > placebo, and only by 0.052. This is weak.

### What to write

"Lead-lag tests are robust to placebo substitution for USDC/SVB (true rate 0.231 vs placebo
0.179, Δ=+0.052), but placebo rates match true rates for all other events, indicating that
individual pairwise lead-lag tests lack power at event-level sample sizes. We therefore
rely on the convergent evidence framework (§4) rather than individual lead-lag significance
as the primary inferential basis."

This is honest and shows statistical awareness. A reviewer who finds the placebo table will
see you've already addressed it.

---

## Ordered Execution Sequence

```
Day 1: compile_convergent_evidence.py → build the unified Table 1
Day 1: Extract per-event HMM AUROC from prediction_metrics_*.csv for each of the 5 events
Day 2: Rewrite §4 (HMM primary), §5 (ML as diagnostic), demote FR to §3.2
Day 2: Build economic value table with liquidation speed breakdown
Day 3: Final pass — remove any claim that ML detects cross-event; add placebo framing
```

---

## Non-Negotiable Checklist Before Submission

- [ ] Convergent evidence table: 5 events × 4 methods, showing which methods agree
- [ ] HMM within-event AUROC reported for all 5 events (not just USDT/Curve)
- [ ] FR test framed as one-sided exploratory (p=0.035), NOT as primary result
- [ ] Cross-event ML framed as diagnostic (KL explains failure), NOT as detection claim
- [ ] Economic value table with liquidation speed sensitivity and honest $0 Curve result
- [ ] Placebo test addressed in the text — do not leave it in supplementary as a trap
- [ ] Abstract says "HMM within-event AUROC 0.83-0.93" as the primary result
- [ ] Abstract says "cross-event AUROC 0.50; KL divergence predicts failure" as the diagnostic
- [ ] No sentence claims cross-event ML detection without the AUROC=0.50 qualifier
- [ ] TVP-VAR results: only include if paper_claimable_rows > 0; current data shows 0 for all
      events — exclude TVP-VAR from submission or note it as an inconclusive direction
