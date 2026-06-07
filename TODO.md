# TODO — Research Improvement Plans

## Current weaknesses

- Only 5 stress episodes — thin evidence base for statistical claims about cross-episode patterns
- Only 1 of 5 events (USDT/Curve) shows full Forbes-Rigobon contagion — pattern is interesting but the paper rests on a single positive case
- Supervised ML section is predominantly negative ("cross-event prediction fails") — adds little predictive contribution as currently framed
- HMM detection is unsupervised and simple — reviewers may demand a more sophisticated ML contribution or ablation over HMM order/states
- "Non-detections are mechanism findings" is intellectually honest but hard to sell as a positive result without a sharper theoretical anchor
- No quantification of economic impact — the 116h early-warning advantage for Terra is striking but its dollar value is unspecified

---

## Plans

### Plan A — Quantify the economic value of the 116h early-warning signal

**What to code:**
- `scripts/early_warning_value.py`: given the Terra HMM detection lead of 116h and observed price path, compute mark-to-market loss avoided for three stylised positions: $1M long LUNA, $1M long UST, $1M in Curve pool
- Parameterise by detection threshold and fraction of position unwound at alert time
- Output a sensitivity table: (detection lag, fraction liquidated) → $ loss avoided

**What to run:**
- `python scripts/early_warning_value.py --episode terra --lead_hours 116 --positions luna ust curve`
- Cross-check against publicly reported depegging timeline and Curve pool TVL drain

**Target result:**
- A 2×3 table showing that acting on the HMM signal 116h early would have avoided $X–$Y of losses depending on position size and liquidation fraction
- Even a conservative figure (e.g., >$10M avoided on a $50M pool) makes the finding concrete and publishable

**Write into paper:**
- Section 4.3 (HMM detection results): add one paragraph + Table 3 "Economic value of early detection"
- Replace the sentence "116h earlier" with "116h earlier, corresponding to $X–Y avoided loss per $50M in pool exposure (Table 3)"

---

### Plan B — Re-frame the supervised ML section as a regime-transfer diagnostic

**What to code:**
- `scripts/concept_shift_analysis.py`: for each leave-one-episode-out cross-validation fold, compute:
  - feature distribution shift (KL divergence or MMD on the AMM-flow feature vectors between train and test episode)
  - model performance (AUROC, precision-recall)
  - scatter: feature shift magnitude vs AUROC degradation
- This reframes "prediction fails" as "concept shift is measurable and predicts failure"

**What to run:**
- `python scripts/concept_shift_analysis.py --features amm_flow_features.parquet --labels episode_labels.csv`
- Produce a 5×5 train/test AUROC heatmap and a shift-vs-AUROC scatter plot

**Target result:**
- A statistically meaningful correlation (r > 0.6 or p < 0.05) between distribution shift and AUROC degradation
- Finding: "Cross-episode prediction fails, and the failure is predictable from feature-space shift (r=X, p=Y) — suggesting a shift detector could gate when AMM signals are reliable"

**Write into paper:**
- Rename Section 5 from "Supervised prediction" to "When AMM signals transfer: a concept-shift diagnostic"
- Add Figure 5: heatmap + scatter; add one-paragraph interpretation in Section 5.2

---

### Plan C — Add a provenance-stratified robustness table (Tier-A vs Tier-B only)

**What to code:**
- `scripts/tier_robustness.py`: re-run the Forbes-Rigobon z-test and the HMM AUROC using only Tier-A on-chain data (no CEX feeds), and separately using only Tier-B
- Compare z-statistics and AUROC across tiers for each episode

**What to run:**
- `python scripts/tier_robustness.py --config configs/provenance_gate.yaml`
- Should reuse existing Forbes-Rigobon and HMM pipeline with a `--data_tier` flag

**Target result:**
- A 5-row × 4-column table: episode | FR z (Tier-A only) | FR z (Tier-B only) | FR z (both)
- Expected: USDT/Curve result holds on Tier-A alone; exogenous episodes still non-significant
- Strengthens the "execution-grade data" claim and the provenance-gating contribution

**Write into paper:**
- Section 3.4 (data provenance): add Table 1b "Forbes-Rigobon results by data tier"
- Section 6 (robustness): one paragraph citing that the USDT/Curve finding is robust to removing CEX feeds

---

### Plan D — Bootstrap confidence intervals on Forbes-Rigobon z-statistics

**What to code:**
- `scripts/bootstrap_fr.py`: for each episode, block-bootstrap the Forbes-Rigobon Fisher z-statistic (block length = 5 days to preserve autocorrelation structure)
- Report 95% CI for z under H0 and under the observed data
- Plot: z-statistic with CI error bars across all 5 episodes

**What to run:**
- `python scripts/bootstrap_fr.py --n_bootstrap 2000 --block_length 5`
- Runtime: ~10 min on existing data

**Target result:**
- Figure showing USDT/Curve z=2.82 with non-overlapping CI vs exogenous episodes near zero
- Strengthens the claim that exogenous non-detections are not noise artefacts but true zeros

**Write into paper:**
- Section 4.1: replace point estimates with "z=2.82 [95% CI: 2.1–3.6]" style reporting
- Add Figure 2b: z-statistic comparison across episodes with bootstrap CIs

---

### Plan E — Extend to 3–5 additional AMM pool episodes from 2023–2024

**What to code:**
- `scripts/fetch_additional_episodes.py`: query The Graph or Dune Analytics for Curve pool TokenExchange logs for:
  - crvUSD depeg event (Aug 2023)
  - DAI/USDC Curve pool activity during USDC de-peg echoes (late 2023)
  - Any 2024 stablecoin stress episode with Curve TVL > $500M
- Apply existing provenance pipeline and Forbes-Rigobon test

**What to run:**
- `python scripts/fetch_additional_episodes.py --pools 3pool crvusd --start 2023-07-01 --end 2024-06-01`
- `python scripts/run_forbes_rigobon.py --episodes all`

**Target result:**
- 2–3 additional episodes; aim for 1 more positive contagion detection (endogenous pool-level shock)
- If crvUSD Aug 2023 is endogenous: z > 2 would bring the positive case count to 2 and strengthen the pattern
- If all new episodes are non-detections: confirms the endogenous/exogenous distinction holds out-of-sample

**Write into paper:**
- Section 4 headline: change "5 stress episodes" to "8 stress episodes"
- Robustness section: "The endogenous/exogenous distinction replicates on 3 additional 2023–2024 episodes"

---

### Plan F — Add an HMM order/state ablation and a CUSUM benchmark

**What to code:**
- `scripts/hmm_ablation.py`: sweep HMM number of states (2, 3, 4) and emission distributions (Gaussian, Student-t) for all 5 episodes; report AUROC for each configuration
- `scripts/cusum_detector.py`: implement a standard CUSUM change-point detector on the same AMM-flow features as a non-ML baseline

**What to run:**
- `python scripts/hmm_ablation.py --states 2 3 4 --emissions gaussian student`
- `python scripts/cusum_detector.py --episodes all`

**Target result:**
- Table showing 3-state Gaussian HMM is robust (AUROC within ±0.02 of best configuration)
- CUSUM AUROC clearly lower than HMM for Terra (target: CUSUM < 0.80 vs HMM 0.954)
- Demonstrates the 3-state HMM is not arbitrarily chosen and beats a naive statistical baseline

**Write into paper:**
- Section 4.3: add Table 4 "HMM configuration ablation" and one column for CUSUM
- Removes the reviewer objection that the HMM state count is ad hoc

---

### Plan G — Compute pool-level arbitrage intensity and link to contagion strength

**What to code:**
- `scripts/arbitrage_intensity.py`: from TokenExchange logs, compute per-block arbitrage trade fraction (trades that push price toward external CEX price vs trades that push away), rolling over 6h windows
- Correlate arbitrage intensity time-series with the Forbes-Rigobon regime indicator

**What to run:**
- `python scripts/arbitrage_intensity.py --episode usdt_curve --window_hours 6`
- Scatter plot: arbitrage intensity vs pool price deviation from CEX

**Target result:**
- For USDT/Curve: arbitrage shifts from net-stabilising pre-stress to net-amplifying during stress period (z=+3.84 already found); now show the intensity correlates with z-statistic level
- Provides a micro-mechanism story: "arbitrageurs flip sign because the depeg exceeds their cost-of-capital threshold"

**Write into paper:**
- Section 4.2 (arbitrage flip results): add Figure 4 "Arbitrage intensity vs pool-CEX deviation"
- One paragraph: "The flip from stabilising to amplifying arbitrage (z=+3.84) coincides with a X% increase in arbitrage trade intensity, consistent with a cost-of-capital threshold effect"

---

### Plan H — Write a structured "signal taxonomy" table as the paper's conceptual anchor

**What to code:**
- No new code needed — synthesise existing results into a 2×2 or 3×3 taxonomy table
- Dimensions: shock type (endogenous pool vs exogenous market) × signal layer (on-chain AMM vs CEX market)
- Cells: detected/not detected, which method wins, detection lead

**What to run:**
- Manual synthesis of Tables 1–3 from existing results
- One script `scripts/make_taxonomy_figure.py` to render as a LaTeX table or matplotlib heatmap

**Target result:**
- A single figure that summarises when on-chain AMM signals work (endogenous shocks, pool-level stress) and when they don't (exogenous shocks where CEX leads)
- Turns the "non-detections are mechanism findings" narrative into a visually clear positive contribution

**Write into paper:**
- Abstract: add one sentence citing the taxonomy
- Section 2 (framework): place the taxonomy table as Figure 1 to set up all subsequent results
- Conclusion: "The taxonomy predicts signal utility from shock origin, providing a decision rule for practitioners"
