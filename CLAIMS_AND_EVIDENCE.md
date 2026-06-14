# Claims and Evidence

What the paper argues, and for every headline number, the committed file it comes from. Artifacts live
under `results/paper/tables/` and `results/tables/`.

## The narrative

Empirical contagion studies routinely conflate genuine *contagion* — a crisis-driven shift in linkage —
with constant *interdependence*. Using Tier-A on-chain Curve `TokenExchange` flow across five stablecoin
stress episodes, we show the distinction is empirically sharp and mechanism-specific.

For the June-2023 USDT/Curve episode, an endogenous pool-imbalance shock, cross-pool flow coupling rises
from near-zero in calm (ρ̂=0.09) to ρ̂=0.53 during the acute window — a contemporaneous lag-0 co-movement,
not an early warning. None of the four exogenous episodes (Terra/LUNA, USDC/SVB, FTX, BUSD) shows this
activation despite identical Tier-A data, so the null is mechanism-specific, not a data gap.

The machine-learning result follows the same split. An unsupervised online Hidden Markov Model, run
causally on Tier-A on-chain state, detects the endogenous Terra/LUNA collapse at AUROC 0.95 versus 0.50
on Tier-B market data, raising the alarm 116 hours earlier; exchange-borne shocks are detectable only on
the market layer. The same split governs structure: on-chain arbitrage is stabilizing in calm markets but
flips to amplifying under endogenous stress, and price discovery is on-chain for DeFi-native shocks (the
Curve pool deviates 37× more than Binance) but CEX-led for bank-run shocks.

The headline does not rest on any single statistic — the acute Forbes–Rigobon window is small (n=53) —
but on the *convergence* of independent methods on the same episode, and on the mechanism-specificity of
the nulls. The operational implication is that DeFi stress surveillance needs separate monitoring
architectures for endogenous versus exchange-originating shocks, and a provenance-aware three-gate
pipeline underpins every claim.

## Where each number lives

| Claim | Number | File | Field / row |
|---|---|---|---|
| Cross-pool coupling rises in the acute window | ρ̂ 0.09 (calm, n=95) → 0.53 (acute, n=53), Fisher z=2.82 | `results/tables/table_regime_contagion.csv`, `results/tables/table_tier_robustness_fr.csv` | `r_calm`=0.0905, `r_acute`=0.5273, `fisher_z`=2.8209 (usdt_curve_2023) |
| Block-bootstrap significance (primary) | one-sided p=0.035 (96.5% of replicates positive) | `results/tables/table_convergent_evidence.csv` | `fr_p_gt0`=0.965 (usdt_curve_2023) |
| HMM detects Terra on-chain vs market, earlier | AUROC 0.954 vs 0.499, +116 h | `results/tables/table_online_detection.csv` | `auroc_onchain_causal`, `auroc_market_causal`, `earlier_by_h`=116 (terra_luna_2022) |
| Per-event HMM detection (default 3-state) | Terra 0.954, USDT/Curve 0.934, FTX 0.401, BUSD 0.609 | `results/tables/table_hmm_regime.csv` | `auroc` per `event_id` |
| Best within-event HMM (ablation; tab:convergent) | Terra 0.975, SVB 0.935, USDT/Curve 0.927 | `results/tables/table_hmm_ablation.csv` (best config) | `auroc` max per event — distinct from the default-config 0.954 above |
| Arbitrage flips stabilizing → amplifying | z=+3.84 USDT/Curve, +3.69 BUSD, −2.80 FTX | `results/tables/table_arbitrage_regime.csv` | `r_calm`/`r_panic`, `fisher_z`, `stabilizing_to_amplifying_flip` |
| On-chain price discovery: Curve 37× more sensitive | ratio 37.2 (peak on-chain 0.1249 vs CEX 0.00336) | `results/tables/table_price_discovery.csv` | `onchain_dev_ratio`, `max_onchain_dev`, `max_cex_dev`, `price_discovery_venue` |
| Convergent evidence (5 events × 4 methods) | FR / lead-lag / TE / HMM per event | `results/tables/table_convergent_evidence.csv` (+ `.json`) | one row per event; `convergence` = STRONG/MODERATE/NONE |
| TVP-VAR spillover concentrates post-onset (corroborating) | FEVD shares; spillover post-onset | `results/paper/tables/table_tvp_var_summary_usdt_curve_2023.csv`, `table_tvp_var_spillovers_usdt_curve_2023.csv` | `fevd_share_mean` per edge (post/pre-onset split is a corroborating robustness check, secondary to FR + HMM) |
| Supervised cross-event prediction fails under concept shift (negative control) | LOEO AUROC, r=0.525 | `results/paper/tables/table_prediction_metrics.csv` | per-model AUROC by `event_id` |

All numbers regenerate from `scripts/`. Provenance tiering (A/B) for every edge is recorded in the
`table_*_provenance_*` and `table_claim_gate_*` files.
