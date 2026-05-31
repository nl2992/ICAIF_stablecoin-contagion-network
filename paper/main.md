# Cross-Protocol Stablecoin Stress Propagation

This Markdown companion tracks the current working-paper narrative in
`paper/main.tex`. The paper is now framed as a provenance-aware AMM benchmark
for Curve Finance and Uniswap v3, not as a Curve-only pilot.

## Core Claim

Stablecoin stress is a multi-protocol liquidity-flow event. The paper studies
five stress episodes, but only the USDT/Curve 2023 event currently yields
paper-claimable A/A evidence after both provenance and statistical gates.

The paper does not claim structural causality. It reports significant
lead-lag relations and directed predictive dependence.

## Current Headline Results

| Result | Evidence |
|---|---|
| Within-Curve co-movement | Curve 3pool and Curve crvUSD/USDT show positive AMM-flow co-movement, `rho = 0.386`, 95% CI `[0.249, 0.507]`, FDR `p <= 0.006`. |
| Cross-protocol counter-movement | Curve pools and Uniswap v3 USDC/USDT show negative AMM-flow co-movement, `rho = -0.486`, 95% CI `[-0.594, -0.361]`, FDR `p < 0.001`. |
| Directional timing signal | Curve crvUSD/USDT leads Uniswap by one hour, `rho = -0.268`, FDR `p < 0.001`. |
| Prediction validation | Adding Uniswap v3 Tier-A lags improves next-hour Curve stress AUROC from `0.677` to `0.711`. |

## Provenance Gate

Every edge is filtered by:

1. Endpoint tier.
2. Feature tier.
3. Statistical significance.

An A/A claim requires execution-grade on-chain evidence at both endpoints.
Curve `TokenExchange` logs and Uniswap v3 `Swap` logs are Tier A. Public CEX
candles, BBO, and public OHLCV feeds are Tier B context only because they are
not historical full-depth L2 order books.

Fixture data are explicitly blocked from paper claims.

## Paper-Safe Outputs

The key paper-facing tables are:

- `results/paper/tables/table_cross_protocol_leadlag_usdt_curve_2023.csv`
- `results/paper/tables/table_prediction_cross_protocol.csv`
- `results/paper/tables/table_stress_propagation_score.csv`
- `results/paper/tables/table_propagation_intensity.csv`
- `results/paper/tables/table_node_coverage_empirical.csv`

The paper figures should be generated from claim-gated outputs only.

## Non-Claims

- No structural causality is claimed.
- No CEX microstructure claim is made without full historical L2 data.
- No fixture-derived row is paper-claimable.
- Terra/LUNA, USDC/SVB, FTX, and BUSD are retained for external-validity and
  negative/sparse-evidence comparisons, not as equal-strength headline events.

## Build

```bash
python scripts/21_run_cross_protocol_analysis.py --event usdt_curve_2023
python scripts/22_run_cross_protocol_prediction.py --event usdt_curve_2023
python scripts/20_compute_stress_propagation_score.py
python scripts/19_compute_propagation_intensity.py
```

Then compile `paper/main.tex`.
