# Preliminary Results Checkpoint

This document records the current empirical checkpoint in paper-safe language.
It should not be read as confirming structural causality. The correct phrasing is
"directed predictive dependence" or "significant Granger relations."

## Provenance Gate

Current outputs are mixed-provenance. Every event still contains at least one
`fixture_non_empirical` node, so all claims remain preliminary and must be
reported with real-node-only filters.

| Event | Total nodes | Real nodes | Fixture nodes | Paper claim tier |
| --- | ---: | ---: | ---: | --- |
| USDC/SVB 2023 | 11 | 3 | 8 | real-node preliminary, fixture contaminated |
| Terra/LUNA 2022 | 4 | 2 | 2 | real-node preliminary, fixture contaminated |
| USDT/Curve 2023 | 8 | 1 | 7 | insufficient real coverage |
| FTX 2022 | 6 | 2 | 4 | real-node preliminary, fixture contaminated |
| BUSD 2023 | 4 | 3 | 1 | real-node preliminary, fixture contaminated |

## Directed-Dependence Summary

| Event | Lead-lag all | Lead-lag real-only | TE all | TE real-only | Real-only network edges | VAR status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| USDC/SVB 2023 | 18 | 6 | 49 | 4 | 2 | 2/2 significant Granger relations |
| Terra/LUNA 2022 | 3 | 1 | 7 | 2 | 2 | 1/2 significant Granger relations |
| USDT/Curve 2023 | 12 | 0 | 16 | 0 | 0 | skipped: fewer than 2 real nodes |
| FTX 2022 | 5 | 2 | 10 | 2 | 2 | 2/2 significant Granger relations |
| BUSD 2023 | 6 | 6 | 9 | 4 | 2 | 1/2 significant Granger relations |

Interpretation: the all-node outputs are directionally promising but are not
paper-ready because fixtures remain in the panels. Real-node-only counts are
nonzero for four events. USDT/Curve currently has only one real node and should
not be used for VAR, Granger, or network-prediction claims until coverage is
repaired.

## Placebo Gate

| Event | True lead-lag sig rate | Placebo lead-lag sig rate | Status |
| --- | ---: | ---: | --- |
| USDC/SVB 2023 | 0.231 | 0.179 | passes weakly |
| Terra/LUNA 2022 | 0.250 | 0.250 | fails |
| USDT/Curve 2023 | 0.214 | 0.214 | fails |
| FTX 2022 | 0.167 | 0.167 | fails |
| BUSD 2023 | 0.750 | 0.750 | fails |

Interpretation: placebo validation is not yet strong enough for paper claims.
The next empirical milestone is true-window significance rates that dominate
placebo rates after real-node filtering and multiple-testing correction.

## Prediction Baselines

Prediction baselines currently run for USDC/SVB, FTX, and BUSD. Terra/LUNA
produces no usable test metric because the test split has only one label class.
USDT/Curve is deferred because it has only one real node.

| Event | Best model | AUROC | AUPRC | Caveat |
| --- | --- | ---: | ---: | --- |
| USDC/SVB 2023 | XGBoost | 0.566 | 1.000 | test prevalence is near 1, so AUPRC is inflated |
| FTX 2022 | LightGBM | 0.930 | 0.966 | promising, but mixed with fixture nodes |
| BUSD 2023 | Random Forest | 0.335 | 0.464 | weak baseline; not paper-supportive |

## Paper-Safe Narrative

Across five stablecoin stress episodes, the pipeline identifies nontrivial
directed predictive dependence through lead-lag and transfer-entropy channels.
USDC/SVB produces the broadest all-node network, while BUSD and Terra show
smaller but denser structures. However, real-node-only filtering materially
reduces the claim set, and USDT/Curve is not yet usable for VAR because it has
only one real node. The current evidence supports a preliminary-results
checkpoint, not a final contagion claim.

## GNN Gate

Do not start GNN as a paper claim until:

- Real-node-only panels exist for at least three events.
- Prediction labels have nonzero train and test prevalence.
- Logistic regression, random forest, XGBoost, and LightGBM baselines run on at
  least three events.
- Placebo rates are below true-event rates after real-node filtering.
- Shuffled-network placebo logic exists.

The GNN enters the paper only if it improves AUROC or AUPRC by more than 5%
over the best non-graph baseline and shuffled-edge ablations perform worse.
