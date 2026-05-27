# Stablecoin Contagion Network: Full Project-Control Report

## Current Status

The repository now has a runnable control pipeline for the USDC/SVB event:

```text
windows -> ingest -> reconstruct -> panel -> eventmaps -> leadlag -> var -> te -> network
```

The current data artefacts are deterministic fixtures marked
`fixture_non_empirical`. They validate orchestration, schema contracts,
manifests, plotting, and model-table plumbing. They do not support empirical
claims about contagion.

## Main Claim To Prove

Stablecoin stress propagates through a multi-layer network: centralized venues
may lead price discovery at shock onset, decentralized pools may amplify or
prolong stress through inventory imbalance, and settlement, bridge, mint-burn,
and exchange-flow channels may explain persistence.

## Evidence Standard

Use the word "contagion" only when an edge is supported by time-directed model
evidence and provenance:

- Lead-lag with timestamp alignment, bootstrap inference, and FDR adjustment.
- VAR/Granger/FEVD or rolling VAR spillovers.
- Hawkes branching ratios with confidence intervals, if available.
- Transfer entropy as nonlinear robustness.
- Network centrality and role labels as operational summaries, not ground truth.

## Current Non-Claims

- No real contagion result is established by fixture data.
- No causal effect is established without an event-study or identification layer.
- No GNN claim exists while `temporal_gnn.py` remains a stub.

## Immediate Priority

Replace fixture ingestion with real USDC/SVB raw sources while preserving the
bronze, silver, gold, manifest, and model-output contracts.
