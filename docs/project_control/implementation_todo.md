# Implementation TODO

## P0: Repo Control

- Keep `docs/project_control/` on `main`.
- Keep `make mvp EVENT=usdc_svb_2023` as the acceptance test for the control
  pipeline.
- Treat fixture outputs as non-empirical until real raw data lands.

## P1: Data Pipeline

- Implement real CEX ingestion for Binance, Coinbase, and Kraken.
- Implement DEX event ingestion for Curve 3pool and Uniswap USDC/USDT 0.05%.
- Implement on-chain flow ingestion for exchange flows and mint/burn.
- Ensure every bronze, silver, and gold artefact has a manifest row.

## P2: Feature Panel

- Complete layer-specific silver reconstruction.
- Add grid-aware sampling at 1s, 5s, and 60s.
- Preserve no-lookahead semantics for rolling flow features and downstream
  labels.

## P3: Models

- Add FDR-adjusted lead-lag and TE outputs.
- Add rolling VAR / TVP-VAR.
- Make Hawkes optional and non-blocking.
- Standardize graph edge schemas across methods.

## P4: Prediction and GNN

- Build prediction baselines before GNN work.
- Include GNN only if it beats strong non-graph baselines by more than 5% AUROC
  or AUPRC and passes shuffled-graph ablations.
