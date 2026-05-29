# Stablecoin Contagion Network

A multi-layer benchmark for measuring how stablecoin stress propagates across
centralized venues, decentralized liquidity pools, and on-chain settlement channels.

## Research question

> **How does stress propagate across stablecoins, venues, and liquidity pools once a shock starts?**

This project extends the execution-aware logic of [Stablecoin StressBench](../ICAIF/).
The previous benchmark asked whether a local stablecoin dislocation is executable after
depth, fees, and settlement frictions. This project asks how one node's stress becomes
another node's stress across a multi-layer financial network.

## Core thesis

Stablecoin stress propagates through a layered network. Centralized venues often lead
price discovery at shock onset, decentralized pools amplify or prolong deviations through
inventory imbalance, and on-chain settlement or bridge-flow frictions help explain persistence.

## Event set

| Event | Mechanism | Analysis window |
|---|---|---|
| USDC/SVB 2023 | Fiat-reserve bank shock | 2023-03-08 → 2023-03-20 |
| Terra/LUNA 2022 | Algorithmic/reflexive collapse | 2022-05-01 → 2022-05-31 |
| USDT/Curve 2023 | DeFi pool imbalance | 2023-06-10 → 2023-06-25 |
| FTX 2022 | Exchange credit/liquidity shock | 2022-11-01 → 2022-11-30 |
| BUSD 2023 | Regulatory/issuer wind-down | 2023-02-06 → 2023-03-13 |

## Node taxonomy

The network contains three node layers:

| Layer | Examples | Primary features |
|---|---|---|
| Market nodes | USDC-Coinbase, USDT-Binance | mid, spread, depth, imbalance, executable price |
| Pool nodes | Curve 3pool, Uniswap USDC/USDT | reserve imbalance, implied price, swap flow, slippage |
| Flow nodes | ETH exchange flows, bridge flows, mint/burn | inflow, outflow, netflow, mint/burn, gas |

## Data provenance tiers

Every node and every claim is assigned a provenance tier:

| Tier | What it means | Permitted claim |
|---|---|---|
| A | On-chain logs, verified pool events, high-quality timestamped data | On-chain AMM flow, directional propagation |
| B | Trades, OHLCV, pool snapshots, aggregate on-chain flows | Price/liquidity context, weaker directional evidence |
| C | Partial, proxy, or sparse data | Taxonomy and qualitative context only |

No edge can support a stronger claim than the weaker of its two endpoint tiers.

Price co-movement is not treated as contagion. Contagion requires time-directed evidence:
lag structure, Granger/VAR relation, Hawkes excitation, transfer entropy, or temporal-graph
predictive lift.

## Methods

| Method | Role |
|---|---|
| Lead-lag cross-correlation | First-pass directional timing |
| Block-bootstrap inference | Robust significance testing |
| VAR / Granger causality | Linear directed dependence |
| Time-varying VAR spillovers | Dynamic edge strength across phases |
| Multivariate Hawkes processes | Mutual excitation of stress arrivals |
| Transfer entropy | Non-linear directional information flow |
| Temporal network centrality | Node roles: originator / amplifier / sink |
| Temporal GNN (TGN/DySAT) | Predictive extension |

## Primary outputs

| File | Description |
|---|---|
| `results/tables/table_event_windows.csv` | Event definitions and quality flags |
| `results/tables/table_node_coverage.csv` | Node availability and provenance tiers |
| `data/gold/dataset_contagion_features.parquet` | Final event-time feature panel |
| `results/tables/table_leadlag_tests.csv` | Pairwise lag tests with bootstrap p-values |
| `results/tables/table_var_spillovers.csv` | VAR/FEVD spillover estimates |
| `results/tables/table_hawkes_params.csv` | Hawkes branching ratios and CIs |
| `results/tables/table_transfer_entropy.csv` | Directional information flow estimates |
| `results/tables/table_node_centrality.csv` | Weighted centrality by event |
| `results/tables/table_prediction_metrics.csv` | AUC, PR-AUC, Brier, lift by event |
| `results/figures/figure_heatmap_coverage.png` | Data coverage heatmap |
| `results/figures/figure_event_time_map.png` | Event-time stress map |
| `results/figures/figure_contagion_map.png` | Directed multi-layer stress network |

## Success criteria

| Dimension | Criterion |
|---|---|
| Lead-lag | Upstream-to-downstream lags significant at p < 0.01 after block bootstrap |
| VAR/FEVD | At least one key off-diagonal relation per major event, spillover share > 10% |
| Hawkes | Off-diagonal branching ratio > 0.1 with CI excluding zero |
| Flow relevance | On-chain/bridge flows reduce downstream variance or RMSE by > 10% |
| Predictive | Graph model improves AUC or PR-AUC by > 5% over best non-graph baseline |
| Replication | Core propagation ordering in at least 4 of 5 event windows |

## Quick start

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env   # add API keys

# 2. generate event window definitions and coverage table
make windows

# 3. MVP: USDC/SVB full analysis pipeline
make mvp EVENT=usdc_svb_2023

# 4. paper outputs
make paper
```

See `Makefile` for per-step targets (`leadlag`, `var`, `hawkes`, `te`, `network`, `predict`).

> **⚠ Fixture data notice**
>
> Until real vendor/API ingestion is configured, `make ingest` writes
> **deterministic fixture files** into `data/bronze/` and marks them
> `tier_actual = fixture_non_empirical` in `data/manifests/`.
> These fixtures are valid for **pipeline testing only** and must
> **never be used as paper evidence**.  All empirical claims in the paper
> must be reproduced using real data; see `docs/reproducibility.md` and
> `docs/provenance_tiers.md`.

## Current empirical status

**See [`DATA_INVENTORY.md`](DATA_INVENTORY.md) for the full, verified inventory of what data is real, what is Tier B proxy, and what is fixture.**

The repo is pipeline-complete. Verified provenance as of 2026-05-29:

| Tier | Nodes | Events |
|---|---|---|
| **A (real)** | `usdc_mint_burn`, all `curve_*` pool nodes | usdc_svb_2023; all 5 events for Curve |
| **B (real proxy)** | all Binance nodes, `usdc_coinbase`, CoinMetrics flows | All 5 events |
| **fixture** | `usdc_kraken`, `usdt_kraken`, `uniswap_usdc_usdt_005`, `eth_bridge_flows`, `usdt_mint_burn`, `tron_usdt_exchange_flows` | Various |

**A/A provenance-valid pairs available in 3 of 5 events:**

| A/A pair | Event |
|---|---|
| `usdc_mint_burn` ↔ `curve_3pool` | usdc_svb_2023 |
| `curve_3pool` ↔ `curve_ust_wormhole` | terra_luna_2022 |
| `curve_3pool` ↔ `curve_crvusd_usdt` | usdt_curve_2023 |

All A/A pairs use `usdc_net_sold_1h` (direct hourly sum from on-chain `TokenExchange` logs).
Claims using derived features (`reserve_imbalance`, `implied_pool_price`) are A/B, not A/A.

**Provenance-valid ≠ statistically supported.** Directional paper claims require both gates:
the provenance gate (data source tier, no fixture, feature-level cap) *and* the statistical
gate (FDR-adjusted significance). Only rows where `paper_claim_allowed == True` in the
claim-gated result tables are used for headline directional claims in the paper.

For ftx_2022 and busd_2023, A/B directional evidence is available (curve_3pool A + Binance B nodes).

```bash
make coveragegate EVENT=usdc_svb_2023
python scripts/00c_claim_gate.py --all-events --strict
```

## Repository structure

```
stablecoin-contagion-network/
├── configs/           # event windows, nodes, features, sources, models, paper
├── data/
│   ├── raw/           # never committed
│   ├── bronze/        # normalized raw payloads
│   ├── silver/        # reconstructed books / pools / flows
│   ├── gold/          # final feature panels
│   └── manifests/     # source hashes and query manifests
├── src/stressnet/
│   ├── utils/         # time, I/O, logging, validation
│   ├── data/          # ingestion: binance, coinbase, kraken, uniswap, curve, etherscan, dune, coinmetrics
│   ├── reconstruct/   # orderbook, dex_pool, flows
│   ├── features/      # market, dex, onchain, basis, panels
│   ├── graph/         # nodes, edges, temporal_graph, centrality
│   ├── models/        # leadlag, var_granger, tvp_var, hawkes, transfer_entropy, baselines, temporal_gnn
│   ├── evaluation/    # bootstrap, placebo, metrics, robustness
│   └── plotting/      # coverage, event_maps, networks, sankey, paper_figures
├── scripts/           # 00_make_event_windows.py … 99_make_paper_outputs.py
├── notebooks/
├── results/
│   ├── tables/
│   └── figures/
├── docs/
│   ├── data_card.md
│   ├── node_taxonomy.md
│   ├── provenance_tiers.md
│   ├── event_windows.md
│   ├── methodology.md
│   ├── limitations.md
│   ├── reproducibility.md
│   └── project_control/
│       ├── full_project_control_report.md
│       ├── implementation_todo.md
│       ├── icaif_2026_submission_plan.md
│       └── gnn_extension_plan.md
├── tests/
└── paper/
```

## Reproducibility

All raw data sources are recorded in `data/manifests/`. Raw and reconstructed data are
not committed to git. Curated feature panels and paper-ready tables are generated by
numbered scripts and saved under `data/gold/` and `results/`.

## Non-claims

- Not all cross-node co-movement is causal contagion.
- Route-complete Tier-A L2 coverage is not available for every historical event.
- Correlation does not imply directed transmission.
- Temporal GNN prediction improvements do not imply causal mechanisms.
