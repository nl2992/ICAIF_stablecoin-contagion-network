# Data Card

## Dataset: Stablecoin Contagion Network Feature Panel

**Version:** 0.1.0 (USDC/SVB MVP)
**File:** `data/gold/dataset_contagion_features.parquet`
**License:** Research use only. Data sourced from public exchange APIs, on-chain logs, and optional paid archives.

---

## Overview

Event-synchronized, provenance-tiered feature panel for studying stress propagation
across a multi-layer stablecoin network. Each row corresponds to one node at one
timestamp within an event analysis window.

## Schema

| Column | Type | Description |
|---|---|---|
| `event_id` | str | Event identifier from `configs/events.yaml` |
| `node_id` | str | Node identifier from `configs/nodes.yaml` |
| `node_layer` | str | `CEX` / `DEX` / `onchain_flow` / `bridge_flow` / `mint_burn` |
| `wall_clock_utc` | datetime64[ns, UTC] | Absolute UTC timestamp |
| `event_time_seconds` | float | Seconds relative to shock onset (T=0) |
| `tier_nominal` | str | Provenance tier from config: `A` / `B` / `C` |
| `tier_actual` | str | Effective tier after downgrade checks |
| `mid_price` | float | (best_bid + best_ask) / 2; null if unavailable |
| `spread_bps` | float | (best_ask - best_bid) / mid × 10,000 |
| `depth_10bps_bid_usd` | float | Cumulative bid depth within 10 bps |
| `depth_10bps_ask_usd` | float | Cumulative ask depth within 10 bps |
| `orderbook_imbalance` | float | (bid - ask depth) / (bid + ask depth) |
| `executable_price_10k_buy` | float | Book-walk VWAP for $10k buy order |
| `executable_price_10k_sell` | float | Book-walk VWAP for $10k sell order |
| `basis_vs_usd` | float | log(mid) - 0; deviation from $1.00 peg in log points |
| `reserve_imbalance` | float | DEX pool reserve skew; null for non-pool nodes |
| `implied_pool_price` | float | DEX marginal price; null for non-pool nodes |
| `pool_slippage_10k` | float | DEX $10k swap impact in bps; null for non-pool nodes |
| `exchange_netflow_1h` | float | Net exchange inflow (USD); null for non-flow nodes |
| `bridge_netflow_1h` | float | Net bridge flow (USD); null for non-flow nodes |
| `mint_burn_net_1h` | float | Net on-chain minting (USD); null for non-mint nodes |
| `gas_base_fee_gwei` | float | Ethereum base fee; ETH-chain nodes only |
| `label_basis_gt10bps` | bool | \|basis_vs_usd\| > 10 bps |
| `label_basis_gt50bps` | bool | \|basis_vs_usd\| > 50 bps |
| `label_downstream_gt10bps_1m` | bool | Will any downstream node exceed 10 bps within 1 min? |

## Coverage (MVP: USDC/SVB 2023)

| Node | Tier | Rows | Date range | Source |
|---|---|---|---|---|
| usdc_coinbase | A | TBD | 2023-03-08 to 2023-03-20 | Coinbase WS Level2 |
| usdc_binance | A | TBD | 2023-03-08 to 2023-03-20 | Binance Vision |
| usdt_binance | A | TBD | 2023-03-08 to 2023-03-20 | Binance Vision |
| usdt_kraken | A | TBD | 2023-03-08 to 2023-03-20 | Kraken WS |
| curve_3pool | A | TBD | 2023-03-08 to 2023-03-20 | Etherscan on-chain logs |
| uniswap_usdc_usdt_005 | A | TBD | 2023-03-08 to 2023-03-20 | The Graph / Etherscan |
| eth_usdc_exchange_flows | B | TBD | 2023-03-08 to 2023-03-20 | Coin Metrics / Dune |
| eth_usdt_exchange_flows | B | TBD | 2023-03-08 to 2023-03-20 | Coin Metrics / Dune |

## Data collection

All raw data is fetched by `scripts/01_fetch_raw_data.py` and saved with source hashes
in `data/manifests/`. No raw data is committed to git.

Reconstruction steps:
1. Bronze: normalised raw payloads (JSON → Parquet)
2. Silver: reconstructed order books / pool states / flow aggregates
3. Gold: event-time aligned feature panel

## Known data quality issues

See `docs/limitations.md` for full details. Key issues:
- FTX L2 unavailable; FTX event uses downstream venues only.
- Exchange flow labelling ~70-85% complete for large wallets.
- Terra/LUNA UST CEX archives may be incomplete; Curve pool state is Tier-A.

## Reproducibility

Run `make windows && make panel EVENT=usdc_svb_2023` to regenerate the MVP panel.
All intermediate artefacts are deterministic given fixed API responses.
