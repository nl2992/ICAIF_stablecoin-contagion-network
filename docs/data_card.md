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
| `depth_10bps_bid_usd` | float | Cumulative bid depth within 10 bps when L2 exists; proxy/null otherwise |
| `depth_10bps_ask_usd` | float | Cumulative ask depth within 10 bps when L2 exists; proxy/null otherwise |
| `orderbook_imbalance` | float | (bid - ask depth) / (bid + ask depth) |
| `executable_price_10k_buy` | float | Book-walk VWAP for $10k buy order when L2 exists; proxy/null otherwise |
| `executable_price_10k_sell` | float | Book-walk VWAP for $10k sell order when L2 exists; proxy/null otherwise |
| `depth_source` | str | Depth provenance, e.g. `l2_bookwalk`, `best_level_bbo_proxy`, `unavailable_ohlcv` |
| `executable_price_source` | str | Executable-price provenance, e.g. `l2_bookwalk`, `best_level_bbo_proxy`, `pool_slippage_proxy` |
| `microstructure_quality` | str | Operational feature-quality label such as `bbo_proxy`, `ohlcv_proxy`, `dex_pool_proxy` |
| `is_executable_bookwalk` | bool | True only when executable prices are computed from depth-level book-walk reconstruction |
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

## Coverage

Coverage is generated from `data/manifests/manifest_*.csv` into
`results/tables/table_node_coverage.csv`. The table records actual tier, row
counts, source names, coverage percentage when available, sequence-gap counts,
resync counts, and clock-offset diagnostics. Nominal Tier-A rows are downgraded
when coverage is weak or sequence diagnostics fail.

## Data collection

All raw data is fetched by `scripts/01_ingest_raw_data.py` and saved with source hashes
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

Run `make mvp EVENT=usdc_svb_2023` to regenerate the fixture-safe MVP panel, or
`make empirical EVENT=usdc_svb_2023` to run the no-fixture empirical path.
All intermediate artefacts are deterministic given fixed API responses.
