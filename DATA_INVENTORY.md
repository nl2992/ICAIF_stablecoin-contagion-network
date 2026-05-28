# Data Inventory

**Last verified: 2026-05-29**

This document records every data source actually present in the repository,
its verified provenance tier, what features it populates, and what is missing or fixture.

---

## Quick reference

| Symbol | Meaning |
|--------|---------|
| ✅ A | Tier A — execution-grade, directly from on-chain logs or L2 book |
| ⚠️ B | Tier B — real data, sufficient for context; not execution-grade |
| 🔴 FIX | `fixture_non_empirical` — synthetic pipeline data, **not paper-claimable** |
| ❌ | Not fetched / missing |

---

## usdc_svb_2023 (USDC de-peg, March 2023)

| Node | `tier_actual` | Source | Populated features | Notes |
|---|---|---|---|---|
| `usdc_mint_burn` | ✅ **A** | Etherscan ERC-20 Transfer logs | `mint_burn_net_1h`, `basis_vs_usd` | Only genuine Tier A node in the whole dataset |
| `curve_3pool` | ⚠️ B | Etherscan TokenExchange events | `usdc_net_sold_1h`, `reserve_imbalance`\*, `implied_pool_price`\* | Raw events are Tier A in substance; derived ratio Tier B (see §Curve below) |
| `usdc_binance` | ⚠️ B | Binance Vision bookTicker (BBO) | `spread_bps`, `basis_vs_usd` | `depth_10bps_bid_usd = null`, `orderbook_imbalance = null` |
| `usdt_binance` | ⚠️ B | Binance Vision bookTicker (BBO) | `spread_bps`, `basis_vs_usd` | Same as above |
| `usdc_coinbase` | ⚠️ B | Coinbase REST 1-min candles | `spread_bps`, `basis_vs_usd` | OHLCV proxy only |
| `eth_usdc_exchange_flows` | ⚠️ B | CoinMetrics exchange netflows | `exchange_netflow_1h` | Pre-aggregated; label confidence ~80 % |
| `eth_usdt_exchange_flows` | ⚠️ B | CoinMetrics exchange netflows | `exchange_netflow_1h` | Same |
| `usdc_kraken` | 🔴 **FIX** | `deterministic_pipeline_fixture` | Fake depth/imbalance/bookwalk | **Do not use.** Generated when `ETHERSCAN_API_KEY` absent |
| `usdt_kraken` | 🔴 **FIX** | `deterministic_pipeline_fixture` | Fake depth/imbalance/bookwalk | Same |
| `uniswap_usdc_usdt_005` | 🔴 **FIX** | `deterministic_pipeline_fixture` | Fake `reserve_imbalance` | The Graph subgraph not fetched |
| `eth_bridge_flows` | 🔴 **FIX** | `deterministic_pipeline_fixture` | Fake flow values | Dune query not executed |

---

## terra_luna_2022 (UST collapse, May 2022)

| Node | `tier_actual` | Source | Populated features | Notes |
|---|---|---|---|---|
| `curve_3pool` | ⚠️ B | Etherscan TokenExchange events | `usdc_net_sold_1h`, `reserve_imbalance`\* | 701 rows |
| `curve_ust_wormhole` | ⚠️ B | Etherscan TokenExchange events | `reserve_imbalance`\*, `implied_pool_price`\* | 180 rows; pool nearly drained (extreme values expected) |
| `usdt_binance` | ⚠️ B | Binance Vision bookTicker | `spread_bps`, `basis_vs_usd` | |
| `ust_binance` | ⚠️ B | Binance Vision aggTrades | `spread_bps`, `basis_vs_usd` | Trade-level only; `depth_10bps_bid_usd = null` |

---

## usdt_curve_2023 (USDT/Curve stress, June 2023)

| Node | `tier_actual` | Source | Populated features | Notes |
|---|---|---|---|---|
| `curve_3pool` | ⚠️ B | Etherscan TokenExchange events | `usdc_net_sold_1h`, `reserve_imbalance`\* | 379 rows |
| `curve_crvusd_usdt` | ⚠️ B | Etherscan TokenExchange events | `usdc_net_sold_1h`, `reserve_imbalance`\*\* | 285 rows; **scaling bug** — `reserve_imbalance` / `implied_pool_price` use wrong normalizer for this pool |
| `usdt_binance` | ⚠️ B | Binance Vision bookTicker | `spread_bps`, `basis_vs_usd` | |
| `eth_usdt_exchange_flows` | ⚠️ B | CoinMetrics | `exchange_netflow_1h` | |
| `usdt_mint_burn` | 🔴 **FIX** | `deterministic_pipeline_fixture` | Fake mint/burn data | Not fetched; no Etherscan key at ingest time |
| `usdt_kraken` | 🔴 **FIX** | `deterministic_pipeline_fixture` | Fake depth/imbalance/bookwalk | |
| `uniswap_usdc_usdt_005` | 🔴 **FIX** | `deterministic_pipeline_fixture` | Fake pool state | |
| `tron_usdt_exchange_flows` | 🔴 **FIX** | `deterministic_pipeline_fixture` | Fake flows | |

---

## ftx_2022 (FTX collapse, November 2022)

| Node | `tier_actual` | Source | Populated features | Notes |
|---|---|---|---|---|
| `curve_3pool` | ⚠️ B | Etherscan TokenExchange events | `usdc_net_sold_1h`, `reserve_imbalance`\* | 718 rows |
| `usdt_binance` | ⚠️ B | Binance Vision bookTicker | `spread_bps`, `basis_vs_usd` | |
| `busd_binance` | ⚠️ B | Binance Vision bookTicker | `spread_bps`, `basis_vs_usd` | |
| `eth_usdc_exchange_flows` | ⚠️ B | CoinMetrics | `exchange_netflow_1h` | |
| `eth_usdt_exchange_flows` | ⚠️ B | CoinMetrics | `exchange_netflow_1h` | |
| `eth_bridge_flows` | 🔴 **FIX** | `deterministic_pipeline_fixture` | Fake flow values | |

---

## busd_2023 (BUSD wind-down, February 2023)

| Node | `tier_actual` | Source | Populated features | Notes |
|---|---|---|---|---|
| `curve_3pool` | ⚠️ B | Etherscan TokenExchange events | `usdc_net_sold_1h`, `reserve_imbalance`\* | 864 rows |
| `busd_binance` | ⚠️ B | Binance Vision bookTicker + klines | `spread_bps`, `basis_vs_usd` | |
| `usdc_binance` | ⚠️ B | Binance Vision klines | `spread_bps`, `basis_vs_usd` | |
| `usdt_binance` | ⚠️ B | Binance Vision bookTicker | `spread_bps`, `basis_vs_usd` | |

---

## What is genuinely NOT available (no free source exists)

| Data | Why missing | Required for |
|---|---|---|
| Full L2 order book for any Binance node | No free historical archive; Tardis/Kaiko paid only | `depth_10bps_bid_usd`, `orderbook_imbalance`, bookwalk VWAP, Tier A for any CEX node |
| Real Kraken depth data | `kraken.py` returns `(None, 'fixture')` when key absent; Kraken free REST is OHLC only | Tier A CEX |
| Uniswap historical pool state | The Graph API not called; requires `THE_GRAPH_API_KEY` | Tier A DEX for Uniswap node |
| USDT mint/burn (Tether) | Etherscan ingest returns fixture when key absent | Tier A for usdt_curve_2023 |
| Tron USDT flows | TronGrid not implemented | Tier B flow context |
| Bridge flows (Dune) | Dune queries not executed; requires `DUNE_API_KEY` | Tier B flow context |

---

## The Curve tier situation

Curve pool `TokenExchange` events are fetched directly from Etherscan and are
**Tier A in substance** — they are on-chain transaction logs with exact block timestamps.

However, the ingestion script (`src/stressnet/data/curve.py`) computes `reserve_imbalance`
as `cumulative_usdc_net_sold / $500_000_000` (hardcoded 3pool normalizer) and
`implied_pool_price` as `1 / (1 + |reserve_imbalance|)`. These are **approximations**
because:

1. The normalizer is hardcoded to $500M and wrong for other pools (crvUSD/USDT, UST/3CRV)
2. True on-chain reserves require `get_balances()` at each block (needs archive RPC)

**What is correctly Tier A:** `usdc_net_sold_1h` (direct hourly sum of TokenExchange amounts).  
**What is correctly Tier B:** `reserve_imbalance`, `implied_pool_price` (derived proxies).

The `curve_crvusd_usdt` pool shows extreme values (`reserve_imbalance ~ -1e9`,
`implied_pool_price ~ 9e-10`) due to decimal-scaling mismatch between crvUSD (18 dec)
and USDT (6 dec) in the raw event parser. This needs fixing before that pool can be used.

---

## Current A/A eligible pairs (no additional data required)

| Pair | Event | Claim | Condition |
|---|---|---|---|
| `usdc_mint_burn` ↔ `curve_3pool` via `usdc_net_sold_1h` | usdc_svb_2023 | Directional on-chain flow: CEX redemption pressure vs. AMM swap flow | Both re-tagged to Tier A for raw event features only |

To claim this A/A edge:
1. Fix the `ingest_curve_pool_events` return value from `'B'` to `'A'` for the raw `usdc_net_sold_1h` column
2. In the manifest, tag `usdc_net_sold_1h` feature as `depth_source="on_chain_event_log"` and `tier_actual="A"`
3. Rebuild the panel and re-run the claim gate

---

## To unlock additional real data (free, just needs keys)

| Action | Key needed | Nodes unlocked |
|---|---|---|
| Set `ETHERSCAN_API_KEY` and re-run `scripts/01_fetch_raw_data.py` | `ETHERSCAN_API_KEY` | `usdt_mint_burn` (usdt_curve_2023), real Curve data for all events |
| Set `THE_GRAPH_API_KEY` and re-run | `THE_GRAPH_API_KEY` | `uniswap_usdc_usdt_005` (Tier B → maybe A with archive) |
| Set `DUNE_API_KEY` and run `scripts/01c_ingest_dune_queries.py` | `DUNE_API_KEY` | `eth_bridge_flows`, `tron_usdt_exchange_flows` |

Etherscan free tier: 5 calls/second, no daily limit. The re-ingest for all 5 events
takes approximately 30 minutes.

---

## Paper claim ceiling (current state, no new data)

| Claim level | Evidence available |
|---|---|
| **A_A** | Zero confirmed — pending Curve tier fix |
| **A_B** | `usdc_mint_burn` (A) paired with any Tier B node |
| **B_B** | All 881 existing result rows — Binance/CoinMetrics/Curve proxy features |

The 881 rows of results from the empirical pipeline are real and use real data.
The claims are correctly labeled B_B ("contextual co-movement").
No A/A edge has been demonstrated yet.
