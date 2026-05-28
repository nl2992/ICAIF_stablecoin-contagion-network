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
| `usdc_mint_burn` | ✅ **A** | Etherscan ERC-20 Transfer logs | `mint_burn_net_1h`, `basis_vs_usd` | Genuine Tier A: 4 real hourly mint/burn events |
| `curve_3pool` | ✅ **A** | Etherscan TokenExchange events | `usdc_net_sold_1h`\*, `reserve_imbalance`†, `implied_pool_price`† | 312 hourly rows; raw on-chain flow is Tier A; derived reserve proxy is Tier B |
| `usdc_binance` | ⚠️ B | Binance Vision bookTicker (BBO) | `spread_bps`, `basis_vs_usd` | `depth_10bps_bid_usd = null` |
| `usdt_binance` | ⚠️ B | Binance Vision bookTicker (BBO) | `spread_bps`, `basis_vs_usd` | Same as above |
| `usdc_coinbase` | ⚠️ B | Coinbase REST 1-min candles | `spread_bps`, `basis_vs_usd` | OHLCV proxy only |
| `eth_usdc_exchange_flows` | ⚠️ B | CoinMetrics exchange netflows | `exchange_netflow_1h` | Pre-aggregated |
| `eth_usdt_exchange_flows` | ⚠️ B | CoinMetrics exchange netflows | `exchange_netflow_1h` | Same |
| `usdc_kraken` | 🔴 **FIX** | `deterministic_pipeline_fixture` | Synthetic depth/imbalance | No free historical depth; spread also synthetic |
| `usdt_kraken` | 🔴 **FIX** | `deterministic_pipeline_fixture` | Synthetic depth/imbalance | Same |
| `uniswap_usdc_usdt_005` | 🔴 **FIX** | `deterministic_pipeline_fixture` | Synthetic `reserve_imbalance` | The Graph subgraph not fetched |
| `eth_bridge_flows` | 🔴 **FIX** | `deterministic_pipeline_fixture` | Synthetic flow values | Dune query not executed |

**A/A edge confirmed**: `usdc_mint_burn (A) ↔ curve_3pool (A)` via `usdc_net_sold_1h` and `mint_burn_net_1h`

---

## terra_luna_2022 (UST collapse, May 2022)

| Node | `tier_actual` | Source | Populated features | Notes |
|---|---|---|---|---|
| `curve_3pool` | ✅ **A** | Etherscan TokenExchange events | `usdc_net_sold_1h`\*, `reserve_imbalance`†, `implied_pool_price`† | 701 hourly rows |
| `curve_ust_wormhole` | ✅ **A** | Etherscan TokenExchange events | `usdc_net_sold_1h`\*, `reserve_imbalance`†, `implied_pool_price`† | 180 hourly rows; pool nearly drained (extreme values expected) |
| `usdt_binance` | ⚠️ B | Binance Vision bookTicker | `spread_bps`, `basis_vs_usd` | |
| `ust_binance` | ⚠️ B | Binance Vision aggTrades | `spread_bps`, `basis_vs_usd` | Trade-level; `depth_10bps_bid_usd = null` |

**A/A edge confirmed**: `curve_3pool (A) ↔ curve_ust_wormhole (A)` via `usdc_net_sold_1h` (3pool USDC flow vs UST/3CRV pool imbalance)

---

## usdt_curve_2023 (USDT/Curve stress, June 2023)

| Node | `tier_actual` | Source | Populated features | Notes |
|---|---|---|---|---|
| `curve_3pool` | ✅ **A** | Etherscan TokenExchange events | `usdc_net_sold_1h`\*, `reserve_imbalance`†, `implied_pool_price`† | 379 hourly rows |
| `curve_crvusd_usdt` | ✅ **A** | Etherscan TokenExchange events | `usdc_net_sold_1h`\*, `reserve_imbalance`†, `implied_pool_price`† | 285 hourly rows; decimal bug **fixed** (was ~1e10, now -0.41 to +0.02) |
| `usdt_binance` | ⚠️ B | Binance Vision bookTicker | `spread_bps`, `basis_vs_usd` | |
| `eth_usdt_exchange_flows` | ⚠️ B | CoinMetrics | `exchange_netflow_1h` | |
| `usdt_mint_burn` | 🔴 **FIX** | `deterministic_pipeline_fixture` | Synthetic mint/burn data | USDT uses `Issue`/`Redeem` events, not standard ERC-20 Transfer; Etherscan approach insufficient |
| `usdt_kraken` | 🔴 **FIX** | `deterministic_pipeline_fixture` | Synthetic depth/imbalance | |
| `uniswap_usdc_usdt_005` | 🔴 **FIX** | `deterministic_pipeline_fixture` | Synthetic pool state | |
| `tron_usdt_exchange_flows` | 🔴 **FIX** | `deterministic_pipeline_fixture` | Synthetic flows | TronGrid not implemented |

**A/A edge confirmed**: `curve_3pool (A) ↔ curve_crvusd_usdt (A)` — both Etherscan on-chain flow features

---

## ftx_2022 (FTX collapse, November 2022)

| Node | `tier_actual` | Source | Populated features | Notes |
|---|---|---|---|---|
| `curve_3pool` | ✅ **A** | Etherscan TokenExchange events | `usdc_net_sold_1h`\*, `reserve_imbalance`†, `implied_pool_price`† | 718 hourly rows |
| `usdt_binance` | ⚠️ B | Binance Vision bookTicker | `spread_bps`, `basis_vs_usd` | |
| `busd_binance` | ⚠️ B | Binance Vision bookTicker | `spread_bps`, `basis_vs_usd` | |
| `eth_usdc_exchange_flows` | ⚠️ B | CoinMetrics | `exchange_netflow_1h` | |
| `eth_usdt_exchange_flows` | ⚠️ B | CoinMetrics | `exchange_netflow_1h` | |
| `eth_bridge_flows` | 🔴 **FIX** | `deterministic_pipeline_fixture` | Synthetic flow values | |

**A/B edges only**: `curve_3pool (A) ↔ usdt_binance (B)` etc. No second A node in this event yet.

---

## busd_2023 (BUSD wind-down, February 2023)

| Node | `tier_actual` | Source | Populated features | Notes |
|---|---|---|---|---|
| `curve_3pool` | ✅ **A** | Etherscan TokenExchange events | `usdc_net_sold_1h`\*, `reserve_imbalance`†, `implied_pool_price`† | 864 hourly rows |
| `busd_binance` | ⚠️ B | Binance Vision bookTicker + klines | `spread_bps`, `basis_vs_usd` | |
| `usdc_binance` | ⚠️ B | Binance Vision klines | `spread_bps`, `basis_vs_usd` | |
| `usdt_binance` | ⚠️ B | Binance Vision bookTicker | `spread_bps`, `basis_vs_usd` | |

**A/B edges only**: No second A node in this event. A/B claim possible with `curve_3pool (A)` as anchor.

---

## Feature-level tier notes

\* **`usdc_net_sold_1h`**: **Tier A** — direct hourly sum of on-chain `TokenExchange` amounts.
  For Curve 3pool and meta-pools: denominated in USDC/stablecoin native units (6 dec).
  For `curve_crvusd_usdt` (StableSwap-ng): denominated in 18-dec internal units, correctly normalised.

† **`reserve_imbalance`, `implied_pool_price`**: **Tier B** — derived proxy.
  `reserve_imbalance = usdc_net_sold_cum / pool_size_usd` where `pool_size_usd` is a
  hardcoded per-pool estimate (not a real-time archive-RPC balance call). Claims using
  these features must be stated at Tier B.

---

## What is genuinely NOT available (no free source exists)

| Data | Why missing | Required for |
|---|---|---|
| Full L2 order book for any Binance node | No free historical archive; Tardis/Kaiko paid only | `depth_10bps_bid_usd`, `orderbook_imbalance`, bookwalk VWAP, Tier A for any CEX node |
| Real Kraken depth data | Public REST is OHLC only; depth returns `None` | Tier A CEX |
| Uniswap historical pool state | The Graph API not called; requires `THE_GRAPH_API_KEY` | Tier A DEX for Uniswap node |
| USDT mint/burn (Tether ERC-20) | Tether uses `Issue`/`Redeem` events not standard Transfer; current etherscan.py only queries Transfer | Tier A for usdt_curve_2023 |
| Tron USDT flows | TronGrid not implemented | Tier B flow context |
| Bridge flows (Dune) | Dune queries not executed; requires `DUNE_API_KEY` | Tier B flow context |

---

## The Curve tier situation

Curve pool `TokenExchange` events are fetched directly from Etherscan and are
**Tier A in substance** — they are on-chain transaction logs with exact block timestamps.
As of 2026-05-29, `ingest_curve_pool_events` now returns `'A'` for all Curve pools.

**Feature-level tiers:**
- `usdc_net_sold_1h` = **Tier A** (direct on-chain event sum)
- `reserve_imbalance` = **Tier B** (derived: `cum_sum / pool_size_usd`, normaliser is approximate)
- `implied_pool_price` = **Tier B** (derived: `1 / (1 + |reserve_imbalance|)`, approximation)

**StableSwap-ng decimal fix** (2026-05-29): The `curve_crvusd_usdt` pool uses
StableSwap-ng, which emits all token amounts in 18-decimal internal units regardless of
the token's native decimals (USDT has 6). The old code divided by 1e6 (USDT dec) instead
of 1e18, producing `reserve_imbalance ~ -2.4e10`. The fix adds a per-pool `PoolConfig`
with `ng_scaled=True`, dividing by 1e18 instead. After the fix:
`reserve_imbalance ∈ [-0.41, 0.02]` (was ±2.5e10).

---

## Current A/A confirmed pairs (2026-05-29)

| Pair | Event | Claim type |
|---|---|---|
| `usdc_mint_burn` ↔ `curve_3pool` via `usdc_net_sold_1h` / `mint_burn_net_1h` | usdc_svb_2023 | Directed on-chain flow: CEX redemption pressure vs. AMM swap flow |
| `curve_3pool` ↔ `curve_ust_wormhole` via `usdc_net_sold_1h` | terra_luna_2022 | Cross-pool USDC stress propagation during Terra collapse |
| `curve_3pool` ↔ `curve_crvusd_usdt` via `usdc_net_sold_1h` | usdt_curve_2023 | Cross-pool USDT stress: 3pool vs crvUSD pool imbalance |

Note: All A/A pairs use `usdc_net_sold_1h` (Tier A) as the linking feature. Claims
using `reserve_imbalance` or `implied_pool_price` are A/B, not A/A.

---

## To unlock additional real data (free, just needs keys)

| Action | Key needed | Nodes unlocked |
|---|---|---|
| Set `THE_GRAPH_API_KEY` and re-run | `THE_GRAPH_API_KEY` | `uniswap_usdc_usdt_005` (Tier B → maybe A with archive) |
| Set `DUNE_API_KEY` and run `scripts/01c_ingest_dune_queries.py` | `DUNE_API_KEY` | `eth_bridge_flows`, `tron_usdt_exchange_flows` |
| Implement Tether `Issue`/`Redeem` event decoder in etherscan.py | `ETHERSCAN_API_KEY` (already set) | `usdt_mint_burn` as real Tier A for usdt_curve_2023 |

---

## Paper claim ceiling (current state, 2026-05-29)

| Claim level | Evidence available | Events |
|---|---|---|
| **A_A** | 3 confirmed pairs (see table above) | usdc_svb_2023, terra_luna_2022, usdt_curve_2023 |
| **A_B** | `curve_3pool (A)` paired with any B node | All 5 events |
| **B_B** | All Binance/CoinMetrics results (876 rows) | All 5 events |

The 881-row result set from the empirical pipeline uses real data (no fixture).
With A/A pairs now confirmed in 3 events, the paper gate for headline
microstructure claims is unlocked for those 3 events.

For ftx_2022 and busd_2023, A/A claims remain blocked (single A node each).
These events are supported by A/B directional evidence.
