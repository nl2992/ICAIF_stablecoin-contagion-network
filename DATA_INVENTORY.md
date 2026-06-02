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

**A/A provenance-valid pair**: `usdc_mint_burn (A) ↔ curve_3pool (A)` via `usdc_net_sold_1h`
and `mint_burn_net_1h`.

**Paper-claimable result**: Not statistically supported. The sparse-flow event study (4 arrivals,
1000 permutations) finds no significant response enrichment (p = 1.0 for curve_3pool, NaN for
CEX nodes which lack `usdc_net_sold_1h`). Reported as **high-provenance descriptive evidence**.
The mint/burn series is too sparse for continuous lead-lag; a Hawkes or binomial test with larger
time windows may recover power.

**Limitations**: Only 4 mint/burn events in the 12-day window. CEX nodes do not have
`usdc_net_sold_1h` (Tier-A feature); responses of CEX nodes to mint/burn arrivals cannot be
tested with this feature. `reserve_imbalance` is Tier B derived.

---

## terra_luna_2022 (UST collapse, May 2022)

| Node | `tier_actual` | Source | Populated features | Notes |
|---|---|---|---|---|
| `curve_3pool` | ✅ **A** | Etherscan TokenExchange events | `usdc_net_sold_1h`\*, `reserve_imbalance`†, `implied_pool_price`† | 701 hourly rows |
| `curve_ust_wormhole` | ✅ **A** | Etherscan TokenExchange events | `usdc_net_sold_1h`\*, `reserve_imbalance`†, `implied_pool_price`† | 180 hourly rows; pool nearly drained (extreme values expected) |
| `usdt_binance` | ⚠️ B | Binance Vision bookTicker | `spread_bps`, `basis_vs_usd` | |
| `ust_binance` | ⚠️ B | Binance Vision aggTrades | `spread_bps`, `basis_vs_usd` | Trade-level; `depth_10bps_bid_usd = null` |

**A/A provenance-valid pair**: `curve_3pool (A) ↔ curve_ust_wormhole (A)` via `usdc_net_sold_1h`
(3pool USDC flow vs UST/3CRV pool imbalance).

**Paper-claimable result**: Not statistically supported at the hourly grid in the AMM-only
lead-lag analysis (0/2 pairs pass at p < 0.01). Reported as **high-provenance descriptive
evidence** (`claim_strength = suggestive`). The pool drain is extreme and may reduce
cross-correlation power.

**Limitations**: `curve_ust_wormhole` coverage < 50% for some sub-windows (pool drained mid-event);
may be downgraded to Tier B in robustness checks. `reserve_imbalance` is Tier B derived.

---

## usdt_curve_2023 (USDT/Curve stress, June 2023)  ← **primary technical case study**

| Node | `tier_actual` | Source | Populated features | Notes |
|---|---|---|---|---|
| `curve_3pool` | ✅ **A** | Etherscan TokenExchange events | `usdc_net_sold_1h`\*, `reserve_imbalance`†, `implied_pool_price`† | 379 hourly rows |
| `curve_crvusd_usdt` | ✅ **A** | Etherscan TokenExchange events | `usdc_net_sold_1h`\*, `reserve_imbalance`†, `implied_pool_price`† | 285 hourly rows; decimal bug **fixed** (was ~1e10, now -0.41 to +0.02) |
| `usdt_binance` | ⚠️ B | Binance Vision bookTicker | `spread_bps`, `basis_vs_usd` | |
| `eth_usdt_exchange_flows` | ⚠️ B | CoinMetrics | `exchange_netflow_1h` | |
| `usdt_mint_burn` | ✅ **A** | Etherscan `eth_getLogs` (Issue/Redeem) | `mint_burn_net_1h` | USDT uses `Issue(uint256)`/`Redeem(uint256)` events; decoded via `ingest_tether_issue_redeem()` in `etherscan.py`; `TETHER_MINT_CONFIGS` registry used |
| `usdt_kraken` | 🔴 **FIX** | `deterministic_pipeline_fixture` | Synthetic depth/imbalance | |
| `uniswap_usdc_usdt_005` | 🔴 **FIX** | `deterministic_pipeline_fixture` | Synthetic pool state | |
| `tron_usdt_exchange_flows` | 🔴 **FIX** | `deterministic_pipeline_fixture` | Synthetic flows | TronGrid not implemented |

**A/A provenance-valid pair**: `curve_3pool (A) ↔ curve_crvusd_usdt (A)` — both Etherscan on-chain flow features.

**Paper-claimable result**: Both directions (`curve_3pool → curve_crvusd_usdt` and reverse) are
**Bonferroni-significant (p ≤ 0.014) on `usdc_net_sold_1h` at the hourly grid** in the AMM-only
lead-lag analysis. `claim_strength = robust`, `paper_claim_allowed = True`. These 2 rows are in
`results/paper/tables/table_aa_paper_claimable_edges.csv`.

**A/A settlement pair newly available**: `usdt_mint_burn (A) ↔ curve_3pool (A)` via `mint_burn_net_1h` and
`usdc_net_sold_1h`.  The settlement-layer Tether Issue/Redeem events can now be fetched; run
`make empirical EVENT=usdt_curve_2023` with `ETHERSCAN_API_KEY` set to populate the real data.

**Limitations**: `reserve_imbalance` and `implied_pool_price` are Tier-B derived proxies.

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

## Current A/A provenance-valid pairs (2026-05-29)

These pairs pass the **provenance gate** only. Whether they also pass the **statistical gate**
is determined by `paper_claim_allowed == True` in the claim-gated result tables after running
`python scripts/00c_claim_gate.py --all-events`.

| Pair | Event | Provenance claim type |
|---|---|---|
| `usdc_mint_burn` ↔ `curve_3pool` via `usdc_net_sold_1h` / `mint_burn_net_1h` | usdc_svb_2023 | A/A on-chain settlement + AMM flow |
| `curve_3pool` ↔ `curve_ust_wormhole` via `usdc_net_sold_1h` | terra_luna_2022 | A/A DEX flow (cross-pool) |
| `curve_3pool` ↔ `curve_crvusd_usdt` via `usdc_net_sold_1h` | usdt_curve_2023 | A/A DEX flow (cross-pool) |

Note: All A/A pairs use `usdc_net_sold_1h` (Tier A direct on-chain AMM flow) as the linking feature.
Claims using `reserve_imbalance` or `implied_pool_price` are A/B, not A/A.
**Provenance-valid ≠ paper-claimable.** Paper claims require both gates to pass.

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
| **A_A_dex_flow** | 2 provenance-valid pairs (`curve_3pool`↔`curve_ust_wormhole`, `curve_3pool`↔`curve_crvusd_usdt`) | terra_luna_2022, usdt_curve_2023 |
| **A_A_onchain_settlement** | 1 provenance-valid pair (`usdc_mint_burn`↔`curve_3pool`) — sparse; use event-arrival method | usdc_svb_2023 |
| **A_B** | `curve_3pool (A)` paired with any B node | All 5 events |
| **B_B** | All Binance/CoinMetrics results | All 5 events |

The paper gate for headline **Tier-A on-chain AMM-flow claims** is unlocked when at least one
A/A DEX-flow edge is `paper_claim_allowed == True` after statistical testing. Run
`python scripts/00c_claim_gate.py --all-events --strict` to verify.

Historical CEX microstructure claims (executable L2 depth, order-book imbalance) remain
**unavailable** without vendor/live L2 data (Tardis or Kaiko). Public Binance bookTicker
and OHLCV data are Tier B and support contextual co-movement only.

For ftx_2022 and busd_2023, A/A claims remain blocked (single A node each).
These events are supported by A/B directional evidence (`curve_3pool A` + Binance/CoinMetrics B).
