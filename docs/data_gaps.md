# Data Gaps

This document records every known gap between the data we **have** and the data we **need**
for each event, along with the estimated impact on claim quality.

Gap severity:
- **P0** — blocks the headline result (no A/A edge possible without it)
- **P1** — degrades evidence tier but headline result survives at B/B
- **P2** — reduces robustness or completeness; paper still publishable

---

## usdc_svb_2023 (USDC de-peg, March 2023)

| Gap | Severity | Affected nodes | Current proxy | Required for Tier A |
|---|---|---|---|---|
| No full L2 for `usdc_binance` | P0 | usdc_binance | BBO (bookTicker) | Tardis `USDCUSDT` incremental_book_L2 |
| No full L2 for `usdt_binance` | P0 | usdt_binance | BBO (bookTicker) | Tardis `USDCUSDT` or `BTCUSDT` |
| No full L2 for `usdc_coinbase` | P0 | usdc_coinbase | 1-min klines | Tardis `USDC-USD` snapshot25 |
| `usdt_kraken` / `usdc_kraken` | P1 | kraken nodes | 1-min klines | Tardis `USDTUSD` / `USDCUSD` |
| Uniswap subgraph lag | P2 | uniswap_usdc_usdt_005 | The Graph (5 min lag) | Archive RPC (block-level) |
| Tron USDT flows | P2 | tron_usdt_exchange_flows | missing | TronGrid API |

## terra_luna_2022 (UST collapse, May 2022)

| Gap | Severity | Affected nodes | Current proxy | Notes |
|---|---|---|---|---|
| UST/Binance L2 archive | P0 | ust_binance | aggTrades only | Tardis historical; may be available; UST delisted |
| `usdt_binance` L2 | P1 | usdt_binance | BBO | Tardis (event outside free window) |
| On-chain Curve UST pool | P1 | curve_ust_wormhole | Etherscan logs ✅ | Already Tier A |
| Terra on-chain (non-EVM) | P2 | — | missing | Terra blockchain archive (out of scope v1) |

## usdt_curve_2023 (USDT/Curve stress, June 2023)

| Gap | Severity | Affected nodes | Current proxy | Notes |
|---|---|---|---|---|
| `usdt_binance` L2 | P0 | usdt_binance | BBO | Tardis `USDCUSDT`; June 2023 in free window |
| `curve_crvusd_usdt` block gaps | P1 | curve_crvusd_usdt | Etherscan logs | Block coverage may be uneven |
| Tron USDT flows | P2 | tron_usdt_exchange_flows | partial | Tron archive limited |

## ftx_2022 (FTX collapse, November 2022)

| Gap | Severity | Affected nodes | Current proxy | Notes |
|---|---|---|---|---|
| `usdt_binance` L2 | P0 | usdt_binance | BBO | Tardis Nov 2022 |
| `busd_binance` L2 | P0 | busd_binance | BBO | Tardis `BUSDUSDT`; BUSD still active Nov 2022 |
| Bridge flows | P1 | eth_bridge_flows | Dune partial | Dune query `usdc_bridge_flows.sql` |
| FTX-specific addresses | P2 | eth_usdt_exchange_flows | CoinMetrics | FTX address label confidence ~70 % |

## busd_2023 (BUSD wind-down, February 2023)

| Gap | Severity | Affected nodes | Current proxy | Notes |
|---|---|---|---|---|
| `busd_binance` L2 | P0 | busd_binance | BBO | Tardis `BUSDUSDT` Feb 2023 |
| `usdc_binance` L2 | P1 | usdc_binance | BBO | Tardis `USDCUSDT` |
| BUSD mint/burn on BSC | P2 | — | missing | BSC Etherscan scan; low priority |

---

## Summary: P0 gaps (blocks A/A headline)

All five events lack full L2 data for at least two CEX nodes.  The critical path is:

1. Ingest Tardis `incremental_book_L2` for `USDCUSDT` on Binance — covers
   usdc_svb_2023, usdt_curve_2023, and ftx_2022 in one download.
2. Ingest Tardis `BUSDUSDT` — covers busd_2023 and ftx_2022.
3. Ingest Tardis `USTUSDT` — covers terra_luna_2022 (if archive available).

See `docs/data_acquisition_plan.md` for the fetch sequence and cost estimates.

---

## Schema: gap tracking fields in manifest

Each node's manifest entry should include:

| Field | Type | Description |
|---|---|---|
| `coverage_pct` | float | Fraction of analysis window covered by non-null observations |
| `gap_rate` | float | Sequence gaps / total messages (for diff-depth streams) |
| `resync_count` | int | Number of forced book resyncs |
| `clock_skew_ms` | float | Median (local_ts − exchange_ts) over the window |
| `sequence_gap_count` | int | Count of messages where sequence_id jumped > 1 |
| `tier_downgrade_reason` | str | Empty string or one of: "incomplete_coverage", "sequence_gaps", "resync", "clock_unreliable" |
| `depth_source` | str | `"full_l2_book"`, `"bbo_only"`, `"ohlcv_proxy"` |
| `is_executable_bookwalk` | bool | True if bookwalk VWAP is available for this node/window |
