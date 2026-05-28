# Vendor Decision Log

This document records every decision to include, exclude, or defer a data vendor.
It exists so that reviewers can trace the provenance of each data source choice.

---

## Tardis — Selected as primary CEX L2 vendor

**Decision date:** 2026-05-28  
**Decision:** Use Tardis as the primary vendor for Tier-A CEX L2 data.

**Rationale:**
1. Tardis provides `incremental_book_L2` (diff-depth) and `book_snapshot_25` (25-level
   snapshots) for Binance, Coinbase, Kraken and ~50 other exchanges.
2. Exchange-assigned timestamps (`exchange_ts`) are preserved alongside local ingestion
   timestamps (`local_ts`), enabling clock-skew diagnostics.
3. Sequence IDs are included, allowing gap detection without reference to a separate
   normalised feed.
4. Historical archive extends back to 2017 for major pairs; covers all five events.
5. Academic programme provides free access for research use.
6. Python client (`tardis-dev`) is well-maintained and returns normalized CSV/Parquet.

**Limitations:**
- Paid subscription required for historical data outside free tier window (last 3 days).
- UST/Binance (delisted May 2022) availability uncertain; must verify before relying on it.
- Coinbase `USDC-USD` pair does not exist on Coinbase Exchange (USDC redeems at par).
  Fallback: `USDC-USDT` or the `book_snapshot_25` for `BTC-USD` as spread proxy.

**Alternative considered:** Kaiko — comparable coverage; selected Tardis because of
simpler Python client and better sequence-ID coverage for gap detection.

---

## Kaiko — Deferred (backup)

**Decision date:** 2026-05-28  
**Decision:** Deferred to backup only.

**Rationale:** Kaiko `full_order_book` and `market_depth` endpoints are equivalent to
Tardis but require REST polling rather than streaming.  The `src/stressnet/data/kaiko_l2.py`
stub exists and can be activated by setting `DATA_VENDOR=kaiko` in `.env`.

---

## Binance Vision — Retained for Tier B (public, no key)

**Decision date:** initial setup  
**Decision:** Use Binance Vision aggTrades, klines, and bookTicker as the Tier-B proxy
data until Tardis L2 is available.

**Rationale:**
- Freely available, no API key, no rate limits for bulk download.
- Covers all five events.
- bookTicker provides BBO (single best bid/ask level) — sufficient for spread proxy.
- klines provide 1-minute OHLCV for price context.

**Limitation:** bookTicker is BBO only — no depth, so `depth_10bps_bid_usd` cannot
be computed from it.  `depth_source = "bbo_only"` is set in the manifest.

---

## Etherscan — Retained for on-chain logs (Tier A/B)

Transfer events and contract logs fetched at the block level provide Tier-A timestamps
for `usdc_mint_burn`, `usdt_mint_burn`, `curve_3pool`, and `curve_ust_wormhole`.

**Limitation:** Etherscan API key required; `ETHERSCAN_API_KEY` in `.env`.
Block-range pagination limited to 10,000 results per call; the ingestion script applies
recursive halving to avoid dropping events.

---

## The Graph — Retained for Uniswap subgraph (Tier B, upgrading)

**Current status:** The Graph Uniswap v3 subgraph for `uniswap_usdc_usdt_005`.

**Limitation:** 5–15 minute block-confirmation lag; not Tier A without archive RPC.
`THE_GRAPH_API_KEY` required for production-level queries.

**Upgrade path:** Archive RPC (`ETH_ARCHIVE_RPC_URL`) + `scripts/01d_ingest_archive_pool_state.py`
will reconstruct exact per-block pool state, enabling Tier A for this node.

---

## Coin Metrics — Retained for exchange flow aggregates (Tier B)

Pre-aggregated exchange netflows from Coin Metrics provide `exchange_netflow_1h` for
USDC and USDT.  These are Tier B because the underlying address label confidence is ~80 %.

`COINMETRICS_API_KEY` required.

---

## TronGrid — Deferred to Phase 6

Tron USDT (`tron_usdt_exchange_flows`) is relevant for `usdt_curve_2023` but the
impact on overall results is P2.  Deferred until all five events have at least one
A/A edge from Ethereum/CEX sources.

`TRONGRID_API_KEY` needed when this phase is activated.

---

## Archive RPC — Deferred to Phase 4

`ETH_ARCHIVE_RPC_URL` needed for exact block-level Uniswap pool state.  A free Alchemy
or Infura archive endpoint is sufficient.  Deferred until Tardis L2 is delivering Tier A
for CEX nodes (higher marginal impact).
