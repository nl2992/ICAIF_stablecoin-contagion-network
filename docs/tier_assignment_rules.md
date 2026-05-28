# Tier Assignment Rules

This document is the authoritative specification for how every data source is assigned
to Tier A, B, or C.  It supersedes any implicit ordering in the codebase and is checked
by the coverage gate (`scripts/00d_check_empirical_coverage.py`).

---

## Tier A — Execution-grade microstructure

A source qualifies for Tier A if and only if **all** of the following hold:

| Criterion | Requirement |
|-----------|-------------|
| **Granularity** | Tick-level or full L2 (snapshot + incremental updates) for CEX; per-block or per-event for DEX/on-chain |
| **Timestamps** | Exchange-assigned timestamps available and monotonically non-decreasing |
| **Completeness** | `coverage_pct ≥ 50 %` over the analysis window |
| **Sequence integrity** | `gap_rate < 1 %` (CEX diff-depth only) |
| **No forced resync** | `resync_count == 0` OR resync spans < 5 min total |
| **Bid/ask depth** | Full L2 or at minimum 10-level depth sufficient for bookwalk VWAP |

### Tier A source types (exhaustive)

| Source type | `source_type` tag | Example nodes |
|---|---|---|
| Vendor L2 (Tardis `incremental_book_L2`) | `tardis_l2` | `usdc_binance_l2`, `usdt_binance_l2` |
| Vendor L2 (Tardis `book_snapshot_25`) | `tardis_snapshot25` | `usdc_coinbase_l2` |
| DEX on-chain pool events | `dex_pool_onchain` | `curve_3pool`, `uniswap_usdc_usdt_005` |
| Mint/burn Transfer events | `onchain_event` | `usdc_mint_burn`, `usdt_mint_burn` |

### Automatic Tier A → B downgrade conditions

The manifest pipeline (`src/stressnet/features/manifest.py`) computes diagnostics and
applies the following downgrade rules **before** the claim gate reads provenance:

```
coverage_pct  < 0.50   →  downgrade  ("incomplete coverage")
gap_rate      > 0.01   →  downgrade  ("sequence gaps")
resync_count  > 0      →  warn; if cumulative resync span > 300 s → downgrade
clock_skew_ms > 5000   →  downgrade  ("clock unreliable")
```

Downgraded nodes keep their `tier_nominal` value (A) but receive
`tier_actual = B` plus a `tier_downgrade_reason` string.

---

## Tier B — Context-grade

A source qualifies for Tier B if it provides **price and/or quantity** information at
≥ 1-minute resolution but does not meet one or more Tier-A criteria.

| Source type | `source_type` tag | Typical gap |
|---|---|---|
| Binance Vision klines (1 m OHLCV) | `binance_vision_klines` | No bid/ask depth |
| Binance Vision bookTicker (BBO) | `binance_vision_bbo` | Single best level, no depth |
| Binance Vision aggTrades | `binance_vision_trades` | No order book at all |
| Coinbase REST candles | `coinbase_candles` | No real-time L2 |
| Kraken REST OHLC | `kraken_ohlc` | No real-time L2 |
| The Graph Uniswap subgraph | `thegraph_subgraph` | 5–15 min block delay |
| Coin Metrics exchange flows | `coin_metrics_flows` | Pre-aggregated netflows |
| Dune decoded on-chain | `dune_onchain` | Query latency; no tick timestamps |
| Etherscan ERC-20 transfer logs | `etherscan_transfers` | Label confidence ~80 % |

### Permitted Tier B claims

- Approximate stress timing (±5 min resolution)
- Price context and spread proxy
- Exchange flow co-movement
- Auxiliary covariates in VAR / regression
- Corroborating evidence for Tier-A directional claims

---

## Tier C — Taxonomy-grade

Tier C covers sources that **cannot** support timing or directional claims.  They remain
in the pipeline for event classification and narrative context only.

| Trigger | Examples |
|---|---|
| `coverage_pct < 0.20` | Sporadic snapshots |
| Daily OHLCV or coarser | CoinGecko daily candles |
| Delisted / defunct source with gaps | UST Binance archive with missing days |
| Interpolated values between sparse anchors | Pool state between two far-apart blocks |
| Address labels with < 50 % coverage | Exchange-flow label set |

### fixture_non_empirical (special tag)

Rows tagged `tier_actual = fixture_non_empirical` are synthetic data generated for
testing only.  They are **never** paper-claimable.  The strict-mode claim gate
(`scripts/00c_claim_gate.py --strict`) will exit non-zero if any fixture row appears
in the final panel.

---

## Edge claim derivation

No edge may support a stronger claim than the weaker endpoint tier.

```
tier_i = A, tier_j = A  →  claim_level = A_A_directional_microstructure
tier_i = A, tier_j = B  →  claim_level = A_B_suggestive_directional
tier_i = B, tier_j = B  →  claim_level = B_B_context_only
Any endpoint = C         →  claim_level = C_taxonomy_only
Any endpoint = fixture   →  not claimable
```

Claim sentences written to result tables (`claim_language` column):
- `A_A`: "directional microstructure transmission"
- `A_B`: "suggestive timing evidence"
- `B_B`: "contextual co-movement"
- `C`:   "event co-occurrence context only"

---

## Review checklist before paper submission

- [ ] Every node has a `tier_actual` (not just `tier_nominal`) populated in the manifest.
- [ ] At least one `A_A_directional_microstructure` edge exists for the headline event.
- [ ] No `fixture_non_empirical` rows in the paper panel.
- [ ] All Tier-A CEX nodes have `is_executable_bookwalk = True` or a documented
      fallback explaining why bookwalk VWAP is unavailable.
- [ ] All manifest diagnostics (`coverage_pct`, `gap_rate`, `resync_count`,
      `clock_skew_ms`) are populated for Tier-A nodes.
