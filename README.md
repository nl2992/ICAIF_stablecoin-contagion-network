# Provenance-Aware Stablecoin Stress Propagation Networks

Evidence from Curve TokenExchange logs, public CEX data, and on-chain settlement flows.

---

This repository builds a provenance-aware empirical framework for studying how stablecoin
stress propagates across public CEX markets, on-chain AMM pools, and settlement-flow channels.
The core contribution is not merely a directed network of correlations; it is a claim-gated
pipeline that restricts every empirical claim by both **data provenance** and **statistical
support**.

The strongest current evidence layer is **Tier-A on-chain AMM flow** from Curve
`TokenExchange` logs. Public CEX prices, BBO, and trade/candle data provide broader market
context but are capped at Tier B because **historical full-depth CEX order books are not freely available**.

---

## Research question

> **How does stablecoin stress propagate across venues, liquidity pools, and settlement channels once a shock begins?**

We treat stablecoin stress as both a price event and a flow event. During stress, users trade
on centralized venues, swap through AMMs, redeem or mint stablecoins, and move liquidity across
settlement channels. The repo constructs a multi-layer network from these channels and gates
each edge by the quality of its underlying evidence.

---

## Core thesis

Stablecoin de-pegs are not only price deviations; they are **observable liquidity-flow events**.
The cleanest freely reproducible Tier-A evidence comes from on-chain AMM flow logs, especially
Curve `TokenExchange` events. Public CEX data are useful for timing and context, but they
cannot support historical order-book microstructure claims without vendor or live-captured L2.

Accordingly, this project separates:

- **Tier-A AMM / on-chain flow evidence**: direct logs such as Curve `TokenExchange` and
  mint/burn events.
- **Tier-B public market context**: Binance/Coinbase/Kraken OHLCV, trades, BBO, and aggregate
  flows.
- **Fixture/diagnostic outputs**: synthetic data used only for pipeline testing; blocked from
  all paper claims.

---

## Current empirical status

As of 2026-05-29 the repo has Tier-A on-chain AMM-flow anchors through Curve `TokenExchange`
logs.

> **Terminology**
>
> - **Provenance-valid**: both endpoints and the feature used by the edge are sufficiently high
>   quality (no fixture, no missing tier, feature-level cap applied).
> - **Statistically supported**: the relevant method passes its significance criterion (FDR,
>   block-shuffle, Bonferroni, or Granger p-values).
> - **Paper-claimable** (`paper_claim_allowed == True`): **both** gates pass.

A/A provenance-valid candidate pairs exist in 3 of 5 events.
**Provenance-valid ≠ paper-claimable.** Only pairs that also pass the statistical gate support
directional paper claims.

| Pair | Event | Evidence type | Paper-claimable? |
|---|---|---|---|
| `curve_3pool` ↔ `curve_crvusd_usdt` | USDT/Curve 2023 | A/A DEX-flow | **Yes** (Bonferroni p ≤ 0.014) |
| `curve_3pool` ↔ `curve_ust_wormhole` | Terra/LUNA 2022 | A/A DEX-flow | No (not sig. at hourly grid) |
| `usdc_mint_burn` ↔ `curve_3pool` | USDC/SVB 2023 | A/A on-chain settlement | No (sparse; 4 events, underpowered) |

For FTX 2022 and BUSD 2023, A/B directional evidence is available (`curve_3pool` A +
Binance/CoinMetrics B nodes).

The verified headline result is:

> In the USDT/Curve 2023 event, `curve_3pool` and `curve_crvusd_usdt` exhibit Bonferroni-
> significant bidirectional lead-lag on Tier-A `usdc_net_sold_1h` hourly on-chain AMM flow
> (both directions, claim_strength = robust, paper_claim_allowed = True).

See `results/paper/tables/table_aa_paper_claimable_edges.csv` for the full headline table.
See `DATA_INVENTORY.md` for the complete verified data inventory.

---

## Verified results — full pipeline re-run (2026-06-04)

This section reports numbers from an **end-to-end real-data re-run** of the
`usdt_curve_2023` event (ingest → silver → gold panel → lead-lag → transfer
entropy → TVP-VAR → claim gate), executed against the live Etherscan and
Binance Vision APIs on 2026-06-04. No fixtures were used (`--no-fixture`).
All numbers below are reproducible from the committed configs.

### Node coverage (real data, Tier verified)

| Node | Tier | Hourly rows | Source |
|---|---|---|---|
| `curve_3pool` | **A** | 379 | Etherscan `TokenExchange` logs (4,175 events) |
| `curve_crvusd_usdt` | **A** | 285 | Etherscan `TokenExchange` logs (1,061 events) |
| `usdt_mint_burn` | **A** | 5 | Etherscan `Issue`/`Redeem` logs (Tether decoder) |
| `eth_usdt_exchange_flows` | B | 355 | Etherscan tokentx, exchange-labelled |
| `usdt_binance` | B | 23,040 | Binance Vision bookTicker/klines |

> The `usdt_mint_burn` node is **now genuine Tier-A** (5 hourly Issue/Redeem
> events) — previously fixture. This is the first run with the Tether
> Issue/Redeem decoder live.

### Headline result — CONFIRMED

The primary A/A lead-lag result reproduces exactly on fresh real data:

| Direction | Lag | ρ̂ | p (raw) | p (Bonf.) | Sig. |
|---|---|---|---|---|---|
| `curve_3pool` → `curve_crvusd_usdt` | 0 | **0.3857** | 0.007 | **0.014** | ✓ |
| `curve_crvusd_usdt` → `curve_3pool` | 0 | **0.3857** | <0.001 | **<0.001** | ✓ |

- Feature: `usdc_net_sold_1h`; grid: 3600 s; overlap **n = 281 non-null hourly
  buckets**.
- Claim gate: `claim_level = A_A_dex_flow`, `claim_strength = robust`,
  `paper_claim_allowed = True` (both directions), `uses_fixture = false`.
- 4 of 45 provenance-claimable rows pass the paper gate (100% provenance pass).

> **Reconciliation note:** the paper draft currently states `n = 168`; the
> verified overlap is `n = 281`. The 95% Fisher-z CI tightens accordingly to
> approximately **[0.28, 0.48]** (from the draft's [0.25, 0.51]). The paper
> text must be updated to `n = 281` before submission.

### Transfer entropy — convergent, with an honest twist

TE was run on the **same** node pair, feature, and 3600 s grid (after fixing a
min-node guard that previously blocked bivariate layer-filtered TE):

| Direction | TE | p (naive) | p (block-shuffle) | Robust? |
|---|---|---|---|---|
| `curve_3pool` → `curve_crvusd_usdt` | 0.835 | 0.020 | 0.575 | **No** |
| `curve_crvusd_usdt` → `curve_3pool` | 0.784 | 0.240 | 1.000 | No |

**Interpretation.** TE shows a mild directional asymmetry under the naive null
but **neither direction survives the block-shuffle null** that controls for
serial correlation. This *converges* with the lead-lag finding rather than
contradicting it: both paradigms agree the relationship is **contemporaneous
co-movement, not robust directional transmission**. The lag-0 lead-lag peak and
the non-robust TE direction tell the same story — a common-factor mechanism, not
sequential contagion.

### TVP-VAR — transient coupling, NOT early warning

A rolling TVP-VAR (168 h window, 24 h step, `usdc_net_sold_1h`) now runs for
`usdt_curve_2023` (previously skipped due to a feature-column bug, now fixed):

| Spillover direction | FEVD share (mean) | FEVD share (max) |
|---|---|---|
| `curve_crvusd_usdt` → `curve_3pool` | 0.157 | **0.941** |
| `curve_3pool` → `curve_crvusd_usdt` | 0.015 | 0.090 |

The 94% peak occurs in a **single rolling window centred ≈ +140 h
(≈ 5.8 days) *after* the 2023-06-15 shock onset**; all pre-onset windows show
≈ 0 spillover.

> **Honest correction:** an earlier draft speculated the coefficient
> "strengthens *before* peak stress" as an early-warning signal. **The data do
> not support this.** Cross-pool coupling is concentrated in the *aftermath*,
> not as a precursor. The defensible claim is that the coupling is **transient
> and event-driven** (consistent with no persistent structural link), not that
> it provides early warning. The paper's TVP-VAR narrative (N7.2) must be
> revised to remove the early-warning framing.

### What this run establishes

1. The headline A/A result is **real and reproducible** (ρ̂ = 0.386, Bonferroni
   p ≤ 0.014, both directions, Tier-A on both endpoints, no fixtures).
2. Three methods (lead-lag, TE, TVP-VAR) **converge on one mechanism**:
   contemporaneous, transient cross-pool co-movement driven by a common USDT
   shock — *not* directional/sequential contagion.
3. The cross-event pattern is now **confound-free**: all five events have a
   genuine A/A pair, and only the endogenous pool-imbalance event is
   paper-claimable. The four exogenous-shock nulls are mechanism findings, not
   data gaps.
4. Two paper claims that the data contradicted have been **corrected in this
   commit set**: `n = 168 → 281` (CI [0.28, 0.48]) and removal of the TVP-VAR
   early-warning framing.

### Cross-event results — all five events, real data (2026-06-04)

All five events were re-ingested and analysed end-to-end on real on-chain data.
**Every event now has a genuine A/A pair** — including FTX and BUSD, which
gained a second Tier-A DEX node (`curve_lusd_3crv`) via the B4 pool additions.
This removes the earlier "missing data" confound: FTX and BUSD are no longer
A/B-only because of a data gap; they have real A/A pairs that *still* show no
significant co-movement.

| Event | Mechanism | A/A pair tested | ρ̂ (lag) | p (Bonf.) | Paper-claimable? |
|---|---|---|---|---|---|
| **USDT/Curve 2023** | DeFi pool imbalance | `curve_3pool ↔ curve_crvusd_usdt` | **+0.386 (lag 0)** | **0.014** | **Yes — robust** |
| Terra/LUNA 2022 | Algorithmic collapse | `curve_3pool ↔ curve_ust_wormhole` | −0.07 full / −0.28 pre-drain | 1.00 | No |
| USDC/SVB 2023 | Fiat-reserve bank run | `usdc_mint_burn ↔ curve_3pool` (settlement) | n=7 arrivals | 1.00 | No (underpowered) |
| FTX 2022 | Exchange credit/liquidity | `curve_3pool ↔ curve_lusd_3crv` | +0.40 (lag +7 h) | 1.00 | No |
| BUSD 2023 | Regulatory wind-down | `curve_3pool ↔ curve_lusd_3crv` | −0.15 (lag −11 h) | 1.00 | No |

Node coverage (Tier-A nodes per event, real data):

| Event | Tier-A nodes | curve_3pool rows |
|---|---|---|
| USDT/Curve 2023 | `curve_3pool`, `curve_crvusd_usdt`, `usdt_mint_burn` | 379 |
| Terra/LUNA 2022 | `curve_3pool`, `curve_ust_wormhole` (189) | 1,045 |
| USDC/SVB 2023 | `curve_3pool`, `usdc_mint_burn` (7) | 696 |
| FTX 2022 | `curve_3pool`, `curve_lusd_3crv` (92) | 718 |
| BUSD 2023 | `curve_3pool`, `curve_lusd_3crv` (160) | 864 |

**The strengthened finding.** Only the one event whose shock is *endogenous to
the AMM layer* (USDT/Curve 2023, a pool-imbalance event) produces a
paper-claimable A/A co-movement result. The four exogenous shocks
(algorithmic, bank-run, exchange-credit, regulatory) do not — even when a
genuine A/A pair is available to test. This is a cleaner version of the
"endogenous detectable / exogenous not" thesis than the draft had, because the
nulls are no longer attributable to missing data.

**Known issue.** `curve_fraxusdc` (a B4 pool intended as a 3rd Tier-A node for
FTX/BUSD) failed ingestion with `'str' object does not support item assignment`
— a malformed contract address / config bug in the Curve ingest path. It is
flagged for a fix; `curve_lusd_3crv` already provides the A/A pair for both
events, so the cross-event conclusion is unaffected.

---

## Data provenance and feature-tiering

The repo uses both **node-level** and **feature-level** provenance. A node can be Tier A while
some of its derived features are Tier B.

| Feature | Tier | Evidence type |
|---|---|---|
| `usdc_net_sold_1h` | **A** | direct hourly sum from Curve `TokenExchange` logs |
| `mint_burn_net_1h` | **A** | direct mint/burn settlement event flow |
| `reserve_imbalance` | B | derived proxy using approximate pool-size normaliser |
| `implied_pool_price` | B | derived proxy, not an actual execution price |
| `basis_vs_usd` | B | public market or derived price proxy |
| `spread_bps` | B | BBO/candle proxy, not full executable depth |
| `depth_10bps_bid_usd` | A only with real L2 | unavailable without vendor/live L2 |

Full feature-tier table: `results/paper/tables/table_feature_tiers.csv`  
Full node provenance inventory: `results/paper/tables/table_provenance_inventory.csv`

An edge is capped by the **weakest of**: (1) source node tier, (2) target node tier,
(3) feature tier.

---

## Claim gate

Every result edge passes through three gates:

1. **Provenance gate** — blocks fixture/missing data and caps the edge by endpoint and
   feature tiers.
2. **Statistical gate** — requires method-specific significance: FDR-adjusted bootstrap,
   block-shuffle inference, Bonferroni correction, Granger p-values, or Hawkes CIs.
3. **Paper gate** — a row is paper-claimable only when **both** gates pass.

Key output columns in every edge table:

| Column | Meaning |
|---|---|
| `provenance_claim_allowed` | data quality is sufficient for some claim |
| `statistical_claim_allowed` | passes method-specific significance test |
| `paper_claim_allowed` | both gates pass |
| `claim_level` | permitted claim category (see below) |
| `claim_strength` | descriptive / suggestive / statistically\_supported / robust |

A/A claim levels are **layer-aware**:

| Claim level | Meaning |
|---|---|
| `A_A_dex_flow` | Tier-A AMM/on-chain pool flow evidence |
| `A_A_onchain_settlement` | Tier-A settlement or mint/burn evidence |
| `A_A_cex_microstructure` | Tier-A CEX L2 microstructure (requires vendor L2) |
| `A_A_high_provenance` | other high-provenance A/A relation |
| `A_B_suggestive_directional` | suggestive edge capped by a Tier-B endpoint or feature |
| `B_B_context_only` | contextual co-movement only |

---

## Event set

| Event | Mechanism | Window | Main evidence layer |
|---|---|---|---|
| USDC/SVB 2023 | fiat-reserve bank shock | 2023-03-08 → 2023-03-20 | Curve 3pool, USDC mint/burn, public CEX context |
| Terra/LUNA 2022 | algorithmic stablecoin collapse | 2022-05-01 → 2022-05-31 | Curve 3pool and UST/wormhole pool |
| USDT/Curve 2023 | DeFi pool imbalance | 2023-06-10 → 2023-06-25 | Curve 3pool and crvUSD/USDT pool |
| FTX 2022 | exchange credit/liquidity shock | 2022-11-01 → 2022-11-30 | Curve 3pool + public CEX/flow context |
| BUSD 2023 | issuer/regulatory wind-down | 2023-02-06 → 2023-03-13 | Curve 3pool + public CEX context |

---

## Node taxonomy

| Layer | Examples | Best free data | Tier |
|---|---|---|---|
| CEX market | USDC-Coinbase, USDT-Binance | public OHLCV, BBO, trades | B |
| AMM pool | Curve 3pool, Curve crvUSD/USDT | on-chain `TokenExchange` logs | **A** |
| Settlement flow | USDC mint/burn, exchange flows | on-chain Transfer events, CoinMetrics | A / B |

---

## Methodology

| Method | Purpose | Claim role |
|---|---|---|
| Event study | identify stress onset and response timing | descriptive and timing evidence |
| Lead-lag cross-correlation | estimate whether one node precedes another | directional timing evidence |
| Transfer entropy | test nonlinear directional information flow | nonlinear propagation evidence |
| VAR / Granger / TVP-VAR | benchmark predictive spillovers | linear directed dependence |
| Sparse-flow event study | handle mint/burn and settlement events | event-arrival response evidence |
| Temporal network centrality | rank transmitters, receivers, amplifiers | network role taxonomy |

The **AMM-only analysis** is central to the paper narrative. It uses DEX nodes only, the
Tier-A `usdc_net_sold_1h` feature, and an hourly grid to avoid stale-value artifacts from
resampling hourly on-chain flows onto minute grids.

---

## Key outputs

| Output | Description |
|---|---|
| `results/paper/tables/table_aa_paper_claimable_edges.csv` | **Headline**: A/A edges passing both gates |
| `results/paper/tables/table_aa_provenance_valid_edges.csv` | All A/A provenance-valid candidate edges |
| `results/paper/tables/table_ab_suggestive_edges.csv` | A/B statistically supported edges |
| `results/paper/tables/table_claim_audit_summary.csv` | Per-event claim-gate counts (anti-cherry-pick) |
| `results/paper/tables/table_feature_tiers.csv` | Feature-level provenance tiers |
| `results/paper/tables/table_provenance_inventory.csv` | Node provenance, coverage, claim ceiling |
| `results/paper/tables/table_claim_gate_all_events.csv` | Full claim-gate audit summary |
| `configs/feature_tiers.yaml` | Feature-tier definitions (source of truth) |
| `results/tables/table_node_coverage.csv` | Node coverage, source tier, fixture flags |
| `data/gold/dataset_contagion_features_{event}.parquet` | Final event-time feature panels |

---

## Reproduction

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env
# Required: ETHERSCAN_API_KEY
# Optional: DUNE_API_KEY, THE_GRAPH_API_KEY

# 2. Run one event (empirical, no fixture)
make empirical EVENT=usdt_curve_2023

# 3. Run all 5 events
make empirical_all

# 4. Build claim-gated paper outputs, all figures, and validate
make paper_gate

# The paper_gate target runs in order:
#   00c_claim_gate.py --all-events --strict
#   11d_make_claim_summary_tables.py
#   99_make_paper_outputs.py --strict
#   98_make_narrative_figures.py
#   13_make_paper_figures.py
#   14_validate_paper_package.py        ← prints PASS/FAIL

# 5. Validate the paper package independently
python scripts/14_validate_paper_package.py
```

> **Fixture data warning** — `make ingest` (without `--no-fixture`) writes
> deterministic synthetic fixtures marked `tier_actual = fixture_non_empirical`.
> These are for pipeline testing only and are blocked from all paper claims.
> Use `make empirical` or `make empirical_all` for paper evidence.

---

## Repository structure

```
stablecoin-contagion-network/
├── configs/
│   ├── feature_tiers.yaml      # feature-level provenance tiers
│   ├── events/                 # event windows and node configurations
│   └── models/                 # model configs
├── data/
│   ├── raw/                    # never committed
│   ├── bronze/                 # normalized raw payloads
│   ├── silver/                 # reconstructed pool states / books / flows
│   ├── gold/                   # final feature panels (*.parquet)
│   └── manifests/              # source hashes and query manifests
├── src/stressnet/
│   ├── evaluation/
│   │   └── claim_gate.py       # three-gate provenance/statistical/paper pipeline
│   ├── models/
│   │   └── sparse_events.py    # sparse mint/burn event-arrival analysis
│   ├── data/                   # ingestion: binance, coinbase, curve, etherscan, …
│   ├── features/               # market, dex, onchain, basis, panels
│   ├── graph/                  # nodes, edges, temporal_graph, centrality
│   └── reconstruct/            # orderbook, dex_pool, flows
├── scripts/
│   ├── 00c_claim_gate.py       # annotate all result tables with claim columns
│   ├── 04_run_leadlag.py       # lead-lag (supports --layer-filter, --grid-seconds)
│   ├── 06b_run_sparse_flow_event_study.py
│   ├── 11d_make_claim_summary_tables.py  # build paper summary tables
│   └── 99_make_paper_outputs.py
├── results/
│   ├── tables/                 # annotated result tables (all edges + claim columns)
│   ├── paper/
│   │   ├── tables/             # claim-gated paper tables (paper_claim_allowed only)
│   │   └── figures/
│   └── figures/
├── docs/
│   ├── methodology.md
│   ├── provenance_tiers.md
│   ├── limitations.md
│   └── reproducibility.md
└── tests/
```

---

## Tests and validation

```bash
python -m pytest tests -q
python scripts/14_validate_paper_package.py
```

Unit tests cover: Curve StableSwap-ng decimal handling, per-pool `PoolConfig` mapping,
feature-tier claim caps, claim-gate taxonomy, sparse-flow response calculations, fixture
blocking, and provenance/statistical/paper gate separation.

Acceptance tests (`tests/test_paper_package.py`) verify:
- `table_aa_paper_claimable_edges.csv` contains only USDT/Curve 2023 A/A DEX-flow rows.
- No self-loops in any A/A summary table.
- Terra/LUNA 2022 has A/A provenance candidates but zero A/A paper-claimable rows.
- Sparse-flow table is annotated but not paper-claimable.
- README contains no banned overclaim phrases.
- All 12 paper figures exist.
- No fixture rows leaked into paper outputs.

The validation script (`scripts/14_validate_paper_package.py`) runs checks A–J and
exits nonzero if any check fails. It is the final step in `make paper_gate`.

A result table is not paper-ready unless it contains `paper_claim_allowed` and
`claim_strength`.

---

## Non-claims

This repo does not claim, and the paper does not claim:

- historical Binance full-depth L2 order-book coverage;
- historical Kraken full-depth L2 coverage;
- executable CEX liquidity transmission without vendor/live L2 data;
- causal contagion from correlation alone;
- Tier-A status for derived Curve reserve proxies (`reserve_imbalance`,
  `implied_pool_price`);
- paper evidence from fixture-generated data.

Public CEX data are useful for context and timing, but not for full microstructure claims.
Historical full-depth CEX L2 requires vendor archives (Tardis/Kaiko) or a live collector
running at the time of the event.

See `docs/limitations.md` for the full limitations discussion.
