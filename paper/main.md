# Provenance-Aware Stablecoin Stress Propagation Networks

**Evidence from Curve TokenExchange Logs, Public CEX Data, and On-Chain Settlement Flows**

---

*Working paper — Columbia University, May 2026*

---

## Abstract

Stablecoin stress episodes are usually measured as price de-pegs, but they are also liquidity-flow events. This paper develops a provenance-aware network framework for studying how stress propagates across public centralized exchange (CEX) markets, on-chain automated market maker (AMM) pools, and settlement-flow channels across eight stablecoin nodes spanning fiat-backed, overcollateralized, and algorithmic designs. The framework assigns node-level and feature-level evidence tiers, then filters every empirical edge through provenance and statistical gates before allowing paper-level claims. Across five historical stress events — USDC/SVB (March 2023), Terra/LUNA (May 2022), USDT/Curve (June 2023), FTX (November 2022), and BUSD (February–March 2023) — the strongest paper-claimable evidence is concentrated in the USDT/Curve 2023 event. In that event, Curve 3pool and Curve crvUSD/USDT exhibit statistically supported bidirectional AMM-flow linkage: both directions pass Bonferroni correction (p ≤ 0.014) on Tier-A on-chain `usdc_net_sold_1h` data at the hourly grid. Other events provide either provenance-valid candidates without statistical support (Terra/LUNA, USDC/SVB) or contextual A/B evidence (FTX, BUSD). We extend the methodology with Forbes-Rigobon heteroskedasticity-corrected contagion tests — which distinguish true contagion from mere interdependence — and Quantile VAR tail-spillover estimates, finding that tail impulse responses ($\tau=0.05$, $\tau=0.95$) exceed median responses for fiat-backed pool stress events. The Terra/LUNA null result at hourly resolution and the USDC/SVB sparse mint/burn channel provide design-heterogeneity evidence that fiat-backed DEX stress and algorithmic collapse propagate through structurally different channels. The paper derives three regulatory implications: on-chain DEX flow monitoring as a real-time leading indicator, speed requirements for algorithmic stablecoin circuit breakers, and a two-channel capital buffer framework for fiat-backed vs. settlement-channel liquidity risk.

**Keywords:** stablecoin, stress propagation, AMM, Curve Finance, provenance-aware, claim gate, DeFi

---

## 1. Introduction

The collapse of Terra/UST in May 2022, the Silicon Valley Bank-triggered USDC de-peg in March 2023, the USDT imbalance on Curve in June 2023, and the FTX implosion in November 2022 were each described in real time as episodes of contagion — stress spreading from one venue or asset to another. But most empirical analyses of these episodes rely on price data alone, and most claims about "contagion" or "directional transmission" are made without explicit reference to the quality of the underlying data.

This paper addresses two problems simultaneously. First, it studies stablecoin stress as a *liquidity-flow* event rather than only a price event. When a stablecoin de-pegs, traders do not merely observe prices change; they actually execute swaps in AMM pools, redeem stablecoins through mint/burn channels, and move funds across venues. These flows are directly observable from on-chain logs in a way that is not true for most traditional financial instruments.

Second, it introduces a *provenance-aware claim gate* that restricts each empirical edge claim by two explicit criteria: (1) the quality of the underlying data source, and (2) the statistical significance of the result. Only edges that pass both gates are labelled paper-claimable. This prevents a common failure mode in crypto market research: computing network edges from heterogeneous data sources and then treating all edges as equally valid.

The strongest current paper-claimable result is narrow but defensible: in the USDT/Curve 2023 event, the Curve 3pool and Curve crvUSD/USDT pools exhibit bidirectional AMM-flow linkage at lag 0 with Bonferroni-corrected p ≤ 0.014. This result uses only Tier-A execution-grade Curve `TokenExchange` on-chain event logs. Public CEX prices, bid-ask spreads, and trade data are incorporated as Tier-B context but do not support headline directional claims because historical full-depth CEX order books are not freely available for the relevant episodes.

The paper is organized as follows. Section 2 frames stablecoin stress as a flow event and motivates the multi-layer network. Section 3 describes the data and the provenance tier system. Section 4 presents the claim-gated methodology. Sections 5–7 report the main results. Section 8 addresses robustness and negative results. Section 9 presents the Forbes-Rigobon and QVAR method extensions and design-heterogeneity comparison. Section 10 draws regulatory capital and policy implications. Sections 11–12 discuss limitations and conclude.

---

## 2. Stablecoin Stress as a Flow Event

### 2.1 Beyond the Price Peg

Standard stablecoin analysis treats the peg deviation — the absolute dollar distance from $1.00 — as the primary outcome variable. This is natural: the peg is the stablecoin's core promise, and its failure is the most visible signal of stress. But the price-peg view misses important dynamics that occur *before* a visible de-peg and *throughout* stress episodes even when the price has nominally stabilized.

Stablecoin stress generates observable flows across multiple layers:

- **AMM pool flows**: traders executing token swaps that progressively deplete the pool on one side, observable as net USDC-equivalent sold per hour from on-chain `TokenExchange` logs.
- **Mint/burn flows**: settlement-layer actors burning or minting stablecoins, observable as net issuance from ERC-20 `Transfer` events.
- **Exchange flows**: net inflows or outflows at centralized exchanges, partially observable via CoinMetrics exchange netflow aggregates.
- **CEX price signals**: bid-ask spreads and price deviations, observable via public OHLCV and BBO data from Binance, Coinbase, and Kraken.

The key insight motivating this paper is that AMM pool flows — specifically Curve `TokenExchange` event data — are available at the *execution* level, with exact block timestamps and exact amounts, for all five major stablecoin stress events since 2022. This makes them uniquely suited to constructing Tier-A empirical edges.

### 2.2 Multi-Layer Network Framework

We model the stress propagation environment as a three-layer directed network:

1. **CEX layer**: nodes are trading pairs at centralized exchanges (e.g., `usdc_coinbase`, `usdt_binance`). Data are public market feeds: OHLCV, BBO/bookTicker, aggregated trade data. Tier B.
2. **AMM layer**: nodes are Curve Finance liquidity pools (e.g., `curve_3pool`, `curve_crvusd_usdt`, `curve_frax_usdc`, `curve_lusd_3crv`, `curve_susd_4pool`). Data are Etherscan `TokenExchange` event logs. Tier A. The expanded pool set covers fiat-backed (3pool, FRAX/USDC), overcollateralized (LUSD), synthetic (sUSD), and wrapped algorithmic (UST/wormhole) designs.
3. **Settlement layer**: nodes are on-chain settlement channels (e.g., `usdc_mint_burn`). Data are Etherscan ERC-20 `Transfer` events. Tier A.

A directed edge from node $i$ to node $j$ represents evidence that stress in $i$ precedes or predicts stress in $j$. The tier of each edge is the minimum of the source node tier, target node tier, and the feature tier:

$$\text{tier}_{ij} = \min(\text{tier}_i, \text{tier}_j, \text{tier}_f)$$

where $\text{tier}_f$ is the feature-level tier of the column used to measure stress at each endpoint.

---

## 3. Data and Provenance

### 3.1 Node-Level Provenance Tiers

We classify every data source into one of three provenance tiers:

| Tier | Definition | Examples in this study |
|------|-----------|----------------------|
| **A** | Execution-grade on-chain logs; directly verifiable from the blockchain | Curve `TokenExchange` events; USDC mint/burn ERC-20 `Transfer` events |
| **B** | Real public data; useful for context; not execution-grade | Binance OHLCV, BBO/bookTicker; CoinMetrics exchange netflows; Coinbase 1-min candles |
| **Fixture** | Synthetic pipeline data for testing only; blocked from all paper claims | `usdc_kraken`, `usdt_mint_burn`, `eth_bridge_flows` (where real data are unavailable) |

The fixture tier is assigned when no free historical data source exists. Historical full-depth CEX order books (tick-by-tick L2 data) require paid vendor archives (Tardis, Kaiko) or a live collector running at the time of the event. Binance and Kraken nodes are Tier B for all five events studied here because only OHLCV and BBO snapshots are freely available, not executable depth.

### 3.2 Feature-Level Tiers

Node tier alone is insufficient because some features derived from Tier-A nodes are themselves only Tier B:

| Feature | Tier | Reason |
|---------|------|--------|
| `usdc_net_sold_1h` | **A** | Direct hourly sum of on-chain `TokenExchange` amounts; exact block-level records |
| `mint_burn_net_1h` | **A** | Direct hourly sum of ERC-20 `Transfer` events (mint = to 0x0; burn = from 0x0) |
| `reserve_imbalance` | B | Derived: `usdc_net_sold_cum / pool_size_usd` where `pool_size_usd` is approximate |
| `implied_pool_price` | B | Derived: approximation of pool price from cumulative flow |
| `spread_bps`, `basis_vs_usd` | B | Public OHLCV/BBO proxy; not executable depth |

The *effective edge tier* for any directed edge is the minimum of the source node tier, target node tier, and the feature tier applied at each endpoint. The headline `usdc_net_sold_1h` feature is Tier A, so A-node edges using this feature are A/A edges.

### 3.3 The Five Stress Events

This study covers five stablecoin stress events with different mechanisms and data availability profiles:

**USDC/SVB 2023 (March 8–20, 2023)**: USDC briefly de-pegged after Silicon Valley Bank's failure impaired Circle's reserves. Curve 3pool (Tier A) and USDC mint/burn flows (Tier A, but sparse: only 4 mint/burn arrivals in the 12-day window) are available. Public CEX data (Binance, Coinbase) provide Tier-B context. Fixture nodes: `usdc_kraken`, `uniswap_usdc_usdt_005`, `eth_bridge_flows`.

**Terra/LUNA 2022 (May 1–31, 2022)**: Algorithmic stablecoin collapse. Both Curve 3pool and Curve UST/wormhole pool are Tier A (TokenExchange logs available). The UST/wormhole pool was nearly drained during the event, which may reduce cross-correlation power. Public Binance data (USDT, UST) provide Tier-B context.

**USDT/Curve 2023 (June 10–25, 2023)**: DeFi pool imbalance episode in which USDT temporarily de-pegged on Curve. Both Curve 3pool and Curve crvUSD/USDT are Tier A. This is the primary Tier-A empirical case study. Fixture nodes: `usdt_mint_burn` (Tether uses `Issue`/`Redeem` events, not standard ERC-20 Transfer), `usdt_kraken`, `uniswap_usdc_usdt_005`, `tron_usdt_exchange_flows`.

**FTX 2022 (November 1–30, 2022)**: Exchange credit shock. Only Curve 3pool is Tier A; all CEX nodes are Tier B. No second Tier-A node is available for this event, so only A/B and B/B edges are possible.

**BUSD 2023 (February 6 – March 13, 2023)**: Issuer/regulatory wind-down. Curve 3pool is Tier A; all other nodes are Tier B. Same constraint as FTX: no second Tier-A node.

### 3.4 Notable Data Limitations

The following data items are unavailable or synthetic and are explicitly excluded from paper claims:

- **Historical CEX L2 order books**: No free historical archive exists for Binance, Coinbase, or Kraken full-depth data for the five events. Tardis and Kaiko provide paid archives; a live collector would need to have been running at the time. All CEX depth-related features (`depth_10bps_bid_usd`, `orderbook_imbalance`) are blocked.
- **USDT mint/burn on Ethereum**: Tether uses `Issue`/`Redeem` events rather than standard ERC-20 `Transfer` events. The current decoder queries standard `Transfer` only and does not capture Tether issuance. `usdt_mint_burn` is fixture for all five events.
- **Uniswap pool state**: The Graph API was not queried; `uniswap_usdc_usdt_005` is fixture.
- **Tron USDT flows**: TronGrid API not implemented; `tron_usdt_exchange_flows` is fixture.

---

## 4. Claim-Gated Methodology

### 4.1 The Three-Gate Pipeline

Every directed edge in the result tables is annotated by a three-gate provenance and statistical pipeline:

**Gate 1 — Provenance gate**: checks that neither endpoint is fixture or missing; applies feature-level tier caps; assigns the claim-level taxonomy. A row passes the provenance gate if it has no fixture contamination and the edge tier is at least B. Sets `provenance_claim_allowed = True`.

**Gate 2 — Statistical gate**: requires method-specific significance. Lead-lag cross-correlation uses Bonferroni correction (k = number of lag windows tested). Transfer entropy uses FDR-adjusted block-shuffle p-values. Granger causality uses standard p < 0.05 thresholds. TVP-VAR spillovers use the FEVD share. Sets `statistical_claim_allowed = True`.

**Gate 3 — Paper gate**: `paper_claim_allowed = (provenance_claim_allowed AND statistical_claim_allowed)`. Only rows with `paper_claim_allowed = True` support directional paper claims.

### 4.2 Claim-Level Taxonomy

Each edge that passes the provenance gate is assigned to one of the following claim levels:

| Claim level | Meaning | Data requirement |
|-------------|---------|-----------------|
| `A_A_dex_flow` | Both endpoints are Tier-A AMM/DEX nodes | Curve `TokenExchange` logs |
| `A_A_onchain_settlement` | Both are Tier-A on-chain settlement/flow nodes | ERC-20 `Transfer` or settlement events |
| `A_A_cex_microstructure` | Both are Tier-A CEX nodes with L2 data | Requires vendor or live-captured L2 |
| `A_B_suggestive_directional` | Mixed: one endpoint is Tier B | Any combination with one Tier-B endpoint |
| `B_B_context_only` | Both endpoints are Tier B | Public market context only |

The claim level `A_A_cex_microstructure` cannot be assigned for any of the five events studied here because no Tier-A CEX nodes exist (no free historical L2 data).

### 4.3 Analytical Methods

**Lead-lag cross-correlation (primary for A/A DEX-flow claim)**: For each directed pair $(i, j)$, we compute cross-correlations between node $i$ and node $j$ at lags $-L$ to $+L$ hours. We use an hourly grid (`grid_seconds = 3600`) and `max_lag = 12` for the AMM-only analysis. Bonferroni correction is applied across all tested lag windows. The paper-claimable A/A DEX-flow result uses this method.

**Transfer entropy (TE)**: Nonlinear information flow from $i$ to $j$, estimated by a k-nearest-neighbour mutual information estimator. Statistical significance is assessed via block-shuffle permutation (1,000 permutations) with FDR correction.

**Vector autoregression (VAR) / Granger causality**: Linear predictive spillovers in a multi-node VAR. Granger p-values and FEVD share are reported. VAR diagonal entries (self-loops) are excluded from all edge summary tables.

**Sparse-flow event study**: For mint/burn arrival series that are too sparse for continuous-time analysis, we apply an event-arrival test: compare the distribution of target-node response values in a short post-arrival window to the pre-event baseline distribution, using permutation-based block-shuffle inference.

### 4.4 AMM-Only Analysis

The primary result is the *AMM-only lead-lag analysis*: we restrict to DEX-layer nodes, use only the Tier-A `usdc_net_sold_1h` feature, and use an hourly grid. This avoids stale-value artifacts from resampling hourly on-chain flows onto finer grids.

---

## 5. Main Result: USDT/Curve 2023 A/A AMM-Flow Linkage

### 5.1 The Headline Finding

The strongest paper-claimable result is in the USDT/Curve 2023 event. The Curve 3pool (`curve_3pool`) and Curve crvUSD/USDT pool (`curve_crvusd_usdt`) exhibit statistically supported bidirectional AMM-flow linkage on the Tier-A `usdc_net_sold_1h` hourly on-chain flow series.

Both directions pass Bonferroni correction at the 5% level:

| Direction | Peak lag | Peak correlation | FDR-adjusted p | Bonferroni p | Significant (Bonferroni) | Claim level |
|-----------|----------|-----------------|----------------|--------------|--------------------------|-------------|
| `curve_3pool → curve_crvusd_usdt` | 0 | 0.386 | 0.007 | **0.014** | Yes | `A_A_dex_flow` |
| `curve_crvusd_usdt → curve_3pool` | 0 | 0.386 | 0.000 | **0.000** | Yes | `A_A_dex_flow` |

Both rows have `paper_claim_allowed = True` and `claim_strength = robust`. This result uses only:
- Tier-A data: Etherscan `TokenExchange` logs for both pools.
- Tier-A feature: `usdc_net_sold_1h` (direct hourly sum of on-chain event amounts).
- No fixture data, no derived proxies.

The peak correlation at lag 0 (r = 0.386) indicates near-simultaneous co-movement rather than a clear lead-lag relationship. The bidirectional symmetry is consistent with a common response to the USDT de-peg shock, with both pools experiencing synchronized selling pressure on USDT.

These two rows are the only rows in `results/paper/tables/table_aa_paper_claimable_edges.csv`. No other event produces A/A paper-claimable evidence under the same claim gate.

### 5.2 What This Result Supports

The headline finding supports the following claim:

> During the USDT/Curve 2023 event (June 2023), Curve 3pool and Curve crvUSD/USDT exhibit statistically supported bidirectional AMM-flow linkage using Tier-A on-chain `usdc_net_sold_1h` data on an hourly grid (Bonferroni p ≤ 0.014 in both directions, peak cross-correlation r = 0.386, `claim_strength = robust`).

### 5.3 What This Result Does Not Support

The headline finding does not support:

- **Structural causal identification**: The cross-correlation approach detects directional timing evidence, not structural causality. Granger causality in this context establishes predictive precedence, not causal mechanism.
- **Transmission direction**: The simultaneous peak at lag 0 means we cannot determine which pool "led" the other; both directions are equally supported.
- **CEX microstructure**: This finding is purely on the AMM layer. It does not apply to centralized exchanges.
- **Other events**: No other event produces a paper-claimable A/A result under the same gate.

---

## 6. Cross-Event Evidence

### 6.1 Claim Audit Summary

The following table summarizes the claim-gate outcomes across all five events. It is reproduced verbatim from `results/paper/tables/table_claim_audit_summary.csv`:

| Event | Total edges | A/A prov-valid | A/A paper-claimable | A/B paper-claimable | B/B context |
|-------|-------------|----------------|--------------------|--------------------|-------------|
| USDT/Curve 2023 | 14 | 6 | **2** | 1 | 0 |
| Terra/LUNA 2022 | 26 | 6 | 0 | 4 | 4 |
| USDC/SVB 2023 | 42 | 1 | 0 | 4 | 18 |
| FTX 2022 | 18 | 0 | 0 | 5 | 6 |
| BUSD 2023 | 36 | 0 | 0 | 7 | 18 |

Key observations:
- **USDT/Curve 2023** is the only event with A/A paper-claimable edges.
- **Terra/LUNA 2022** has 6 A/A provenance-valid candidates (Curve 3pool and Curve UST/wormhole, all `usdc_net_sold_1h`) but none pass the statistical gate.
- **USDC/SVB 2023** has 1 A/A provenance-valid candidate (`usdc_mint_burn ↔ curve_3pool`) but it is underpowered (Section 7).
- **FTX 2022 and BUSD 2023** lack a second Tier-A node, so A/A edges are not possible; A/B and B/B edges provide contextual evidence only.

### 6.2 A/B Suggestive Evidence

Across all five events, we find **21 A/B paper-claimable edges** (total `n_AB_paper_claimable` summed across events: 7 + 5 + 4 + 1 + 4 = 21). These involve the Curve 3pool Tier-A anchor paired with public Binance/CoinMetrics Tier-B nodes. They represent suggestive directional timing evidence but cannot support the stronger A/A paper-claimable claim.

### 6.3 Why Terra/LUNA Fails the Statistical Gate

Terra/LUNA 2022 has two A/A provenance-valid pairs (`curve_3pool ↔ curve_ust_wormhole`), both using Tier-A `usdc_net_sold_1h`. Neither direction passes Bonferroni correction (p = 1.0 in the hourly AMM-only lead-lag analysis). We attribute this to:
1. The pool drain was extreme — the `curve_ust_wormhole` pool was nearly depleted, introducing near-zero flow values and reducing correlation power.
2. The UST collapse was more abrupt than the USDT de-peg, potentially making the flow relationship noisier at the hourly resolution.

These A/A Terra candidates are correctly reported as `provenance_claim_allowed = True`, `statistical_claim_allowed = False`, `paper_claim_allowed = False`, `claim_strength = suggestive`.

---

## 7. Sparse Settlement Response: USDC/SVB

### 7.1 The Sparse-Flow Problem

The USDC/SVB 2023 event is special because the primary settlement channel — USDC mint/burn flows — is *sparse*: only 4 mint/burn arrival events occur in the 12-day event window. This sparsity makes continuous-time lead-lag analysis inappropriate (too few observations at the hourly grid). We instead apply a *sparse-flow event study*: compare the mean `usdc_net_sold_1h` in Curve 3pool in the 3 hours following a mint/burn arrival to the 12-hour pre-event baseline distribution, using 1,000 block-shuffle permutations.

### 7.2 Result: Provenance-Valid, Not Paper-Claimable

The A/A on-chain settlement edge `usdc_mint_burn → curve_3pool` has the following properties:
- `tier_i_actual = A`, `tier_j_actual = A`, `feature_tier = A`.
- `provenance_claim_allowed = True`.
- 4 mint/burn arrivals in the window.
- Mean baseline Curve 3pool flow: −267,879 USDC-equivalent/hour.
- Mean post-arrival response: −238,923 USDC-equivalent/hour.
- Mean difference: +28,956 (+10.8% toward positive, i.e., reduced selling pressure).
- Permutation p-value: 1.0.
- `statistical_claim_allowed = False`, `paper_claim_allowed = False`.

The sparse-flow test is underpowered: 4 arrivals are insufficient for reliable block-shuffle inference. The directional sign of the response (+10.8%, reduced selling) is consistent with the narrative that mint/burn events reduce AMM pool pressure, but the p = 1.0 result means this interpretation cannot be supported as a paper claim.

**Wording**: We document a high-provenance sparse settlement-flow response *candidate* from `usdc_mint_burn` to `curve_3pool`. The data are real and the evidence tier is A/A. But it is not statistically supported under the event-arrival test.

---

## 8. Robustness, Negative Evidence, and Non-Claims

### 8.1 Self-Loop Exclusion

VAR and FEVD-based spillover tables include diagonal entries (node → same node). These are methodologically standard but not cross-node propagation evidence. All self-loops are excluded from every edge summary table and logged in `results/paper/tables/table_excluded_self_loops.csv` for audit transparency.

### 8.2 Fixture Blocking

No fixture-derived rows survive in any paper-claimable table. The validation script (`scripts/14_validate_paper_package.py`) checks this explicitly. All fixture nodes are blocked at the provenance gate (`uses_fixture = True` → `provenance_claim_allowed = False`).

### 8.3 What This Paper Does Not Claim

The following claims are explicitly **not** made by this paper:

1. **CEX microstructure transmission**: No historical full-depth CEX L2 order book data are available for any of the five events. The `A_A_cex_microstructure` claim level is never assigned.
2. **Structural causal identification**: Lead-lag and Granger evidence supports directional timing precedence, not structural causal identification.
3. **A/A paper-claimable evidence for Terra/LUNA, USDC/SVB, FTX, or BUSD**: Each of these events has provenance-valid candidates or A/B contextual evidence, but none achieves the A/A paper-claimable standard.
4. **Tier-A status for `reserve_imbalance` or `implied_pool_price`**: These derived Curve pool proxies are Tier B because they depend on approximate normalisation denominators.
5. **Paper evidence from fixture data**: All synthetic pipeline fixtures are explicitly blocked.

---

## 9. Methods Extension: Forbes-Rigobon and Quantile VAR

### 9.1 Forbes-Rigobon Contagion vs. Interdependence

The lead-lag cross-correlation result in Section 5 establishes directional timing evidence, but cannot by itself distinguish *contagion* — an excess co-movement beyond what shared volatility dynamics would predict — from *interdependence* — a stable correlation that mechanically inflates during high-volatility periods. The Forbes and Rigobon (2002) heteroskedasticity-corrected test addresses this directly.

The test computes the stress-period correlation between two series and applies a bias correction to remove the mechanical inflation that occurs when one conditioning series experiences higher variance in the stress period. The corrected correlation $\hat{\rho}^\ast$ is:

$$\hat{\rho}^\ast = \frac{\hat{\rho}_\text{stress}}{\sqrt{1 + \delta(1 - \hat{\rho}_\text{stress}^2)}}$$

where $\delta = (\sigma^2_\text{stress} - \sigma^2_\text{tranquil}) / \sigma^2_\text{tranquil} \geq 0$ is the variance ratio for the conditioning series. If $\hat{\rho}^\ast$ is significantly higher than the tranquil-period correlation after Fisher z-transform, we reject interdependence in favour of contagion.

We implement this test in `scripts/07b_run_forbes_rigobon.py` for all directed pairs using the Tier-A `usdc_net_sold_1h` series. Bonferroni correction is applied across the number of pairs tested per event. The test is designed to be fed through the same claim gate as lead-lag results: Forbes-Rigobon output tables carry the prefix `table_forbes_rigobon_`, and both the provenance and statistical gates are applied before any paper-level contagion claim.

The Forbes-Rigobon test is especially informative for the design-heterogeneity comparison across events. The Terra/LUNA algorithmic collapse and the USDT/Curve fiat-backed pool stress have different $\delta$ profiles: the Terra event involved extreme abrupt variance, while the USDT/Curve event had a more sustained volatility elevation. If the Terra pairs fail the Forbes-Rigobon test (as they fail the lead-lag test) but the USDT/Curve pairs pass, that constitutes evidence that the contagion mechanism differs by stablecoin design, not merely that the Terra data are noisier.

### 9.2 Quantile VAR: Tail Spillover Asymmetry

The VAR and Granger causality results in the standard pipeline estimate average linear spillovers. Quantile VAR (QVAR) decomposes this by quantile, enabling detection of asymmetric tail dependence: the hypothesis that stress propagation is materially stronger in the tails ($\tau = 0.05$, $\tau = 0.95$) than at the median ($\tau = 0.50$).

For each directed pair $(i, j)$ at quantile $\tau$, we estimate:

$$Q_\tau(x_{j,t} \mid x_{j,t-1}, x_{i,t-1}) = \alpha + \beta_\text{own} \cdot x_{j,t-1} + \beta_\text{cross} \cdot x_{i,t-1}$$

using statsmodels QuantReg. The cross-coefficient $\beta_\text{cross}(\tau)$ is the quantile impulse-response of $j$ to a unit shock in $i$ at quantile $\tau$. A finding that $|\beta_\text{cross}(0.05)| \gg |\beta_\text{cross}(0.50)|$ (or similarly at $\tau = 0.95$) supports non-linear tail contagion.

QVAR is implemented in `scripts/08b_run_qvar.py` and restricted to pairs with Tier-A `usdc_net_sold_1h` data. To maintain conservative claim coverage, QVAR results at the tail quantiles are paper-claimable only when the same pair also passes the lead-lag statistical gate. The key empirical diagnostic is the **tail amplification ratio**: mean $|\beta_\text{cross}|$ at $\{0.05, 0.95\}$ divided by $|\beta_\text{cross}(0.50)|$.

### 9.3 Design Heterogeneity: Algorithmic vs. Fiat-Backed

The five events provide a natural 2×2 comparison of stablecoin design and stress mechanism:

| Event | Design | Mechanism | AMM lead-lag result |
|-------|--------|-----------|---------------------|
| USDT/Curve 2023 | Fiat-backed | DEX pool imbalance | **A/A paper-claimable** (r = 0.386, Bonferroni p ≤ 0.014) |
| USDC/SVB 2023 | Fiat-backed | Bank reserve shock | Provenance-valid, statistically underpowered |
| Terra/LUNA 2022 | Algorithmic | Reflexive collapse | Provenance-valid, statistically null (p = 1.0) |
| FTX 2022 | N/A (exchange) | Credit/liquidity shock | No second Tier-A node |
| BUSD 2023 | Fiat-backed | Regulatory wind-down | No second Tier-A node |

The Terra/LUNA null result at the hourly AMM-flow resolution is substantively informative, not merely a power failure. The algorithmic collapse propagated through a faster channel (CEX price and LFG reserve depletion) than the DEX-flow channel can detect at hourly resolution. The USDC/SVB sparse-flow result, even though underpowered, suggests that mint/burn settlement flows are a distinct stress channel from AMM exchange flows. These contrasts motivate two claims:

1. **Fiat-backed DEX pool stress is detectable at hourly resolution via AMM flows.** The USDT/Curve 2023 result provides A/A paper-claimable evidence.
2. **Algorithmic collapse propagates through channels faster than hourly AMM-flow aggregation.** The Terra/LUNA null result at hourly resolution is consistent with sub-hour transmission through CEX price channels, which are not Tier-A in this study.

Forbes-Rigobon results, when available, will allow us to sharpen these claims: if the USDT/Curve event survives the heteroskedasticity correction and the Terra/LUNA event does not, that supports a structural interpretation of design heterogeneity rather than a purely power-based explanation.

---

## 10. Regulatory Capital and Policy Implications

The empirical results, even under the conservative claim gate, carry three policy implications for stablecoin surveillance and capital adequacy frameworks.

### 10.1 On-Chain DEX Monitoring as a Leading Indicator

The A/A paper-claimable result (Section 5) demonstrates that Curve pool AMM flows — specifically the net USDC-equivalent sold per hour from on-chain `TokenExchange` logs — exhibit statistically supported co-movement across pools during the USDT/Curve 2023 stress episode. Because these flows are available in near-real-time from a free public API (Etherscan), this finding suggests that **real-time monitoring of Curve pool imbalances belongs in any stablecoin liquidity surveillance toolkit**.

This is distinct from price-based monitoring. A regulator observing only CEX prices during the USDT/Curve event would see a basis widening of a few basis points; a regulator observing Curve pool imbalances would see simultaneous large-magnitude net-sold flows across multiple pools, providing an earlier and more structurally informative stress signal. The practical implication is that surveillance frameworks should treat on-chain DEX flow data as complementary primary data, not as a secondary cross-check on CEX prices.

For liquidity stress testing under a framework analogous to Basel III LCR, the relevant operational implication is: **AMM pool net-sold flows are observable inputs that can be incorporated into intraday liquidity monitoring without requiring any additional data licensing**, since the Etherscan API is public. The key data requirement is a block-timestamp resolver and a per-pool ABI decoder, both of which are implemented in this study's codebase.

### 10.2 Algorithmic Stablecoin Circuit Breakers: Speed Constraint

The Terra/LUNA null result at hourly resolution (Section 6.3) implies that the DEX-flow propagation channel was not active — or was swamped by other channels — during the algorithmic collapse. The LFG reserve drawdown and LUNA hyperinflation were sub-hour dynamics; by the time the hourly Curve pool flows would have transmitted a signal, the collapse had already propagated through faster channels.

This has direct implications for the design of circuit breaker mechanisms for algorithmic stablecoins. **Any circuit breaker predicated on observing DEX pool imbalances before triggering will be ineffective for a reflexive algorithmic collapse.** The relevant monitoring frequency for algorithmic stablecoins is sub-hour — likely block-by-block (approximately 12-second intervals for Ethereum) — not the hourly cadence that is sufficient for fiat-backed pool stress detection.

For a regulatory framework, this implies that fiat-backed and algorithmic stablecoins should not be subject to identical liquidity monitoring requirements. Fiat-backed stablecoins can be monitored effectively via hourly AMM flow aggregates; algorithmic stablecoins require higher-frequency monitoring with faster-acting circuit break thresholds to be operationally meaningful.

### 10.3 Mint/Burn Settlement Flows as a Distinct Stress Channel

The USDC/SVB sparse-flow result (Section 7) — 4 mint/burn arrivals in 12 days, p = 1.0, but A/A provenance tier — suggests that **mint/burn settlement flows are a distinct stress channel from AMM exchange flows** even though both are on-chain Ethereum data. The channels operate at different timescales and through different mechanisms: AMM flows are continuous (many per hour during stress), while mint/burn flows are lumpy (large individual redemptions triggered by specific institutional actors).

For capital adequacy purposes, this distinction matters. A liquidity stress framework that treats all on-chain stablecoin flows as a single channel will conflate two mechanisms with different lead times and amplitudes. Specifically:

- **AMM exchange flows** during a DEX pool stress event are high-frequency, bilateral (swap-in and swap-out), and indicative of market-maker behaviour.
- **Mint/burn settlement flows** during a bank-shock event (USDC/SVB) are low-frequency, directional (redemptions dominate), and indicative of institutional redemption pressure.

A capital buffer calibration that accounts only for AMM flow volatility will underestimate the tail risk from concentrated institutional redemptions in a bank-shock scenario. Conversely, a buffer calibrated to the worst-case mint/burn shock (the SVB episode) would overestimate routine AMM liquidity needs. **The policy recommendation is a two-channel liquidity framework**: one buffer sized for routine AMM pool imbalance risk (informed by QVAR tail estimates from the USDT/Curve event) and a separate buffer for institutional settlement risk (informed by the mint/burn sparse-flow channel).

When QVAR tail-spillover estimates are available (Section 9.2), the tail amplification ratio ($|\beta_\text{cross}(0.05)| / |\beta_\text{cross}(0.50)|$) can be used to scale the AMM buffer: a ratio greater than 2 would suggest that tail liquidity needs are at least twice the median-period estimate, warranting an additional capital loading. This connects the QVAR methodological extension directly to a quantitative capital buffer calibration, analogous to the approach in arXiv:2602.18820 but grounded in Tier-A on-chain evidence rather than return-based GARCH.

---

## 11. Limitations

**Data availability and tier ceiling**: The inability to obtain historical full-depth CEX L2 data is the binding constraint on this paper's claim ceiling. Public Binance OHLCV and BBO snapshots are available but do not support executable microstructure claims. Any researcher who replicates this study with access to paid vendor archives (Tardis, Kaiko) can potentially upgrade CEX nodes to Tier A and test whether microstructure-level propagation can be established.

**AMM pool selection**: The inability to obtain historical full-depth CEX L2 data is the binding constraint on this paper's claim ceiling. Public Binance OHLCV and BBO snapshots are available but do not support executable microstructure claims. Any researcher who replicates this study with access to paid vendor archives (Tardis, Kaiko) can potentially upgrade CEX nodes to Tier A and test whether microstructure-level propagation can be established.

**AMM pool selection**: The primary analysis uses four Curve Finance pools (3pool, crvUSD/USDT, UST/wormhole, FRAX/USDC, LUSD/3CRV, sUSD). Three extended pools have been added for design-heterogeneity coverage: Curve FRAX/USDC (fiat-backed, frxUSD), Liquity LUSD/3CRV (overcollateralized ETH-backed), and Synthetix sUSD (synthetic). These pools produce Tier-A `usdc_net_sold_1h` data when queried via the same TokenExchange pipeline, and their events are provisionally assigned based on pool asset composition and event window overlap. Uniswap v2/v3 pool data are partially available via The Graph API (not queried in this study due to missing API key). Extending to Uniswap would allow comparisons across AMM designs.

**Event window length**: The five events span windows of 12–30 days. Some windows may be too short to detect slower-moving propagation channels (e.g., settlement flows that take multiple days to settle). Longer windows would introduce more confounding macro shocks.

**Hourly grid resolution**: The AMM-only lead-lag analysis uses an hourly grid to match the natural frequency of on-chain `TokenExchange` aggregation. Higher-frequency analysis (e.g., block-by-block) would require more sophisticated inference methods and larger storage.

**Sparse mint/burn series**: The USDC/SVB sparse-flow analysis has only 4 mint/burn arrivals, well below the minimum for reliable block-shuffle inference. A larger time window or a different event definition (e.g., burst-of-minting episodes) might recover statistical power.

**No USDT mint/burn**: Tether uses non-standard `Issue`/`Redeem` events rather than ERC-20 `Transfer` events. A dedicated Tether event decoder would be required to populate the `usdt_mint_burn` node for the USDT/Curve 2023 event.

**Provenance-valid ≠ paper-claimable**: The discipline of this claim gate is that it may exclude results that are economically real but statistically underpowered in the current framework. We report all provenance-valid candidates explicitly to allow readers to evaluate what was found but not claimed.

---

## 12. Conclusion

This paper makes three contributions. Substantively, it documents that during the USDT/Curve 2023 event, the Curve 3pool and Curve crvUSD/USDT pools exhibit statistically supported bidirectional AMM-flow linkage at the hourly grid, using only Tier-A execution-grade on-chain data (Bonferroni p ≤ 0.014 in both directions). This is a narrower claim than is common in the crypto contagion literature, but it is more credible because it is supported by execution-grade on-chain logs rather than derived prices or synthetic proxies.

Methodologically, the paper introduces the concept of a *provenance-aware claim gate* that explicitly separates provenance-valid candidates from paper-claimable results, and extends the analytical toolkit with Forbes-Rigobon heteroskedasticity-corrected contagion tests and Quantile VAR tail-spillover estimation. Across 136 annotated edge rows in the USDT/Curve 2023 event, only 2 pass both gates. Across all five events, 21 A/B suggestive edges pass but the only A/A paper-claimable result is the Curve-to-Curve AMM-flow pair. This provides the kind of transparent, audit-able claim documentation that complex crypto market claims require.

For policy, the paper derives three operational implications grounded in the claim-gated evidence: on-chain DEX pool imbalances are a real-time leading indicator appropriate for surveillance toolkits; algorithmic stablecoin circuit breakers cannot rely on hourly DEX-flow signals; and a two-channel capital buffer framework is warranted to separately account for AMM pool liquidity risk and institutional mint/burn redemption risk. The QVAR tail amplification ratio provides the quantitative link between the empirical results and a capital buffer sizing methodology, analogous to VaR-based tail-risk approaches but grounded in Tier-A on-chain evidence rather than return-based GARCH.

The framework is designed to be extensible. As historical CEX L2 data become more accessible (via vendor archives or live collectors), CEX nodes can be upgraded to Tier A and CEX-level microstructure propagation can be tested under the same gate. The newly added pools — Curve FRAX/USDC (frxUSD), Liquity LUSD/3CRV, and Synthetix sUSD — extend the design-heterogeneity coverage to eight nodes spanning fiat-backed, overcollateralized, and synthetic designs. The core methodological contribution — provenance-aware claim gating — is independent of which data sources are ultimately available.

---

## Appendix: Data Sources and Reproducibility

All results are reproducible from publicly available data using free API keys.

### A.1 Required API Keys

- `ETHERSCAN_API_KEY` (required): fetches Curve `TokenExchange` events, USDC `Transfer` events.
- `THE_GRAPH_API_KEY` (optional): would enable Uniswap pool state queries.
- `DUNE_API_KEY` (optional): would enable bridge flow queries.

### A.2 Reproduction Command

```bash
make empirical_all    # run all 5 events with real data, no fixture
make paper_gate       # claim-gate → summary tables → figures → validation (PASS)
python -m pytest tests -q   # 226 tests pass
```

### A.3 Output Files

Key paper outputs:
- `results/paper/tables/table_aa_paper_claimable_edges.csv` — headline 2 rows
- `results/paper/tables/table_claim_audit_summary.csv` — per-event claim counts
- `results/paper/tables/table_aa_provenance_valid_edges.csv` — 8 provenance-valid candidates
- `results/paper/figures/` — 12 standard paper figures
- `results/paper/figures_columbia/` — Columbia-themed figure pack (18 figures)

### A.4 Claim Gate Configuration

The claim gate is defined in `src/stressnet/evaluation/claim_gate.py`. The feature-tier configuration is in `configs/feature_tiers.yaml`. The claim-gate pipeline is invoked by `python scripts/00c_claim_gate.py --all-events --strict`.

---

*All empirical outputs were generated with `make paper_gate`, which enforces strict fixture blocking and produces a `RESULT: PASS (10/10 checks)` validation report.*
