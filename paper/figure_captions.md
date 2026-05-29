# Paper Figure Captions

**Paper title:** Provenance-Aware Stablecoin Stress Propagation Networks: Evidence from Curve TokenExchange Logs, Public CEX Data, and On-Chain Settlement Flows

---

## Figure 1

**Multi-layer stress-propagation architecture.**
The framework organises nodes into three layers: CEX markets (Binance, Coinbase, Kraken — Tier B public context), AMM pools (Curve 3pool, Curve crvUSD/USDT — Tier A on-chain `TokenExchange` logs), and settlement-flow channels (USDC mint/burn — Tier A on-chain Transfer events). An edge tier is the minimum of the source-node tier, target-node tier, and feature tier. Every edge passes through a provenance gate, a statistical gate, and a paper gate before supporting a claim. CEX nodes are classified Tier B because historical full-depth L2 order books are not freely available.

---

## Figure 2

**Three-gate claim pipeline.**
A directed edge is paper-claimable only when both the provenance gate and the statistical gate pass. The provenance gate checks that neither endpoint is fixture or missing, applies feature-level tier caps, and assigns a claim-level taxonomy (A/A DEX-flow, A/A on-chain settlement, A/B suggestive, B/B context-only). The statistical gate requires method-specific significance: Bonferroni correction for lead-lag, FDR-adjusted block-shuffle for transfer entropy, Granger p-values for VAR. Rows that pass only the provenance gate are labelled `provenance_claim_allowed = True`, `paper_claim_allowed = False`, and reported as provenance-valid candidates.

---

## Figure 3

**Claim-gate audit across all five stress events (anti-cherry-pick transparency).**
Stacked bars show the composition of annotated edge rows per event: B/B context-only (light grey), A/B suggestive paper-claimable (blue), A/A provenance-valid but statistically unsupported (hatched green), and A/A paper-claimable (amber, headline). The USDT/Curve 2023 event is the only event with A/A paper-claimable evidence. Terra/LUNA 2022 and USDC/SVB 2023 contain A/A provenance-valid candidates that do not pass the statistical gate. FTX 2022 and BUSD 2023 provide A/B directional evidence only. A/A provenance-valid rows shown here are distinct from A/A paper-claimable rows (no double-counting); the hatched segment represents edges that pass the provenance gate but fail the statistical gate.

---

## Figure 4

**USDT/Curve 2023 AMM-flow timeline (main empirical figure).**
Hourly net USDC-equivalent sold (`usdc_net_sold_1h`) for Curve 3pool (top panel) and Curve crvUSD/USDT (middle panel), with cumulative flows (bottom panel). Both series are derived directly from Curve `TokenExchange` on-chain event logs (Tier A). The co-movement between pools during the June 2023 de-peg episode motivates the lead-lag analysis. Shaded regions mark positive (green) and negative (red) net flows. All values are in thousands of USDC-equivalent units per hour.

---

## Figure 5

**USDT/Curve 2023 lead-lag correlation profile (headline evidence).**
Cross-correlation between `curve_3pool` and `curve_crvusd_usdt` hourly `usdc_net_sold_1h` series at lags −12 to +12 hours. Both directions exhibit peak correlation at lag 0 with Bonferroni-corrected p ≤ 0.014, qualifying as `A_A_dex_flow` paper-claimable evidence (`claim_strength = robust`). The symmetry of the profile at lag 0 is consistent with near-simultaneous AMM-flow linkage rather than a clear lead-lag relationship. The shaded band marks the Bonferroni significance threshold.

---

## Figure 6

**A/A paper-claimable network (headline pairs only).**
The only two rows that survive both the provenance gate and the statistical gate: `curve_3pool → curve_crvusd_usdt` and `curve_crvusd_usdt → curve_3pool`, both in the USDT/Curve 2023 event, `feature = usdc_net_sold_1h`, `p_bonferroni ≤ 0.014`. Both nodes are Tier A (on-chain `TokenExchange` logs). Node shapes indicate layer (square = DEX/AMM). This network does not include Terra/LUNA, USDC/SVB, FTX, or BUSD A/A rows, none of which pass the statistical gate.

---

## Figure 7

**A/A provenance-valid versus A/A paper-claimable, by event.**
Grouped bars show the number of A/A edges that pass the provenance gate (hatched green) versus those that also pass the statistical gate (amber). Several events have high-provenance A/A candidates: Terra/LUNA 2022 has 6 provenance-valid rows, USDC/SVB 2023 has 1, and USDT/Curve 2023 has 6. Only USDT/Curve 2023 produces any A/A paper-claimable rows (2). This demonstrates that provenance-valid does not imply paper-claimable; statistical support is required for the headline claim.

---

## Figure 8

**Terra/LUNA 2022 A/A AMM-flow candidate (negative result, high-provenance descriptive evidence).**
Hourly `usdc_net_sold_1h` for Curve 3pool and Curve UST/wormhole pool during the May 2022 UST collapse. Both series use Tier-A `TokenExchange` logs, forming an A/A provenance-valid candidate pair. However, neither direction passes the statistical gate in the AMM-only hourly lead-lag analysis (p = 1.0 for both directions, claim_strength = suggestive). The pool drain is extreme and may reduce cross-correlation power. These rows are reported as high-provenance descriptive evidence, not paper-claimable directional claims.

---

## Figure 9

**USDC/SVB 2023 sparse settlement-flow response (underpowered, not paper-claimable).**
Event-study bars show the mean Curve 3pool `usdc_net_sold_1h` response following USDC mint/burn arrivals (4 events, March 2023). The `usdc_mint_burn` node is Tier A (on-chain Transfer logs); `curve_3pool` is Tier A (TokenExchange logs), giving a genuine A/A on-chain-settlement provenance-valid pair. Despite real data, the sparse-event permutation test is underpowered (p = 1.0 for curve_3pool response; 4 arrivals insufficient for block-shuffle inference). This result is classified as `provenance_claim_allowed = True`, `paper_claim_allowed = False`. It is reported as a high-provenance descriptive candidate, not a directional paper claim.

---

## Figure 10

**Feature-tier matrix.**
Each row is a feature column; the colour encodes its evidence tier (green = Tier A, grey = Tier B). Direct on-chain flow features — `usdc_net_sold_1h` and `mint_burn_net_1h` — are Tier A because they are direct hourly sums of on-chain event logs. Derived proxies — `reserve_imbalance` and `implied_pool_price` — are Tier B because they depend on approximate normalisation factors. CEX features (`spread_bps`, `basis_vs_usd`) are Tier B because they rely on public OHLCV/BBO data without full L2 depth. The headline claim uses only Tier-A features.

---

## Figure 11

**Node provenance coverage heatmap.**
Each cell shows the coverage percentage and provenance tier for a (node, event) pair. Green cells are Tier A (Etherscan TokenExchange or Transfer logs); grey cells are Tier B (public CEX data); red/cross-hatched cells are fixture (synthetic pipeline data, not paper-claimable). The heatmap reveals that only Curve AMM nodes are Tier A across events; CEX nodes remain Tier B; and several nodes are fixture for events where data are unavailable. Coverage is measured as the fraction of event-window hours with non-null observations.

---

## Figure 12

**Full paper-claimable stress-propagation network.**
Directed network showing all edges that pass both the provenance and statistical gates: A/A paper-claimable AMM-flow edges (thick amber arrows) and A/B suggestive paper-claimable edges (blue arrows). Node colour encodes tier (green = Tier A, grey = Tier B); node shape encodes layer (square = AMM/DEX, circle = CEX, diamond = settlement/flow). Fixture-derived nodes are omitted. The headline `curve_3pool ↔ curve_crvusd_usdt` pair (USDT/Curve 2023, amber) is the only A/A paper-claimable edge. This figure does not claim causal contagion; it presents the subset of directional timing evidence supported by both data provenance and statistical inference.
