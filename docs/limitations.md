# Known Limitations

## 1. Order-cancellation richness is uneven across exchanges

Coinbase's `full` / `level3` channel provides explicit order lifecycle messages
(open, change, done/cancel). Binance and Kraken do not publish order IDs or cancel
events; cancellations must be inferred from price-level size reductions.

**Impact:** Cancellation intensity is Tier-A on Coinbase and an explicit proxy on
Binance/Kraken. Claims about cancellation-driven contagion should note this asymmetry.

## 2. Address labelling is incomplete

Exchange-wallet and bridge labels from Etherscan nametags, Dune community labels, and
vendor products (Coin Metrics, Nansen) cover the largest wallets but miss smaller or
newly-created addresses. Label confidence is roughly 70–85% for major exchange wallets.

**Impact:** Exchange netflow and bridge flow metrics are Tier-B, not Tier-A. They
provide directional evidence but should not support precise quantitative claims without
triangulation.

## 3. Defunct-venue L2 history may be incomplete

FTX ceased operations in November 2022 and archived data is limited. Full-depth L2
history for FTX is likely unrecoverable. Third-party archives (Tardis, Kaiko) may have
partial coverage up to the withdrawal halt.

**Impact:** The FTX event analysis relies on downstream venue data (Binance, Coinbase,
Kraken) as transmission recipients, not on FTX itself as a Tier-A source.

## 4. Synthetic control is vulnerable to network spillovers

In a stablecoin stress event, "untreated" donor nodes are rarely fully untreated —
they participate in the same arbitrage network and may already reflect contagion.
This violates the stable unit treatment value assumption (SUTVA) required by classical
synthetic control.

**Impact:** Synthetic control results should be interpreted as providing suggestive
causal evidence, not clean identification. The paper should note this explicitly and
rely on event-study identification as the primary causal strategy.

## 5. Data coverage varies substantially across events

| Event | CEX L2 | DEX pool | On-chain flows |
|---|---|---|---|
| USDC/SVB 2023 | High | High | Medium |
| Terra/LUNA 2022 | Medium | High | High |
| USDT/Curve 2023 | Medium | High | Medium |
| FTX 2022 | Medium (no FTX) | Low | Medium |
| BUSD 2023 | High | Low | High |

Claims that involve pooling all five events must account for differential data quality
or restrict the claim to events with sufficient coverage.

## 6. AMM pricing is not equivalent to CEX pricing

DEX pool implied prices are the marginal swap price, not a bid-ask spread. Comparing
pool price dynamics to CEX mid or executable prices requires explicit accounting for
AMM fee tiers, liquidity concentration, and slippage. Apparent basis between CEX and
DEX may include a structural fee wedge unrelated to stress.

## 7. Event timing is approximate for algorithmic stress

Terra/LUNA and the USDT/Curve episode do not have a single, externally verifiable
T=0 comparable to a regulatory announcement. The shock onset is identified from the
data itself, which introduces some circularity in event-study designs for those events.

## 8. Temporal GNNs are not interpretable by default

TGN/DySAT-style models improve prediction but do not produce straightforward causal
attributions. Predictive lift cannot be translated directly into "DEX caused CEX stress."
All causal language in the paper refers to the directed methods (Hawkes, transfer entropy,
VAR/Granger), not the GNN predictions.
