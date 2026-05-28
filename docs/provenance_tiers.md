# Provenance Tiers

Every node, edge, and claim in this project is assigned a provenance tier. The tiers
restrict what inferences can be made from each data source and prevent conflating
high-precision microstructure evidence with coarser price-context data.

## Tier A — Execution-grade

**Definition:** Real L2 order book (snapshot + incremental updates with timestamps),
verified DEX pool state from on-chain event logs, or another high-quality source with
sufficient continuity and timing precision to support microstructure analysis.

**Examples:**
- Binance WebSocket diff-depth stream with server timestamps
- Coinbase Advanced Trade Level2 channel
- Kraken WebSocket book snapshots and updates
- Curve pool state from on-chain AddLiquidity / TokenExchange / RemoveLiquidity events
- Uniswap v3 pool Swap events (sqrtPriceX96, tick, liquidity)
- USDC/USDT mint/burn Transfer events from Etherscan

**Permitted claims:**
- Directional lead-lag at second or sub-second resolution
- Order-book stress transmission (spread widening, depth erosion, imbalance)
- Microstructure propagation across venues
- Primary input to Hawkes event definitions
- Tier-A prediction targets

## Tier B — Context-grade

**Definition:** Trades, OHLCV candles, DEX pool snapshots, or aggregate on-chain flows.
Sufficient for price/liquidity context but insufficient for precise microstructure claims.

**Examples:**
- Binance Vision spot aggTrades and klines (1m OHLCV)
- Binance Vision `bookTicker` best bid/ask and best-level quantities (BBO, not full L2)
- Coinbase REST candles
- The Graph Uniswap v3 subgraph pool statistics
- Coin Metrics exchange flow metrics (pre-aggregated netflows)
- Dune Analytics queries on decoded on-chain data
- Etherscan ERC-20 token transfer logs (address label confidence ~80%)

**Permitted claims:**
- Approximate stress timing and price context
- Weaker directional evidence (must be corroborated by Tier-A)
- Exchange flow co-movement analysis
- Auxiliary covariates in VAR/regression models

## Tier C — Taxonomy-grade

**Definition:** Partial, proxy, sporadic, or reconstructed data that cannot support
directional timing claims.

**Examples:**
- OHLCV from defunct or delisted venues with missing intervals
- Address labels with < 50% coverage
- Interpolated pool states between sparse snapshots
- Aggregated daily flow data

**Permitted claims:**
- Event taxonomy and mechanism classification
- Qualitative narrative context
- Presence/absence of stress (not timing or direction)

## Edge provenance rule

No edge can support a stronger claim than the weaker of its two endpoint provenance tiers.

| Node i tier | Node j tier | Edge claim tier |
|---|---|---|
| A | A | A — directional transmission |
| A | B | B — suggestive co-movement |
| B | B | B — suggestive co-movement |
| Any | C | C — taxonomy only |

## Downgrade conditions

A Tier-A source is downgraded in the following circumstances:

1. **Clock gaps:** Source timestamps are missing or implausible for >1% of messages.
2. **Sequence gaps:** Update sequence numbers show unexplained jumps.
3. **Resync periods:** A book resync was triggered (data quality flag is set).
4. **Incomplete coverage:** Source covers < 50% of the event analysis window.

The manifest records `coverage_pct`, `sequence_gap_count`, `gap_rate`,
`resync_count`, and `clock_offset_ms` when those diagnostics are available.
The consolidated node-coverage table applies the Tier-A downgrade rule
automatically before paper gates read provenance.

Downgraded nodes remain in the panel but carry a `tier_actual` field distinct from
`tier_nominal`, and their edges are labelled accordingly.
