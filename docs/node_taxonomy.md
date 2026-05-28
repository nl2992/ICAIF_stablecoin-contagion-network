# Node Taxonomy

The contagion network contains three distinct node layers. This multi-layer design is
what distinguishes the project from a standard price-correlation study: it captures
transmission through order books, AMM pool states, and settlement flows simultaneously.

## Layer 1: Market nodes

A market node is a stablecoin–venue pair that participates in secondary market trading.

**Node ID format:** `{asset}_{venue}`, e.g. `usdc_coinbase`, `usdt_binance`.

| Node | Asset | Venue | Primary events |
|---|---|---|---|
| usdc_coinbase | USDC | Coinbase | USDC/SVB 2023 |
| usdc_binance | USDC | Binance | USDC/SVB 2023, BUSD 2023 |
| usdt_binance | USDT | Binance | All five events |
| usdt_kraken | USDT | Kraken | USDC/SVB, USDT/Curve |
| usdc_kraken | USDC | Kraken | USDC/SVB 2023 |
| busd_binance | BUSD | Binance | BUSD 2023, FTX 2022 |
| ust_binance | UST | Binance | Terra/LUNA 2022 |

**Features at each market node:**

- `mid_price`: (best_bid + best_ask) / 2
- `spread_bps`: (best_ask - best_bid) / mid × 10,000
- `depth_10bps_bid_usd`, `depth_10bps_ask_usd`: cumulative depth within 10 bps
- `orderbook_imbalance`: (bid_depth − ask_depth) / (bid_depth + ask_depth)
- `signed_trade_imbalance_1m`: net buyer volume minus seller volume, 1-minute window
- `cancellation_proxy`: price-level removal rate (true cancellations on Coinbase; proxy elsewhere)
- `executable_price_10k`: VWAP book-walk price for $10k notional when L2 depth is available; otherwise explicitly flagged as proxy/null by `executable_price_source`
- `basis_vs_usd`: log deviation from $1.00 peg

## Layer 2: Pool nodes

A pool node is a DEX liquidity pool. Pool nodes capture stress amplification through
AMM inventory imbalance and slippage curve shifts.

**Node ID format:** `{protocol}_{assets}`, e.g. `curve_3pool`, `uniswap_usdc_usdt_005`.

| Node | Protocol | Assets | Chain | Primary events |
|---|---|---|---|---|
| curve_3pool | Curve v1 | DAI/USDC/USDT | Ethereum | All five events |
| curve_ust_wormhole | Curve v1 | UST/3CRV | Ethereum | Terra/LUNA 2022 |
| uniswap_usdc_usdt_005 | Uniswap v3 | USDC/USDT 0.05% | Ethereum | USDC/SVB, USDT/Curve |
| curve_crvusd_usdt | Curve v2 | crvUSD/USDT | Ethereum | USDT/Curve 2023 |

**Features at each pool node:**

- `reserve_imbalance`: (reserve_0 / total_reserves) − 0.5; indicates which token dominates
- `implied_pool_price`: marginal spot price from pool invariant or sqrtPriceX96
- `pool_slippage_10k`: price impact in bps for a $10k notional swap
- `lp_mint_burn_net`: net LP share mints minus burns (liquidity provider confidence signal)
- `swap_imbalance_1m`: (buy_volume − sell_volume) / total_volume, 1-minute window
- `virtual_price`: Curve virtual_price() or equivalent pool health invariant

## Layer 3: Flow nodes

A flow node represents a settlement or movement channel. Flow nodes can transmit or
absorb stress by altering the cost or speed of capital movement between other nodes.

**Node ID format:** `{chain}_{asset}_{type}`, e.g. `eth_usdc_exchange_flows`.

| Node | Type | Chain | Asset | Primary events |
|---|---|---|---|---|
| eth_usdc_exchange_flows | exchange_flow | Ethereum | USDC | USDC/SVB, FTX |
| eth_usdt_exchange_flows | exchange_flow | Ethereum | USDT | USDT/Curve, FTX, USDC/SVB |
| tron_usdt_exchange_flows | exchange_flow | Tron | USDT | USDT/Curve |
| eth_bridge_flows | bridge_flow | multi | multi | FTX, USDC/SVB |
| usdc_mint_burn | mint_burn | Ethereum | USDC | USDC/SVB |
| usdt_mint_burn | mint_burn | Ethereum | USDT | USDT/Curve |

**Features at each flow node:**

- `exchange_inflow_1h`, `exchange_outflow_1h`, `exchange_netflow_1h`
- `bridge_inflow_1h`, `bridge_outflow_1h`
- `mint_burn_net_1h`
- `gas_base_fee_gwei`: Ethereum congestion proxy

## Node roles in the contagion network

After estimating directed edges, nodes are classified into functional roles:

| Role | Definition |
|---|---|
| Originator | High out-degree, low in-degree; consistently leads at stress onset |
| Amplifier | High betweenness; stress passing through increases in intensity |
| Sink | High in-degree, low out-degree; absorbs stress without transmitting |

These roles are computed from centrality measures (weighted out-degree, in-degree,
eigenvector centrality, betweenness) and may vary across events.
