# Source Coverage Matrix

Status key: full = empirical data in pipeline  |  part = partial/proxy  |  — = missing  |  n/a = not applicable

## CEX market nodes

| Node | Tier nominal | usdc_svb_2023 | terra_luna_2022 | usdt_curve_2023 | ftx_2022 | busd_2023 | L2 vendor available |
|---|---|---|---|---|---|---|---|
| usdc_binance | A | part BBO | — | part BBO | — | part BBO | Tardis `USDCUSDT` |
| usdt_binance | A | part BBO | part BBO | part BBO | part BBO | part BBO | Tardis `USDCUSDT`/`BTCUSDT` |
| usdc_coinbase | A | part klines | — | — | — | — | Tardis `USDC-USD` (limited) |
| usdt_kraken | A | part klines | — | part klines | — | — | Tardis `USDTUSD` |
| usdc_kraken | A | part klines | — | — | — | — | Tardis `USDCUSD` |
| busd_binance | A | — | — | — | part BBO | part BBO | Tardis `BUSDUSDT` |
| ust_binance | B | — | part trades | — | — | — | Tardis `USTUSDT` (delisted) |

**part BBO** = Binance Vision bookTicker (best bid/ask single level, Tier B).  
**part klines** = 1-minute OHLCV candles, no bid/ask depth (Tier B).

## DEX / pool nodes

| Node | Tier nominal | usdc_svb_2023 | terra_luna_2022 | usdt_curve_2023 | ftx_2022 | busd_2023 | On-chain status |
|---|---|---|---|---|---|---|---|
| curve_3pool | A | full on-chain | full on-chain | full on-chain | full on-chain | full on-chain | Etherscan logs full |
| curve_ust_wormhole | A | — | full on-chain | — | — | — | Etherscan logs full |
| uniswap_usdc_usdt_005 | A | full subgraph | — | full subgraph | — | — | The Graph part (5min lag) |
| curve_crvusd_usdt | A | — | — | full on-chain | — | — | Etherscan logs full |

## On-chain flow nodes

| Node | Tier nominal | usdc_svb_2023 | terra_luna_2022 | usdt_curve_2023 | ftx_2022 | busd_2023 |
|---|---|---|---|---|---|---|
| eth_usdc_exchange_flows | B | full CoinMetrics | — | — | full CoinMetrics | — |
| eth_usdt_exchange_flows | B | full CoinMetrics | — | full CoinMetrics | full CoinMetrics | — |
| tron_usdt_exchange_flows | B | — | — | part partial | — | — |
| eth_bridge_flows | B | part Dune | — | — | part Dune | — |
| usdc_mint_burn | A | full Etherscan | — | — | full Etherscan | — |
| usdt_mint_burn | A | — | — | full Etherscan | — | — |

---

## A/A DEX-flow edges (current status)

Tier-A DEX-flow (`A_A_dex_flow`) edges come from Curve `TokenExchange` logs on-chain.
CEX microstructure (`A_A_cex_microstructure`) edges require vendor/live L2 and are not
currently available.

| Event | A/A DEX-flow status | Paper-claimable? |
|---|---|---|
| usdt_curve_2023 | `curve_3pool` ↔ `curve_crvusd_usdt` (Bonferroni p ≤ 0.014) | **Yes** |
| terra_luna_2022 | `curve_3pool` ↔ `curve_ust_wormhole` (not significant at hourly grid) | No |
| usdc_svb_2023 | `usdc_mint_burn` ↔ `curve_3pool` (sparse; event-arrival underpowered) | No |
| ftx_2022 | Single Tier-A node only; A/B edges available | No A/A |
| busd_2023 | Single Tier-A node only; A/B edges available | No A/A |

For CEX `A_A_cex_microstructure` edges, the minimum requirement is Tardis/Kaiko L2 for
the relevant CEX pairs. This is an optional future extension; the paper narrative leads
with Tier-A AMM-flow evidence.
| ftx_2022 | `busd_binance` Tier B; `usdt_binance` Tier B | Tardis `BUSDUSDT`+`USDCUSDT` |
| busd_2023 | Same as ftx_2022 | Tardis `BUSDUSDT` |

---

## .env keys required per source

| Source | `.env` key |
|---|---|
| Tardis vendor L2 | `TARDIS_API_KEY` |
| Kaiko (alternative) | `KAIKO_API_KEY` |
| Dune Analytics | `DUNE_API_KEY` |
| Etherscan | `ETHERSCAN_API_KEY` |
| Coin Metrics | `COINMETRICS_API_KEY` |
| The Graph | `THE_GRAPH_API_KEY` |
| Archive RPC | `ETH_ARCHIVE_RPC_URL` |
| Tron / TronGrid | `TRONGRID_API_KEY` |
