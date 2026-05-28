# Source Coverage Matrix

Status key: ✅ empirical data in pipeline  |  ⚠️ partial/proxy  |  ❌ missing  |  — not applicable

## CEX market nodes

| Node | Tier nominal | usdc_svb_2023 | terra_luna_2022 | usdt_curve_2023 | ftx_2022 | busd_2023 | L2 vendor available |
|---|---|---|---|---|---|---|---|
| usdc_binance | A | ⚠️ BBO | ❌ | ⚠️ BBO | ❌ | ⚠️ BBO | Tardis `USDCUSDT` |
| usdt_binance | A | ⚠️ BBO | ⚠️ BBO | ⚠️ BBO | ⚠️ BBO | ⚠️ BBO | Tardis `USDCUSDT`/`BTCUSDT` |
| usdc_coinbase | A | ⚠️ klines | ❌ | ❌ | ❌ | ❌ | Tardis `USDC-USD` (limited) |
| usdt_kraken | A | ⚠️ klines | ❌ | ⚠️ klines | ❌ | ❌ | Tardis `USDTUSD` |
| usdc_kraken | A | ⚠️ klines | ❌ | ❌ | ❌ | ❌ | Tardis `USDCUSD` |
| busd_binance | A | ❌ | ❌ | ❌ | ⚠️ BBO | ⚠️ BBO | Tardis `BUSDUSDT` |
| ust_binance | B | — | ⚠️ trades | — | — | — | Tardis `USTUSDT` (delisted) |

**⚠️ BBO** = Binance Vision bookTicker (best bid/ask single level, Tier B).  
**⚠️ klines** = 1-minute OHLCV candles, no bid/ask depth (Tier B).

## DEX / pool nodes

| Node | Tier nominal | usdc_svb_2023 | terra_luna_2022 | usdt_curve_2023 | ftx_2022 | busd_2023 | On-chain status |
|---|---|---|---|---|---|---|---|
| curve_3pool | A | ✅ on-chain | ✅ on-chain | ✅ on-chain | ✅ on-chain | ✅ on-chain | Etherscan logs ✅ |
| curve_ust_wormhole | A | — | ✅ on-chain | — | — | — | Etherscan logs ✅ |
| uniswap_usdc_usdt_005 | A | ✅ subgraph | — | ✅ subgraph | — | — | The Graph ⚠️ (5min lag) |
| curve_crvusd_usdt | A | — | — | ✅ on-chain | — | — | Etherscan logs ✅ |

## On-chain flow nodes

| Node | Tier nominal | usdc_svb_2023 | terra_luna_2022 | usdt_curve_2023 | ftx_2022 | busd_2023 |
|---|---|---|---|---|---|---|
| eth_usdc_exchange_flows | B | ✅ CoinMetrics | — | — | ✅ CoinMetrics | — |
| eth_usdt_exchange_flows | B | ✅ CoinMetrics | — | ✅ CoinMetrics | ✅ CoinMetrics | — |
| tron_usdt_exchange_flows | B | — | — | ⚠️ partial | — | — |
| eth_bridge_flows | B | ⚠️ Dune | — | — | ⚠️ Dune | — |
| usdc_mint_burn | A | ✅ Etherscan | — | — | ✅ Etherscan | — |
| usdt_mint_burn | A | — | — | ✅ Etherscan | — | — |

---

## Critical gaps blocking Tier-A edges

The following are the minimum changes required to produce at least one
`A_A_directional_microstructure` edge per headline event:

| Event | Blocking gap | Required action |
|---|---|---|
| usdc_svb_2023 | CEX nodes are Tier B (BBO/klines) | Ingest Tardis `incremental_book_L2` for `usdc_binance` + `usdt_binance` |
| terra_luna_2022 | `ust_binance` is Tier B only; L2 may be missing | Tardis `USTUSDT` archive check; fallback Tier B/B acceptable |
| usdt_curve_2023 | CEX nodes are Tier B | Tardis `USDCUSDT` L2 for `usdt_binance` |
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
