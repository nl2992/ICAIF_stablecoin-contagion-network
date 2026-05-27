# Event Windows

## Rationale for event selection

The five events cover distinct stress mechanisms, ensuring that any propagation patterns
found are not artefacts of one mechanism type:

| Event | Mechanism class | Why included |
|---|---|---|
| USDC/SVB 2023 | Fiat-reserve bank shock | Cleanest CEX-led price discovery shock with good L2 coverage |
| Terra/LUNA 2022 | Algorithmic reflexive | DEX-native propagation; Curve pool imbalance primary channel |
| USDT/Curve 2023 | DeFi pool imbalance | Pool-originated stress; tests whether DEX can lead CEX |
| FTX 2022 | Exchange credit/liquidity | Cross-stablecoin contagion; exchange outflows primary channel |
| BUSD 2023 | Regulatory wind-down | Slow-moving shock; tests persistence and mint/burn channel |

## Window definitions

### USDC/SVB 2023 (primary MVP event)

- **Shock onset:** FDIC receivership of Silicon Valley Bank announced ~2023-03-10 06:00 UTC
- **Core shock window:** 2023-03-10 to 2023-03-15
- **Analysis window:** 2023-03-08 to 2023-03-20
- **Event time T=0:** 2023-03-10 06:00:00 UTC
- **Key milestones:**
  - T=0: SVB FDIC announcement; USDC/USD on Coinbase begins trading below $0.99
  - T+12h: USDC reaches ~$0.87 on some venues
  - T+2d: Circle confirms $3.3B exposure; USDC recovery begins
  - T+5d: USDC back near $0.9990–$0.9995

### Terra/LUNA 2022

- **Shock onset:** ~2022-05-08 00:00 UTC (initial UST depeg from arbitrage attack)
- **Core shock window:** 2022-05-09 to 2022-05-16
- **Analysis window:** 2022-05-01 to 2022-05-31
- **Event time T=0:** 2022-05-08 00:00:00 UTC
- **Key milestones:**
  - T=0: Large UST withdrawals from Anchor; initial depeg below $0.99
  - T+1d: LFG deploys BTC reserves; temporary recovery
  - T+2d: Death spiral begins; LUNA supply hyperinflation
  - T+7d: UST trades at ~$0.10

### USDT/Curve 2023

- **Shock onset:** ~2023-06-15 00:00 UTC
- **Core shock window:** 2023-06-15 to 2023-06-18
- **Analysis window:** 2023-06-10 to 2023-06-25
- **Key milestones:**
  - Pre-event: 3pool becomes heavily USDT-weighted
  - Onset: USDT/USDC basis widens briefly on CEXs
  - Resolution: LP rebalancing and arbitrage flow restore pool

### FTX 2022

- **Shock onset:** 2022-11-06 ~20:00 UTC (CZ announces Binance will sell FTT)
- **Core shock window:** 2022-11-06 to 2022-11-14
- **Analysis window:** 2022-11-01 to 2022-11-30
- **Key milestones:**
  - T=0: CZ tweet and FTT sell announcement
  - T+3d: FTX halts withdrawals (~2022-11-08)
  - T+5d: FTX files for bankruptcy (2022-11-11)
  - Data limitation: FTX L2 unavailable post-collapse; focus on Binance/Coinbase/Kraken

### BUSD 2023

- **Shock onset:** 2023-02-13 (NYDFS instruction to Paxos to stop minting BUSD)
- **Core shock window:** 2023-02-13 to 2023-02-21
- **Analysis window:** 2023-02-06 to 2023-03-13
- **Key milestones:**
  - 2023-02-13: NYDFS order announced
  - Gradual: BUSD supply declines as users redeem or convert to USDT/USDC
  - No acute price shock; focus on flow channels and conversion dynamics

## Event-time alignment

All events are aligned on event time relative to shock onset (T=0). Both wall-clock
UTC and event-time frames are retained in the feature panel:

- `wall_clock_utc`: absolute UTC timestamp
- `event_time_seconds`: seconds relative to shock onset (negative = pre-event)
- `event_time_minutes`: convenience alias

## Notes on data availability by event

| Event | CEX L2 quality | DEX pool quality | On-chain flow quality |
|---|---|---|---|
| USDC/SVB 2023 | High (Coinbase, Binance) | High (Curve 3pool on-chain) | Medium (labelling) |
| Terra/LUNA 2022 | Medium (UST delisted) | High (Curve UST/3CRV) | High (LFG wallet known) |
| USDT/Curve 2023 | Medium | High (3pool, crvUSD) | Medium |
| FTX 2022 | Medium (no FTX L2) | Low | Medium |
| BUSD 2023 | High (Binance BUSD pairs) | Low | High (Paxos redemption) |
