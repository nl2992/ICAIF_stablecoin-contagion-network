# Paper Package README

## Title

**Cross-Protocol Stablecoin Stress Propagation: A Provenance-Aware AMM Network Analysis of Curve Finance and Uniswap v3**

---

## Abstract

Stablecoin stress episodes are usually measured as price de-pegs, but they are also liquidity-flow events. This project develops a provenance-aware network framework for studying how stress propagates across public CEX markets, on-chain AMM pools, and settlement-flow channels. The framework assigns node-level and feature-level evidence tiers, then filters every empirical edge through provenance and statistical gates before allowing paper-level claims. Across five stress events, the strongest paper-claimable A/A evidence appears in the USDT/Curve 2023 event, where Curve 3pool, Curve crvUSD/USDT, and Uniswap v3 USDC/USDT exhibit statistically supported AMM-flow linkages using Tier-A on-chain logs. The upgraded paper reports six A/A lead-lag rows: positive within-Curve co-movement and negative Curve--Uniswap counter-movement, plus a one-hour directional cross-protocol signal. A Tier-A prediction exercise shows that adding Uniswap v3 lags improves next-hour Curve stress AUROC from 0.677 to 0.711.

---

## Headline Result

During the USDT/Curve 2023 event, Tier-A Curve and Uniswap v3 AMM-flow logs produce six statistically supported A/A lead-lag rows. Within-Curve flow co-movement is positive (`rho = 0.386`), while Curve--Uniswap flow co-movement is negative (`rho = -0.486`), consistent with counter-flow routing during stress. Curve crvUSD/USDT leads Uniswap by one hour (`rho = -0.268`, FDR p < 0.001).

**What this does and does not claim:**
- **Does claim**: statistically supported AMM-flow linkage between Curve and Uniswap v3 pools during the USDT stress episode, using execution-grade on-chain data.
- **Does not claim**: structural causal identification. Lead-lag evidence shows directional timing correlation, not structural causality.
- **Does not claim**: CEX microstructure transmission. Historical full-depth CEX order books are not freely available; public Binance/Kraken data are Tier B.
- **Does not claim**: structural causality; the paper reports significant directed predictive dependence and lead-lag timing evidence.
- **Does not claim**: that Terra/LUNA, USDC/SVB, FTX, or BUSD events exhibit the same A/A paper-claimable cross-protocol pattern. Each provides either provenance-valid candidates (Terra, USDC/SVB) or A/B contextual evidence (FTX, BUSD).

---

## Data Tiers

| Tier | Definition | Examples |
|------|-----------|---------|
| **A** | Execution-grade on-chain logs; directly verifiable from the blockchain | Curve `TokenExchange` events, Uniswap v3 `Swap` events, USDC mint/burn Transfer events |
| **B** | Real public market data; useful for context but not execution-grade | Binance OHLCV, BBO/bookTicker; CoinMetrics exchange netflows |
| **Fixture** | Synthetic pipeline data for testing only; **never paper-claimable** | `usdc_kraken`, `usdt_mint_burn`, `eth_bridge_flows` |

An edge tier equals `min(tier_i, tier_j, feature_tier)`. The headline feature `usdc_net_sold_1h` is Tier A (direct on-chain sum). Derived proxies `reserve_imbalance` and `implied_pool_price` are Tier B.

---

## Claim-Gate Definition

Every result edge is filtered by three gates in sequence:

1. **Provenance gate** — blocks fixture/missing data; caps edge tier to the minimum of node and feature tiers. Sets `provenance_claim_allowed`.
2. **Statistical gate** — requires method-specific significance (Bonferroni, FDR-adjusted block-shuffle, Granger p-values). Sets `statistical_claim_allowed`.
3. **Paper gate** — `paper_claim_allowed = (provenance_claim_allowed AND statistical_claim_allowed)`.

Only `paper_claim_allowed = True` rows support directional paper claims. **Provenance-valid ≠ paper-claimable.**

---

## Table List

| Table | Description |
|-------|-------------|
| `results/paper/tables/table_cross_protocol_leadlag_usdt_curve_2023.csv` | **Headline**: Curve--Uniswap A/A cross-protocol lead-lag rows |
| `results/paper/tables/table_prediction_cross_protocol.csv` | Tier-A prediction results comparing Curve-only and cross-protocol features |
| `results/paper/tables/table_stress_propagation_score.csv` | Event-level stress propagation score |
| `results/paper/tables/table_aa_paper_claimable_edges.csv` | Legacy within-Curve A/A paper-claimable rows |
| `results/paper/tables/table_aa_provenance_valid_edges.csv` | All A/A provenance-valid candidates (may not pass statistical gate) |
| `results/paper/tables/table_ab_suggestive_edges.csv` | A/B suggestive paper-claimable edges |
| `results/paper/tables/table_claim_audit_summary.csv` | Per-event claim-gate counts (anti-cherry-pick audit) |
| `results/paper/tables/table_claim_gate_all_events.csv` | Full claim-gate audit summary with row counts |
| `results/paper/tables/table_feature_tiers.csv` | Feature-level provenance tiers from `configs/feature_tiers.yaml` |
| `results/paper/tables/table_provenance_inventory.csv` | Node provenance, coverage %, and claim ceiling |
| `results/paper/tables/table_excluded_self_loops.csv` | Diagnostic: VAR/FEVD self-loop rows excluded from edge tables |

---

## Figure List

| Figure | Filename | Description |
|--------|----------|-------------|
| 1 | `figure_01_multilayer_architecture.png` | Multi-layer architecture with claim gate |
| 2 | `figure_02_claim_gate_pipeline.png` | Three-gate pipeline diagram |
| 3 | `figure_03_claim_audit_by_event.png` | Anti-cherry-pick audit bars by event |
| 4 | `figure_04_usdt_curve_amm_flow_timeline.png` | USDT/Curve 2023 AMM-flow timeline |
| 5 | `figure_05_usdt_curve_leadlag_profile.png` | Lead-lag correlation profile (headline) |
| 6 | `figure_06_aa_paper_claimable_network.png` | A/A paper-claimable network |
| 7 | `figure_07_aa_provenance_vs_paper_claimable.png` | Provenance-valid vs paper-claimable comparison |
| 8 | `figure_08_terra_amm_flow_candidate.png` | Terra/LUNA A/A candidate (negative result) |
| 9 | `figure_09_usdc_svb_sparse_settlement_response.png` | USDC/SVB sparse settlement response |
| 10 | `figure_10_feature_tier_matrix.png` | Feature-tier matrix |
| 11 | `figure_11_node_provenance_coverage.png` | Node provenance coverage heatmap |
| 12 | `figure_12_full_paper_claimable_network.png` | Full paper-claimable network |

---

## Non-Claims

This paper package does **not** claim:

- Historical full-depth CEX order-book coverage (Binance, Kraken, Coinbase L2 data require paid vendor archives such as Tardis or Kaiko, or a live collector operating at the time of the event).
- Structural causal identification from correlation or lead-lag evidence alone.
- Tier-A status for derived Curve pool proxies (`reserve_imbalance`, `implied_pool_price`).
- Paper-claimable A/A evidence for Terra/LUNA 2022 (provenance-valid candidate, not statistically supported at the hourly grid).
- Paper-claimable evidence for the USDC/SVB 2023 sparse settlement-flow test (underpowered: 4 mint/burn arrivals, p = 1.0).
- Paper-claimable A/A evidence for FTX 2022 or BUSD 2023 (single Tier-A node in each event; A/B contextual evidence only).
- Paper evidence from fixture-generated data (all fixture rows are blocked by the provenance gate).

---

## Reproduction

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env
# Required: ETHERSCAN_API_KEY
# Optional: DUNE_API_KEY, THE_GRAPH_API_KEY

# 2. Run all 5 events empirically (real data only, no fixture)
make empirical_all

# 3. Build claim-gated paper outputs + figures + validation
make paper_gate

# 4. Validate the paper package
python scripts/14_validate_paper_package.py

# 5. Run all tests
python -m pytest tests -q
```

The `make paper_gate` command runs in this order:
1. `python scripts/00c_claim_gate.py --all-events --strict`
2. `python scripts/11d_make_claim_summary_tables.py`
3. `python scripts/99_make_paper_outputs.py --strict`
4. `python scripts/98_make_narrative_figures.py`
5. `python scripts/13_make_paper_figures.py`
6. `python scripts/14_validate_paper_package.py`

After a successful `make paper_gate`, the validation script prints `RESULT: PASS`.
