# Methodology

## 1. Observable state definition

Let $i$ index a node and $t$ denote event time. Define the **mid-price** as:

$$m_{i,t} = (a_{i,t} + b_{i,t}) / 2$$

For claims about propagation through tradable dislocations, use the **executable price**
from a book-walk or pool impact function for a fixed notional $q$:

$$p^{\text{exec}}_{i,t}(q) = \text{VWAP}(q, \text{book}_{i,t})$$

Define the **directional basis** between nodes $i$ and $j$ as:

$$\beta_{i \to j, t} = \log p^{\text{exec}}_{j,t}(q) - \log p^{\text{exec}}_{i,t}(q)$$

Pair this with liquidity state variables: spread, depth, imbalance, cancellation intensity,
and on-chain netflow.

## 2. Lead-lag analysis

Estimate whether stress at node $i$ systematically precedes stress at node $j$.

- Compute cross-correlation of basis, spread, and depth sequences at lags $\{-60, \ldots, 60\}$ seconds.
- Preferred grid: 1-second primary; 100 ms and 5 seconds as robustness.
- Handle asynchronous CEX/DEX feeds with Hayashi-Yoshida-style overlap correction.
- Report block-bootstrap 99% confidence intervals (block size: 5 minutes).

## 3. VAR / Granger causality

Estimate directed linear dependence among node states:

$$y_t = A_t y_{t-1} + B_t x_t + u_t, \qquad A_t = A_{t-1} + \eta_t$$

where $y_t$ stacks node-level basis, spread, depth, and imbalance, and $x_t$ contains
exogenous controls (gas fees, exchange status flags, bridge congestion).

Time-varying VAR preferred because node relationships shift across onset, panic, and
recovery phases. Report Forecast-Error Variance Decomposition (FEVD) spillover shares.

Success criterion: at least one key off-diagonal relation per major event with FEVD
spillover share > 10%.

## 4. Multivariate Hawkes processes

Let $N_i(t)$ count stress events at node $i$ (basis threshold exceedances, rapid depth
collapses, large swap imbalances, or cancellation spikes). The conditional intensity is:

$$\lambda_i(t) = \mu_i(t) + \sum_j \int_0^t \phi_{ij}(t-s) \, dN_j(s)$$

Off-diagonal kernels $\phi_{ij}$ measure directed excitation. The **branching ratio**:

$$n_{ij} = \int_0^\infty \phi_{ij}(u) \, du$$

is the main contagion statistic. Success criterion: $n_{ij} > 0.1$ with 95% CI excluding
zero for core edges.

Event definitions use two thresholds: $|\beta| > 10$ bps and $|\beta| > 50$ bps.

## 5. Transfer entropy

Estimates non-linear directional information flow:

$$TE_{i \to j} = I(Y_{j,t}; Y_{i,t-1:t-L} \mid Y_{j,t-1:t-L})$$

In plain terms: does node $i$'s past reduce uncertainty about node $j$'s future, beyond
what node $j$'s own past already explains?

Used as a robustness check when Granger-style linear dynamics understate non-linear,
threshold-type depeg propagation. Null distribution from $n=200$ time shuffles.

## 6. Temporal contagion network

Build a directed network where:
- **Nodes** are stablecoin-venue pairs, DEX pools, and bridge/flow channels.
- **Edge weights** are estimated from one of: VAR lag coefficients, Hawkes branching
  ratios, or transfer entropy values.

Compute:
- Weighted in-degree and out-degree (transmission and reception strength)
- Eigenvector centrality (systemic importance)
- Betweenness centrality (bridge / amplifier role)
- Community detection via Louvain/Leiden

Classify nodes as originators, amplifiers, or sinks based on in/out-degree asymmetry
and betweenness.

## 7. Predictive extension (temporal GNNs)

**Task:** Binary classification — will node $j$ exceed $|\beta| > 10$ bps (or 50 bps)
within the next minute, given the current graph state?

**Architecture:** TGN or DySAT-style temporal self-attention model.

Node features: basis, spread, depth, imbalance, pool skew, bridge netflows, gas, venue-status.
Edge features: fee wedge, settlement latency, chain connectivity, historical spillover strength.

**Baselines:** Logistic regression, random forest, XGBoost/LightGBM, LSTM.

**Evaluation split:** Event-based (never random). Out-of-event evaluation.

Success criterion: > 5% AUC/PR-AUC improvement over best non-graph baseline,
or a clearly documented informative null.

## 8. Causal identification

Causal claims rely on two strategies:

1. **Event studies** around plausibly exogenous shocks (FDIC announcement, Paxos order,
   bankruptcy filing). Focus on sharp discontinuities at T=0.

2. **Synthetic control / spillover-aware SCM** on treated nodes (e.g. USDC-Coinbase,
   BUSD-Binance) with donor pool of less-affected nodes.

**IV strategy:** Instrument settlement frictions with chain congestion (base fee spikes,
unusual confirmation delays) or venue-specific maintenance/withdrawal interruptions.

All causal claims are labelled as *partial* and *triangulated*, not definitive.
Model counterfactuals (setting edges or covariates to zero in the estimated model) are
distinguished from claims about literal market history.

## 9. Robustness checks

- Block bootstrap with 5-minute and 15-minute blocks
- Placebo events (matched high-volatility days, no stablecoin news)
- Subsample analysis (with/without Binance)
- Alternative sampling grids (100 ms, 1 s, 5 s, 60 s)
- Alternative basis definitions (mid vs. executable; 1 bps vs. 10 bps threshold)
- CEX-only vs. CEX+DEX network
- Alternative node definitions (including/excluding flow nodes)
