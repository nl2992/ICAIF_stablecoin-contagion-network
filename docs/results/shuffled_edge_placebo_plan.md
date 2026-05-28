# Shuffled-Edge Placebo Plan

This is the required GNN placebo design. It must be implemented before any GNN
result is considered paper-worthy.

## Objective

Test whether graph structure adds predictive information beyond node features
and class imbalance. A true-edge GNN should outperform the same architecture
trained on shuffled edges.

## Inputs

- Real-node-only gold panels.
- Prediction labels with nonzero train and test prevalence.
- Baseline edge table from lead-lag, VAR/FEVD, TE, or ensemble graph.
- Chronological or leave-one-event-out splits.

## Procedure

1. Build the true graph from real-node-only directed edges.
2. Freeze the node feature matrix, timestamps, labels, train/test split, and
   class weights.
3. Generate at least 20 shuffled-edge graphs:
   - Preserve number of edges.
   - Preserve edge-weight distribution.
   - Prefer preserving in/out-degree sequence when possible.
   - Disallow self-loops unless the true graph includes them.
4. Train the same GNN architecture and hyperparameters on:
   - True graph.
   - Shuffled graph replicas.
   - No-edge or identity-edge baseline.
5. Report AUROC, AUPRC, Brier, and lift over prevalence.

## Passing Criterion

The GNN is paper-eligible only if:

- True-edge GNN beats the best non-graph baseline by more than 5% in AUROC or
  AUPRC.
- True-edge GNN beats the median shuffled-edge GNN.
- The shuffled-edge 95% interval does not contain the true-edge score, or the
  true-edge score is at least practically separated and stable across events.

## Failure Interpretation

If the true graph does not beat shuffled edges, the correct result is:

> The graph architecture did not add robust predictive information beyond node
> features and class imbalance controls.

Do not present that as evidence of contagion.
