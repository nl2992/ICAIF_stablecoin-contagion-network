# GNN Extension Plan

## Rule

No GNN before the gold panel, directed edge tables, and non-graph prediction
baselines are working on real data.

## Dataset

Create:

- `src/stressnet/models/gnn_dataset.py`
- `scripts/09c_build_gnn_dataset.py`

Expected artefacts:

- `node_id_to_idx.json`
- `edge_index.pt`
- `edge_features.pt`
- `node_features.pt`
- `timestamps.pt`
- `labels.pt`
- `splits.json`

## Models

Implement snapshot GraphSAGE or GCN first, then TGN.

Required ablations:

- True graph.
- Shuffled graph.
- No DEX nodes.
- No flow nodes.
- No edge features.
- Static graph.
- Temporal graph.

## Paper Inclusion Criterion

The graph model must improve AUROC or AUPRC by more than 5% over the best
non-graph baseline, and shuffled edges must perform worse than true edges.
Otherwise, report the GNN result as repo-only or negative evidence.
