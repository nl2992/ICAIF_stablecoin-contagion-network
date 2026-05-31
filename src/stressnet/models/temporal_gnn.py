"""Snapshot GCN with LSTM temporal aggregation for stress propagation prediction.

Architecture (SnapshotGCN):
    Per snapshot t:
        H^(0) = X_t                             node features [N, F]
        A_hat = D^{-1/2}(A+I)D^{-1/2}          symmetric-normalised adjacency
        H^(l) = ReLU(A_hat H^(l-1) W^(l))      GCN message passing
        z_t   = mean(H^(L))                     global mean pool  [hidden_dim]

    Temporal:
        h_T = LSTM(z_0, ..., z_T)[-1]          sequence → hidden state
        p   = sigmoid(MLP(h_T))                 stress probability

    Ablations tracked by build_tgn():
        - TGNClassifier (torch-geometric): upgrade path, disabled when pyg absent

Paper inclusion criterion (configs/paper.yaml → gnn_improvement_threshold):
    AUROC or AUPRC must exceed best non-graph baseline by > 5 pp.
    Shuffled-edge placebo must score lower than true-edge model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    logger.warning("PyTorch not installed; SnapshotGCN unavailable.")

try:
    from torch_geometric.nn import TGNMemory, TransformerConv
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

_GNN_IMPROVEMENT_THRESHOLD = 0.05   # 5 pp AUROC/AUPRC lift required


# ---------------------------------------------------------------------------
# SnapshotGCN (torch-only, no PyG required)
# ---------------------------------------------------------------------------

def build_gcn(
    node_feat_dim: int,
    hidden_dim: int = 64,
    n_gcn_layers: int = 2,
    dropout: float = 0.1,
) -> Any:
    """Build a SnapshotGCN + LSTM model.

    Returns None with a warning if PyTorch is not available.
    """
    if not _HAS_TORCH:
        logger.warning("SnapshotGCN requires torch; returning None.")
        return None

    class SnapshotGCN(nn.Module):
        """GCN over temporal snapshots with LSTM temporal aggregation."""

        def __init__(self) -> None:
            super().__init__()
            # GCN layers: input proj + n_gcn_layers hidden
            dims = [node_feat_dim] + [hidden_dim] * n_gcn_layers
            self.gcn_layers = nn.ModuleList(
                [nn.Linear(dims[i], dims[i + 1]) for i in range(n_gcn_layers)]
            )
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=False)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
            self.drop = nn.Dropout(dropout)

        # -----------------------------------------------------------------
        def _adj_hat(self, A: "torch.Tensor") -> "torch.Tensor":
            """Symmetric normalisation with self-loops: D^{-½}(A+I)D^{-½}."""
            N = A.size(0)
            A_tilde = A + torch.eye(N, device=A.device, dtype=A.dtype)
            deg = A_tilde.sum(dim=1)
            d_inv_sqrt = torch.where(
                deg > 0, deg.pow(-0.5), torch.zeros_like(deg)
            )
            D_hat = torch.diag(d_inv_sqrt)
            return D_hat @ A_tilde @ D_hat

        def _gcn_pass(
            self, X: "torch.Tensor", A_hat: "torch.Tensor"
        ) -> "torch.Tensor":
            """Apply all GCN layers to node feature matrix X."""
            H = X
            for layer in self.gcn_layers:
                H = F.relu(layer(A_hat @ H))
                H = self.drop(H)
            return H                  # [N, hidden_dim]

        # -----------------------------------------------------------------
        def forward(
            self,
            snapshots: list[tuple["torch.Tensor", "torch.Tensor"]],
        ) -> "torch.Tensor":
            """
            Args:
                snapshots: list of (X, A) per time step.
                    X: [N_t, node_feat_dim]  node features
                    A: [N_t, N_t]            adjacency matrix (weighted OK)

            Returns:
                Scalar prediction p ∈ (0, 1).
            """
            embeddings: list["torch.Tensor"] = []
            for X, A in snapshots:
                if X.shape[0] == 0:
                    continue
                A_hat = self._adj_hat(A)
                H = self._gcn_pass(X, A_hat)
                embeddings.append(H.mean(dim=0))      # global mean pool

            if not embeddings:
                return torch.zeros(1)

            # Sequence over time: [T, 1, hidden_dim]
            seq = torch.stack(embeddings).unsqueeze(1)
            _, (h_n, _) = self.lstm(seq)              # h_n: [1, 1, hidden_dim]
            return self.classifier(h_n.squeeze())     # scalar

    return SnapshotGCN()


# ---------------------------------------------------------------------------
# TGNClassifier (requires torch-geometric) — upgrade path
# ---------------------------------------------------------------------------

def build_tgn(
    node_feat_dim: int,
    edge_feat_dim: int,
    memory_dim: int = 64,
    time_dim: int = 16,
    n_heads: int = 4,
    dropout: float = 0.1,
) -> Any:
    """Build a Temporal Graph Network (TGN) classifier.

    Returns None with a warning if PyTorch/PyG is not available.
    Prefer build_gcn() for the primary model; this is an upgrade path.
    """
    if not _HAS_TORCH or not _HAS_PYG:
        logger.warning(
            "TGN requires torch and torch-geometric. "
            "Falling back to SnapshotGCN via build_gcn()."
        )
        return build_gcn(node_feat_dim)

    class TGNClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.memory = TGNMemory(
                num_nodes=100,
                raw_msg_dim=node_feat_dim + edge_feat_dim,
                memory_dim=memory_dim,
                time_dim=time_dim,
                message_module=None,
                aggregator_module=None,
            )
            self.conv = TransformerConv(
                in_channels=memory_dim,
                out_channels=memory_dim,
                heads=n_heads,
                dropout=dropout,
                edge_dim=edge_feat_dim,
            )
            self.classifier = nn.Sequential(
                nn.Linear(memory_dim * n_heads, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, batch: Any) -> Any:
            # TGN event batch: (src, dst, t, msg, edge_feat)
            src, dst, t, raw_msg, edge_attr, edge_index = batch
            z, _ = self.memory(src, dst, t, raw_msg)
            h = self.conv(z, edge_index, edge_attr)
            return self.classifier(h.mean(dim=0))

    return TGNClassifier()


# ---------------------------------------------------------------------------
# Snapshot construction from panel + edge tables
# ---------------------------------------------------------------------------

def _load_edges(results_dir: Path, event_id: str) -> pl.DataFrame | None:
    """Load TE or lead-lag edge estimates for the event."""
    for stem in (
        f"table_transfer_entropy_{event_id}",
        f"table_leadlag_tests_{event_id}",
    ):
        path = results_dir / "tables" / f"{stem}.csv"
        if path.exists():
            df = pl.read_csv(path)
            logger.info("Loaded edge estimates from %s (%d rows)", path.name, df.height)
            return df
    return None


def _panel_to_snapshots(
    panel: pl.DataFrame,
    edges: pl.DataFrame,
    feature_cols: list[str],
    window_s: float = 3600.0,
    step_s: float = 3600.0,
) -> list[tuple[Any, Any, int, list[int]]]:
    """Convert a flat gold panel + edge table into snapshot tensors.

    Returns list of (X_tensor, A_tensor, majority_label, row_labels) per window.
    row_labels contains per-row binary labels for row-level AUROC evaluation.
    """
    if not _HAS_TORCH:
        return []

    # Build node index
    all_nodes = sorted(panel["node_id"].unique().to_list())
    node_idx = {n: i for i, n in enumerate(all_nodes)}
    N = len(all_nodes)

    # Build adjacency matrix from edges (static for now)
    A_base = np.zeros((N, N), dtype=np.float32)
    for row in edges.iter_rows(named=True):
        src = row.get("node_i") or row.get("causing_node")
        tgt = row.get("node_j") or row.get("caused_node")
        weight = float(row.get("peak_corr") or row.get("te_i_to_j") or 1.0)
        if src in node_idx and tgt in node_idx:
            A_base[node_idx[src], node_idx[tgt]] = abs(weight)

    A_tensor = torch.from_numpy(A_base)

    # Resolve feature columns present in the panel
    feat_cols = [c for c in feature_cols if c in panel.columns]
    if not feat_cols:
        logger.warning("No feature columns found in panel for GNN snapshots.")
        return []

    label_col = "label_downstream_gt10bps_1m"
    has_label = label_col in panel.columns

    ts_col = "event_time_seconds"
    t_min = float(panel[ts_col].min())
    t_max = float(panel[ts_col].max())

    snapshots = []
    t = t_min
    while t < t_max:
        t_end = t + window_s
        win = panel.filter((pl.col(ts_col) >= t) & (pl.col(ts_col) < t_end))
        if win.height == 0:
            t += step_s
            continue

        # Build node feature matrix
        X = np.zeros((N, len(feat_cols)), dtype=np.float32)
        for row in (
            win.group_by("node_id")
            .agg([pl.col(c).mean() for c in feat_cols])
            .iter_rows(named=True)
        ):
            nid = row["node_id"]
            if nid in node_idx:
                i = node_idx[nid]
                for j, c in enumerate(feat_cols):
                    v = row.get(c)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        X[i, j] = float(v)

        X_tensor = torch.from_numpy(X)

        # Labels: per-row list + majority vote for seq2one training target
        if has_label:
            row_labels = (
                win.filter(pl.col(label_col).is_not_null())[label_col]
                .cast(pl.Int8)
                .to_list()
            )
            label = int(np.mean(row_labels) >= 0.5) if row_labels else 0
        else:
            row_labels = []
            label = 0

        snapshots.append((X_tensor, A_tensor, label, row_labels))
        t += step_s

    logger.info(
        "Built %d snapshots for event (N=%d nodes, F=%d features)",
        len(snapshots), N, len(feat_cols),
    )
    return snapshots


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

_FEATURE_COLS = [
    "basis_vs_usd", "spread_bps", "reserve_imbalance",
    "implied_pool_price", "pool_slippage_10k",
    "exchange_netflow_1h", "mint_burn_net_1h",
    "mid_price", "orderbook_imbalance",
]


def train_tgn_stub(
    model: Any = None,
    dataset_path: Any = None,
    event_id: str = "",
    architecture: str = "SnapshotGCN",
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 16,
    baseline_auc: float | None = None,
    output_dir: Any = None,
) -> dict[str, Any]:
    """Train a GNN on the temporal contagion graph for *event_id*.

    Data path priority:
        1. dataset_path (parquet): pre-built prediction dataset from script 09b
        2. gold panel: data/gold/dataset_contagion_features_{event_id}.parquet
        3. Fails with FileNotFoundError if neither exists.

    Edges are loaded from results/tables/table_transfer_entropy_{event_id}.csv
    or table_leadlag_tests_{event_id}.csv.

    Writes results/tables/table_gnn_metrics_{event_id}.csv.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required to train the GNN.")

    # ---- resolve output dir ----
    from stressnet.config import gold_root, results_root as _results_root
    out_dir = Path(output_dir) if output_dir else _results_root()

    # ---- load panel ----
    panel_path: Path | None = None
    if dataset_path is not None and Path(dataset_path).exists():
        panel_path = Path(dataset_path)
    else:
        candidate = gold_root() / f"dataset_contagion_features_{event_id}.parquet"
        if candidate.exists():
            panel_path = candidate

    if panel_path is None:
        raise FileNotFoundError(
            f"No panel found for event '{event_id}'. "
            "Run: make panel EVENT={event_id}  (or: make usdc)"
        )

    panel = pl.read_parquet(panel_path)
    uses_fixture = (
        "tier_actual" in panel.columns
        and "fixture_non_empirical" in panel["tier_actual"].unique().to_list()
    )
    if uses_fixture:
        logger.info(
            "Panel for %s includes fixture_non_empirical nodes. "
            "GNN will train on all nodes; result is NOT paper-claimable.",
            event_id,
        )

    # ---- load edges ----
    edges = _load_edges(_results_root(), event_id)
    if edges is None or edges.height == 0:
        logger.warning(
            "No edge estimates found for %s; using fully-connected graph.", event_id
        )
        nodes = panel["node_id"].unique().to_list()
        edges = pl.DataFrame(
            [{"node_i": a, "node_j": b, "peak_corr": 1.0}
             for a in nodes for b in nodes if a != b]
        )

    # ---- build snapshots ----
    snapshots = _panel_to_snapshots(
        panel, edges, _FEATURE_COLS,
        window_s=3600.0, step_s=3600.0,
    )
    if len(snapshots) < 4:
        raise RuntimeError(
            f"Too few snapshots ({len(snapshots)}) for event '{event_id}'. "
            "Need at least 4 hourly windows."
        )

    node_feat_dim = snapshots[0][0].shape[1]

    # ---- (re)build model if not provided ----
    if model is None:
        model = build_gcn(node_feat_dim=node_feat_dim)
    if model is None:
        raise RuntimeError("Could not build GNN model (torch unavailable).")

    # ---- temporal train/test split (80/20, no shuffling) ----
    n_train = max(1, int(len(snapshots) * 0.8))
    train_snaps = snapshots[:n_train]
    test_snaps = snapshots[n_train:]

    if not test_snaps:
        test_snaps = snapshots[-max(1, len(snapshots) // 5):]

    # ---- training loop ----
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    logger.info(
        "Training %s on %s: %d train / %d test snapshots, %d epochs",
        architecture, event_id, len(train_snaps), len(test_snaps), epochs,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        np.random.shuffle(train_snaps)   # type: ignore[arg-type]
        for X, A, label, _row_labels in train_snaps:
            optimizer.zero_grad()
            pred = model([(X, A)])
            target = torch.tensor([float(label)], dtype=torch.float32)
            loss = F.binary_cross_entropy(pred.view(1), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        if epoch % 10 == 0:
            logger.info(
                "Epoch %3d/%d  loss=%.4f", epoch, epochs, epoch_loss / len(train_snaps)
            )

    # ---- evaluation: expand predictions to row level for robust AUROC ----
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for X, A, _label, row_labels in test_snaps:
            pred = model([(X, A)])
            score = float(pred.view(1))
            if row_labels:
                # Each row in the snapshot gets the snapshot's predicted score
                y_true.extend(row_labels)
                y_score.extend([score] * len(row_labels))
            else:
                y_true.append(_label)
                y_score.append(score)

    from sklearn.metrics import roc_auc_score, average_precision_score
    auroc = float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan")
    auprc = float(average_precision_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan")

    # ---- improvement gate ----
    # Gate fails automatically if fixture data was used (not paper-claimable)
    passes_gate = (
        not uses_fixture
        and baseline_auc is not None
        and not np.isnan(auroc)
        and (auroc - baseline_auc) >= _GNN_IMPROVEMENT_THRESHOLD
    )

    results = {
        "event_id": event_id,
        "model": architecture,
        "status": "trained",
        "reason": "fixture_data" if uses_fixture else "",
        "AUROC": auroc,
        "AUPRC": auprc,
        "n_train_snapshots": len(train_snaps),
        "n_test_snapshots": len(test_snaps),
        "node_feat_dim": node_feat_dim,
        "epochs": epochs,
        "uses_fixture_data": uses_fixture,
        "passes_gnn_gate": passes_gate,
    }

    # ---- write metrics ----
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / f"table_gnn_metrics_{event_id}.csv"
    pl.DataFrame([results]).write_csv(out_path)
    logger.info(
        "GNN results for %s: AUROC=%.4f AUPRC=%.4f passes_gate=%s → %s",
        event_id, auroc, auprc, passes_gate, out_path,
    )
    return results
