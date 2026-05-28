"""Temporal GNN stub for downstream stress prediction.

Full implementation requires PyTorch and PyTorch Geometric.
This stub defines the interface and training scaffold; the model layers
are implemented when torch-geometric is available.
"""

from __future__ import annotations

from typing import Any

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    logger.warning("PyTorch not installed; temporal GNN unavailable.")

try:
    from torch_geometric.nn import TGNMemory, TransformerConv
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


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
    """
    if not _HAS_TORCH or not _HAS_PYG:
        logger.warning("TGN requires torch and torch-geometric; returning None.")
        return None

    class TGNClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.memory = TGNMemory(
                num_nodes=100,          # will be reset at runtime
                raw_msg_dim=node_feat_dim + edge_feat_dim,
                memory_dim=memory_dim,
                time_dim=time_dim,
                message_module=None,    # use default
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
            raise NotImplementedError("Forward pass requires full TGN event batch.")

    return TGNClassifier()


def train_tgn_stub(
    model: Any = None,
    dataset_path: Any = None,
    event_id: str = "",
    architecture: str = "TGN",
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 200,
    baseline_auc: float | None = None,
    output_dir: Any = None,
) -> dict[str, Any]:
    """Training scaffold for the TGN.

    In the final implementation, this should:
    1. Convert temporal snapshots to TGN-compatible event streams.
    2. Train with event-based negative sampling.
    3. Evaluate on held-out event windows.

    Currently raises NotImplementedError as a reminder to implement after
    the non-graph baselines are validated.
    """
    raise NotImplementedError(
        "TGN training not yet implemented. "
        "Complete non-graph baselines (src/stressnet/models/baselines.py) first. "
        "Then implement temporal GNN here using the TGN architecture."
    )
