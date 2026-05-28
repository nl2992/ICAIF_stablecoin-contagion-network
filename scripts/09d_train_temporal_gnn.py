"""Train a temporal GNN on the contagion prediction dataset.

This script is a structured stub.  The non-graph baselines in script 09 must
be validated and their AUC/AUPRC numbers recorded before implementing this
script; the GNN's purpose is to demonstrate a >5% lift over the best non-graph
baseline (configs/paper.yaml: gnn_improvement_threshold).

Architecture options (configs/models.yaml):
    TGN   – Temporal Graph Network (Rossi et al., 2020)
    DySAT – Dynamic Self-Attention Network (Sankar et al., 2020)

Dependencies (optional, not in base requirements.txt):
    torch>=2.0
    torch-geometric>=2.3
    torch-geometric-temporal (for TGN/DySAT)

Run after:
    make panel EVENT=<event>
    make predict EVENT=<event>          # validates non-graph baselines
    python scripts/09b_make_prediction_dataset.py --event <event>

Writes (when fully implemented):
    results/tables/table_gnn_metrics_{event}.csv
    results/models/gnn_checkpoint_{event}.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stressnet.config import gold_root, results_root
from stressnet.models.temporal_gnn import build_tgn, train_tgn_stub
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train temporal GNN on contagion dataset.")
    parser.add_argument("--event", required=True)
    parser.add_argument(
        "--architecture",
        default="TGN",
        choices=["TGN", "DySAT"],
        help="GNN architecture (default: TGN).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument(
        "--baseline-auc",
        type=float,
        default=None,
        help="Best non-graph baseline AUROC from script 09. GNN must exceed this by >5%%.",
    )
    args = parser.parse_args()

    # Require the prediction dataset (temporal graph dataset built by script 09b)
    pred_path = gold_root() / f"dataset_prediction_{args.event}.parquet"
    if not pred_path.exists():
        logger.info(
            "GNN deferred: temporal graph dataset not yet built (run 09b first). "
            "Expected: %s", pred_path
        )
        raise SystemExit(0)

    # Check non-graph baselines exist first
    baseline_table = results_root() / "tables" / f"table_prediction_metrics_{args.event}.csv"
    if not baseline_table.exists():
        raise SystemExit(
            f"Non-graph baselines not found: {baseline_table}.\n"
            f"Run: python scripts/09_run_prediction.py --event {args.event}"
        )

    logger.info("Loading prediction dataset from %s", pred_path)

    # Build model (returns None if PyTorch/PyG unavailable)
    # TODO: node_feat_dim and edge_feat_dim should come from the prediction
    # dataset once the temporal graph dataset is implemented (script 09b).
    # Using placeholder dims until that dataset is available.
    model = build_tgn(node_feat_dim=16, edge_feat_dim=8)
    if model is None:
        raise SystemExit(
            "PyTorch / torch-geometric not installed.\n"
            "Install with: pip install torch torch-geometric torch-geometric-temporal\n"
            "Then re-run this script."
        )

    # Delegate to model stub — raises NotImplementedError until implemented
    try:
        train_tgn_stub(
            model=model,
            dataset_path=pred_path,
            event_id=args.event,
            architecture=args.architecture,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            baseline_auc=args.baseline_auc,
            output_dir=results_root(),
        )
    except NotImplementedError as exc:
        logger.error(
            "GNN training stub not yet implemented: %s\n"
            "Validate non-graph baselines first (script 09), then implement "
            "src/stressnet/models/temporal_gnn.py.",
            exc,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
