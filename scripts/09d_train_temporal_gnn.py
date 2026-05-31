"""Train a temporal GNN on the contagion feature panel.

Architecture options:
    SnapshotGCN – GCN over temporal snapshots + LSTM (torch-only, default)
    TGN         – Temporal Graph Network (requires torch-geometric)
    DySAT       – Dynamic Self-Attention Network (requires torch-geometric)

Run after:
    make panel EVENT=<event>
    make predict EVENT=<event>   # validates non-graph baselines

Writes:
    results/tables/table_gnn_metrics_{event}.csv
    results/models/gnn_checkpoint_{event}.pt   (if torch available)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from stressnet.config import gold_root, results_root
from stressnet.models.temporal_gnn import build_gcn, build_tgn, train_tgn_stub
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_ARCHITECTURES = ["SnapshotGCN", "TGN", "DySAT"]


def _write_deferred_status(event_id: str, reason: str, architecture: str) -> None:
    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"table_gnn_metrics_{event_id}.csv"
    pl.DataFrame(
        [
            {
                "event_id": event_id,
                "model": architecture,
                "status": "deferred",
                "reason": reason,
                "AUROC": None,
                "AUPRC": None,
                "passes_gnn_gate": False,
            }
        ]
    ).write_csv(out_path)
    logger.info("Wrote deferred GNN status: %s", out_path)


def _best_baseline_auc(event_id: str) -> float | None:
    """Read best AUROC from the non-graph baseline table."""
    path = results_root() / "tables" / f"table_prediction_metrics_{event_id}.csv"
    if not path.exists():
        return None
    df = pl.read_csv(path)
    if "AUROC" not in df.columns:
        return None
    # Use the full-feature, non-ablation best AUROC
    if "ablation" in df.columns:
        df = df.filter(pl.col("ablation") == "full")
    val = df["AUROC"].max()
    return float(val) if val is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train temporal GNN on contagion dataset.")
    parser.add_argument("--event", required=True)
    parser.add_argument(
        "--architecture",
        default="SnapshotGCN",
        choices=_ARCHITECTURES,
        help="GNN architecture (default: SnapshotGCN).",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--baseline-auc",
        type=float,
        default=None,
        help=(
            "Best non-graph baseline AUROC. If omitted, read from "
            "results/tables/table_prediction_metrics_{event}.csv."
        ),
    )
    args = parser.parse_args()

    # ---- check gold panel ----
    gold_panel = gold_root() / f"dataset_contagion_features_{args.event}.parquet"
    pred_dataset = gold_root() / f"dataset_prediction_{args.event}.parquet"

    if not gold_panel.exists() and not pred_dataset.exists():
        logger.error(
            "No gold panel found for %s. Run: make panel EVENT=%s",
            args.event, args.event,
        )
        _write_deferred_status(args.event, "gold_panel_missing", args.architecture)
        raise SystemExit(1)

    dataset_path = pred_dataset if pred_dataset.exists() else gold_panel

    # ---- resolve baseline AUC ----
    baseline_auc = args.baseline_auc or _best_baseline_auc(args.event)
    if baseline_auc is not None:
        logger.info("Non-graph baseline AUROC for gate check: %.4f", baseline_auc)
    else:
        logger.info(
            "No baseline AUROC found; run: make predict EVENT=%s  first.", args.event
        )

    # ---- build model ----
    if args.architecture == "SnapshotGCN":
        model = None  # train_tgn_stub builds it from node_feat_dim
    elif args.architecture in ("TGN", "DySAT"):
        model = build_tgn(node_feat_dim=16, edge_feat_dim=8)
        if model is None:
            logger.warning(
                "%s unavailable (requires torch-geometric). Falling back to SnapshotGCN.",
                args.architecture,
            )
    else:
        model = None

    # ---- train ----
    try:
        results = train_tgn_stub(
            model=model,
            dataset_path=dataset_path,
            event_id=args.event,
            architecture=args.architecture,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            baseline_auc=baseline_auc,
            output_dir=results_root(),
        )
        auroc = results.get("AUROC", float("nan"))
        passes = results.get("passes_gnn_gate", False)
        logger.info(
            "GNN training complete: AUROC=%.4f  passes_gate=%s",
            auroc, passes,
        )
        if passes:
            logger.info(
                "Gate PASSED: GNN improves over non-graph baseline by >=5pp. "
                "Include in paper."
            )
        elif baseline_auc is not None:
            logger.info(
                "Gate NOT passed (need +%.1f pp lift over %.4f). "
                "Report as repo-only or negative evidence.",
                5.0, baseline_auc,
            )

        # ---- save checkpoint ----
        try:
            import torch
            ckpt_dir = results_root() / "models"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"gnn_checkpoint_{args.event}.pt"
            if model is not None:
                torch.save(model.state_dict(), ckpt_path)
                logger.info("Saved checkpoint: %s", ckpt_path)
        except Exception as exc:
            logger.warning("Could not save checkpoint: %s", exc)

    except FileNotFoundError as exc:
        logger.error("Data not found: %s", exc)
        _write_deferred_status(args.event, "data_not_found", args.architecture)
        raise SystemExit(1) from exc
    except RuntimeError as exc:
        logger.error("Training failed: %s", exc)
        _write_deferred_status(args.event, str(exc)[:80], args.architecture)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
