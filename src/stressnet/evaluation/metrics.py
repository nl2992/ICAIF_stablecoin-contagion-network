"""Evaluation metrics for prediction and contagion estimation."""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
)


def compute_prediction_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "model",
    event_id: str | None = None,
) -> dict:
    """Compute AUROC, AUPRC, Brier score, and precision@k for a binary classifier."""
    if len(np.unique(y_true)) < 2:
        return {"model": model_name, "event_id": event_id, "error": "single_class"}

    auroc = float(roc_auc_score(y_true, y_proba))
    auprc = float(average_precision_score(y_true, y_proba))
    brier = float(brier_score_loss(y_true, y_proba))

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    best_f1 = float(np.max(f1_scores))

    return {
        "model": model_name,
        "event_id": event_id,
        "AUROC": auroc,
        "AUPRC": auprc,
        "Brier": brier,
        "best_F1": best_f1,
        "n_positive": int(y_true.sum()),
        "n_total": len(y_true),
        "prevalence": float(y_true.mean()),
    }


def gnn_improvement(
    baseline_metrics: pl.DataFrame,
    gnn_metrics: dict,
    metric: str = "AUROC",
    threshold: float = 0.05,
) -> dict:
    """Compute relative improvement of GNN over best baseline.

    Returns a dict with: best_baseline, baseline_score, gnn_score, improvement, meets_threshold.
    """
    best_row = baseline_metrics.sort(metric, descending=True).row(0, named=True)
    baseline_score = float(best_row[metric])
    gnn_score = float(gnn_metrics.get(metric, 0))
    improvement = gnn_score - baseline_score

    return {
        "best_baseline": best_row.get("model"),
        f"baseline_{metric}": baseline_score,
        f"gnn_{metric}": gnn_score,
        "improvement": improvement,
        "meets_threshold": improvement >= threshold,
    }
