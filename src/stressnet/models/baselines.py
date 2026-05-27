"""Non-graph baseline classifiers for downstream stress prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

_BASELINE_MODELS = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000, C=1.0),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
}
if _HAS_LGBM:
    _BASELINE_MODELS["LightGBM"] = lambda: LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
if _HAS_XGB:
    _BASELINE_MODELS["XGBoost"] = lambda: XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss")


def prepare_Xy(
    panel: pl.DataFrame,
    feature_cols: list[str],
    label_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and label vector from a panel DataFrame.

    Drops rows with any null feature or null label.
    """
    subset = panel.select(feature_cols + [label_col]).drop_nulls()
    X = subset.select(feature_cols).to_numpy().astype(np.float32)
    y = subset[label_col].cast(pl.Int8).to_numpy()
    return X, y


def run_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pl.DataFrame:
    """Train and evaluate all baseline classifiers.

    Returns a DataFrame with columns: model, AUROC, AUPRC, Brier.
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    rows = []
    for name, factory in _BASELINE_MODELS.items():
        model = factory()
        try:
            model.fit(X_tr, y_train)
            proba = model.predict_proba(X_te)[:, 1]
            rows.append({
                "model": name,
                "AUROC": float(roc_auc_score(y_test, proba)),
                "AUPRC": float(average_precision_score(y_test, proba)),
                "Brier": float(brier_score_loss(y_test, proba)),
            })
            logger.info("%s: AUROC=%.4f AUPRC=%.4f", name, rows[-1]["AUROC"], rows[-1]["AUPRC"])
        except Exception as exc:
            logger.warning("Baseline %s failed: %s", name, exc)

    return pl.DataFrame(rows).sort("AUROC", descending=True)
