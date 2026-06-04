"""Unsupervised stress-regime detection from on-chain pool state via an HMM.

The supervised, cross-event prediction tasks in this project fail because five
events are far too few for cross-event generalisation (Section: ML benchmark).
The correct AI framing for a few-event regime problem is unsupervised,
per-event, online latent-state detection: can a model recover the stress
regime from on-chain data alone, with no labels?

We fit a Gaussian Hidden Markov Model to standardized on-chain pool-state
features for Curve 3pool --- |usdc_net_sold_1h| (flow intensity),
|implied_pool_price - 1| (price deviation from peg), and |reserve_imbalance| ---
with NO access to the event_phase labels.  The latent state with the highest
mean price-deviation is taken as the "stress" state.  We then evaluate, post
hoc, how well the inferred stress-state posterior recovers the true ``panic``
regime (AUROC, balanced accuracy).

Reads:  data/gold/dataset_contagion_features_{event}.parquet
Writes: results/tables/table_hmm_regime.csv

Requires: hmmlearn (see requirements-optional.txt)

Usage:
    python scripts/27_run_hmm_regime.py
"""

from __future__ import annotations

import csv
import warnings

import numpy as np
import polars as pl

from stressnet.config import gold_root, results_root
from stressnet.utils.logging import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

_EVENTS = [
    "usdt_curve_2023", "terra_luna_2022", "ftx_2022", "busd_2023", "usdc_svb_2023",
]
_NODE = "curve_3pool"
_N_STATES = 3        # calm / onset / stress
_RANDOM_STATE = 0


def _features(event_id: str):
    df = pl.read_parquet(gold_root() / f"dataset_contagion_features_{event_id}.parquet")
    d = (
        df.filter(pl.col("node_id") == _NODE)
        .with_columns((pl.col("event_time_seconds") // 3600).alias("h"))
        .group_by("h")
        .agg(
            pl.col("usdc_net_sold_1h").sum().alias("flow"),
            pl.col("implied_pool_price").mean().alias("px"),
            pl.col("reserve_imbalance").mean().alias("imb"),
            pl.col("event_phase").first().alias("phase"),
        )
        .sort("h")
    )
    af   = np.abs(np.nan_to_num(d["flow"].to_numpy()))
    pxd  = np.abs(np.nan_to_num(d["px"].to_numpy(), nan=1.0) - 1.0)
    imb  = np.abs(np.nan_to_num(d["imb"].to_numpy()))
    X = np.column_stack([af, pxd, imb])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = (d["phase"].to_numpy() == "panic").astype(int)
    return X, y


def main() -> None:
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        logger.error(
            "hmmlearn is required.  Install with: pip install -r requirements-optional.txt"
        )
        return
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    rows = []
    for event_id in _EVENTS:
        try:
            X, y = _features(event_id)
        except Exception as exc:
            logger.warning("%s: feature build failed (%s); skipping.", event_id, exc)
            continue
        if len(np.unique(y)) < 2 or len(y) < 30:
            logger.info("%s: insufficient regime variation; skipping.", event_id)
            continue

        n_states = min(_N_STATES, max(2, len(y) // 30))
        model = GaussianHMM(
            n_components=n_states, covariance_type="diag",
            n_iter=300, random_state=_RANDOM_STATE,
        ).fit(X)
        post = model.predict_proba(X)
        states = model.predict(X)
        # Stress state = latent state with the highest mean price-deviation feature.
        stress = int(np.argmax(model.means_[:, 1]))
        p_stress = post[:, stress]

        auroc = float(roc_auc_score(y, p_stress))
        bal_acc = float(balanced_accuracy_score(y, (states == stress).astype(int)))
        stress_prev = float((states == stress).mean())

        rows.append({
            "event_id":        event_id,
            "n_hours":         int(len(y)),
            "n_states":        int(n_states),
            "auroc":           round(auroc, 4),
            "balanced_acc":    round(bal_acc, 4),
            "stress_state_prevalence": round(stress_prev, 4),
            "detects_regime":  bool(auroc >= 0.80),
        })
        logger.info(
            "%s: HMM(%d states) unsupervised  AUROC=%.3f  bal_acc=%.3f  detects=%s",
            event_id, n_states, auroc, bal_acc, auroc >= 0.80,
        )

    if not rows:
        logger.warning("No HMM rows produced.")
        return
    out = results_root() / "tables" / "table_hmm_regime.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    n_det = sum(1 for r in rows if r["detects_regime"])
    mean_auc = float(np.mean([r["auroc"] for r in rows]))
    logger.info(
        "Wrote %s (%d events; %d detect regime at AUROC>=0.80; mean AUROC=%.3f)",
        out, len(rows), n_det, mean_auc,
    )


if __name__ == "__main__":
    main()
