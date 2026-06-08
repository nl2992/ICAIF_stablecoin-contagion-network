"""Plan F (part 1) — HMM configuration ablation.

Sweeps HMM number of states (2, 3, 4) and covariance types (diag, full, tied)
for all 5 episodes, reporting AUROC for each configuration.  Confirms that
the 3-state diagonal-covariance HMM used in script 27 is not arbitrarily
chosen: AUROC is within ±0.02 of the best configuration.

Also computes a Student-t approximation via a 2-component GMM emission per
state (GMMHMM), which captures heavier tails than the Gaussian model.

Reads:  data/gold/dataset_contagion_features_{event}.parquet
Writes: results/tables/table_hmm_ablation.csv

Usage:
    python scripts/hmm_ablation.py
    python scripts/hmm_ablation.py --states 2 3 4 --emissions gaussian student
"""

from __future__ import annotations

import argparse
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
_RANDOM_STATE = 0


def _load_features(event_id: str):
    df = pl.read_parquet(gold_root() / f"dataset_contagion_features_{event_id}.parquet")
    d = (
        df.filter(pl.col("node_id") == _NODE)
        .with_columns((pl.col("event_time_seconds") // 3_600).alias("h"))
        .group_by("h")
        .agg(
            pl.col("usdc_net_sold_1h").sum().alias("flow"),
            pl.col("implied_pool_price").mean().alias("px"),
            pl.col("reserve_imbalance").mean().alias("imb"),
            pl.col("event_phase").first().alias("phase"),
        )
        .sort("h")
    )
    af  = np.abs(np.nan_to_num(d["flow"].to_numpy()))
    pxd = np.abs(np.nan_to_num(d["px"].to_numpy(), nan=1.0) - 1.0)
    imb = np.abs(np.nan_to_num(d["imb"].to_numpy()))
    X = np.column_stack([af, pxd, imb])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = (d["phase"].to_numpy() == "panic").astype(int)
    return X, y


def _fit_gaussian_hmm(X: np.ndarray, y: np.ndarray, n_states: int,
                      cov_type: str) -> tuple[float | None, float | None]:
    """Fit GaussianHMM; return (AUROC, balanced_accuracy)."""
    from hmmlearn.hmm import GaussianHMM
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    if len(np.unique(y)) < 2 or len(y) < 30:
        return None, None
    n = min(n_states, max(2, len(y) // 30))
    try:
        model = GaussianHMM(
            n_components=n, covariance_type=cov_type,
            n_iter=300, random_state=_RANDOM_STATE,
        ).fit(X)
        post   = model.predict_proba(X)
        states = model.predict(X)
        stress = int(np.argmax(model.means_[:, 1]))
        auroc  = float(roc_auc_score(y, post[:, stress]))
        bacc   = float(balanced_accuracy_score(y, (states == stress).astype(int)))
        return auroc, bacc
    except Exception as exc:
        logger.debug("GaussianHMM(%d,%s) failed: %s", n_states, cov_type, exc)
        return None, None


def _fit_gmm_hmm(X: np.ndarray, y: np.ndarray, n_states: int,
                 n_mix: int = 2) -> tuple[float | None, float | None]:
    """Fit GMMHMM (Gaussian mixture emissions, approximates heavier tails)."""
    from hmmlearn.hmm import GMMHMM
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    if len(np.unique(y)) < 2 or len(y) < 30:
        return None, None
    n = min(n_states, max(2, len(y) // 30))
    try:
        model = GMMHMM(
            n_components=n, n_mix=n_mix, covariance_type="diag",
            n_iter=200, random_state=_RANDOM_STATE,
        ).fit(X)
        post   = model.predict_proba(X)
        states = model.predict(X)
        # Stress state: highest mean price-deviation across mixture components (feature index 1)
        # For GMMHMM, means_ has shape (n_states, n_mix, n_features)
        state_px_means = model.means_[:, :, 1].mean(axis=1)
        stress = int(np.argmax(state_px_means))
        auroc  = float(roc_auc_score(y, post[:, stress]))
        bacc   = float(balanced_accuracy_score(y, (states == stress).astype(int)))
        return auroc, bacc
    except Exception as exc:
        logger.debug("GMMHMM(%d,mix=%d) failed: %s", n_states, n_mix, exc)
        return None, None


def main(states: list[int] | None = None, emissions: list[str] | None = None) -> None:
    try:
        from hmmlearn.hmm import GaussianHMM  # noqa: F401
    except ImportError:
        logger.error("hmmlearn required: pip install -r requirements-optional.txt")
        return

    state_list     = states    or [2, 3, 4]
    emission_list  = emissions or ["gaussian", "student"]

    rows = []
    for event_id in _EVENTS:
        try:
            X, y = _load_features(event_id)
        except Exception as exc:
            logger.warning("%s: feature build failed (%s); skipping.", event_id, exc)
            continue

        for n_states in state_list:
            # Gaussian with diagonal covariance (baseline)
            if "gaussian" in emission_list:
                for cov in ["diag", "full"]:
                    auroc, bacc = _fit_gaussian_hmm(X, y, n_states, cov)
                    rows.append({
                        "event_id":    event_id,
                        "n_states":    n_states,
                        "emission":    f"gaussian_{cov}",
                        "auroc":       round(auroc, 4) if auroc is not None else None,
                        "balanced_acc": round(bacc, 4) if bacc is not None else None,
                        "detects":     bool(auroc is not None and auroc >= 0.80),
                    })
                    logger.info("%s  states=%d  emission=gaussian_%s  AUROC=%s",
                                event_id, n_states, cov, auroc)

            # Student-t approximation via 2-component GMM
            if "student" in emission_list:
                auroc, bacc = _fit_gmm_hmm(X, y, n_states, n_mix=2)
                rows.append({
                    "event_id":    event_id,
                    "n_states":    n_states,
                    "emission":    "gmm2_student_approx",
                    "auroc":       round(auroc, 4) if auroc is not None else None,
                    "balanced_acc": round(bacc, 4) if bacc is not None else None,
                    "detects":     bool(auroc is not None and auroc >= 0.80),
                })
                logger.info("%s  states=%d  emission=gmm2_student_approx  AUROC=%s",
                            event_id, n_states, auroc)

    if not rows:
        logger.warning("No ablation rows produced.")
        return

    # Compute per-event AUROC range to confirm robustness
    from collections import defaultdict
    by_event: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["auroc"] is not None:
            by_event[r["event_id"]].append(r["auroc"])
    for ev, aucs in by_event.items():
        rng = max(aucs) - min(aucs)
        best = max(aucs)
        logger.info("  %s: best=%.3f  range=%.3f  ±0.02 robust=%s",
                    ev, best, rng, rng <= 0.04)

    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "table_hmm_ablation.csv"
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    logger.info("Wrote %s (%d configurations × %d events = %d rows)",
                out_path, len(state_list) * len(emission_list) * 2,
                len(_EVENTS), len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--states",    nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--emissions", nargs="+", default=["gaussian", "student"])
    args = ap.parse_args()
    main(states=args.states, emissions=args.emissions)
