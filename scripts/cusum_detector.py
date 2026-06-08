"""Plan F (part 2) — CUSUM change-point detector as non-ML baseline.

Implements a standard CUSUM (Page's cumulative sum test) change-point
detector on the same AMM-flow features as the HMM (usdc_net_sold_1h,
|implied_pool_price - 1|, reserve_imbalance) for all 5 episodes.

Uses the CUSUM posterior score as a soft detector to compute AUROC,
enabling direct comparison with the HMM result (target: CUSUM < 0.80 vs
HMM 0.954 for Terra, confirming the HMM is not trivially beaten by a
simple statistical baseline).

CUSUM scores each timestep t as:
  S_t = max(0, S_{t-1} + (X_t - mu_0 - k))
where mu_0 is the pre-crisis mean, k = h/2 is the allowable slack, and h
is the decision threshold.  We normalise S_t to [0,1] for AUROC computation.

Reads:  data/gold/dataset_contagion_features_{event}.parquet
Writes: results/tables/table_cusum_baseline.csv

Usage:
    python scripts/cusum_detector.py
    python scripts/cusum_detector.py --episodes all
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
    X_raw = np.column_stack([af, pxd, imb])
    y = (d["phase"].to_numpy() == "panic").astype(int)
    return X_raw, y


def _cusum_score(x: np.ndarray, mu0: float, sigma0: float,
                 k_sigma: float = 0.5) -> np.ndarray:
    """One-sided upper CUSUM score (normalised to [0,1]).

    k_sigma: allowable slack as fraction of sigma0 (default 0.5 = half-sigma).
    """
    k = k_sigma * sigma0
    S = np.zeros(len(x))
    for t in range(1, len(x)):
        S[t] = max(0.0, S[t - 1] + (x[t] - mu0 - k))
    s_max = S.max()
    if s_max > 0:
        S = S / s_max
    return S


def _combined_cusum(X: np.ndarray, y: np.ndarray,
                    k_sigma: float = 0.5) -> np.ndarray:
    """Compute per-feature CUSUM scores, combine by taking the max across features."""
    # Estimate in-control parameters from the first 20% of data (pre-crisis)
    n_pre = max(10, int(0.2 * len(X)))
    scores = []
    for j in range(X.shape[1]):
        x = X[:, j]
        mu0    = x[:n_pre].mean()
        sigma0 = x[:n_pre].std() + 1e-9
        scores.append(_cusum_score(x, mu0, sigma0, k_sigma))
    # Combine: average normalised scores (uniform weighting)
    return np.stack(scores, axis=1).mean(axis=1)


def main(episodes: list[str] | None = None) -> None:
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    try:
        from hmmlearn.hmm import GaussianHMM
        has_hmm = True
    except ImportError:
        has_hmm = False

    event_list = episodes if episodes and episodes != ["all"] else _EVENTS
    rows = []
    for event_id in event_list:
        try:
            X, y = _load_features(event_id)
        except Exception as exc:
            logger.warning("%s: feature build failed (%s); skipping.", event_id, exc)
            continue
        if len(np.unique(y)) < 2 or len(y) < 30:
            logger.info("%s: insufficient regime variation; skipping.", event_id)
            continue

        # CUSUM score
        X_std = (X - X.mean(0)) / (X.std(0) + 1e-9)
        cusum = _combined_cusum(X_std, y)
        cusum_auroc = float(roc_auc_score(y, cusum))
        cusum_pred  = (cusum > 0.5).astype(int)
        cusum_bacc  = float(balanced_accuracy_score(y, cusum_pred))

        # HMM comparison (if available)
        hmm_auroc = None
        if has_hmm:
            from hmmlearn.hmm import GaussianHMM
            n = min(3, max(2, len(y) // 30))
            try:
                model  = GaussianHMM(n_components=n, covariance_type="diag",
                                     n_iter=300, random_state=0).fit(X_std)
                post   = model.predict_proba(X_std)
                stress = int(np.argmax(model.means_[:, 1]))
                hmm_auroc = float(roc_auc_score(y, post[:, stress]))
            except Exception:
                pass

        rows.append({
            "event_id":         event_id,
            "n_hours":          int(len(y)),
            "cusum_auroc":      round(cusum_auroc, 4),
            "cusum_balanced_acc": round(cusum_bacc, 4),
            "cusum_detects":    bool(cusum_auroc >= 0.80),
            "hmm_auroc":        round(hmm_auroc, 4) if hmm_auroc is not None else None,
            "hmm_vs_cusum_delta": round(
                (hmm_auroc - cusum_auroc), 4
            ) if hmm_auroc is not None else None,
            "hmm_beats_cusum":  bool(
                hmm_auroc is not None and hmm_auroc > cusum_auroc
            ),
        })
        logger.info(
            "%s: CUSUM AUROC=%.3f  HMM AUROC=%s  HMM beats CUSUM=%s",
            event_id, cusum_auroc,
            f"{hmm_auroc:.3f}" if hmm_auroc is not None else "n/a",
            hmm_auroc is not None and hmm_auroc > cusum_auroc,
        )

    if not rows:
        logger.warning("No CUSUM rows produced.")
        return

    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "table_cusum_baseline.csv"
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    mean_cusum = float(np.mean([r["cusum_auroc"] for r in rows]))
    mean_hmm   = float(np.mean([r["hmm_auroc"] for r in rows if r["hmm_auroc"] is not None]))
    n_beats    = sum(1 for r in rows if r["hmm_beats_cusum"])
    logger.info("Mean CUSUM AUROC=%.3f  Mean HMM AUROC=%.3f  HMM beats CUSUM in %d/%d events",
                mean_cusum, mean_hmm, n_beats, len(rows))
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", default=["all"])
    args = ap.parse_args()
    main(episodes=args.episodes)
