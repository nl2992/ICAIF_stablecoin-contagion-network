"""Plan B — Re-frame the supervised ML section as a regime-transfer diagnostic.

For each leave-one-episode-out (LOSO) cross-validation fold, compute:
  - Feature distribution shift between train and test episodes
    (symmetric KL divergence on Gaussian-fitted feature marginals)
  - Model performance (AUROC, precision-recall AUC) via logistic regression
  - Scatter: feature shift magnitude vs AUROC degradation

Reframes "prediction fails" as "concept shift is measurable and predicts failure".
A statistically significant correlation (r > 0.6 or p < 0.05) between distribution
shift and AUROC degradation confirms the shift-as-failure hypothesis.

Reads:  data/gold/dataset_contagion_features_{event}.parquet
Writes: results/tables/table_concept_shift.csv
        results/tables/table_shift_auroc_scatter.csv

Usage:
    python scripts/concept_shift_analysis.py
    python scripts/concept_shift_analysis.py --features amm_flow_features.parquet --labels episode_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import warnings

import numpy as np
import polars as pl
from scipy import stats

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
    X = np.column_stack([af, pxd, imb])
    y = (d["phase"].to_numpy() == "panic").astype(int)
    return X, y


def _sym_kl_gaussian(X_a: np.ndarray, X_b: np.ndarray) -> float:
    """Symmetric KL divergence between per-feature Gaussian fits (summed over features)."""
    eps = 1e-8
    kl_total = 0.0
    for j in range(X_a.shape[1]):
        mu_a, s_a = X_a[:, j].mean(), X_a[:, j].std() + eps
        mu_b, s_b = X_b[:, j].mean(), X_b[:, j].std() + eps
        # KL(N_a || N_b) + KL(N_b || N_a)
        kl_ab = np.log(s_b / s_a) + (s_a**2 + (mu_a - mu_b)**2) / (2 * s_b**2) - 0.5
        kl_ba = np.log(s_a / s_b) + (s_b**2 + (mu_b - mu_a)**2) / (2 * s_a**2) - 0.5
        kl_total += float(kl_ab + kl_ba)
    return kl_total


def _mmd_rbf(X_a: np.ndarray, X_b: np.ndarray, gamma: float = 1.0) -> float:
    """Unbiased maximum mean discrepancy with RBF kernel."""
    from sklearn.metrics.pairwise import rbf_kernel
    m, n = len(X_a), len(X_b)
    K_aa = rbf_kernel(X_a, X_a, gamma=gamma)
    K_bb = rbf_kernel(X_b, X_b, gamma=gamma)
    K_ab = rbf_kernel(X_a, X_b, gamma=gamma)
    mmd = (K_aa.sum() - np.diag(K_aa).sum()) / (m * (m - 1)) \
        + (K_bb.sum() - np.diag(K_bb).sum()) / (n * (n - 1)) \
        - 2.0 * K_ab.mean()
    return float(max(0.0, mmd))


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score

    datasets = {}
    for ev in _EVENTS:
        try:
            datasets[ev] = _load_features(ev)
        except Exception as exc:
            logger.warning("Skipping %s: %s", ev, exc)
    usable = [ev for ev in _EVENTS if ev in datasets and len(np.unique(datasets[ev][1])) >= 2]

    # ── Pairwise LOSO: train on A, test on B ──────────────────────────────────
    scatter_rows = []
    for train_ev in usable:
        X_tr, y_tr = datasets[train_ev]
        sc = StandardScaler().fit(X_tr)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
        clf.fit(sc.transform(X_tr), y_tr)
        for test_ev in usable:
            if train_ev == test_ev:
                continue
            X_te, y_te = datasets[test_ev]
            if len(np.unique(y_te)) < 2:
                continue
            proba = clf.predict_proba(sc.transform(X_te))[:, 1]
            auroc = float(roc_auc_score(y_te, proba))
            auprc = float(average_precision_score(y_te, proba))
            kl    = _sym_kl_gaussian(X_tr, X_te)
            mmd   = _mmd_rbf(sc.transform(X_tr), sc.transform(X_te))
            scatter_rows.append({
                "train_event":         train_ev,
                "test_event":          test_ev,
                "kl_divergence":       round(kl, 4),
                "mmd":                 round(mmd, 6),
                "auroc":               round(auroc, 4),
                "auprc":               round(auprc, 4),
                "auroc_degradation":   round(0.5 - auroc, 4),   # >0 means below chance
            })

    # ── Within-event AUROC (5-fold CV) as reference ──────────────────────────
    from sklearn.model_selection import StratifiedKFold
    within_auroc = {}
    for ev in usable:
        X, y = datasets[ev]
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        aucs = []
        for tr, te in skf.split(X, y):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
            clf.fit(sc.transform(X[tr]), y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
        within_auroc[ev] = float(np.mean(aucs)) if aucs else float("nan")

    # ── Correlation: KL divergence vs AUROC ──────────────────────────────────
    kl_vals   = np.array([r["kl_divergence"] for r in scatter_rows])
    auc_vals  = np.array([r["auroc"] for r in scatter_rows])
    r_kl_auc, p_kl_auc = stats.pearsonr(kl_vals, auc_vals)

    mmd_vals  = np.array([r["mmd"] for r in scatter_rows])
    r_mmd_auc, p_mmd_auc = stats.pearsonr(mmd_vals, auc_vals)

    # ── Summary table ─────────────────────────────────────────────────────────
    summary = {
        "n_train_test_pairs":         len(scatter_rows),
        "n_usable_events":            len(usable),
        "mean_within_event_auroc":    round(float(np.mean(list(within_auroc.values()))), 4),
        "mean_cross_event_auroc":     round(float(np.mean(auc_vals)), 4),
        "pearson_r_kl_auroc":         round(float(r_kl_auc), 4),
        "p_kl_auroc":                 round(float(p_kl_auc), 4),
        "pearson_r_mmd_auroc":        round(float(r_mmd_auc), 4),
        "p_mmd_auroc":                round(float(p_mmd_auc), 4),
        "shift_predicts_failure":     bool(p_kl_auc < 0.05),   # significant at 5%
    }

    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "table_concept_shift.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        w.writeheader(); w.writerow(summary)
    with (out_dir / "table_shift_auroc_scatter.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(scatter_rows[0].keys()))
        w.writeheader(); w.writerows(scatter_rows)

    logger.info("=== Concept shift vs AUROC degradation ===")
    logger.info("Within-event AUROC: %.3f  |  Cross-event AUROC: %.3f",
                summary["mean_within_event_auroc"], summary["mean_cross_event_auroc"])
    logger.info("KL-div vs AUROC: r=%.3f p=%.4f", r_kl_auc, p_kl_auc)
    logger.info("MMD     vs AUROC: r=%.3f p=%.4f", r_mmd_auc, p_mmd_auc)
    logger.info("Shift predicts failure: %s", summary["shift_predicts_failure"])
    for r in scatter_rows:
        logger.info("  %s->%s: KL=%.2f MMD=%.4f AUROC=%.3f",
                    r["train_event"][:10], r["test_event"][:10],
                    r["kl_divergence"], r["mmd"], r["auroc"])
    logger.info("Wrote table_concept_shift.csv and table_shift_auroc_scatter.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=None, help="(unused; reads gold parquets directly)")
    ap.add_argument("--labels",   default=None, help="(unused; labels in gold parquets)")
    _ = ap.parse_args()
    main()
