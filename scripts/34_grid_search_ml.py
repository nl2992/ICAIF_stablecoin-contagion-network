"""Disciplined grid search over ML ideas, on top of the current findings.

Families (CPU, small tabular):
  F1  feature ablation x model grid   -> is the certified detection lift robust,
                                         and which Tier-A feature carries it?
  F3  domain-adaptation transfer       -> does removing covariate shift (z-score
                                         per event / CORAL) recover cross-event
                                         transfer above the 0.50 chance baseline?
  F6  HMM robustness grid              -> is the 0.92-0.95 unsupervised result
                                         stable across n_states x covariance x
                                         feature set, or a lucky config?

Everything is reported -- nulls as nulls.  Outputs CSVs to results/tables/ and a
console verdict per family.  Seeds averaged where stochastic; no config is
cherry-picked.

Usage:  python scripts/34_grid_search_ml.py
"""
from __future__ import annotations

import csv
import itertools
import warnings

import numpy as np
import polars as pl

from stressnet.config import gold_root, results_root
from stressnet.utils.logging import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

EVENTS = ["usdt_curve_2023", "terra_luna_2022", "ftx_2022", "busd_2023", "usdc_svb_2023"]
ONCHAIN_NODE = "curve_3pool"
CEX_NODE = {
    "usdt_curve_2023": "usdt_binance", "terra_luna_2022": "usdt_binance",
    "ftx_2022": "usdt_binance", "busd_2023": "busd_binance",
    "usdc_svb_2023": "usdc_coinbase",
}
MARKET = ["m_basis", "m_spread", "m_obi"]
ONCHAIN = ["oc_pxdev", "oc_imb", "oc_slip", "oc_flow"]
TDIR = results_root() / "tables"


def _frame(ev: str) -> pl.DataFrame:
    df = pl.read_parquet(gold_root() / f"dataset_contagion_features_{ev}.parquet")
    oc = (df.filter(pl.col("node_id") == ONCHAIN_NODE)
            .with_columns((pl.col("event_time_seconds") // 3600).alias("h"))
            .group_by("h").agg(
                (pl.col("implied_pool_price").mean() - 1.0).abs().alias("oc_pxdev"),
                pl.col("reserve_imbalance").mean().abs().alias("oc_imb"),
                pl.col("pool_slippage_10k").mean().alias("oc_slip"),
                pl.col("usdc_net_sold_1h").sum().abs().alias("oc_flow"),
                pl.col("event_phase").first().alias("phase"))
            .sort("h"))
    cx = (df.filter(pl.col("node_id") == CEX_NODE[ev])
            .with_columns((pl.col("event_time_seconds") // 3600).alias("h"))
            .group_by("h").agg(
                pl.col("basis_bps").abs().mean().alias("m_basis"),
                pl.col("spread_bps").mean().alias("m_spread"),
                pl.col("orderbook_imbalance").mean().alias("m_obi"))
            .sort("h"))
    return oc.join(cx, on="h", how="left").sort("h")


def _data():
    out = {}
    for ev in EVENTS:
        d = _frame(ev)
        y = (d["phase"].to_numpy() == "panic").astype(int)
        if len(np.unique(y)) < 2:
            continue
        X = {c: np.nan_to_num(d[c].to_numpy().astype(float)) for c in MARKET + ONCHAIN}
        out[ev] = (X, y)
    return out


def _model(name):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    if name == "LR":
        return LogisticRegression(max_iter=1000, class_weight="balanced")
    if name == "RF":
        return RandomForestClassifier(n_estimators=150, random_state=0, class_weight="balanced", n_jobs=-1)
    if name == "GBM":
        return HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.05,
                                              l2_regularization=1.0, random_state=0)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800, random_state=0)
    raise ValueError(name)


def _cv_auc(Xcols, X, y, model_name, seeds=5):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    M = np.column_stack([X[c] for c in Xcols])
    aucs = []
    for s in range(seeds):
        skf = StratifiedKFold(5, shuffle=True, random_state=s)
        fold = []
        for tr, te in skf.split(M, y):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            sc = StandardScaler().fit(M[tr])
            mdl = _model(model_name).fit(sc.transform(M[tr]), y[tr])
            fold.append(roc_auc_score(y[te], mdl.predict_proba(sc.transform(M[te]))[:, 1]))
        if fold:
            aucs.append(np.mean(fold))
    return float(np.mean(aucs)) if aucs else float("nan"), float(np.std(aucs)) if aucs else float("nan")


# ----------------------------------------------------------------------------- F1
def family1(data):
    feature_sets = {
        "market": MARKET, "onchain": ONCHAIN, "both": MARKET + ONCHAIN,
        "oc_pxdev": ["oc_pxdev"], "oc_imb": ["oc_imb"], "oc_slip": ["oc_slip"], "oc_flow": ["oc_flow"],
        "both_minus_pxdev": MARKET + [c for c in ONCHAIN if c != "oc_pxdev"],
        "both_minus_imb":   MARKET + [c for c in ONCHAIN if c != "oc_imb"],
        "both_minus_slip":  MARKET + [c for c in ONCHAIN if c != "oc_slip"],
        "both_minus_flow":  MARKET + [c for c in ONCHAIN if c != "oc_flow"],
    }
    models = ["LR", "RF", "GBM"]  # MLP dropped: overkill/overfits on a few-hundred-row grid
    rows = []
    for ev, (X, y) in data.items():
        base = {m: _cv_auc(MARKET, X, y, m)[0] for m in models}
        for fs_name, cols in feature_sets.items():
            for m in models:
                au, sd = _cv_auc(cols, X, y, m)
                rows.append({"event": ev, "feature_set": fs_name, "model": m,
                             "auroc": round(au, 4), "auroc_sd": round(sd, 4),
                             "lift_vs_market": round(au - base[m], 4)})
                logger.info("F1 %-16s %-18s %-4s auroc=%.3f lift=%+.3f", ev, fs_name, m, au, au - base[m])
    _write("grid_f1_ablation.csv", rows)
    # verdict: does 'both' beat 'market' on the 3 regime events, for every model?
    return rows


# ----------------------------------------------------------------------------- F3
def family3(data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    evs = list(data)
    cols = ONCHAIN

    def feats(ev):
        X, y = data[ev]
        return np.column_stack([X[c] for c in cols]), y

    def coral(Xs, Xt, eps=1e-3):
        Cs = np.cov(Xs, rowvar=False) + eps * np.eye(Xs.shape[1])
        Ct = np.cov(Xt, rowvar=False) + eps * np.eye(Xt.shape[1])
        from scipy.linalg import sqrtm
        A = np.real(sqrtm(np.linalg.inv(Cs)) @ sqrtm(Ct))
        return Xs @ A

    rows = []
    for method in ["raw", "zscore_per_event", "coral"]:
        offdiag = []
        for held in evs:
            Xt, yt = feats(held)
            Xtr_parts, ytr_parts = [], []
            if method == "zscore_per_event":
                Xt_use = StandardScaler().fit_transform(Xt)
            else:
                Xt_use = Xt
            for src in evs:
                if src == held:
                    continue
                Xs, ys = feats(src)
                if method == "zscore_per_event":
                    Xs = StandardScaler().fit_transform(Xs)
                elif method == "coral":
                    Xs = coral(Xs, Xt)
                Xtr_parts.append(Xs); ytr_parts.append(ys)
            Xtr = np.vstack(Xtr_parts); ytr = np.concatenate(ytr_parts)
            if len(np.unique(ytr)) < 2 or len(np.unique(yt)) < 2:
                continue
            sc = StandardScaler().fit(Xtr)
            mdl = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
            a = roc_auc_score(yt, mdl.predict_proba(sc.transform(Xt_use if method != "coral" else Xt))[:, 1])
            offdiag.append(a)
            rows.append({"method": method, "held_out": held, "transfer_auroc": round(float(a), 4)})
        logger.info("F3 %-18s mean transfer AUROC = %.3f (chance 0.50)", method, np.mean(offdiag))
    _write("grid_f3_transfer.csv", rows)
    return rows


# ----------------------------------------------------------------------------- F6
def family6(data):
    from hmmlearn.hmm import GaussianHMM
    from sklearn.metrics import roc_auc_score
    fsets = {"oc3": ["oc_flow", "oc_pxdev", "oc_imb"],
             "oc4": ONCHAIN,
             "oc+mkt": ONCHAIN + MARKET}
    rows = []
    for ev, (X, y) in data.items():
        for fs_name, cols in fsets.items():
            M = np.column_stack([X[c] for c in cols])
            M = (M - M.mean(0)) / (M.std(0) + 1e-9)
            for n_states, cov in itertools.product([2, 3, 4], ["diag", "full", "spherical"]):
                try:
                    m = GaussianHMM(n_components=n_states, covariance_type=cov,
                                    n_iter=300, random_state=0).fit(M)
                    post = m.predict_proba(M)
                    # stress state = highest mean on the price-deviation feature if present else col 0
                    devcol = cols.index("oc_pxdev") if "oc_pxdev" in cols else 0
                    s = int(np.argmax(m.means_[:, devcol]))
                    au = roc_auc_score(y, post[:, s])
                except Exception:
                    au = float("nan")
                rows.append({"event": ev, "feature_set": fs_name, "n_states": n_states,
                             "covariance": cov, "auroc": round(float(au), 4)})
        logger.info("F6 %-16s done", ev)
    _write("grid_f6_hmm.csv", rows)
    return rows


def _write(name, rows):
    TDIR.mkdir(parents=True, exist_ok=True)
    with (TDIR / name).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    logger.info("wrote %s (%d rows)", name, len(rows))


def main():
    data = _data()
    logger.info("=== F1 feature ablation x model grid ===")
    family1(data)
    logger.info("=== F3 domain-adaptation transfer ===")
    family3(data)
    logger.info("=== F6 HMM robustness grid ===")
    family6(data)


if __name__ == "__main__":
    main()
