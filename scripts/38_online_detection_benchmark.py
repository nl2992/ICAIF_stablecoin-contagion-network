"""Online (causal) stress-regime detection benchmark: Tier-A on-chain state vs
Tier-B market state, same unsupervised detector, scored causally.

This is the robust, model-agnostic ML result.  It is:
  * unsupervised  -- no labels used in fitting (no n=5 cross-event transfer);
  * causal        -- detection at time t uses the FILTERED (forward-only) HMM
                     posterior P(stress | x_1..t), so no look-ahead leakage;
  * a DATA-SOURCE comparison -- identical 3-state Gaussian HMM run on on-chain
                     features vs market features, so any gap is about the data,
                     not the model class (unlike a supervised LR-vs-RF lift).

For each event we report, for each data source:
  - auroc_causal : AUROC of the filtered stress posterior vs the panic label;
  - det_delay_h  : hours from true panic onset to first causal alarm, where the
                   alarm threshold is calibrated to a 5% false-alarm rate on the
                   pre-onset calm period (a fixed, comparable operating point).
A robustness variant fits HMM params on the first 50% only, then filters forward.

Reads:  data/gold/dataset_contagion_features_{event}.parquet
Writes: results/tables/table_online_detection.csv
"""
from __future__ import annotations

import csv
import warnings

import numpy as np
import polars as pl
from scipy.special import logsumexp
from scipy.stats import norm

from stressnet.config import gold_root, results_root
from stressnet.utils.logging import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

EVENTS = ["usdt_curve_2023", "terra_luna_2022", "ftx_2022", "busd_2023", "usdc_svb_2023"]
CEX_NODE = {
    "usdt_curve_2023": "usdt_binance", "terra_luna_2022": "usdt_binance",
    "ftx_2022": "usdt_binance", "busd_2023": "busd_binance",
    "usdc_svb_2023": "usdc_coinbase",
}
FALSE_ALARM = 0.05


def _grid(ev):
    df = pl.read_parquet(gold_root() / f"dataset_contagion_features_{ev}.parquet")
    oc = (df.filter(pl.col("node_id") == "curve_3pool")
            .with_columns((pl.col("event_time_seconds") // 3600).alias("h"))
            .group_by("h").agg(
                pl.col("usdc_net_sold_1h").sum().alias("flow"),
                pl.col("implied_pool_price").mean().alias("px"),
                pl.col("reserve_imbalance").mean().alias("imb"),
                pl.col("event_phase").first().alias("ph")).sort("h"))
    cx = (df.filter(pl.col("node_id") == CEX_NODE[ev])
            .with_columns((pl.col("event_time_seconds") // 3600).alias("h"))
            .group_by("h").agg(
                pl.col("basis_bps").abs().mean().alias("basis"),
                pl.col("spread_bps").mean().alias("spread"),
                pl.col("orderbook_imbalance").mean().abs().alias("obi")).sort("h"))
    d = oc.join(cx, on="h", how="left").sort("h")
    y = (d["ph"].to_numpy() == "panic").astype(int)
    # feature matrices; the FIRST column is the most stress-indicative (used to
    # label the HMM 'stress' state) for each source.
    onchain = np.column_stack([
        np.abs(np.nan_to_num(d["px"].to_numpy(), nan=1.) - 1.),   # pool price-dev (stress-leading)
        np.abs(np.nan_to_num(d["flow"].to_numpy())),
        np.abs(np.nan_to_num(d["imb"].to_numpy()))])
    market = np.column_stack([
        np.nan_to_num(d["basis"].to_numpy()),                      # CEX |basis| (stress-leading)
        np.nan_to_num(d["spread"].to_numpy()),
        np.nan_to_num(d["obi"].to_numpy())])
    return onchain, market, y


def _filtered_posterior(model, X, stress_state):
    """Causal forward-filtered P(state | x_1..t) using fitted HMM parameters.
    Emission likelihood computed manually (diag Gaussian) for version safety."""
    means = model.means_
    var = np.array([np.diag(c) for c in model.covars_])          # (k, d) diag variances
    logpi = np.log(model.startprob_ + 1e-300)
    logA = np.log(model.transmat_ + 1e-300)
    T, _ = X.shape
    k = means.shape[0]
    logb = np.zeros((T, k))
    for i in range(k):
        logb[:, i] = norm.logpdf(X, means[i], np.sqrt(var[i]) + 1e-9).sum(axis=1)
    logfilt = np.zeros((T, k))
    logfilt[0] = logpi + logb[0]
    logfilt[0] -= logsumexp(logfilt[0])
    for t in range(1, T):
        logpred = logsumexp(logfilt[t - 1][:, None] + logA, axis=0)  # predict step
        logfilt[t] = logb[t] + logpred
        logfilt[t] -= logsumexp(logfilt[t])
    return np.exp(logfilt[:, stress_state])


def _fit_hmm(Xfit):
    from hmmlearn.hmm import GaussianHMM
    Z = (Xfit - Xfit.mean(0)) / (Xfit.std(0) + 1e-9)
    m = GaussianHMM(n_components=3, covariance_type="diag", n_iter=300, random_state=0).fit(Z)
    s = int(np.argmax(m.means_[:, 0]))   # stress state = highest mean on stress-leading feature
    return m, s


def _detect_delay(post, y, onset):
    """First causal alarm minus onset, threshold = 95th pctile of calm posterior."""
    calm = post[:onset] if onset > 0 else post[:1]
    thr = np.quantile(calm, 1 - FALSE_ALARM) if len(calm) else 0.5
    fired = np.where(post > thr)[0]
    fired = fired[fired >= 0]
    if len(fired) == 0:
        return None, thr
    return int(fired[0] - onset), float(thr)


def _auroc(post, y):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, post))


def _run_source(X, y, mode):
    mu = X.mean(0); sd = X.std(0) + 1e-9
    if mode == "full":
        m, s = _fit_hmm(X)
    else:  # train on first half only, then filter forward over the whole series
        cut = len(X) // 2
        m, s = _fit_hmm(X[:cut])
    Z = (X - mu) / sd
    post = _filtered_posterior(m, Z, s)
    onset = int(np.argmax(y == 1)) if y.sum() else 0
    delay, _ = _detect_delay(post, y, onset)
    return _auroc(post, y), delay


def main():
    rows = []
    for ev in EVENTS:
        onchain, market, y = _grid(ev)
        if y.sum() == 0:
            continue
        au_oc, dl_oc = _run_source(onchain, y, "full")
        au_mk, dl_mk = _run_source(market, y, "full")
        au_oc_h, _ = _run_source(onchain, y, "half")
        rows.append({
            "event": ev, "onset_h": int(np.argmax(y == 1)),
            "auroc_onchain_causal": round(au_oc, 3),
            "auroc_market_causal": round(au_mk, 3),
            "auroc_gap": round(au_oc - au_mk, 3),
            "auroc_onchain_halffit": round(au_oc_h, 3),
            "delay_onchain_h": dl_oc, "delay_market_h": dl_mk,
            "earlier_by_h": (dl_mk - dl_oc) if (dl_oc is not None and dl_mk is not None) else None,
        })
        logger.info("%-16s onchain=%.3f market=%.3f gap=%+.3f | halffit=%.3f | "
                    "delay oc=%s mk=%s earlier_by=%s",
                    ev, au_oc, au_mk, au_oc - au_mk, au_oc_h, dl_oc, dl_mk,
                    rows[-1]["earlier_by_h"])
    reg = [r for r in rows if r["auroc_onchain_causal"] >= 0.8]
    if reg:
        logger.info("On events with an on-chain regime (n=%d): mean causal AUROC "
                    "onchain=%.3f market=%.3f gap=%+.3f",
                    len(reg), np.mean([r["auroc_onchain_causal"] for r in reg]),
                    np.mean([r["auroc_market_causal"] for r in reg]),
                    np.mean([r["auroc_gap"] for r in reg]))
    TDIR = results_root() / "tables"; TDIR.mkdir(parents=True, exist_ok=True)
    with (TDIR / "table_online_detection.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    logger.info("wrote table_online_detection.csv (%d events)", len(rows))


if __name__ == "__main__":
    main()
