"""One honest supervised test: do Tier-A on-chain pool-state features carry
incremental information about the stress regime BEYOND a Tier-B market-only
baseline, within event (cross-validated)?

This is the literal "informational value of on-chain data" question the paper's
provenance thesis raises -- posed as DETECTION (is the system in stress now?),
not cross-event PREDICTION (which concept shift defeats).  For each event we run
5-fold stratified CV logistic regression on the Curve-3pool hourly grid and
compare three feature sets on identical folds:

  - MARKET  (Tier-B): CEX |basis|, spread, order-book imbalance  (asof-joined)
  - ONCHAIN (Tier-A): pool |price-dev|, |reserve imbalance|, slippage, |net sold|
  - BOTH            : market + on-chain

A positive result is ONCHAIN/BOTH AUROC exceeding MARKET by a margin that holds
across the events with a genuine on-chain regime.  Paired across folds so the
lift is a within-fold difference, not a cross-run artefact.

Reads:  data/gold/dataset_contagion_features_{event}.parquet
Writes: results/tables/table_informational_value.csv
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

EVENTS = ["usdt_curve_2023", "terra_luna_2022", "ftx_2022", "busd_2023", "usdc_svb_2023"]
ONCHAIN_NODE = "curve_3pool"
CEX_NODE = {
    "usdt_curve_2023": "usdt_binance", "terra_luna_2022": "usdt_binance",
    "ftx_2022": "usdt_binance", "busd_2023": "busd_binance",
    "usdc_svb_2023": "usdc_coinbase",
}
MARKET = ["m_basis", "m_spread", "m_obi"]
ONCHAIN = ["oc_pxdev", "oc_imb", "oc_slip", "oc_flow"]


def _frame(ev: str) -> pl.DataFrame:
    df = pl.read_parquet(gold_root() / f"dataset_contagion_features_{ev}.parquet")
    oc = (df.filter(pl.col("node_id") == ONCHAIN_NODE)
            .with_columns((pl.col("event_time_seconds") // 3600).alias("h"))
            .group_by("h").agg(
                (pl.col("implied_pool_price").mean() - 1.0).abs().alias("oc_pxdev"),
                pl.col("reserve_imbalance").mean().abs().alias("oc_imb"),
                pl.col("pool_slippage_10k").mean().alias("oc_slip"),
                pl.col("usdc_net_sold_1h").sum().abs().alias("oc_flow"),
                pl.col("event_phase").first().alias("phase"),
                pl.col("wall_clock_utc").min().alias("t"))
            .sort("h"))
    cx = (df.filter(pl.col("node_id") == CEX_NODE[ev])
            .with_columns((pl.col("event_time_seconds") // 3600).alias("h"))
            .group_by("h").agg(
                pl.col("basis_bps").abs().mean().alias("m_basis"),
                pl.col("spread_bps").mean().alias("m_spread"),
                pl.col("orderbook_imbalance").mean().alias("m_obi"))
            .sort("h"))
    return oc.join(cx, on="h", how="left").sort("h")


def _cv_auc(X, y, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    out = []
    for tr, te in skf.split(X, y):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        out.append(roc_auc_score(y[te], m.predict_proba(sc.transform(X[te]))[:, 1]))
    return float(np.nanmean(out)) if out else float("nan")


N_SEEDS = 20


def main() -> None:
    rows = []
    for ev in EVENTS:
        d = _frame(ev)
        y = (d["phase"].to_numpy() == "panic").astype(int)
        if len(np.unique(y)) < 2:
            logger.warning("%s: single-class phase, skipping", ev); continue
        Xm = np.nan_to_num(d.select(MARKET).to_numpy().astype(float))
        Xo = np.nan_to_num(d.select(ONCHAIN).to_numpy().astype(float))
        Xb = np.column_stack([Xm, Xo])
        # seed-averaged AUROC (stability) + paired within-seed lift
        am = np.array([_cv_auc(Xm, y, s) for s in range(N_SEEDS)])
        ao = np.array([_cv_auc(Xo, y, s) for s in range(N_SEEDS)])
        ab = np.array([_cv_auc(Xb, y, s) for s in range(N_SEEDS)])
        lift = ab - am
        rows.append({
            "event_id": ev, "n_hours": int(d.height), "panic_prev": round(float(y.mean()), 3),
            "auroc_market": round(float(np.nanmean(am)), 3),
            "auroc_onchain": round(float(np.nanmean(ao)), 3),
            "auroc_both": round(float(np.nanmean(ab)), 3),
            "lift_both_minus_market": round(float(np.nanmean(lift)), 3),
            "lift_sd": round(float(np.nanstd(lift)), 4),
        })
        logger.info("%-16s market=%.3f onchain=%.3f both=%.3f  lift(+onchain)=%+.3f (sd %.3f, %d seeds)",
                    ev, np.nanmean(am), np.nanmean(ao), np.nanmean(ab), np.nanmean(lift), np.nanstd(lift), N_SEEDS)

    detect = [r for r in rows if r["auroc_onchain"] >= 0.8]
    if detect:
        logger.info("On events w/ on-chain regime (onchain AUROC>=0.8): mean market=%.3f mean both=%.3f mean lift=%+.3f",
                    np.mean([r["auroc_market"] for r in detect]),
                    np.mean([r["auroc_both"] for r in detect]),
                    np.mean([r["lift_both_minus_market"] for r in detect]))
    tdir = results_root() / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    with (tdir / "table_informational_value.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    logger.info("Wrote table_informational_value.csv (%d events)", len(rows))


if __name__ == "__main__":
    main()
