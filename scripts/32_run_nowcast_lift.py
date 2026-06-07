"""Honest within-event nowcast: does current Tier-A on-chain pool state predict
NEAR-FUTURE CEX price deviation beyond a price/market-only baseline?

Motivation.  Cross-event supervised prediction fails under concept shift (n=5).
But our price-discovery result shows the Curve pool *leads* the CEX for
DeFi-native stress.  Expressed as a supervised task, that lead should appear as
incremental predictive lift from on-chain features -- and, if the mechanism is
real, it should appear *only* for the endogenous (pool-led) episode and vanish
for exogenous, CEX-led ones.  This script tests exactly that.

Task.  On each event's CEX timeline (minute resolution, carrying the
forward-looking labels), predict label_downstream_gt{TH}bps_{H} at horizon H.
  - BASELINE (Tier-B, price/market only): current + rolling CEX basis,
    spread, order-book imbalance, depth.  What a price-based monitor sees.
  - +ON-CHAIN (Tier-A): the baseline PLUS current Curve-3pool state
    (reserve imbalance, implied-price deviation, pool slippage, net USDC sold),
    asof-joined BACKWARD onto the CEX clock so only past on-chain info is used.
Split is temporal (first 70%% train, last 30%% test) -- no shuffling, so the
test set is strictly in the future of the training set (causal, leakage-free).

Reads:  data/gold/dataset_contagion_features_{event}.parquet
Writes: results/tables/table_nowcast_lift.csv

Usage:
    python scripts/32_run_nowcast_lift.py
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

# CEX node that carries the forward-looking labels for each event
CEX_NODE = {
    "usdt_curve_2023": "usdt_binance",
    "terra_luna_2022": "usdt_binance",
    "ftx_2022":        "usdt_binance",
    "busd_2023":       "busd_binance",
    "usdc_svb_2023":   "usdc_coinbase",
}
ONCHAIN_NODE = "curve_3pool"
TRAIN_FRAC = 0.70
# Horizon set to the lead measured by the price-discovery test (pool leads CEX
# by ~1h for DeFi-native stress), NOT chosen to maximise lift.  Target: will the
# CEX |basis| exceed THRESH_BPS at any point within the next HORIZON_MIN minutes?
HORIZON_MIN = 60
THRESH_BPS = 10.0


def _cex_frame(df: pl.DataFrame, node: str) -> pl.DataFrame:
    d = df.filter(pl.col("node_id") == node).sort("wall_clock_utc")
    d = d.with_columns([
        pl.col("basis_bps").abs().alias("abs_basis"),
        (pl.col("depth_10bps_bid_usd") + pl.col("depth_10bps_ask_usd")).log1p().alias("log_depth"),
    ])
    # rolling market-state features (all past-only)
    d = d.with_columns([
        pl.col("abs_basis").rolling_mean(15, min_periods=1).alias("abs_basis_ma15"),
        pl.col("abs_basis").rolling_std(15, min_periods=2).fill_null(0).alias("abs_basis_sd15"),
        pl.col("abs_basis").shift(5).fill_null(0).alias("abs_basis_lag5"),
        pl.col("abs_basis").diff().fill_null(0).alias("abs_basis_chg"),
    ])
    # FORWARD label: does |basis| exceed THRESH_BPS within the next HORIZON_MIN
    # minutes?  Built by a forward-looking rolling max, then dropping the tail
    # rows whose forward window is truncated.
    rev = d.select("abs_basis").reverse()
    fwd_max = rev.select(
        pl.col("abs_basis").rolling_max(HORIZON_MIN, min_periods=1)
    ).reverse().to_series()
    d = d.with_columns((fwd_max.shift(-1) > THRESH_BPS).cast(pl.Int8).alias("y_fwd"))
    # drop last HORIZON_MIN rows (incomplete forward window)
    if d.height > HORIZON_MIN:
        d = d[: d.height - HORIZON_MIN]
    return d


def _onchain_frame(df: pl.DataFrame) -> pl.DataFrame:
    d = df.filter(pl.col("node_id") == ONCHAIN_NODE).sort("wall_clock_utc")
    d = d.with_columns([
        (pl.col("implied_pool_price") - 1.0).abs().alias("oc_pxdev"),
        pl.col("reserve_imbalance").abs().alias("oc_imb"),
        pl.col("pool_slippage_10k").alias("oc_slip"),
        pl.col("usdc_net_sold_1h").abs().alias("oc_flow"),
    ])
    return d.select(["wall_clock_utc", "oc_pxdev", "oc_imb", "oc_slip", "oc_flow"])


BASE_FEATS = ["abs_basis", "spread_bps", "orderbook_imbalance", "log_depth",
              "abs_basis_ma15", "abs_basis_sd15", "abs_basis_lag5", "abs_basis_chg"]
OC_FEATS = ["oc_pxdev", "oc_imb", "oc_slip", "oc_flow"]


def _xy(frame: pl.DataFrame, feats: list[str]):
    X = frame.select(feats).to_numpy().astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = frame["y_fwd"].to_numpy().astype(float)
    return X, y


def _fit_eval(Xtr, ytr, Xte, yte):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
    m = HistGradientBoostingClassifier(
        max_depth=3, max_iter=300, learning_rate=0.05,
        l2_regularization=1.0, random_state=0)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, p)), float(average_precision_score(yte, p))


def main() -> None:
    rows = []
    for ev, node in CEX_NODE.items():
        df = pl.read_parquet(gold_root() / f"dataset_contagion_features_{ev}.parquet")
        cex = _cex_frame(df, node)
        oc = _onchain_frame(df)
        # asof BACKWARD: each CEX row gets the most recent *past* on-chain state
        merged = cex.join_asof(oc, on="wall_clock_utc", strategy="backward").sort("wall_clock_utc")
        merged = merged.filter(pl.col("y_fwd").is_not_null())
        n = merged.height
        cut = int(n * TRAIN_FRAC)
        tr, te = merged[:cut], merged[cut:]
        ytr_all = tr["y_fwd"].to_numpy()
        yte_all = te["y_fwd"].to_numpy()
        if len(np.unique(ytr_all)) < 2 or len(np.unique(yte_all)) < 2:
            logger.warning("%s: degenerate split, skipping", ev); continue

        Xtr_b, ytr = _xy(tr, BASE_FEATS); Xte_b, yte = _xy(te, BASE_FEATS)
        Xtr_o, _ = _xy(tr, BASE_FEATS + OC_FEATS); Xte_o, _ = _xy(te, BASE_FEATS + OC_FEATS)

        au_b, ap_b = _fit_eval(Xtr_b, ytr, Xte_b, yte)
        au_o, ap_o = _fit_eval(Xtr_o, ytr, Xte_o, yte)
        rows.append({
            "event_id": ev,
            "n_test": int(te.height),
            "test_prev": round(float(yte.mean()), 4),
            "auroc_baseline": round(au_b, 4),
            "auroc_onchain": round(au_o, 4),
            "auroc_lift": round(au_o - au_b, 4),
            "ap_baseline": round(ap_b, 4),
            "ap_onchain": round(ap_o, 4),
            "ap_lift": round(ap_o - ap_b, 4),
        })
        logger.info("%-16s  AUROC base=%.3f +onchain=%.3f  lift=%+.3f | AP base=%.3f +onchain=%.3f lift=%+.3f",
                    ev, au_b, au_o, au_o - au_b, ap_b, ap_o, ap_o - ap_b)

    tdir = results_root() / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    with (tdir / "table_nowcast_lift.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    logger.info("Wrote table_nowcast_lift.csv (%d events)", len(rows))


if __name__ == "__main__":
    main()
