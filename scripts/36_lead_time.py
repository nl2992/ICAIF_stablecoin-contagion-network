"""F2: within-event lead time.  Does the on-chain stress signal light up BEFORE
the market (CEX basis) signal, relative to the labelled panic onset?

For each event we compare two label-free detectors on the hourly grid:
  - on-chain: posterior of an unsupervised 3-state Gaussian HMM on Curve-3pool
    pool state (|flow|, |price-dev|, |imbalance|), stress state crossing 0.5;
  - market:   the CEX |basis| crossing 10 bps (the standard price-based monitor).
Lead of each detector = (panic-onset hour) - (first detector-crossing hour);
positive = fires before the labelled onset.  on-chain minus market lead = how
many hours earlier the pool tells you, on average.  Descriptive (in-sample fit,
no out-of-sample claim).

Usage:  python scripts/36_lead_time.py
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
CEX_NODE = {
    "usdt_curve_2023": "usdt_binance", "terra_luna_2022": "usdt_binance",
    "ftx_2022": "usdt_binance", "busd_2023": "busd_binance",
    "usdc_svb_2023": "usdc_coinbase",
}


def _first_cross(series, thresh, sustain=2):
    """First index where series exceeds thresh for >= sustain consecutive steps."""
    above = series > thresh
    for i in range(len(above) - sustain + 1):
        if above[i:i + sustain].all():
            return i
    return None


def main():
    from hmmlearn.hmm import GaussianHMM
    rows = []
    for ev in EVENTS:
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
                .group_by("h").agg(pl.col("basis_bps").abs().mean().alias("basis")).sort("h"))
        d = oc.join(cx, on="h", how="left").sort("h")
        h = d["h"].to_numpy()
        y = (d["ph"].to_numpy() == "panic").astype(int)
        if y.sum() == 0:
            continue
        onset_idx = int(np.argmax(y == 1))            # first panic hour
        # on-chain HMM posterior (unsupervised)
        X = np.column_stack([np.abs(np.nan_to_num(d["flow"].to_numpy())),
                             np.abs(np.nan_to_num(d["px"].to_numpy(), nan=1.) - 1.),
                             np.abs(np.nan_to_num(d["imb"].to_numpy()))])
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        m = GaussianHMM(n_components=3, covariance_type="diag", n_iter=300, random_state=0).fit(X)
        post = m.predict_proba(X)[:, int(np.argmax(m.means_[:, 1]))]
        oc_cross = _first_cross(post, 0.5)
        basis = np.nan_to_num(d["basis"].to_numpy())
        mk_cross = _first_cross(basis, 10.0)
        oc_lead = (onset_idx - oc_cross) if oc_cross is not None else None
        mk_lead = (onset_idx - mk_cross) if mk_cross is not None else None
        rel = (oc_lead - mk_lead) if (oc_lead is not None and mk_lead is not None) else None
        rows.append({
            "event": ev, "onset_hour": onset_idx,
            "onchain_cross": oc_cross, "market_cross": mk_cross,
            "onchain_lead_h": oc_lead, "market_lead_h": mk_lead,
            "onchain_minus_market_h": rel,
        })
        logger.info("F2 %-16s onset=%d  onchain_cross=%s market_cross=%s  onchain_earlier_by=%s h",
                    ev, onset_idx, oc_cross, mk_cross, rel)
    TDIR = results_root() / "tables"; TDIR.mkdir(parents=True, exist_ok=True)
    with (TDIR / "grid_f2_lead_time.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    logger.info("wrote grid_f2_lead_time.csv (%d rows)", len(rows))


if __name__ == "__main__":
    main()
