"""On-chain vs CEX price discovery during stablecoin stress.

Positive finding: for DeFi-native stress the Curve pool price moves *before*
the centralized-exchange price (price discovery happens on-chain first), while
for an exogenous fiat-banking shock the CEX leads.  The direction of price
discovery is therefore informative about where the shock originates, and the
magnitudes show how much price-only monitoring misses.

Method (trend-robust).  Per event we take the on-chain pool price deviation
|implied_pool_price - 1| (Curve 3pool) and the matched CEX price-deviation
|basis_vs_usd|, first-difference both (removing the shared stress trend that
inflates naive level correlations), and compare the average lead-lag
cross-correlation at on-chain-leads (k>0) vs CEX-leads (k<0) lags.  We also
report each venue's peak deviation magnitude.

Reads:  data/gold/dataset_contagion_features_{event}.parquet
Writes: results/tables/table_price_discovery.csv

Usage:
    python scripts/26_run_price_discovery.py
"""

from __future__ import annotations

import csv

import numpy as np
import polars as pl
from scipy import stats

from stressnet.config import gold_root, results_root
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_EVENT_CEX: dict[str, str] = {
    "usdt_curve_2023": "usdt_binance",
    "terra_luna_2022": "usdt_binance",
    "ftx_2022":        "usdt_binance",
    "busd_2023":       "busd_binance",
    "usdc_svb_2023":   "usdc_binance",
}

_ONCHAIN_NODE = "curve_3pool"
_MAXLAG = 4


def _hourly(df: pl.DataFrame, node: str, col: str) -> dict[int, float]:
    d = (
        df.filter((pl.col("node_id") == node) & pl.col(col).is_not_null())
        .with_columns((pl.col("event_time_seconds") // 3600).alias("h"))
        .group_by("h").agg(pl.col(col).mean().alias("v")).sort("h")
    )
    return dict(zip(d["h"].to_list(), d["v"].to_list()))


def _discovery(df: pl.DataFrame, cex: str):
    oc = _hourly(df, _ONCHAIN_NODE, "implied_pool_price")
    cx = _hourly(df, cex, "basis_vs_usd")
    H = sorted(set(oc) & set(cx))
    if len(H) < 30:
        return None
    A = np.abs(np.array([oc[h] for h in H]) - 1.0)   # on-chain pool deviation from peg
    B = np.abs(np.array([cx[h] for h in H]))          # CEX basis deviation
    dA, dB = np.diff(A), np.diff(B)
    Hd = H[1:]
    md = dict(zip(Hd, dB.tolist()))

    pos, neg = [], []
    peak1_r, peak1_p = None, None
    for k in range(-_MAXLAG, _MAXLAG + 1):
        xs, ys = [], []
        for h, v in zip(Hd, dA.tolist()):
            if h + k in md:
                xs.append(v); ys.append(md[h + k])
        if len(xs) >= 10 and np.std(xs) > 0 and np.std(ys) > 0:
            r, p = stats.pearsonr(xs, ys)
            if k > 0:
                pos.append(r)
            elif k < 0:
                neg.append(r)
            if k == 1:
                peak1_r, peak1_p = r, p
    if not pos or not neg:
        return None
    return {
        "onchain_leads_mean_r": float(np.mean(pos)),
        "cex_leads_mean_r":     float(np.mean(neg)),
        "lead1h_r":             float(peak1_r) if peak1_r is not None else None,
        "lead1h_p":             float(peak1_p) if peak1_p is not None else None,
        "max_onchain_dev":      float(A.max()),
        "max_cex_dev":          float(B.max()),
    }


def main() -> None:
    rows = []
    for event_id, cex in _EVENT_CEX.items():
        p = gold_root() / f"dataset_contagion_features_{event_id}.parquet"
        if not p.exists():
            continue
        df = pl.read_parquet(p)
        r = _discovery(df, cex)
        if r is None:
            logger.warning("%s: insufficient overlap; skipping.", event_id)
            continue
        verdict = "on_chain_leads" if r["onchain_leads_mean_r"] > r["cex_leads_mean_r"] else "cex_leads"
        dev_ratio = (r["max_onchain_dev"] / r["max_cex_dev"]) if r["max_cex_dev"] > 0 else None
        rows.append({
            "event_id":              event_id,
            "cex_node":              cex,
            "onchain_leads_mean_r":  round(r["onchain_leads_mean_r"], 4),
            "cex_leads_mean_r":      round(r["cex_leads_mean_r"], 4),
            "lead1h_r":              round(r["lead1h_r"], 4) if r["lead1h_r"] is not None else None,
            "lead1h_p":              round(r["lead1h_p"], 4) if r["lead1h_p"] is not None else None,
            "max_onchain_dev":       round(r["max_onchain_dev"], 4),
            "max_cex_dev":           round(r["max_cex_dev"], 5),
            "onchain_dev_ratio":     round(dev_ratio, 1) if dev_ratio is not None else None,
            "price_discovery_venue": verdict,
        })
        logger.info(
            "%s: onchain_leads=%.3f cex_leads=%.3f lead+1h r=%s (p=%s) dev=%.3f vs %.5f -> %s",
            event_id, r["onchain_leads_mean_r"], r["cex_leads_mean_r"],
            f"{r['lead1h_r']:.3f}" if r["lead1h_r"] is not None else "n/a",
            f"{r['lead1h_p']:.3f}" if r["lead1h_p"] is not None else "n/a",
            r["max_onchain_dev"], r["max_cex_dev"], verdict,
        )

    if not rows:
        logger.warning("No price-discovery rows produced.")
        return
    out = results_root() / "tables" / "table_price_discovery.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    n_oc = sum(1 for r in rows if r["price_discovery_venue"] == "on_chain_leads")
    logger.info("Wrote %s (%d events; %d on-chain-leads)", out, len(rows), n_oc)


if __name__ == "__main__":
    main()
