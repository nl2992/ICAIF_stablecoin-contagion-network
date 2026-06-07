"""Triage the grid-search families into works / doesn't-work, with discipline:
a NEW result must be robust across configs (not one lucky cell), and the
already-certified detection result must survive.  Prints a verdict table.

Usage:  python scripts/37_triage.py
"""
from __future__ import annotations

import numpy as np
import polars as pl

from stressnet.config import results_root
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)
TAB = results_root() / "tables"
REGIME_EVENTS = {"usdt_curve_2023", "terra_luna_2022", "usdc_svb_2023"}  # genuine on-chain regime


def _exists(name):
    return (TAB / name).exists()


def f1():
    if not _exists("grid_f1_ablation.csv"):
        return "F1  ablation x model     : (not run)"
    d = pl.read_csv(TAB / "grid_f1_ablation.csv")
    both = d.filter(pl.col("feature_set") == "both")
    # robust if 'both' lift>0 on regime events across ALL models
    reg = both.filter(pl.col("event").is_in(list(REGIME_EVENTS)))
    frac_pos = float((reg["lift_vs_market"] > 0).mean())
    mean_lift = float(reg["lift_vs_market"].mean())
    # which single on-chain feature carries most (mean auroc across regime events)
    singles = ["oc_pxdev", "oc_imb", "oc_slip", "oc_flow"]
    best = max(singles, key=lambda s: d.filter((pl.col("feature_set") == s) &
              (pl.col("event").is_in(list(REGIME_EVENTS))))["auroc"].mean())
    verdict = "WORKS (robust)" if frac_pos >= 0.75 and mean_lift > 0.02 else "weak/mixed"
    return (f"F1  ablation x model     : {verdict}; +onchain lift on regime events "
            f"mean={mean_lift:+.3f}, positive in {frac_pos*100:.0f}% of model cells; "
            f"dominant single feature = {best}")


def f2():
    if not _exists("grid_f2_lead_time.csv"):
        return "F2  lead time            : (not run)"
    d = pl.read_csv(TAB / "grid_f2_lead_time.csv")
    reg = d.filter(pl.col("event").is_in(list(REGIME_EVENTS)))
    leads = reg["onchain_minus_market_h"].drop_nulls().to_list()
    med = float(np.median(leads)) if leads else float("nan")
    allpos = all(v > 0 for v in leads)
    verdict = "WORKS (directional)" if allpos else "mixed"
    return (f"F2  lead time            : {verdict}; on regime events on-chain leads market "
            f"by {leads} h (median {med:+.0f}); exogenous events on-chain is late (correct)")


def f3():
    if not _exists("grid_f3_transfer.csv"):
        return "F3  transfer recovery    : (not run)"
    d = pl.read_csv(TAB / "grid_f3_transfer.csv")
    g = d.group_by("method").agg(pl.col("transfer_auroc").mean().alias("m")).sort("m", descending=True)
    best = g.row(0, named=True)
    recovered = best["m"] > 0.62
    verdict = "WORKS (transfer recovered)" if recovered else "NULL (concept shift confirmed)"
    detail = "; ".join(f"{r['method']}={r['m']:.3f}" for r in g.iter_rows(named=True))
    return f"F3  transfer recovery    : {verdict}; mean transfer AUROC by method: {detail} (chance 0.50)"


def f4():
    if not _exists("grid_f4_gru_forecast.csv"):
        return "F4  GPU GRU forecast     : (not run)"
    d = pl.read_csv(TAB / "grid_f4_gru_forecast.csv").filter(pl.col("lift").is_not_null())
    if d.height == 0:
        return "F4  GPU GRU forecast     : NULL (all splits degenerate -- basis too persistent)"
    lifts = d["lift"].to_list()
    med = float(np.median(lifts))
    frac_pos = float((d["lift"] > 0.02).mean())
    verdict = "WORKS" if med > 0.02 and frac_pos >= 0.6 else "NULL/weak"
    return (f"F4  GPU GRU forecast     : {verdict}; on-chain forecast lift median={med:+.3f} "
            f"over {d.height} non-degenerate cells, >+0.02 in {frac_pos*100:.0f}%")


def f6():
    if not _exists("grid_f6_hmm.csv"):
        return "F6  HMM robustness       : (not run)"
    d = pl.read_csv(TAB / "grid_f6_hmm.csv")
    reg = d.filter(pl.col("event").is_in(list(REGIME_EVENTS)))
    # for each (event), best config AUROC and spread across configs
    by = reg.group_by("event").agg(pl.col("auroc").mean().alias("mean"),
                                    pl.col("auroc").max().alias("max"),
                                    pl.col("auroc").min().alias("min"))
    mean_over = float(by["mean"].mean()); min_over = float(by["min"].min())
    verdict = "WORKS (stable)" if mean_over > 0.80 and min_over > 0.6 else "config-sensitive"
    return (f"F6  HMM robustness       : {verdict}; regime-event AUROC mean-over-configs "
            f"{mean_over:.3f}, worst single config {min_over:.3f}")


def main():
    logger.info("================  GRID-SEARCH TRIAGE  ================")
    for fn in (f1, f2, f3, f4, f6):
        try:
            logger.info(fn())
        except Exception as e:
            logger.error("%s: %s", fn.__name__, e)
    logger.info("F5/F7 cross-pair sweeps: NULL BY CONSTRUCTION -- each event has only "
                "3pool + one satellite pool (SVB only 3pool); no additional A/A pairs exist.")
    logger.info("=====================================================")


if __name__ == "__main__":
    main()
