"""Plan A — Economic value of the 116-hour early-warning signal (Terra).

The on-chain HMM fires 34h before the labeled Terra panic onset;
the CEX basis signal fires 82h after: a 116h total advantage.
This script quantifies the mark-to-market loss that could be avoided
by acting on the HMM signal rather than waiting for the market signal.

Three stylised $1M positions:
  - long_luna   : LUNA/USDT spot (anchor prices from public record)
  - long_ust    : UST/USD (ust_binance mid_price from gold data)
  - curve_pool  : Curve UST/3CRV pool LP (implied_pool_price from gold data)

Sensitivity table: (lead_h, frac_liquidated) → UST loss avoided vs market signal.

Reads:  data/gold/dataset_contagion_features_terra_luna_2022.parquet
        results/tables/table_online_detection.csv
Writes: results/tables/table_early_warning_value.csv
        results/tables/table_early_warning_sensitivity.csv

Usage:
    python scripts/early_warning_value.py
    python scripts/early_warning_value.py --episode terra --lead_hours 116 --positions luna ust curve
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import warnings

import numpy as np
import polars as pl

from stressnet.config import gold_root, results_root
from stressnet.utils.logging import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

EPISODE = "terra_luna_2022"

# event_time_seconds in the gold parquets is offset from shock_onset_utc (origin = 0).
# Pre-crisis data has negative values; post-onset is positive.
ONSET_SECONDS = 0

# From results/tables/table_online_detection.csv (terra_luna_2022 row)
HMM_LEAD_H   = 34   # HMM fires this many hours BEFORE labeled onset
MARKET_LAG_H = 82   # CEX basis signal fires this many hours AFTER labeled onset

HMM_ALERT_S    = ONSET_SECONDS - HMM_LEAD_H   * 3_600   # = -122,400 s
MARKET_ALERT_S = ONSET_SECONDS + MARKET_LAG_H * 3_600   # = +295,200 s

POSITION_USD = 1_000_000

# LUNA anchor prices (public record; stylised).
# Binance LUNAUSDT spot at key times during May 2022:
#   Pre-crisis (long entry):          ~$80   (2022-05-05)
#   HMM alert  (onset − 34h):         ~$68   (2022-05-06 ~14:00 UTC)
#   Labeled panic onset:              ~$55   (2022-05-08 00:00 UTC)
#   Market alert (onset + 82h):        ~$7   (2022-05-11 ~10:00 UTC)
#   Post-collapse:                    ~$0.01 (2022-05-13+)
LUNA_ENTRY        = 80.0
LUNA_AT_HMM       = 68.0
LUNA_AT_MARKET    =  7.0
LUNA_AT_COLLAPSE  =  0.01


def _hourly_series(df: pl.DataFrame, node_id: str, col: str) -> pl.DataFrame:
    return (
        df.filter(pl.col("node_id") == node_id)
        .with_columns((pl.col("event_time_seconds") // 3_600).alias("h"))
        .group_by("h")
        .agg(pl.col(col).mean().alias("val"))
        .sort("h")
    )


def _at_time(series: pl.DataFrame, event_seconds: int) -> float:
    """Return series value at the hourly bucket nearest to event_seconds."""
    target_h = event_seconds // 3_600
    row = series.filter(pl.col("h") == target_h)
    if row.height > 0:
        return float(row["val"][0])
    h_arr = series["h"].to_numpy()
    v_arr = series["val"].to_numpy()
    idx = int(np.argmin(np.abs(h_arr - target_h)))
    return float(v_arr[idx])


def main(position_usd: float = POSITION_USD) -> None:
    df = pl.read_parquet(gold_root() / f"dataset_contagion_features_{EPISODE}.parquet")

    # ── UST position ──────────────────────────────────────────────────────────
    ust_s = _hourly_series(df, "ust_binance", "mid_price")
    ust_entry  = 1.00
    ust_at_hmm = _at_time(ust_s, HMM_ALERT_S)
    ust_at_mkt = _at_time(ust_s, MARKET_ALERT_S)
    ust_at_bot = float(ust_s.sort("h").tail(30)["val"].mean())  # post-collapse avg

    ust_loss_hmm  = max(0.0, (ust_entry - ust_at_hmm) * position_usd)
    ust_loss_mkt  = max(0.0, (ust_entry - ust_at_mkt) * position_usd)
    ust_loss_none = max(0.0, (ust_entry - ust_at_bot) * position_usd)
    ust_avoided   = ust_loss_mkt - ust_loss_hmm

    # ── Curve pool position ──────────────────────────────────────────────────
    pool_s = _hourly_series(df, "curve_ust_wormhole", "implied_pool_price")
    pool_at_hmm = _at_time(pool_s, HMM_ALERT_S)
    pool_at_mkt = _at_time(pool_s, MARKET_ALERT_S)
    pool_at_bot = float(pool_s.sort("h").tail(30)["val"].mean())

    pool_loss_hmm  = max(0.0, (1.0 - pool_at_hmm) * position_usd)
    pool_loss_mkt  = max(0.0, (1.0 - pool_at_mkt) * position_usd)
    pool_loss_none = max(0.0, (1.0 - pool_at_bot) * position_usd)
    pool_avoided   = pool_loss_mkt - pool_loss_hmm

    # ── LUNA position (anchor prices) ────────────────────────────────────────
    luna_loss_hmm  = max(0.0, (LUNA_ENTRY - LUNA_AT_HMM) / LUNA_ENTRY * position_usd)
    luna_loss_mkt  = max(0.0, (LUNA_ENTRY - LUNA_AT_MARKET) / LUNA_ENTRY * position_usd)
    luna_loss_none = max(0.0, (LUNA_ENTRY - LUNA_AT_COLLAPSE) / LUNA_ENTRY * position_usd)
    luna_avoided   = luna_loss_mkt - luna_loss_hmm

    summary = [
        {
            "position":                   "long_luna",
            "entry_price":                LUNA_ENTRY,
            "price_at_hmm_alert":         LUNA_AT_HMM,
            "price_at_market_alert":      LUNA_AT_MARKET,
            "loss_if_hmm_alert_usd":      round(luna_loss_hmm, 0),
            "loss_if_market_alert_usd":   round(luna_loss_mkt, 0),
            "loss_if_no_signal_usd":      round(luna_loss_none, 0),
            "loss_avoided_vs_market_usd": round(luna_avoided, 0),
            "data_source":                "public_anchor",
        },
        {
            "position":                   "long_ust",
            "entry_price":                ust_entry,
            "price_at_hmm_alert":         round(ust_at_hmm, 4),
            "price_at_market_alert":      round(ust_at_mkt, 4),
            "loss_if_hmm_alert_usd":      round(ust_loss_hmm, 0),
            "loss_if_market_alert_usd":   round(ust_loss_mkt, 0),
            "loss_if_no_signal_usd":      round(ust_loss_none, 0),
            "loss_avoided_vs_market_usd": round(ust_avoided, 0),
            "data_source":                "gold_ust_binance_mid_price",
        },
        {
            "position":                   "curve_pool",
            "entry_price":                1.00,
            "price_at_hmm_alert":         round(pool_at_hmm, 4),
            "price_at_market_alert":      round(pool_at_mkt, 4),
            "loss_if_hmm_alert_usd":      round(pool_loss_hmm, 0),
            "loss_if_market_alert_usd":   round(pool_loss_mkt, 0),
            "loss_if_no_signal_usd":      round(pool_loss_none, 0),
            "loss_avoided_vs_market_usd": round(pool_avoided, 0),
            "data_source":                "gold_curve_ust_wormhole_implied_price",
        },
    ]

    # ── Sensitivity: (lead_h_over_market, frac_liquidated) → UST loss avoided ─
    ust_h   = ust_s["h"].to_numpy()
    ust_val = ust_s["val"].to_numpy()
    mkt_price = _at_time(ust_s, MARKET_ALERT_S)
    sens_rows = []
    for lead_h in [0, 24, 48, 72, 116]:
        for frac in [0.25, 0.50, 0.75, 1.00]:
            alert_s = MARKET_ALERT_S - lead_h * 3_600
            idx = int(np.searchsorted(ust_h, alert_s // 3_600))
            idx = min(idx, len(ust_val) - 1)
            p_early = float(ust_val[idx])
            avoided = max(0.0, (mkt_price - p_early) * frac * position_usd)
            # Flip: if acting earlier means exiting at a HIGHER (better) price, avoided > 0
            avoided = max(0.0, (p_early - mkt_price) * frac * position_usd)
            sens_rows.append({
                "lead_h_vs_market_signal": lead_h,
                "frac_liquidated":         frac,
                "ust_price_at_alert":      round(p_early, 4),
                "ust_loss_avoided_usd":    round(avoided, 0),
            })

    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "table_early_warning_value.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    with (out_dir / "table_early_warning_sensitivity.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sens_rows[0].keys()))
        w.writeheader(); w.writerows(sens_rows)

    total_avoided = sum(r["loss_avoided_vs_market_usd"] for r in summary)
    logger.info("=== Economic value of %dh early warning (Terra) ===",
                HMM_LEAD_H + MARKET_LAG_H)
    for r in summary:
        logger.info("  %-12s  HMM-exit loss: $%s  |  Mkt-exit loss: $%s  |  Avoided: $%s",
                    r["position"],
                    f"{r['loss_if_hmm_alert_usd']:,.0f}",
                    f"{r['loss_if_market_alert_usd']:,.0f}",
                    f"{r['loss_avoided_vs_market_usd']:,.0f}")
    logger.info("  Total avoided across 3 x $%s positions: $%s",
                f"{position_usd:,.0f}", f"{total_avoided:,.0f}")
    logger.info("Wrote table_early_warning_value.csv and table_early_warning_sensitivity.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode",     default="terra")
    ap.add_argument("--lead_hours",  type=int, default=116)
    ap.add_argument("--positions",   nargs="+", default=["luna", "ust", "curve"])
    _ = ap.parse_args()   # parameters kept for CLI compat; logic uses constants above
    main()
