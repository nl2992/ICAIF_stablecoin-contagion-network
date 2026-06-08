"""Plan G — Pool-level arbitrage intensity and link to contagion strength.

Computes per-hour arbitrage trade direction from Curve pool TokenExchange
logs (via the reconstructed gold features), then correlates arbitrage intensity
with the Forbes-Rigobon regime indicator (calm vs panic).

Arbitrage classification per hourly bucket:
  - flow      = usdc_net_sold_1h  (positive = selling USDC into pool)
  - price_dev = implied_pool_price - 1.0  (signed: positive = USDC at premium)
  - Stabilising trade: flow and price_dev have opposite signs
    (e.g., sell USDC when USDC is at a premium → drives price back to 1)
  - Destabilising trade: flow and price_dev have same signs
    (e.g., sell USDC when USDC is already at a discount → amplifies depeg)

Rolling 6h window arbitrage intensity = fraction of destabilising hours.

We then compute the Pearson correlation between rolling intensity and the
pool's price deviation, separately for calm (pre) and panic phases.

For USDT/Curve: expect a flip from net-stabilising (negative intensity-deviation
correlation) in calm to net-amplifying (positive) during panic, consistent
with the z=+3.84 arbitrage regime result from script 25.

Reads:  data/gold/dataset_contagion_features_{event}.parquet
Writes: results/tables/table_arbitrage_intensity.csv
        results/tables/table_arb_intensity_timeseries_{event}.csv

Usage:
    python scripts/arbitrage_intensity.py
    python scripts/arbitrage_intensity.py --episode usdt_curve --window_hours 6
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
_WINDOW_H = 6

# CEX node carrying basis_vs_usd for each event (must match script 25)
_EVENT_CEX: dict[str, str] = {
    "usdt_curve_2023": "usdt_binance",
    "terra_luna_2022": "usdt_binance",
    "ftx_2022":        "usdt_binance",
    "busd_2023":       "busd_binance",
    "usdc_svb_2023":   "usdc_binance",
}


def _load_pool_timeseries(event_id: str, pool_node: str = _NODE) -> pl.DataFrame:
    """Load hourly DEX flow (curve_3pool) joined with CEX |basis_vs_usd|.

    Replicates the cross-node join from script 25 but adds a rolling-window
    capability and also retains the implied_pool_price for reference.
    """
    df = pl.read_parquet(gold_root() / f"dataset_contagion_features_{event_id}.parquet")
    cex_node = _EVENT_CEX.get(event_id, "usdt_binance")

    pool = (
        df.filter(pl.col("node_id") == pool_node)
        .with_columns((pl.col("event_time_seconds") // 3_600).alias("h"))
        .group_by("h")
        .agg(
            pl.col("usdc_net_sold_1h").sum().alias("flow"),
            pl.col("implied_pool_price").mean().alias("price"),
            pl.col("reserve_imbalance").mean().alias("imb"),
            pl.col("event_phase").first().alias("phase"),
        )
        .sort("h")
    )
    cex = (
        df.filter((pl.col("node_id") == cex_node) & pl.col("basis_vs_usd").is_not_null())
        .with_columns((pl.col("event_time_seconds") // 3_600).alias("h"))
        .group_by("h")
        .agg(pl.col("basis_vs_usd").abs().mean().alias("abs_basis"))
        .sort("h")
    )
    return pool.join(cex, on="h", how="inner")


def _rolling_arb_intensity(abs_flow: np.ndarray, abs_basis: np.ndarray,
                            window: int) -> np.ndarray:
    """Rolling Pearson correlation of |flow| and |basis|.
    Positive r = amplifying (high flow accompanies high deviation = destabilising).
    Negative r = stabilising (high flow accompanies low deviation).
    """
    intensity = np.full(len(abs_flow), np.nan)
    for i in range(window - 1, len(abs_flow)):
        xw = abs_flow[i - window + 1: i + 1]
        yw = abs_basis[i - window + 1: i + 1]
        if np.std(xw) > 0 and np.std(yw) > 0:
            r, _ = stats.pearsonr(xw, yw)
            intensity[i] = float(r)
    return intensity


def _phase_corr_flow_price(abs_flow: np.ndarray, abs_basis: np.ndarray,
                            phase_mask: np.ndarray) -> float | None:
    """Pearson correlation of |flow| and |basis| within a phase mask."""
    x = abs_flow[phase_mask]
    y = abs_basis[phase_mask]
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return None
    r, _ = stats.pearsonr(x, y)
    return float(r)


def main(episode: str | None = None, window_hours: int = _WINDOW_H) -> None:
    event_list = [episode] if episode and episode != "all" else _EVENTS
    # Normalise episode shorthand
    alias = {"usdt_curve": "usdt_curve_2023", "terra": "terra_luna_2022",
             "ftx": "ftx_2022", "busd": "busd_2023", "usdc_svb": "usdc_svb_2023"}
    event_list = [alias.get(e, e) for e in event_list]

    summary_rows = []
    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    for event_id in event_list:
        try:
            d = _load_pool_timeseries(event_id)
        except Exception as exc:
            logger.warning("%s: load failed (%s); skipping.", event_id, exc)
            continue

        flow      = np.nan_to_num(d["flow"].to_numpy())
        price     = np.nan_to_num(d["price"].to_numpy(), nan=1.0)
        abs_basis = np.nan_to_num(d["abs_basis"].to_numpy())
        phase     = d["phase"].to_numpy()

        abs_flow  = np.abs(flow)

        # Rolling Pearson(|flow|, |basis|): + = amplifying, - = stabilising
        rolling_r = _rolling_arb_intensity(abs_flow, abs_basis, window_hours)

        pre_mask   = phase == "pre"
        panic_mask = phase == "panic"

        # Phase-level correlation of |flow| and |basis| (matches existing script 25)
        r_pre  = _phase_corr_flow_price(abs_flow, abs_basis, pre_mask)
        r_pan  = _phase_corr_flow_price(abs_flow, abs_basis, panic_mask)
        # Flip: negative (stabilising) in calm → positive (amplifying) in panic
        arb_flip = (
            r_pre is not None and r_pan is not None
            and r_pre < 0 and r_pan > 0
        )

        # Rolling intensity correlation with |basis|
        valid = ~np.isnan(rolling_r)
        r_intensity_basis, p_intensity = (None, None)
        if valid.sum() >= 5:
            x = rolling_r[valid]
            y = abs_basis[valid]
            if np.std(x) > 0 and np.std(y) > 0:
                r_intensity_basis, p_intensity = stats.pearsonr(x, y)

        summary_rows.append({
            "event_id":                event_id,
            "n_hours":                 int(len(flow)),
            "window_h":                window_hours,
            "corr_flow_basis_pre":     round(r_pre, 4)  if r_pre  is not None else None,
            "corr_flow_basis_panic":   round(r_pan, 4)  if r_pan  is not None else None,
            "arb_regime_flip":         arb_flip,
            "r_rolling_vs_basis":      round(float(r_intensity_basis), 4)
                                       if r_intensity_basis is not None else None,
            "p_rolling_vs_basis":      round(float(p_intensity), 4)
                                       if p_intensity is not None else None,
        })
        logger.info(
            "%s: r_pre=%s r_panic=%s  flip=%s  rolling-basis r=%s p=%s",
            event_id,
            f"{r_pre:+.3f}" if r_pre is not None else "n/a",
            f"{r_pan:+.3f}" if r_pan is not None else "n/a",
            arb_flip,
            f"{r_intensity_basis:.3f}" if r_intensity_basis is not None else "n/a",
            f"{p_intensity:.4f}" if p_intensity is not None else "n/a",
        )

        # Per-event timeseries
        ts_rows = []
        h_arr = d["h"].to_numpy()
        for i in range(len(h_arr)):
            ts_rows.append({
                "h":                  int(h_arr[i]),
                "abs_flow":           round(float(abs_flow[i]), 6),
                "abs_basis":          round(float(abs_basis[i]), 6),
                "rolling_arb_r":      round(float(rolling_r[i]), 4)
                                      if not np.isnan(rolling_r[i]) else None,
                "phase":              str(phase[i]),
            })
        ts_path = out_dir / f"table_arb_intensity_timeseries_{event_id}.csv"
        with ts_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(ts_rows[0].keys()))
            w.writeheader(); w.writerows(ts_rows)

    if not summary_rows:
        logger.warning("No arbitrage intensity rows produced.")
        return

    out_path = out_dir / "table_arbitrage_intensity.csv"
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    n_flip = sum(1 for r in summary_rows if r["arb_regime_flip"])
    logger.info("Arbitrage regime flip (stabilising -> destabilising): %d/%d episodes",
                n_flip, len(summary_rows))
    logger.info("Wrote table_arbitrage_intensity.csv and per-event timeseries CSVs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode",      default="all")
    ap.add_argument("--window_hours", type=int, default=6)
    args = ap.parse_args()
    main(episode=args.episode, window_hours=args.window_hours)
