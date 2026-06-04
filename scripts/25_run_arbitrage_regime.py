"""Stabilizing-to-amplifying regime flip in on-chain arbitrage flow.

Positive structural finding: in calm markets, on-chain AMM arbitrage flow is
*stabilizing* --- flow intensity is negatively correlated with CEX price
deviation (active arbitrage absorbs dislocations).  During acute stress the
relationship can *flip* to positive --- flow and price dislocation amplify
together as arbitrage capacity is overwhelmed.

For each event we compute the contemporaneous correlation between
|Curve 3pool ``usdc_net_sold_1h``| (Tier-A on-chain flow intensity) and
|CEX ``basis_vs_usd``| (price-deviation magnitude) within the calm (``pre``)
and acute (``panic``) regimes, and test the calm->panic shift with a Fisher
r-to-z statistic.  A sign flip (negative in calm, positive in panic) cannot be
produced by a shared trend, making it robust to the common-trend artefact that
inflates naive level correlations.

Reads:  data/gold/dataset_contagion_features_{event}.parquet
Writes: results/tables/table_arbitrage_regime.csv

Usage:
    python scripts/25_run_arbitrage_regime.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from stressnet.config import gold_root, results_root
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

# (event, CEX node carrying basis_vs_usd) ; flow node is always curve_3pool (Tier A)
_EVENT_CEX: dict[str, str] = {
    "usdt_curve_2023": "usdt_binance",
    "terra_luna_2022": "usdt_binance",
    "ftx_2022":        "usdt_binance",
    "busd_2023":       "busd_binance",
    "usdc_svb_2023":   "usdc_binance",
}

_FLOW_NODE = "curve_3pool"
_FLOW_COL  = "usdc_net_sold_1h"
_PRICE_COL = "basis_vs_usd"
_MIN_N = 5


def _hourly_abs(sub: pl.DataFrame, node: str, col: str) -> pl.DataFrame:
    return (
        sub.filter((pl.col("node_id") == node) & pl.col(col).is_not_null())
        .with_columns((pl.col("event_time_seconds") // 3600).alias("h"))
        .group_by("h")
        .agg(pl.col(col).mean().alias("v"))
    )


def _regime_corr(df: pl.DataFrame, phase: str, cex: str) -> tuple[int, float | None]:
    sub = df.filter(pl.col("event_phase") == phase)
    A = _hourly_abs(sub, _FLOW_NODE, _FLOW_COL).rename({"v": "a"})
    B = _hourly_abs(sub, cex, _PRICE_COL).rename({"v": "b"})
    m = A.join(B, on="h", how="inner").drop_nulls()
    if m.height < _MIN_N:
        return m.height, None
    x = np.abs(m["a"].to_numpy())
    y = np.abs(m["b"].to_numpy())
    if np.std(x) == 0 or np.std(y) == 0:
        return m.height, None
    r, _ = stats.pearsonr(x, y)
    return m.height, float(r)


def _fisher(r1, n1, r2, n2):
    if r1 is None or r2 is None or min(n1, n2) < _MIN_N + 1:
        return None, None
    c = lambda r: max(min(r, 0.999), -0.999)
    z = (math.atanh(c(r2)) - math.atanh(c(r1))) / math.sqrt(1/(n1-3) + 1/(n2-3))
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def main() -> None:
    rows = []
    for event_id, cex in _EVENT_CEX.items():
        p = gold_root() / f"dataset_contagion_features_{event_id}.parquet"
        if not p.exists():
            logger.warning("Panel missing for %s; skipping.", event_id)
            continue
        df = pl.read_parquet(p)
        if "event_phase" not in df.columns:
            continue
        n_c, r_c = _regime_corr(df, "pre", cex)
        n_p, r_p = _regime_corr(df, "panic", cex)
        z, pz = _fisher(r_c, n_c, r_p, n_p)
        flip = (r_c is not None and r_p is not None and r_c < 0 and r_p > 0)
        rows.append({
            "event_id":   event_id,
            "cex_node":   cex,
            "n_calm":     n_c,
            "r_calm":     round(r_c, 4) if r_c is not None else None,
            "n_panic":    n_p,
            "r_panic":    round(r_p, 4) if r_p is not None else None,
            "fisher_z":   round(z, 4) if z is not None else None,
            "p_shift":    round(pz, 4) if pz is not None else None,
            "stabilizing_to_amplifying_flip": flip,
        })
        logger.info(
            "%s: r_calm=%s r_panic=%s Fisher z=%s p=%s flip=%s",
            event_id,
            f"{r_c:+.3f}" if r_c is not None else "n/a",
            f"{r_p:+.3f}" if r_p is not None else "n/a",
            f"{z:+.2f}" if z is not None else "n/a",
            f"{pz:.4f}" if pz is not None else "n/a",
            flip,
        )

    if not rows:
        logger.warning("No arbitrage-regime rows produced.")
        return
    out = results_root() / "tables" / "table_arbitrage_regime.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_flip = sum(1 for r in rows if r["stabilizing_to_amplifying_flip"])
    logger.info("Wrote %s (%d events; %d stabilizing->amplifying flips)",
                out, len(rows), n_flip)


if __name__ == "__main__":
    main()
