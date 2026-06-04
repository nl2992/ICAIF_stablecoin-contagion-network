"""Regime-switching contagion test for AMM-flow cross-pool coupling.

Tests the Forbes-Rigobon contagion hypothesis on Tier-A on-chain AMM flow:
*contagion* is a statistically significant INCREASE in cross-market linkage
during the crisis (``panic``) regime relative to the calm (``pre``) regime,
as distinct from constant *interdependence*.

For each event's Tier-A A/A pool pair we compute the lag-0 Pearson correlation
of hourly ``usdc_net_sold_1h`` within each ``event_phase`` regime, then test the
calm-vs-panic difference with a Fisher r-to-z two-sample statistic.

Reads:  data/gold/dataset_contagion_features_{event}.parquet
Writes: results/tables/table_regime_contagion.csv

Usage:
    python scripts/24_run_regime_contagion.py
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

# Tier-A A/A pool pair per event (both endpoints real on-chain DEX nodes)
_EVENT_PAIRS: dict[str, tuple[str, str]] = {
    "usdt_curve_2023": ("curve_3pool", "curve_crvusd_usdt"),
    "terra_luna_2022": ("curve_3pool", "curve_ust_wormhole"),
    "ftx_2022":        ("curve_3pool", "curve_lusd_3crv"),
    "busd_2023":       ("curve_3pool", "curve_lusd_3crv"),
}

_FEATURE = "usdc_net_sold_1h"
_MIN_N = 5  # minimum overlapping hourly buckets to attempt a correlation


def _hourly_flow(sub: pl.DataFrame, node_id: str) -> pl.DataFrame:
    """Hourly-bucketed net flow for one node within a phase subset."""
    return (
        sub.filter(pl.col("node_id") == node_id)
        .with_columns((pl.col("event_time_seconds") // 3600).alias("h"))
        .group_by("h")
        .agg(pl.col(_FEATURE).sum().alias("f"))
    )


def _phase_corr(df: pl.DataFrame, phase: str, a: str, b: str) -> tuple[int, float | None, float | None]:
    """Lag-0 Pearson correlation of the pair within one regime."""
    sub = df.filter((pl.col("event_phase") == phase) & pl.col(_FEATURE).is_not_null())
    A = _hourly_flow(sub, a).rename({"f": "a"})
    B = _hourly_flow(sub, b).rename({"f": "b"})
    m = A.join(B, on="h", how="inner").drop_nulls()
    if m.height < _MIN_N:
        return m.height, None, None
    x, y = m["a"].to_numpy(), m["b"].to_numpy()
    if np.std(x) == 0 or np.std(y) == 0:
        return m.height, None, None
    r, p = stats.pearsonr(x, y)
    return m.height, float(r), float(p)


def _fisher_diff(r1: float | None, n1: int, r2: float | None,
                 n2: int) -> tuple[float | None, float | None]:
    """Fisher r-to-z two-sample test for the difference r2 - r1.

    Returns (z_stat, two-sided p-value).  z > 0 means linkage rose from
    regime 1 (calm) to regime 2 (panic) — the contagion direction.
    """
    if r1 is None or r2 is None or min(n1, n2) < _MIN_N + 1:
        return None, None
    # clip to avoid infinite atanh at |r|=1
    r1c, r2c = max(min(r1, 0.999), -0.999), max(min(r2, 0.999), -0.999)
    z1, z2 = math.atanh(r1c), math.atanh(r2c)
    se = math.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z = (z2 - z1) / se
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def main() -> None:
    rows: list[dict] = []
    for event_id, (a, b) in _EVENT_PAIRS.items():
        panel_path = gold_root() / f"dataset_contagion_features_{event_id}.parquet"
        if not panel_path.exists():
            logger.warning("Panel missing for %s; skipping.", event_id)
            continue
        df = pl.read_parquet(panel_path)
        if "event_phase" not in df.columns:
            logger.warning("No event_phase in %s; skipping.", event_id)
            continue

        n_pre, r_pre, p_pre = _phase_corr(df, "pre", a, b)
        n_pan, r_pan, p_pan = _phase_corr(df, "panic", a, b)
        z, p_diff = _fisher_diff(r_pre, n_pre, r_pan, n_pan)

        # Contagion = significant POSITIVE shift (linkage rises during panic)
        contagion = (
            z is not None and p_diff is not None and z > 0 and p_diff < 0.05
        )
        rows.append({
            "event_id":        event_id,
            "pool_a":          a,
            "pool_b":          b,
            "n_pre":           n_pre,
            "rho_pre":         round(r_pre, 4) if r_pre is not None else None,
            "p_pre":           round(p_pre, 4) if p_pre is not None else None,
            "n_panic":         n_pan,
            "rho_panic":       round(r_pan, 4) if r_pan is not None else None,
            "p_panic":         round(p_pan, 4) if p_pan is not None else None,
            "fisher_z":        round(z, 4) if z is not None else None,
            "p_regime_shift":  round(p_diff, 4) if p_diff is not None else None,
            "contagion_regime_shift": contagion,
        })
        logger.info(
            "%s: rho_pre=%s rho_panic=%s  Fisher z=%s p=%s  contagion=%s",
            event_id,
            f"{r_pre:+.3f}" if r_pre is not None else "n/a",
            f"{r_pan:+.3f}" if r_pan is not None else "n/a",
            f"{z:+.2f}" if z is not None else "n/a",
            f"{p_diff:.4f}" if p_diff is not None else "n/a",
            contagion,
        )

    if not rows:
        logger.warning("No regime-contagion rows produced.")
        return

    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "table_regime_contagion.csv"
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_contagion = sum(1 for r in rows if r["contagion_regime_shift"])
    logger.info(
        "Wrote %s (%d events; %d with significant positive contagion regime shift)",
        out_path, len(rows), n_contagion,
    )


if __name__ == "__main__":
    main()
