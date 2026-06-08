"""Plan D — Bootstrap confidence intervals on Forbes-Rigobon z-statistics.

For each episode, block-bootstrap the Forbes-Rigobon Fisher z-statistic
(block length = 5 days × 24h = 120 hourly buckets to preserve autocorrelation).
Reports 95% CI for z under the observed data.  Confirms that the USDT/Curve
positive detection (z=2.82) has a non-overlapping CI versus the exogenous
non-detection episodes (z ≈ 0), i.e., non-detections are not noise artefacts.

Algorithm per episode:
  1. Build hourly flow series for calm (pre) and panic phases separately.
  2. Block-bootstrap the calm series and the panic series independently
     (circular block bootstrap, block=5 days).
  3. Compute Fisher z on each bootstrap draw.
  4. Report [2.5%, 97.5%] percentiles as the 95% CI.

Reads:  data/gold/dataset_contagion_features_{event}.parquet
Writes: results/tables/table_bootstrap_fr.csv

Usage:
    python scripts/bootstrap_fr.py
    python scripts/bootstrap_fr.py --n_bootstrap 2000 --block_length 5
"""

from __future__ import annotations

import argparse
import csv
import math
import warnings

import numpy as np
import polars as pl
from scipy import stats

from stressnet.config import gold_root, results_root
from stressnet.utils.logging import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

_EVENT_PAIRS: dict[str, tuple[str, str]] = {
    "usdt_curve_2023": ("curve_3pool", "curve_crvusd_usdt"),
    "terra_luna_2022": ("curve_3pool", "curve_ust_wormhole"),
    "ftx_2022":        ("curve_3pool", "curve_lusd_3crv"),
    "busd_2023":       ("curve_3pool", "curve_lusd_3crv"),
}
_FEATURE = "usdc_net_sold_1h"
_MIN_N   = 6


def _hourly_flow(df: pl.DataFrame, phase: str, node_a: str, node_b: str) -> tuple[np.ndarray, np.ndarray]:
    sub = df.filter((pl.col("event_phase") == phase) & pl.col(_FEATURE).is_not_null())
    A = (sub.filter(pl.col("node_id") == node_a)
         .with_columns((pl.col("event_time_seconds") // 3_600).alias("h"))
         .group_by("h").agg(pl.col(_FEATURE).sum().alias("f")).sort("h"))
    B = (sub.filter(pl.col("node_id") == node_b)
         .with_columns((pl.col("event_time_seconds") // 3_600).alias("h"))
         .group_by("h").agg(pl.col(_FEATURE).sum().alias("f")).sort("h"))
    m = A.rename({"f": "a"}).join(B.rename({"f": "b"}), on="h", how="inner").drop_nulls()
    return m["a"].to_numpy(), m["b"].to_numpy()


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < _MIN_N or np.std(x) == 0 or np.std(y) == 0:
        return None
    r, _ = stats.pearsonr(x, y)
    return float(r)


def _fisher_z(r1: float | None, n1: int, r2: float | None, n2: int) -> float | None:
    if r1 is None or r2 is None or min(n1, n2) < _MIN_N + 1:
        return None
    r1c = max(min(r1, 0.999), -0.999)
    r2c = max(min(r2, 0.999), -0.999)
    z1, z2 = math.atanh(r1c), math.atanh(r2c)
    se = math.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    return (z2 - z1) / se


def _circular_block_sample(x: np.ndarray, y: np.ndarray,
                            block_len: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Circular block bootstrap: sample blocks from x,y jointly to preserve pairing."""
    n = len(x)
    n_blocks = math.ceil(n / block_len)
    starts = rng.integers(0, n, size=n_blocks)
    xs, ys = [], []
    for s in starts:
        idx = np.arange(s, s + block_len) % n
        xs.append(x[idx]); ys.append(y[idx])
    return np.concatenate(xs)[:n], np.concatenate(ys)[:n]


def main(n_bootstrap: int = 2000, block_length_days: int = 5) -> None:
    rng = np.random.default_rng(42)
    requested_block_len = block_length_days * 24  # hours (max target)

    rows = []
    for event_id, (a, b) in _EVENT_PAIRS.items():
        path = gold_root() / f"dataset_contagion_features_{event_id}.parquet"
        if not path.exists():
            logger.warning("Missing panel for %s; skipping.", event_id)
            continue
        df = pl.read_parquet(path)
        if "event_phase" not in df.columns:
            continue

        pre_a, pre_b   = _hourly_flow(df, "pre",   a, b)
        pan_a, pan_b   = _hourly_flow(df, "panic", a, b)
        r_pre   = _pearson_r(pre_a, pre_b)
        r_panic = _pearson_r(pan_a, pan_b)
        z_obs   = _fisher_z(r_pre, len(pre_a), r_panic, len(pan_a))

        if z_obs is None:
            logger.info("%s: insufficient data for bootstrap; skipping.", event_id)
            continue

        # Adaptive block length: aim for ~5-10 blocks so bootstrap has variation.
        # Cap at requested_block_len but don't exceed n//3 (need at least 3 blocks).
        n_min = min(len(pre_a), len(pan_a))
        block_len = max(1, min(requested_block_len, n_min // 3))
        logger.info("%s: n_pre=%d n_panic=%d  block_len=%d",
                    event_id, len(pre_a), len(pan_a), block_len)

        z_boot = []
        for _ in range(n_bootstrap):
            bs_pre_a, bs_pre_b   = _circular_block_sample(pre_a, pre_b, block_len, rng)
            bs_pan_a, bs_pan_b   = _circular_block_sample(pan_a, pan_b, block_len, rng)
            r_b_pre  = _pearson_r(bs_pre_a, bs_pre_b)
            r_b_pan  = _pearson_r(bs_pan_a, bs_pan_b)
            z_b = _fisher_z(r_b_pre, len(bs_pre_a), r_b_pan, len(bs_pan_a))
            if z_b is not None:
                z_boot.append(z_b)

        if len(z_boot) < 100:
            logger.warning("%s: too few valid bootstrap draws (%d).", event_id, len(z_boot))
            continue

        z_arr = np.array(z_boot)
        ci_lo = float(np.percentile(z_arr, 2.5))
        ci_hi = float(np.percentile(z_arr, 97.5))
        ci_se = float(np.std(z_arr))
        p_gt0 = float(np.mean(z_arr > 0))  # bootstrap p(z>0)
        contagion = bool(z_obs > 0 and ci_lo > 0)

        rows.append({
            "event_id":        event_id,
            "pool_a":          a,
            "pool_b":          b,
            "n_pre":           len(pre_a),
            "n_panic":         len(pan_a),
            "rho_pre":         round(r_pre, 4),
            "rho_panic":       round(r_panic, 4),
            "fisher_z_obs":    round(z_obs, 4),
            "ci_lo_95":        round(ci_lo, 4),
            "ci_hi_95":        round(ci_hi, 4),
            "boot_se":         round(ci_se, 4),
            "n_bootstrap":     len(z_boot),
            "p_gt0_bootstrap": round(p_gt0, 4),
            "contagion_ci":    contagion,
        })
        logger.info(
            "%s: z=%.3f [%.3f, %.3f]  p(z>0)=%.3f  contagion(CI)=%s",
            event_id, z_obs, ci_lo, ci_hi, p_gt0, contagion,
        )

    if not rows:
        logger.warning("No bootstrap rows produced.")
        return

    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "table_bootstrap_fr.csv"
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    logger.info("Wrote %s (%d episodes, block=%d days, B=%d)",
                out_path, len(rows), block_length_days, n_bootstrap)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_bootstrap",  type=int, default=2000)
    ap.add_argument("--block_length", type=int, default=5, help="block length in days")
    args = ap.parse_args()
    main(n_bootstrap=args.n_bootstrap, block_length_days=args.block_length)
