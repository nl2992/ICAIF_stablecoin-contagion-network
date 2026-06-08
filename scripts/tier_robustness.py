"""Plan C — Provenance-stratified robustness table (Tier-A vs Tier-B).

Re-runs the Forbes-Rigobon z-test and the HMM AUROC using only Tier-A
on-chain data (DEX pool nodes, no CEX feeds), and separately using only
Tier-B (CEX nodes). Confirms the USDT/Curve finding is not an artefact
of including CEX-sourced data.

Forbes-Rigobon strategy by tier:
  Tier-A (on-chain DEX):  usdc_net_sold_1h correlation on DEX pool pairs
                          (identical to script 24 — reported for completeness)
  Tier-B (CEX):           lag-0 Pearson of |basis_bps| across CEX node pairs
                          where both nodes are Tier-B CEX

HMM strategy by tier:
  Tier-A: fit on curve_3pool on-chain features (identical to script 27)
  Tier-B: fit on CEX node features (|basis_bps|, |mid_price - 1|)

Reads:  data/gold/dataset_contagion_features_{event}.parquet
Writes: results/tables/table_tier_robustness_fr.csv
        results/tables/table_tier_robustness_hmm.csv

Usage:
    python scripts/tier_robustness.py
    python scripts/tier_robustness.py --config configs/provenance_gate.yaml
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

_EVENTS = [
    "usdt_curve_2023", "terra_luna_2022", "ftx_2022", "busd_2023", "usdc_svb_2023",
]
_MIN_N = 5

# Tier-A DEX pairs (on-chain pools only) — replicates script 24 pairs
_TIER_A_PAIRS: dict[str, tuple[str, str]] = {
    "usdt_curve_2023": ("curve_3pool", "curve_crvusd_usdt"),
    "terra_luna_2022": ("curve_3pool", "curve_ust_wormhole"),
    "ftx_2022":        ("curve_3pool", "curve_lusd_3crv"),
    "busd_2023":       ("curve_3pool", "curve_lusd_3crv"),
}

# Tier-B CEX node pairs: pick two CEX nodes per event that both have basis_bps
_TIER_B_PAIRS: dict[str, tuple[str, str]] = {
    "usdt_curve_2023": ("usdt_binance", "usdt_binance"),     # single CEX: no pair → skipped below
    "terra_luna_2022": ("ust_binance",  "usdt_binance"),
    "ftx_2022":        ("usdt_binance", "busd_binance"),
    "busd_2023":       ("busd_binance", "usdt_binance"),
    "usdc_svb_2023":   ("usdc_binance", "usdc_coinbase"),
}


def _hourly_flow(sub: pl.DataFrame, node_id: str, feature: str) -> pl.DataFrame:
    return (
        sub.filter(pl.col("node_id") == node_id)
        .with_columns((pl.col("event_time_seconds") // 3_600).alias("h"))
        .group_by("h")
        .agg(pl.col(feature).sum().alias("f"))
    )


def _phase_corr(df: pl.DataFrame, phase: str, node_a: str, node_b: str,
                feature: str) -> tuple[int, float | None]:
    sub = df.filter((pl.col("event_phase") == phase) & pl.col(feature).is_not_null())
    A = _hourly_flow(sub, node_a, feature).rename({"f": "a"})
    B = _hourly_flow(sub, node_b, feature).rename({"f": "b"})
    m = A.join(B, on="h", how="inner").drop_nulls()
    if m.height < _MIN_N:
        return m.height, None
    x, y = m["a"].to_numpy(), m["b"].to_numpy()
    if np.std(x) == 0 or np.std(y) == 0:
        return m.height, None
    r, _ = stats.pearsonr(x, y)
    return m.height, float(r)


def _fisher_diff(r1, n1, r2, n2):
    if r1 is None or r2 is None or min(n1, n2) < _MIN_N + 1:
        return None, None
    r1c = max(min(r1, 0.999), -0.999)
    r2c = max(min(r2, 0.999), -0.999)
    z1, z2 = math.atanh(r1c), math.atanh(r2c)
    se = math.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z = (z2 - z1) / se
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def _fr_rows(event_id: str, df: pl.DataFrame, tier: str,
             node_a: str, node_b: str, feature: str) -> dict:
    n_pre, r_pre = _phase_corr(df, "pre",   node_a, node_b, feature)
    n_pan, r_pan = _phase_corr(df, "panic", node_a, node_b, feature)
    z, p = _fisher_diff(r_pre, n_pre, r_pan, n_pan)
    contagion = z is not None and p is not None and z > 0 and p < 0.05
    return {
        "event_id":     event_id,
        "tier":         tier,
        "node_a":       node_a,
        "node_b":       node_b,
        "feature":      feature,
        "n_pre":        n_pre,
        "rho_pre":      round(r_pre, 4) if r_pre is not None else None,
        "n_panic":      n_pan,
        "rho_panic":    round(r_pan, 4) if r_pan is not None else None,
        "fisher_z":     round(z, 4) if z is not None else None,
        "p_value":      round(p, 4) if p is not None else None,
        "contagion":    contagion,
    }


def _hmm_auroc_dex(df: pl.DataFrame, node_id: str, n_states: int = 3) -> float | None:
    """HMM on DEX pool features: |flow|, |price-1|, |imbalance| (matches script 27)."""
    try:
        from hmmlearn.hmm import GaussianHMM
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None
    d = (
        df.filter(pl.col("node_id") == node_id)
        .with_columns((pl.col("event_time_seconds") // 3_600).alias("h"))
        .group_by("h")
        .agg(
            pl.col("usdc_net_sold_1h").sum().alias("flow"),
            pl.col("implied_pool_price").mean().alias("px"),
            pl.col("reserve_imbalance").mean().alias("imb"),
            pl.col("event_phase").first().alias("phase"),
        )
        .sort("h")
    )
    af  = np.abs(np.nan_to_num(d["flow"].to_numpy()))
    pxd = np.abs(np.nan_to_num(d["px"].to_numpy(), nan=1.0) - 1.0)
    imb = np.abs(np.nan_to_num(d["imb"].to_numpy()))
    X = np.column_stack([af, pxd, imb])
    y = (d["phase"].to_numpy() == "panic").astype(int)
    if len(np.unique(y)) < 2 or len(y) < 30:
        return None
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    n = min(n_states, max(2, len(y) // 30))
    model = GaussianHMM(n_components=n, covariance_type="diag",
                        n_iter=300, random_state=0).fit(X)
    post   = model.predict_proba(X)
    stress = int(np.argmax(model.means_[:, 1]))
    return float(roc_auc_score(y, post[:, stress]))


def _hmm_auroc_cex(df: pl.DataFrame, node_id: str, n_states: int = 3) -> float | None:
    """HMM on CEX features: |mid_price-1|, |basis_bps|."""
    try:
        from hmmlearn.hmm import GaussianHMM
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None
    d = (
        df.filter(pl.col("node_id") == node_id)
        .with_columns((pl.col("event_time_seconds") // 3_600).alias("h"))
        .group_by("h")
        .agg(
            pl.col("mid_price").mean().alias("px"),
            pl.col("basis_bps").abs().mean().alias("basis"),
            pl.col("event_phase").first().alias("phase"),
        )
        .sort("h")
    )
    pxd   = np.abs(np.nan_to_num(d["px"].to_numpy(), nan=1.0) - 1.0)
    basis = np.abs(np.nan_to_num(d["basis"].to_numpy()))
    X = np.column_stack([pxd, basis])
    y = (d["phase"].to_numpy() == "panic").astype(int)
    if len(np.unique(y)) < 2 or len(y) < 30:
        return None
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    n = min(n_states, max(2, len(y) // 30))
    model = GaussianHMM(n_components=n, covariance_type="diag",
                        n_iter=300, random_state=0).fit(X)
    post   = model.predict_proba(X)
    stress = int(np.argmax(model.means_[:, 0]))  # highest |price-1| = stress state
    return float(roc_auc_score(y, post[:, stress]))


def main() -> None:
    fr_rows  = []
    hmm_rows = []

    for event_id in _EVENTS:
        path = gold_root() / f"dataset_contagion_features_{event_id}.parquet"
        if not path.exists():
            logger.warning("Missing gold panel for %s; skipping.", event_id)
            continue
        df = pl.read_parquet(path)
        if "event_phase" not in df.columns:
            continue

        # ── Forbes-Rigobon: Tier-A (DEX pair, usdc_net_sold_1h) ──────────────
        if event_id in _TIER_A_PAIRS:
            a, b = _TIER_A_PAIRS[event_id]
            row = _fr_rows(event_id, df, "A", a, b, "usdc_net_sold_1h")
            fr_rows.append(row)
            logger.info("FR Tier-A %s: z=%s p=%s contagion=%s",
                        event_id, row["fisher_z"], row["p_value"], row["contagion"])

        # ── Forbes-Rigobon: Tier-B (CEX pair, basis_bps) ─────────────────────
        if event_id in _TIER_B_PAIRS:
            a, b = _TIER_B_PAIRS[event_id]
            if a != b:  # skip if no distinct CEX pair
                row = _fr_rows(event_id, df, "B", a, b, "basis_bps")
                fr_rows.append(row)
                logger.info("FR Tier-B %s: z=%s p=%s contagion=%s",
                            event_id, row["fisher_z"], row["p_value"], row["contagion"])
            else:
                logger.info("FR Tier-B %s: single CEX node, pair skipped.", event_id)

        # ── HMM: Tier-A (curve_3pool, on-chain DEX features) ─────────────────
        tier_a_auroc = _hmm_auroc_dex(df, "curve_3pool")
        hmm_rows.append({
            "event_id": event_id, "tier": "A", "node": "curve_3pool",
            "auroc": round(tier_a_auroc, 4) if tier_a_auroc is not None else None,
            "detects": bool(tier_a_auroc is not None and tier_a_auroc >= 0.80),
        })
        logger.info("HMM Tier-A %s: AUROC=%s", event_id, tier_a_auroc)

        # ── HMM: Tier-B (CEX node, mid_price + basis_bps) ────────────────────
        cex_node_map = {
            "usdt_curve_2023": "usdt_binance",
            "terra_luna_2022": "ust_binance",
            "ftx_2022":        "usdt_binance",
            "busd_2023":       "busd_binance",
            "usdc_svb_2023":   "usdc_coinbase",
        }
        cex_node = cex_node_map.get(event_id)
        if cex_node:
            tier_b_auroc = _hmm_auroc_cex(df, cex_node)
            hmm_rows.append({
                "event_id": event_id, "tier": "B", "node": cex_node,
                "auroc": round(tier_b_auroc, 4) if tier_b_auroc is not None else None,
                "detects": bool(tier_b_auroc is not None and tier_b_auroc >= 0.80),
            })
            logger.info("HMM Tier-B %s: AUROC=%s", event_id, tier_b_auroc)

    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    if fr_rows:
        with (out_dir / "table_tier_robustness_fr.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(fr_rows[0].keys()))
            w.writeheader(); w.writerows(fr_rows)
    if hmm_rows:
        with (out_dir / "table_tier_robustness_hmm.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(hmm_rows[0].keys()))
            w.writeheader(); w.writerows(hmm_rows)

    a_contagion = [r for r in fr_rows if r["tier"] == "A" and r["contagion"]]
    b_contagion = [r for r in fr_rows if r["tier"] == "B" and r["contagion"]]
    logger.info("FR contagion: Tier-A=%d  Tier-B=%d", len(a_contagion), len(b_contagion))
    logger.info("Wrote table_tier_robustness_fr.csv and table_tier_robustness_hmm.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="(unused; reads gold parquets directly)")
    _ = ap.parse_args()
    main()
