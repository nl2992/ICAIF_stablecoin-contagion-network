"""Build unified convergent evidence table: 5 events × 4 methods.

Sources (all pre-computed, just reading CSVs):
  - Forbes-Rigobon:    results/tables/table_bootstrap_fr.csv
  - Lead-lag:          results/tables/table_leadlag_tests_*.csv
  - Transfer entropy:  results/tables/table_transfer_entropy_*.csv
  - Within-event AUROC: results/tables/table_prediction_metrics_*.csv
                        results/tables/table_hmm_ablation.csv

Outputs:
  results/tables/table_convergent_evidence.csv
  results/tables/table_convergent_evidence.json
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parents[1]
TABLES = ROOT / "results/tables"
OUT = TABLES

EVENTS = [
    ("usdc_svb_2023",    "USDC/SVB 2023"),
    ("usdt_curve_2023",  "USDT/Curve 2023"),
    ("terra_luna_2022",  "Terra/Luna 2022"),
    ("ftx_2022",         "FTX 2022"),
    ("busd_2023",        "BUSD 2023"),
]


# ── Forbes-Rigobon ──────────────────────────────────────────────────────────

def load_fr() -> dict:
    path = TABLES / "table_bootstrap_fr.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        eid = str(row.get("event_id", ""))
        out[eid] = {
            "fr_z":      round(float(row["fisher_z_obs"]), 3),
            "fr_p_gt0":  round(float(row["p_gt0_bootstrap"]), 3),
            "fr_ci_lo":  round(float(row["ci_lo_95"]), 3),
            "fr_ci_hi":  round(float(row["ci_hi_95"]), 3),
            "fr_sig":    bool(row.get("contagion_ci", False)),
        }
    return out


# ── Lead-lag ────────────────────────────────────────────────────────────────

def load_leadlag() -> dict:
    """For each event: count Bonferroni-significant pairs; pick strongest p."""
    out = {}
    for eid, _ in EVENTS:
        path = TABLES / f"table_leadlag_tests_{eid}.csv"
        if not path.exists():
            out[eid] = {"ll_sig_pairs": 0, "ll_min_p": None, "ll_best_lag_s": None}
            continue
        df = pd.read_csv(path)
        sig = df[df.get("significant_bonferroni", df.get("significant_p01", pd.Series(False)))]
        min_p = float(df["p_value"].min()) if "p_value" in df else None
        best_lag = None
        if not sig.empty and "peak_lag_seconds" in sig.columns:
            # pick row with smallest p_value
            best_row = sig.loc[sig["p_value"].idxmin()]
            best_lag = int(best_row["peak_lag_seconds"])
        out[eid] = {
            "ll_sig_pairs":  len(sig),
            "ll_min_p":      round(min_p, 5) if min_p is not None else None,
            "ll_best_lag_s": best_lag,
        }
    return out


# ── Transfer entropy ────────────────────────────────────────────────────────

def load_te() -> dict:
    out = {}
    for eid, _ in EVENTS:
        path = TABLES / f"table_transfer_entropy_{eid}.csv"
        if not path.exists():
            out[eid] = {"te_sig_block_fdr": 0, "te_max_te": None}
            continue
        df = pd.read_csv(path)
        col = "significant_block_fdr" if "significant_block_fdr" in df.columns else "significant_fdr"
        sig = df[df[col].astype(str).str.lower() == "true"] if col in df.columns else df.head(0)
        max_te = float(df["te_i_to_j"].max()) if "te_i_to_j" in df.columns else None
        out[eid] = {
            "te_sig_block_fdr": len(sig),
            "te_max_te":        round(max_te, 3) if max_te else None,
        }
    return out


# ── Within-event AUROC ──────────────────────────────────────────────────────

def load_within_auroc() -> dict:
    """Best within-event AUROC: HMM ablation if available, else best ML model."""
    # HMM ablation (currently only usdt_curve_2023)
    hmm_map = {}
    hmm_path = TABLES / "table_hmm_ablation.csv"
    if hmm_path.exists():
        hmm = pd.read_csv(hmm_path)
        for eid in hmm["event_id"].unique():
            sub = hmm[hmm["event_id"] == eid]
            best = sub[sub["detects"].astype(str).str.lower() == "true"]
            if not best.empty:
                hmm_map[eid] = round(float(best["auroc"].max()), 4)

    out = {}
    for eid, _ in EVENTS:
        if eid in hmm_map:
            out[eid] = {"within_auroc": hmm_map[eid], "within_method": "HMM"}
            continue
        # Fallback: prediction_metrics per-event table
        path = TABLES / f"table_prediction_metrics_{eid}.csv"
        if not path.exists():
            out[eid] = {"within_auroc": None, "within_method": None}
            continue
        df = pd.read_csv(path)
        # Use within-event (not cross-event) split if column present
        if "split_type" in df.columns:
            df = df[df["split_type"].str.contains("within", case=False, na=False)]
        if df.empty:
            df = pd.read_csv(path)  # reload and use best model overall
        if "AUROC" in df.columns:
            best_auroc = float(df["AUROC"].max())
        elif "auroc" in df.columns:
            best_auroc = float(df["auroc"].max())
        else:
            best_auroc = None
        out[eid] = {"within_auroc": round(best_auroc, 4) if best_auroc else None,
                    "within_method": "ML-best"}
    return out


# ── Convergence score ────────────────────────────────────────────────────────

def convergence_score(row: dict) -> str:
    signals = 0
    # FR: p_gt0 >= 0.90 counts
    fr_p = row.get("fr_p_gt0")
    if fr_p is not None and fr_p >= 0.90:
        signals += 1
    # Lead-lag: any Bonferroni-significant pair
    if (row.get("ll_sig_pairs") or 0) > 0:
        signals += 1
    # Transfer entropy: any block-FDR significant
    if (row.get("te_sig_block_fdr") or 0) > 0:
        signals += 1
    # Within-event AUROC ≥ 0.75
    auroc = row.get("within_auroc")
    if auroc and auroc >= 0.75:
        signals += 1
    return {0: "NONE", 1: "WEAK", 2: "MODERATE", 3: "STRONG", 4: "VERY STRONG"}.get(signals, "VERY STRONG")


def main():
    fr   = load_fr()
    ll   = load_leadlag()
    te   = load_te()
    auroc = load_within_auroc()

    rows = []
    for eid, label in EVENTS:
        row = {"event_id": eid, "event_label": label}
        row.update(fr.get(eid, {"fr_z": None, "fr_p_gt0": None}))
        row.update(ll.get(eid, {}))
        row.update(te.get(eid, {}))
        row.update(auroc.get(eid, {}))
        row["convergence"] = convergence_score(row)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "table_convergent_evidence.csv", index=False)

    # Print readable table
    print("=" * 95)
    print("CONVERGENT EVIDENCE TABLE — 5 EVENTS × 4 METHODS")
    print("=" * 95)
    hdr = f"{'Event':22s}  {'FR z':>6}  {'FR p>0':>7}  {'LL sig':>6}  {'TE sig':>6}  {'AUROC':>6}  {'Convergence'}"
    print(hdr)
    print("-" * 95)
    for row in rows:
        fz  = f"{row.get('fr_z', '-'):>6}" if row.get("fr_z") is not None else f"{'—':>6}"
        fp  = f"{row.get('fr_p_gt0', '-'):>7.3f}" if row.get("fr_p_gt0") is not None else f"{'—':>7}"
        ll_ = f"{row.get('ll_sig_pairs', 0):>6}"
        te_ = f"{row.get('te_sig_block_fdr', 0):>6}"
        au  = f"{row.get('within_auroc', '-'):>6.3f}" if row.get("within_auroc") else f"{'—':>6}"
        print(f"{row['event_label']:22s}  {fz}  {fp}  {ll_}  {te_}  {au}  {row['convergence']}")

    print()
    (OUT / "table_convergent_evidence.json").write_text(
        json.dumps(rows, indent=2, default=str)
    )
    print(f"Saved: {OUT}/table_convergent_evidence.csv")
    print(f"Saved: {OUT}/table_convergent_evidence.json")


if __name__ == "__main__":
    main()
