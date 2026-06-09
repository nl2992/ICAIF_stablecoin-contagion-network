"""Cross-event transfer detector: does per-event normalization rescue transfer?

The paper reports that supervised cross-event stress prediction fails under
*concept shift* (leave-one-event-out AUROC ~ chance). The SnapshotGCN also fails.
This experiment tests a specific, cheap hypothesis: much of the cross-event gap is
covariate-SCALE shift (each event's microstructure features live on a different
scale / base rate), which a per-event normalization removes -- enabling a learned
detector to transfer where raw-feature models cannot.

Design (leave-one-event-out, 4 events):
  - Task: predict label_downstream_gt50bps_5m (downstream stress onset).
  - Model: HistGradientBoosting (handles NaN, fast).
  - Two feature pipelines:
      RAW : StandardScaler fit on train events, applied to test event.
      PEN : per-event rank-normalization (each event's features -> within-event
            quantile rank in [0,1]); removes per-event scale/distribution shift.
            Transductive on the test event (uses its feature values, not labels).
  - Report per-fold and mean ROC-AUC for both pipelines.

Honest outcomes:
  - PEN >> RAW  => transfer is rescued; learned detector is viable (paper upgrade).
  - PEN ~ RAW ~ chance => confirms genuine concept shift (label-map, not scale);
    strengthens the paper's existing negative rigorously.

Usage:
    python scripts/run_transfer_detector.py
"""
from __future__ import annotations

import glob
import json
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
LABEL = "label_downstream_gt50bps_5m"
SEED = 42


def load_events() -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(glob.glob(str(_ROOT / "data/gold/dataset_prediction_*.parquet"))):
        ev = os.path.basename(f).replace("dataset_prediction_", "").replace(".parquet", "")
        out[ev] = pd.read_parquet(f)
    return out


def feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {c for c in df.columns if c.startswith("label_")}
    return [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def per_event_rank(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    # within-event quantile rank in [0,1]; robust to scale + distribution shift.
    # TRANSDUCTIVE: uses the full within-event distribution (upper bound).
    return df[cols].rank(pct=True)


def per_event_causal_rank(df: pd.DataFrame, cols: list[str], window: int = 2000) -> pd.DataFrame:
    # CAUSAL + distribution-robust: trailing-window percentile rank of each point
    # within the preceding `window` observations of the same event. Online-valid.
    r = df[cols].rolling(window, min_periods=50).rank(pct=True)
    return r.bfill().fillna(0.5)


def per_event_causal_z(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    # CAUSAL within-event normalization: expanding (mean,std) up to each row, so
    # it uses only past observations of the same event -> online/deployment-valid.
    X = df[cols]
    mu = X.expanding(min_periods=20).mean()
    sd = X.expanding(min_periods=20).std()
    z = (X - mu) / sd.replace(0, np.nan)
    return z.bfill().fillna(0.0)


def evaluate(events: dict, cols: list[str], pipeline: str) -> dict:
    fold_auc = {}
    for held in events:
        train_ev = [e for e in events if e != held]
        if pipeline in ("PEN", "CPEN", "CPENR"):
            norm = {"PEN": per_event_rank, "CPEN": per_event_causal_z,
                    "CPENR": per_event_causal_rank}[pipeline]
            Xtr = pd.concat([norm(events[e], cols) for e in train_ev], ignore_index=True)
            ytr = pd.concat([events[e][LABEL] for e in train_ev], ignore_index=True)
            Xte = norm(events[held], cols)
            yte = events[held][LABEL]
            Xtr_v, Xte_v = Xtr.to_numpy(), Xte.to_numpy()
        else:  # RAW
            Xtr = pd.concat([events[e][cols] for e in train_ev], ignore_index=True)
            ytr = pd.concat([events[e][LABEL] for e in train_ev], ignore_index=True)
            Xte = events[held][cols]
            yte = events[held][LABEL]
            sc = StandardScaler()
            Xtr_v = sc.fit_transform(Xtr.to_numpy())
            Xte_v = sc.transform(Xte.to_numpy())
        if yte.nunique() < 2 or ytr.nunique() < 2:
            fold_auc[held] = float("nan"); continue
        clf = HistGradientBoostingClassifier(random_state=SEED, max_iter=200,
                                             learning_rate=0.05)
        clf.fit(Xtr_v, ytr.to_numpy())
        p = clf.predict_proba(Xte_v)[:, 1]
        fold_auc[held] = round(float(roc_auc_score(yte.to_numpy(), p)), 4)
    vals = [v for v in fold_auc.values() if not np.isnan(v)]
    return {"per_fold": fold_auc, "mean_auroc": round(float(np.mean(vals)), 4)}


def main() -> None:
    events = load_events()
    cols = feature_cols(next(iter(events.values())))
    # keep only cols present & numeric in all events
    cols = [c for c in cols if all(c in df.columns for df in events.values())]
    # drop columns that are all-NaN or constant in any event (break HGB binning)
    def ok(c):
        for df in events.values():
            s = df[c]
            if s.notna().sum() == 0 or s.nunique(dropna=True) < 2:
                return False
        return True
    cols = [c for c in cols if ok(c)]
    print(f"events={list(events)} | n_features={len(cols)} | label={LABEL}")
    print(f"base rates: " + ", ".join(f"{e}={events[e][LABEL].mean():.3f}" for e in events))

    raw = evaluate(events, cols, "RAW")
    pen = evaluate(events, cols, "PEN")
    cpen = evaluate(events, cols, "CPEN")
    cpenr = evaluate(events, cols, "CPENR")
    result = {"label": LABEL, "n_features": len(cols),
              "RAW_standardscaler": raw,
              "PEN_per_event_rank_transductive": pen,
              "CPEN_per_event_causal_zscore": cpen,
              "CPENR_per_event_causal_rolling_rank": cpenr,
              "delta_PEN_minus_RAW": round(pen["mean_auroc"] - raw["mean_auroc"], 4),
              "delta_CPENR_minus_RAW": round(cpenr["mean_auroc"] - raw["mean_auroc"], 4)}
    out = _ROOT / "results/eval"; out.mkdir(parents=True, exist_ok=True)
    (out / "transfer_detector.json").write_text(json.dumps(result, indent=2))
    print("\nRAW   (raw features):                       ", raw)
    print("PEN   (per-event rank, transductive):       ", pen)
    print("CPEN  (per-event causal z, online):         ", cpen)
    print("CPENR (per-event causal rolling-rank, online):", cpenr)
    print(f"\nΔ mean AUROC: PEN-RAW={result['delta_PEN_minus_RAW']:+.4f}  "
          f"CPENR-RAW={result['delta_CPENR_minus_RAW']:+.4f}")


if __name__ == "__main__":
    main()
