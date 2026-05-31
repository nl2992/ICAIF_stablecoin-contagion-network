"""Cross-protocol stress prediction experiment.

Demonstrates that Uniswap v3 Tier-A features improve next-hour stress
prediction for Curve 3pool beyond what Curve-only features achieve.

Task: binary classification — will curve_3pool usdc_net_sold_1h be
      in the top quartile (Q75) one hour ahead?

Feature conditions
------------------
  curve_only    : lagged Curve 3pool + crvUSD/USDT features (t-1, t-2, t-3)
  cross_protocol: curve_only features + Uniswap v3 lags + cross-flow differential

Models: LogisticRegression, LightGBM

Evaluation: AUROC with 95% CI via bootstrap (1000 draws).
Split: temporal — first 70% of hours train, last 30% test (no shuffling).

All input data is Tier-A (Etherscan on-chain event logs).

Usage
-----
    python scripts/22_run_cross_protocol_prediction.py
    python scripts/22_run_cross_protocol_prediction.py --event usdt_curve_2023

Writes
------
    results/paper/tables/table_prediction_cross_protocol.csv
    results/paper/figures_cross_protocol/A_prediction_auroc.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import polars as pl

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

# Columbia palette
CNV  = "#003865"
CAMB = "#E67E22"
CTA  = "#27AE60"
CBLU = "#2980B9"
CBKG = "#F8FBFD"
CSL  = "#2C3E50"

EVENT_ID   = "usdt_curve_2023"
BRONZE_DIR = Path("data/bronze") / EVENT_ID
OUT_TABLES = Path("results/paper/tables")
OUT_FIGS   = Path("results/paper/figures_cross_protocol")
LABEL_QUANTILE = 0.75   # top-quartile stress label
TRAIN_FRAC     = 0.70   # first 70% of hours → train
N_BOOTSTRAP    = 1000
RANDOM_STATE   = 42


# ---------------------------------------------------------------------------
# Data loading and feature engineering
# ---------------------------------------------------------------------------

def load_aligned_panel() -> pl.DataFrame:
    """Load and align the three Tier-A pool parquets to a common hourly grid."""
    c3  = pl.read_parquet(BRONZE_DIR / "curve_3pool_pool_events.parquet")
    crv = pl.read_parquet(BRONZE_DIR / "curve_crvusd_usdt_pool_events.parquet")
    uni = pl.read_parquet(BRONZE_DIR / "uniswap_usdc_usdt_005_pool_events.parquet")

    # Rename columns before joining
    c3  = c3.rename({c: f"c3_{c}"  for c in c3.columns  if c != "wall_clock_utc"})
    crv = crv.rename({c: f"crv_{c}" for c in crv.columns if c != "wall_clock_utc"})
    uni = uni.rename({c: f"uni_{c}" for c in uni.columns if c != "wall_clock_utc"})

    # Full outer join on timestamp, fill zeros for missing hours
    df = (
        c3.join(crv, on="wall_clock_utc", how="left")
          .join(uni, on="wall_clock_utc", how="left")
    )
    # Fill nulls with 0 (hours with no trades = zero flow)
    df = df.with_columns([
        pl.col(c).fill_null(0.0)
        for c in df.columns if c != "wall_clock_utc"
    ])
    df = df.sort("wall_clock_utc")
    return df


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add lagged features and cross-protocol differentials."""
    for lag in [1, 2, 3]:
        df = df.with_columns([
            pl.col("c3_usdc_net_sold_1h").shift(lag).alias(f"c3_flow_lag{lag}"),
            pl.col("crv_usdc_net_sold_1h").shift(lag).alias(f"crv_flow_lag{lag}"),
            pl.col("uni_usdc_net_sold_1h").shift(lag).alias(f"uni_flow_lag{lag}"),
            pl.col("c3_n_events").shift(lag).alias(f"c3_nevents_lag{lag}"),
            pl.col("uni_n_events").shift(lag).alias(f"uni_nevents_lag{lag}"),
        ])

    # Cross-protocol flow differential at each lag
    for lag in [1, 2]:
        df = df.with_columns(
            (pl.col(f"c3_flow_lag{lag}") - pl.col(f"uni_flow_lag{lag}"))
            .alias(f"cross_diff_lag{lag}")
        )

    # Drop rows with NaN from lagging (first 3 rows)
    df = df.filter(pl.col("c3_flow_lag3").is_not_null())
    return df


def add_label(df: pl.DataFrame) -> pl.DataFrame:
    """Binary label: is curve_3pool usdc_net_sold_1h in top quartile next hour?"""
    # shift(-1) gives the NEXT hour's value
    next_flow = pl.col("c3_usdc_net_sold_1h").shift(-1)
    q75 = df["c3_usdc_net_sold_1h"].quantile(LABEL_QUANTILE)
    df = df.with_columns(
        pl.when(next_flow > q75)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .cast(pl.Int8)
        .alias("label_stress_next_hour")
    )
    # Drop last row (no next-hour label)
    df = df.filter(pl.col("label_stress_next_hour").is_not_null())
    return df, q75


# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------

CURVE_ONLY_FEATURES = [
    "c3_flow_lag1", "c3_flow_lag2", "c3_flow_lag3",
    "crv_flow_lag1", "crv_flow_lag2",
    "c3_nevents_lag1",
]

CROSS_PROTOCOL_FEATURES = CURVE_ONLY_FEATURES + [
    "uni_flow_lag1", "uni_flow_lag2", "uni_flow_lag3",
    "uni_nevents_lag1",
    "cross_diff_lag1", "cross_diff_lag2",
]


def bootstrap_auc(y_true: np.ndarray, y_score: np.ndarray, n: int = N_BOOTSTRAP
                  ) -> tuple[float, float, float]:
    """Return mean AUROC and 95% CI via percentile bootstrap."""
    rng = np.random.default_rng(RANDOM_STATE)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    aucs = np.array(aucs)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(np.mean(aucs)), lo, hi


def run_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_set: str,
    model_name: str,
    model,
) -> dict:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model.fit(X_tr, y_train)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_te)[:, 1]
    else:
        y_score = model.decision_function(X_te)

    auroc = roc_auc_score(y_test, y_score)
    auprc = average_precision_score(y_test, y_score)
    auc_mean, auc_lo, auc_hi = bootstrap_auc(y_test, y_score)

    return {
        "feature_set": feature_set,
        "model": model_name,
        "n_features": X_train.shape[1],
        "n_train": len(y_train),
        "n_test": len(y_test),
        "prevalence_test": float(y_test.mean()),
        "AUROC": round(auroc, 4),
        "AUROC_CI_lo": round(auc_lo, 4),
        "AUROC_CI_hi": round(auc_hi, 4),
        "AUPRC": round(auprc, 4),
    }


def build_models() -> list[tuple[str, object]]:
    models = [("LogisticRegression", LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE))]
    if _HAS_LGBM:
        models.append(("LightGBM", LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            random_state=RANDOM_STATE, verbose=-1,
        )))
    return models


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_figure(results: pl.DataFrame, out_path: Path) -> None:
    models = results["model"].unique().to_list()
    feature_sets = ["curve_only", "cross_protocol"]
    labels_map = {"curve_only": "Curve-only", "cross_protocol": "Cross-protocol"}
    colours = {m: c for m, c in zip(models, [CNV, CAMB, CTA, CBLU])}

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=CBKG)
    ax.set_facecolor(CBKG)

    n_models = len(models)
    bar_w = 0.35
    x = np.arange(len(feature_sets))

    for i, model in enumerate(models):
        sub = results.filter(pl.col("model") == model).sort("feature_set")
        aucs = [sub.filter(pl.col("feature_set") == fs)["AUROC"][0] for fs in feature_sets]
        lo   = [sub.filter(pl.col("feature_set") == fs)["AUROC_CI_lo"][0] for fs in feature_sets]
        hi   = [sub.filter(pl.col("feature_set") == fs)["AUROC_CI_hi"][0] for fs in feature_sets]
        err  = [[a - l for a, l in zip(aucs, lo)], [h - a for a, h in zip(aucs, hi)]]

        offset = (i - (n_models - 1) / 2) * bar_w * 0.9
        bars = ax.bar(
            x + offset, aucs, bar_w * 0.85,
            color=colours[model], alpha=0.88, label=model,
            yerr=err, capsize=3, error_kw={"elinewidth": 1.2, "ecolor": CSL},
        )
        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{auc:.3f}", ha="center", va="bottom",
                    fontsize=8.5, color=CSL, fontweight="bold")

    # Baseline
    ax.axhline(0.5, color="grey", ls="--", lw=1.0, label="Random baseline")

    ax.set_xticks(x)
    ax.set_xticklabels([labels_map[fs] for fs in feature_sets], fontsize=11)
    ax.set_ylabel("AUROC (next-hour stress, 95% CI)", fontsize=10, color=CSL)
    ax.set_title(
        "Cross-protocol features improve next-hour stress prediction\n"
        "Curve 3pool Q75 label · USDT/Curve 2023 · Tier-A only",
        fontsize=10.5, color=CNV, pad=8,
    )
    ax.set_ylim(0.40, 1.02)
    ax.legend(fontsize=9, framealpha=0.4, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", default=EVENT_ID)
    args = parser.parse_args()

    print(f"\n=== Cross-protocol stress prediction: {args.event} ===\n")

    # Load data
    df = load_aligned_panel()
    df = add_features(df)
    df, q75_threshold = add_label(df)

    print(f"Panel: {len(df)} hours | Stress Q75 threshold: {q75_threshold:,.0f} USDC")

    # Temporal split
    n = len(df)
    n_train = int(n * TRAIN_FRAC)
    train = df[:n_train]
    test  = df[n_train:]

    y_train = train["label_stress_next_hour"].to_numpy()
    y_test  = test["label_stress_next_hour"].to_numpy()

    print(f"Train: {n_train} rows ({y_train.mean():.1%} stressed) | "
          f"Test: {n - n_train} rows ({y_test.mean():.1%} stressed)\n")

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise SystemExit("ERROR: label has only one class in train or test.")

    models = build_models()
    rows = []

    for feature_set, feat_cols in [
        ("curve_only",     CURVE_ONLY_FEATURES),
        ("cross_protocol", CROSS_PROTOCOL_FEATURES),
    ]:
        available = [c for c in feat_cols if c in df.columns]
        X_train = train.select(available).to_numpy().astype(np.float32)
        X_test  = test.select(available).to_numpy().astype(np.float32)

        for model_name, model in models:
            row = run_experiment(
                X_train, y_train, X_test, y_test,
                feature_set, model_name, model,
            )
            rows.append(row)
            print(f"  [{feature_set:20s}] {model_name:20s}  "
                  f"AUROC={row['AUROC']:.4f} [{row['AUROC_CI_lo']:.3f}, {row['AUROC_CI_hi']:.3f}]  "
                  f"AUPRC={row['AUPRC']:.4f}")

    results = pl.DataFrame(rows)
    results = results.with_columns(pl.lit(args.event).alias("event_id"))

    # Lift: cross_protocol AUROC - curve_only AUROC per model
    print("\n--- Cross-protocol AUROC lift ---")
    for model_name, _ in models:
        sub = results.filter(pl.col("model") == model_name)
        auc_co = sub.filter(pl.col("feature_set") == "curve_only")["AUROC"][0]
        auc_cp = sub.filter(pl.col("feature_set") == "cross_protocol")["AUROC"][0]
        lift = auc_cp - auc_co
        print(f"  {model_name:20s}  lift = +{lift:.4f} ({lift*100:.1f} pp)")

    # Save
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_TABLES / "table_prediction_cross_protocol.csv"
    results.write_csv(out_csv)
    print(f"\nSaved: {out_csv}")

    out_fig = OUT_FIGS / "A_prediction_auroc.png"
    make_figure(results, out_fig)


if __name__ == "__main__":
    main()
