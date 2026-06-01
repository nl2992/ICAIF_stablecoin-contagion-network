"""CEX arbitrage execution microstructure: multi-event meta-labeling experiment.

Implements the four training conditions described in the implementation plan:

  Condition A: train secondary LightGBM on Terra/LUNA only → test on SVB
  Condition B: train on Celsius/3AC only           → test on SVB
  Condition C: train on FTX only                   → test on SVB
  Condition D: train on all four pooled            → test on SVB

Threshold is always calibrated on Terra/LUNA. Test is always SVB.

Data sources
------------
  USDCUSDT 1m klines : Binance Vision (spot, daily or monthly archive)
  BTCUSDT  1m klines : Binance Vision (spot, daily)
  BTCUSDC  1m klines : PROXIED from BTCUSDT × USDCUSDT with 20% depth haircut
                       (BTCUSDC spot was illiquid during all 2022-2023 windows)

Usage
-----
    python scripts/23_run_arb_meta_label.py
    python scripts/23_run_arb_meta_label.py --no-download   # use cached klines only

Writes
------
    results/paper/tables/table_arb_meta_label_conditions.csv
    results/paper/tables/table_arb_optical_summary.csv
    results/paper/figures_arb/figure_arb_oracle_capture.png
"""

from __future__ import annotations

import argparse
import math
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from stressnet.models.arb_meta_label import (
    PRICE_PLUS_BOOK_FEATURES,
    PRIMARY_THRESHOLD_BPS,
    apply_primary_filter,
    build_minute_panel,
    calibrate_threshold,
    compute_optical_summary,
    engineer_features,
    evaluate_strategy,
    fetch_klines_range,
    train_secondary,
)

# ---------------------------------------------------------------------------
# Event window definitions
# ---------------------------------------------------------------------------

EVENTS: dict[str, dict] = {
    "terra_luna_2022": {
        "name": "Terra/LUNA",
        "mechanism": "Algorithmic",
        "start": date(2022, 5, 1),
        "end": date(2022, 5, 31),
        "role": "train_A_and_validate",
        # USDCUSDT spot available on Binance Vision through ~Sep 2022
        "stable_sym": "USDCUSDT",
        "basis_mode": "stable_direct",  # b = stable_close - 1
    },
    "celsius_3ac_2022": {
        "name": "Celsius/3AC",
        "mechanism": "Exchange credit",
        "start": date(2022, 6, 12),
        "end": date(2022, 6, 20),
        "role": "train_B",
        "stable_sym": "USDCUSDT",
        "basis_mode": "stable_direct",
    },
    "ftx_2022": {
        "name": "FTX",
        "mechanism": "Exchange credit",
        "start": date(2022, 11, 6),
        "end": date(2022, 11, 14),
        "role": "train_C",
        # USDCUSDT Binance Vision gap Oct–Feb 2022-23; BUSDUSDT daily archives available.
        # BUSD was pegged and liquid during FTX; captures exchange-credit stablecoin stress.
        "stable_sym": "BUSDUSDT",
        "basis_mode": "stable_direct",
    },
    "busd_2023": {
        "name": "BUSD",
        "mechanism": "Regulatory",
        "start": date(2023, 2, 13),
        "end": date(2023, 2, 28),
        "role": "train_D_extra",
        # BUSDUSDT directly captures the regulatory wind-down basis signal.
        "stable_sym": "BUSDUSDT",
        "basis_mode": "stable_direct",
    },
    "usdc_svb_2023": {
        "name": "SVB",
        "mechanism": "Fiat-reserve bank shock",
        "start": date(2023, 3, 8),
        "end": date(2023, 3, 20),
        "role": "test",
        # USDCUSDT monthly archive March 2023 is available on Binance Vision.
        # ingest_binance_range falls back to monthly automatically.
        "stable_sym": "USDCUSDT",
        "basis_mode": "stable_direct",
    },
}

# Columbia palette
CNV  = "#003865"
CAMB = "#E67E22"
CTA  = "#27AE60"
CBLU = "#2980B9"
CBKG = "#F8FBFD"
CSL  = "#2C3E50"
CGRY = "#7F8C8D"

OUT_TABLES = Path("results/paper/tables")
OUT_FIGS   = Path("results/paper/figures_arb")
CACHE_ROOT = Path("data/_arb_kline_cache")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _fetch_sym(sym: str, cfg: dict, cache_dir: Path, download: bool) -> pl.DataFrame | None:
    """Fetch one symbol's klines, using parquet cache if available.

    Tries Binance Vision daily files first, then monthly archives (handled
    transparently by fetch_klines_range). Note: USDCUSDT daily archives on
    Binance Vision are unavailable Oct 2022–Feb 2023; the monthly archive for
    March 2023 is available and covers the SVB test window.
    """
    parquet_path = cache_dir / f"{sym}_1m.parquet"
    event_id = cache_dir.name
    if parquet_path.exists():
        df = pl.read_parquet(parquet_path)
        print(f"  [{event_id}] {sym}: {df.height} rows from cache")
        return df
    if not download:
        print(f"  [{event_id}] {sym}: no cache (run without --no-download to fetch)")
        return None

    print(f"  [{event_id}] {sym}: fetching {cfg['start']} – {cfg['end']} ...")
    df = fetch_klines_range(sym, cfg["start"], cfg["end"], cache_dir=cache_dir / sym)
    if df is None:
        print(f"  [{event_id}] {sym}: NOT FOUND on Binance Vision (daily or monthly)")
        return None
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(parquet_path)
    print(f"  [{event_id}] {sym}: {df.height} rows")
    return df


def load_event_panel(
    event_id: str,
    cfg: dict,
    download: bool = True,
) -> pl.DataFrame | None:
    """Load or download klines for an event and build the minute panel.

    Uses per-event pair config:
      stable_sym   : stablecoin-side symbol (USDCUSDT, BUSDUSDT, or BTCUSDC)
      basis_mode   : "stable_direct" or "triangular"
    """
    cache_dir = CACHE_ROOT / event_id
    stable_sym  = cfg["stable_sym"]
    basis_mode  = cfg["basis_mode"]

    stable = _fetch_sym(stable_sym, cfg, cache_dir, download)
    btcusdt = _fetch_sym("BTCUSDT", cfg, cache_dir, download)

    if stable is None or btcusdt is None:
        return None

    panel = build_minute_panel(stable, btcusdt, basis_mode=basis_mode)
    panel = engineer_features(panel)
    n_fires = apply_primary_filter(panel).height
    print(f"  [{event_id}] Panel: {panel.height} min | {n_fires} primary fires "
          f"({100*n_fires/max(panel.height,1):.1f}%) | stable_sym={stable_sym}")
    return panel


# ---------------------------------------------------------------------------
# Training condition runner
# ---------------------------------------------------------------------------

def run_condition(
    label: str,
    train_panels: list[pl.DataFrame],
    validate_panel: pl.DataFrame,
    test_panel: pl.DataFrame,
    training_event_names: list[str],
    mechanism_classes: str,
) -> dict:
    """Run one training condition. Returns result dict for the summary table."""
    # Pool training data (primary fires only)
    train_fires = pl.concat(
        [apply_primary_filter(p) for p in train_panels],
        how="diagonal",
    )
    val_fires  = apply_primary_filter(validate_panel)
    test_fires = apply_primary_filter(test_panel)

    print(f"\n  Condition {label}: {train_fires.height} training fires | "
          f"{val_fires.height} val fires | {test_fires.height} test fires")

    if train_fires.height < 10:
        print(f"  WARNING: too few training samples for condition {label}")
        return {}

    # Check label balance
    pos_rate = float(train_fires["y_arb"].mean())
    print(f"  Training positive rate: {pos_rate:.1%}")
    if train_fires["y_arb"].n_unique() < 2:
        print(f"  WARNING: single-class label in training set")
        return {}

    # Fit secondary model
    model, used_cols = train_secondary(train_fires)

    # Calibrate threshold on Terra/LUNA validation fires
    if val_fires.height < 5 or val_fires["y_arb"].n_unique() < 2:
        threshold = 0.5
        print("  Using default threshold 0.5 (insufficient val data)")
    else:
        threshold = calibrate_threshold(model, val_fires, used_cols)
        print(f"  Calibrated threshold: {threshold:.3f}")

    # Evaluate on SVB test
    if test_fires.height == 0:
        print("  No test fires — skipping")
        return {}

    result = evaluate_strategy(test_fires, model, used_cols, threshold)

    n_events = len(train_panels)
    result.update({
        "condition": label,
        "training_set": ", ".join(training_event_names),
        "n_training_events": n_events,
        "mechanism_classes": mechanism_classes,
    })

    print(f"  Net bps={result['net_bps']:+.1f} | "
          f"Trades={result['n_trades']} | "
          f"Oracle capture={result['oracle_capture']:.1f}%")
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_capture_figure(rows: list[dict], out_path: Path) -> None:
    """Bar chart: oracle capture by training condition."""
    labels   = [r["condition"] for r in rows]
    captures = [r.get("oracle_capture", 0) for r in rows]
    net_bps  = [r.get("net_bps", 0) for r in rows]

    colours = [CNV, CAMB, CTA, CBLU]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=CBKG)
    for ax in (ax1, ax2):
        ax.set_facecolor(CBKG)
        ax.spines[["top", "right"]].set_visible(False)

    x = np.arange(len(labels))
    bars1 = ax1.bar(x, captures, color=colours[:len(labels)], alpha=0.88, width=0.6)
    ax1.axhline(100, color=CGRY, ls="--", lw=1, label="Oracle ceiling")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel("Oracle capture (%)", fontsize=10, color=CSL)
    ax1.set_title("Meta-labeling oracle capture\n(test: SVB Mar 2023)", fontsize=10.5, color=CNV)
    ax1.set_ylim(0, 120)
    for bar, val in zip(bars1, captures):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color=CSL)
    ax1.legend(fontsize=9)

    colours2 = [c if v >= 0 else "#C0392B" for c, v in zip(colours, net_bps)]
    bars2 = ax2.bar(x, net_bps, color=colours2, alpha=0.88, width=0.6)
    ax2.axhline(0, color=CGRY, ls="-", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylabel("Total net bps (SVB test)", fontsize=10, color=CSL)
    ax2.set_title("Net accumulated bps by training condition\n(test: SVB Mar 2023)",
                  fontsize=10.5, color=CNV)
    for bar, val in zip(bars2, net_bps):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (2 if val >= 0 else -6),
                 f"{val:+.1f}", ha="center", va="bottom", fontsize=9, color=CSL)

    fig.tight_layout(pad=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-download", dest="download", action="store_false",
                        help="Use cached klines only; skip network downloads")
    parser.set_defaults(download=True)
    args = parser.parse_args()

    print("\n=== CEX Arbitrage Meta-Labeling: Multi-Event Training Experiment ===\n")

    # ---- Load all event panels ----
    panels: dict[str, pl.DataFrame | None] = {}
    for ev_id, cfg in EVENTS.items():
        print(f"Loading {ev_id} ({cfg['start']} – {cfg['end']}) ...")
        panels[ev_id] = load_event_panel(ev_id, cfg, download=args.download)

    # ---- Optical summary table ----
    print("\n--- Optical summary ---")
    optical_rows = []
    for ev_id, p in panels.items():
        if p is None:
            continue
        row = compute_optical_summary(p, EVENTS[ev_id]["name"])
        row["event_id"] = ev_id
        row["mechanism"] = EVENTS[ev_id]["mechanism"]
        optical_rows.append(row)
        print(f"  {row['event']:20s}: {row['n_primary_fires']:5d} fires "
              f"({row['fire_rate_pct']:.1f}%) | "
              f"oracle positive rate {row['oracle_positive_rate_pct']:.1f}%")

    if optical_rows:
        df_optical = pl.DataFrame(optical_rows)
        OUT_TABLES.mkdir(parents=True, exist_ok=True)
        df_optical.write_csv(OUT_TABLES / "table_arb_optical_summary.csv")
        print(f"Saved: {OUT_TABLES}/table_arb_optical_summary.csv")

    # ---- Check required panels ----
    terra    = panels.get("terra_luna_2022")
    celsius  = panels.get("celsius_3ac_2022")
    ftx      = panels.get("ftx_2022")
    busd     = panels.get("busd_2023")
    svb      = panels.get("usdc_svb_2023")

    if terra is None or svb is None:
        print("\nERROR: Terra/LUNA and SVB panels are required. Aborting.")
        return

    # ---- Define training conditions ----
    print("\n--- Training conditions ---")
    conditions: list[tuple[str, list, str, list[str]]] = [
        # (label, train_panels, mechanism_str, event_names)
        ("A: Terra only",
         [terra], "Algorithmic",
         ["Terra/LUNA"]),
    ]
    if celsius is not None:
        conditions.append((
            "B: Celsius only",
            [celsius], "Exchange credit",
            ["Celsius/3AC"],
        ))
    if ftx is not None:
        conditions.append((
            "C: FTX only",
            [ftx], "Exchange credit",
            ["FTX"],
        ))

    pool_panels = [terra]
    pool_names  = ["Terra/LUNA"]
    if celsius is not None:
        pool_panels.append(celsius)
        pool_names.append("Celsius/3AC")
    if ftx is not None:
        pool_panels.append(ftx)
        pool_names.append("FTX")
    if busd is not None:
        pool_panels.append(busd)
        pool_names.append("BUSD")

    if len(pool_panels) > 1:
        mechs = "Alg., exchange, regulatory"
        conditions.append((
            f"D: {len(pool_panels)}-event pool",
            pool_panels, mechs, pool_names,
        ))

    # ---- Run each condition ----
    results = []
    for label, train_panels, mechs, ev_names in conditions:
        row = run_condition(
            label=label,
            train_panels=train_panels,
            validate_panel=terra,
            test_panel=svb,
            training_event_names=ev_names,
            mechanism_classes=mechs,
        )
        if row:
            results.append(row)

    if not results:
        print("\nNo results produced.")
        return

    # ---- Oracle row (constant across conditions) ----
    svb_fires = apply_primary_filter(svb)
    oracle_net    = float(svb_fires["net_bps"].filter(svb_fires["y_arb"] == 1).sum())
    oracle_trades = int((svb_fires["y_arb"] == 1).sum())

    # ---- Summary table ----
    print("\n=== Results Summary ===")
    print(f"{'Condition':<26} {'Train events':>13} {'Mech classes':>14} "
          f"{'Net bps':>9} {'Trades':>7} {'Oracle cap':>11}")
    print("-" * 85)
    for r in results:
        print(f"{r['condition']:<26} {r['n_training_events']:>13} "
              f"{r['mechanism_classes']:>14} "
              f"{r['net_bps']:>+9.1f} {r['n_trades']:>7} "
              f"{r['oracle_capture']:>10.1f}%")
    print(f"{'Oracle ceiling':<26} {'—':>13} {'—':>14} "
          f"{oracle_net:>+9.1f} {oracle_trades:>7} {'100.0':>10}%")

    # Save
    df_results = pl.DataFrame(results)
    out_csv = OUT_TABLES / "table_arb_meta_label_conditions.csv"
    df_results.write_csv(out_csv)
    print(f"\nSaved: {out_csv}")

    # ---- Figure ----
    make_capture_figure(results, OUT_FIGS / "figure_arb_oracle_capture.png")

    # ---- Paper-ready numbers ----
    print("\n=== Paper-ready numbers (fill into main.tex) ===")
    cond_a = next((r for r in results if r["condition"].startswith("A")), None)
    cond_d = next((r for r in results if r["condition"].startswith("D")), None)
    if cond_a:
        print(f"  Condition A (Terra only):")
        print(f"    net_bps = {cond_a['net_bps']:+.1f}")
        print(f"    trades  = {cond_a['n_trades']}")
        print(f"    oracle_capture = {cond_a['oracle_capture']:.1f}%")
    if cond_d:
        print(f"  Condition D ({len(pool_panels)}-event pool):")
        print(f"    net_bps = {cond_d['net_bps']:+.1f}")
        print(f"    trades  = {cond_d['n_trades']}")
        print(f"    oracle_capture = {cond_d['oracle_capture']:.1f}%")
    print(f"  Oracle ceiling: net_bps={oracle_net:+.1f}, trades={oracle_trades}")
    print(f"\n  Optical positive rates:")
    for r in optical_rows:
        print(f"    {r['event']:20s}: {r['oracle_positive_rate_pct']:.1f}%")


if __name__ == "__main__":
    main()
