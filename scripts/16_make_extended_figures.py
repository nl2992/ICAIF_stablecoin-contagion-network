"""
scripts/16_make_extended_figures.py
====================================
Generate the extended figure pack — 20 additional Columbia-themed figures
across six thematic groups:

  Group 1 – Event study timeseries (E01–E04)
  Group 2 – Cross-event evidence comparison (C01–C04)
  Group 3 – Robustness and sensitivity (R01–R03)
  Group 4 – Methods deep-dive (M01–M04)
  Group 5 – Network and centrality (N01–N03)
  Group 6 – Prediction and forecasting (P01–P02)

Output: results/paper/figures_extended/

Usage:
    python scripts/16_make_extended_figures.py
    python scripts/16_make_extended_figures.py --only E      # event study group
    python scripts/16_make_extended_figures.py --only C R    # multiple groups
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import polars as pl

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
REPO      = Path(__file__).resolve().parents[1]
RTBL      = REPO / "results" / "tables"
PTBL      = REPO / "results" / "paper" / "tables"
GOLD      = REPO / "data" / "gold"
OUT       = REPO / "results" / "paper" / "figures_extended"
OUT.mkdir(parents=True, exist_ok=True)

# ── Columbia palette ──────────────────────────────────────────────────────────
CU_NAVY   = "#003865"
CU_BLUE   = "#B9D9EB"
CU_SLATE  = "#2C3E50"
CU_GREEN  = "#27AE60"
CU_GREY   = "#7F8C8D"
CU_AMBER  = "#E67E22"
CU_RED    = "#C0392B"
CU_PURPLE = "#8E44AD"
CU_TEAL   = "#16A085"
CU_GOLD   = "#F39C12"

EVENT_COLORS = {
    "usdt_curve_2023":  CU_AMBER,
    "terra_luna_2022":  CU_RED,
    "usdc_svb_2023":    CU_BLUE,
    "ftx_2022":         CU_PURPLE,
    "busd_2023":        CU_GREEN,
}
EVENT_LABELS = {
    "usdt_curve_2023":  "USDT/Curve 2023",
    "terra_luna_2022":  "Terra/LUNA 2022",
    "usdc_svb_2023":    "USDC/SVB 2023",
    "ftx_2022":         "FTX 2022",
    "busd_2023":        "BUSD 2023",
}
ALL_EVENTS = list(EVENT_COLORS.keys())

# ── helpers ───────────────────────────────────────────────────────────────────

def _style(fig=None, ax=None):
    """Apply Columbia base style."""
    if fig:
        fig.patch.set_facecolor("white")
    if ax:
        ax.set_facecolor("#F8FAFB")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CCCCCC")
        ax.tick_params(colors=CU_SLATE, labelsize=9)
        ax.xaxis.label.set_color(CU_SLATE)
        ax.yaxis.label.set_color(CU_SLATE)


def _title(ax, text, sub=None):
    ax.set_title(text, fontsize=11, fontweight="bold", color=CU_NAVY, pad=8)
    if sub:
        ax.set_xlabel(sub, fontsize=8, color=CU_GREY)


def _watermark(fig):
    fig.text(0.98, 0.01, "Nigelli · Columbia MAFN · 2026",
             ha="right", va="bottom", fontsize=6, color="#BBBBBB", style="italic")


def _save(name: str, fig):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved %s", name)


def _read_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def _read_gold(event: str) -> pd.DataFrame | None:
    p = GOLD / f"dataset_contagion_features_{event}.parquet"
    if p.exists():
        return pl.read_parquet(p).to_pandas()
    return None


# ══════════════════════════════════════════════════════════════════════════════
# GROUP E — Event study timeseries
# ══════════════════════════════════════════════════════════════════════════════

def figE01_peg_deviation_all_events():
    """E01: basis_vs_usd for all 5 events — AMM/DEX nodes only, aligned to shock onset."""
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=False)
    fig.patch.set_facecolor("white")

    for idx, ev in enumerate(ALL_EVENTS):
        ax = axes[idx]
        _style(ax=ax)

        df = _read_gold(ev)
        if df is None or "basis_vs_usd" not in df.columns:
            ax.text(0.5, 0.5, f"{EVENT_LABELS[ev]}: no data",
                    transform=ax.transAxes, ha="center", color=CU_GREY)
            continue

        # DEX (AMM) nodes only
        dex = df[df["layer"] == "DEX"] if "layer" in df.columns else df
        if dex.empty:
            dex = df

        dex = dex.copy()
        dex["event_time_h"] = dex["event_time_seconds"] / 3600

        for node_id, grp in dex.groupby("node_id"):
            grp_s = grp.sort_values("event_time_h")
            ax.plot(grp_s["event_time_h"], grp_s["basis_vs_usd"] * 100,
                    lw=1.2, alpha=0.85, label=node_id)

        ax.axvline(0, color=CU_RED, lw=1.5, ls="--", alpha=0.8, label="Shock onset")
        ax.axhline(0, color="#AAAAAA", lw=0.8, ls=":")
        ax.set_ylabel("Basis vs USD (%)", fontsize=8, color=CU_SLATE)
        ax.set_title(EVENT_LABELS[ev], fontsize=10, fontweight="bold", color=EVENT_COLORS[ev])
        ax.legend(fontsize=7, loc="upper left", framealpha=0.7, ncol=2)

        # shade event vs pre
        xlim = ax.get_xlim()
        ax.axvspan(0, max(xlim[1], 1), alpha=0.06, color=EVENT_COLORS[ev])

    fig.supxlabel("Hours relative to shock onset", fontsize=10, color=CU_SLATE, y=0.01)
    fig.suptitle("Peg Deviation (Basis vs USD) — DEX Nodes, All Events",
                 fontsize=13, fontweight="bold", color=CU_NAVY, y=1.01)
    fig.tight_layout(h_pad=1.5)
    _watermark(fig)
    _save("E01_peg_deviation_all_events.png", fig)


def figE02_usdt_curve_cumulative_flow():
    """E02: Cumulative AMM net-sold flow (usdc_net_sold_cum) during USDT/Curve 2023."""
    df = _read_gold("usdt_curve_2023")
    if df is None:
        log.warning("E02: no gold data"); return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.patch.set_facecolor("white")

    dex_nodes = ["curve_3pool", "curve_crvusd_usdt"]
    colors_dex = [CU_NAVY, CU_AMBER]

    for ax in (ax1, ax2):
        _style(ax=ax)
        ax.axvline(0, color=CU_RED, lw=1.5, ls="--", alpha=0.8, label="Shock onset")
        ax.axhline(0, color="#AAAAAA", lw=0.8, ls=":")

    # Top: hourly net sold
    for node, col in zip(dex_nodes, colors_dex):
        sub = df[df["node_id"] == node].sort_values("event_time_seconds")
        if sub.empty or "usdc_net_sold_1h" not in sub.columns:
            continue
        t = sub["event_time_seconds"] / 3600
        ax1.bar(t, sub["usdc_net_sold_1h"] / 1e6, width=0.8,
                color=col, alpha=0.65, label=node)

    ax1.set_ylabel("Net USDC Sold (M/hr)", fontsize=9, color=CU_SLATE)
    _title(ax1, "Hourly AMM Net Flow — USDT/Curve 2023")
    ax1.legend(fontsize=8, framealpha=0.7)

    # Bottom: cumulative
    for node, col in zip(dex_nodes, colors_dex):
        sub = df[df["node_id"] == node].sort_values("event_time_seconds")
        if sub.empty or "usdc_net_sold_cum" not in sub.columns:
            continue
        t = sub["event_time_seconds"] / 3600
        ax2.plot(t, sub["usdc_net_sold_cum"] / 1e6, lw=2, color=col, label=node)
        ax2.fill_between(t, sub["usdc_net_sold_cum"] / 1e6, alpha=0.12, color=col)

    ax2.set_ylabel("Cumulative Net USDC Sold (M)", fontsize=9, color=CU_SLATE)
    ax2.set_xlabel("Hours relative to shock onset", fontsize=9, color=CU_SLATE)
    _title(ax2, "Cumulative AMM Flow — Tier-A Evidence")
    ax2.legend(fontsize=8, framealpha=0.7)

    fig.suptitle("AMM Flow During USDT/Curve 2023 Stress Episode",
                 fontsize=13, fontweight="bold", color=CU_NAVY)
    fig.tight_layout()
    _watermark(fig)
    _save("E02_usdt_curve_cumulative_flow.png", fig)


def figE03_event_study_cab_curves():
    """E03: Cumulative Abnormal Behaviour (CAB) curves — USDT/Curve and Terra/LUNA."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")

    pairs = [
        ("usdt_curve_2023", "USDT/Curve 2023", CU_AMBER),
        ("terra_luna_2022", "Terra/LUNA 2022", CU_RED),
    ]

    for ax, (ev, label, col) in zip(axes, pairs):
        _style(ax=ax)
        p = RTBL / f"table_event_study_timeseries_{ev}.csv"
        df = _read_csv(p)
        if df is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        df["event_time_h"] = df["event_time_seconds"] / 3600

        for node_id, grp in df.groupby("node_id"):
            grp_s = grp.sort_values("event_time_h").dropna(subset=["cab"])
            if grp_s.empty:
                continue
            # Only plot AMM/DEX-like nodes or all if few
            ax.plot(grp_s["event_time_h"], grp_s["cab"] * 100,
                    lw=1.4, alpha=0.8, label=node_id)

        ax.axvline(0, color=CU_RED, lw=1.5, ls="--", alpha=0.8, label="Shock onset")
        ax.axhline(0, color="#AAAAAA", lw=0.8, ls=":")
        ax.axvspan(0, ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 200,
                   alpha=0.05, color=col)
        ax.set_xlabel("Hours relative to shock onset", fontsize=9, color=CU_SLATE)
        ax.set_ylabel("Cumulative Abnormal Basis (%)", fontsize=9, color=CU_SLATE)
        _title(ax, f"CAB Curves — {label}")
        ax.legend(fontsize=7, framealpha=0.7, ncol=2)

    fig.suptitle("Event Study: Cumulative Abnormal Peg Deviation",
                 fontsize=13, fontweight="bold", color=CU_NAVY)
    fig.tight_layout()
    _watermark(fig)
    _save("E03_event_study_cab_curves.png", fig)


def figE04_basis_bps_distribution():
    """E04: Distribution of basis_bps pre vs. during stress — headline USDT/Curve 2023."""
    df = _read_gold("usdt_curve_2023")
    if df is None:
        log.warning("E04: no gold data"); return

    dex_nodes = ["curve_3pool", "curve_crvusd_usdt"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    for ax, node in zip(axes, dex_nodes):
        _style(ax=ax)
        sub = df[df["node_id"] == node].dropna(subset=["basis_bps"])
        pre   = sub[sub["event_phase"] == "pre"]["basis_bps"]
        event = sub[sub["event_phase"] == "event"]["basis_bps"]

        bins = np.linspace(sub["basis_bps"].quantile(0.01),
                           sub["basis_bps"].quantile(0.99), 50)
        ax.hist(pre, bins=bins, alpha=0.55, color=CU_BLUE, label="Pre-stress",
                edgecolor="white", linewidth=0.3)
        ax.hist(event, bins=bins, alpha=0.65, color=CU_AMBER, label="Event window",
                edgecolor="white", linewidth=0.3)

        ax.axvline(pre.mean(), color=CU_BLUE, lw=1.5, ls="--",
                   label=f"Pre mean = {pre.mean():.0f} bps")
        ax.axvline(event.mean(), color=CU_AMBER, lw=1.5, ls="--",
                   label=f"Event mean = {event.mean():.0f} bps")

        ax.set_xlabel("Basis (bps)", fontsize=9, color=CU_SLATE)
        ax.set_ylabel("Frequency", fontsize=9, color=CU_SLATE)
        _title(ax, node.replace("_", " ").title())
        ax.legend(fontsize=8, framealpha=0.8)

    fig.suptitle("Peg-Basis Distribution: Pre-Stress vs. Event Window\nUSDT/Curve 2023 — Tier-A DEX Nodes",
                 fontsize=12, fontweight="bold", color=CU_NAVY)
    fig.tight_layout()
    _watermark(fig)
    _save("E04_basis_bps_distribution.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
# GROUP C — Cross-event comparison
# ══════════════════════════════════════════════════════════════════════════════

def figC01_cross_event_leadlag_dotplot():
    """C01: Peak correlation and lag across all paper-claimable and suggestive edges."""
    rows = []
    for ev in ALL_EVENTS:
        p = PTBL / f"table_leadlag_tests_{ev}.csv"
        df = _read_csv(p)
        if df is None or df.empty:
            continue
        df["event_id"] = ev
        rows.append(df)
    if not rows:
        log.warning("C01: no lead-lag data"); return

    all_df = pd.concat(rows, ignore_index=True)

    # Create an edge label
    node_i_col = "node_i" if "node_i" in all_df.columns else "causing_node"
    node_j_col = "node_j" if "node_j" in all_df.columns else "caused_node"
    all_df["edge"] = all_df[node_i_col].str[:12] + "→" + all_df[node_j_col].str[:12]

    # Filter to significant rows
    sig_col = "significant_p01" if "significant_p01" in all_df.columns else "significant_fdr"
    sig = all_df[all_df[sig_col] == True].copy()
    if sig.empty:
        sig = all_df.head(20)

    # Limit to top-30 by |peak_corr|
    sig = sig.reindex(sig["peak_corr"].abs().sort_values(ascending=False).index).head(30)

    # Compute lag in hours — prefer peak_lag_seconds if available, else use steps
    if "peak_lag_seconds" in sig.columns and sig["peak_lag_seconds"].notna().any():
        sig = sig.copy()
        sig["peak_lag_h"] = sig["peak_lag_seconds"] / 3600
    elif "peak_lag_steps" in sig.columns:
        sig = sig.copy()
        # steps are small integers; treat as hours directly for display
        sig["peak_lag_h"] = sig["peak_lag_steps"].astype(float)
    else:
        sig = sig.copy()
        sig["peak_lag_h"] = 0.0

    # Clamp lag to ±48 h so the figure stays sensibly sized
    sig["peak_lag_h"] = sig["peak_lag_h"].clip(-48, 48)

    fig, ax = plt.subplots(figsize=(11, 8))
    _style(fig=fig, ax=ax)

    for ev in ALL_EVENTS:
        sub = sig[sig["event_id"] == ev]
        if sub.empty:
            continue
        ax.scatter(sub["peak_lag_h"], sub["peak_corr"].abs(),
                   color=EVENT_COLORS[ev], label=EVENT_LABELS[ev],
                   s=80, alpha=0.8, edgecolors="white", linewidth=0.5, zorder=3)

    # Fix axes before adding annotations
    ax.set_xlim(-50, 50)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.3, color=CU_GREY, lw=1, ls=":", alpha=0.6)
    ax.axvline(0, color=CU_GREY, lw=1, ls="--", alpha=0.4)
    ax.set_xlabel("Peak Lag (hours)", fontsize=10, color=CU_SLATE)
    ax.set_ylabel("|Peak Correlation|", fontsize=10, color=CU_SLATE)
    _title(ax, "Cross-Event Lead-Lag: Peak Correlation vs. Lag",
           "Significant pairs only (p < 0.01). Headline USDT/Curve result circled.")

    # Circle the headline result
    headline = sig[(sig["event_id"] == "usdt_curve_2023") &
                   (sig["peak_corr"].abs() > 0.35)]
    if not headline.empty:
        ax.scatter(headline["peak_lag_h"], headline["peak_corr"].abs(),
                   s=250, facecolors="none", edgecolors=CU_AMBER, lw=2.5, zorder=4)
        ax.annotate("Headline result\np≤0.014",
                    xy=(float(headline["peak_lag_h"].iloc[0]), 0.386),
                    xytext=(10, 0.55), fontsize=8, color=CU_AMBER,
                    arrowprops=dict(arrowstyle="->", color=CU_AMBER))

    ax.legend(fontsize=9, framealpha=0.8)
    fig.suptitle("Cross-Event Lead-Lag Comparison — All Significant Pairs",
                 fontsize=13, fontweight="bold", color=CU_NAVY)
    fig.tight_layout()
    _watermark(fig)
    _save("C01_cross_event_leadlag_dotplot.png", fig)


def figC02_transfer_entropy_across_events():
    """C02: Transfer-entropy heatmaps for all 5 events (multi-panel)."""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(5)]

    for ax, ev in zip(axes, ALL_EVENTS):
        _style(ax=ax)
        p = PTBL / f"table_transfer_entropy_{ev}.csv"
        df = _read_csv(p)
        if df is None or df.empty:
            ax.text(0.5, 0.5, f"{EVENT_LABELS[ev]}\nNo TE data",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color=CU_GREY)
            ax.set_title(EVENT_LABELS[ev], fontsize=9, color=EVENT_COLORS[ev], fontweight="bold")
            continue

        ni = "node_i" if "node_i" in df.columns else "source_node"
        nj = "node_j" if "node_j" in df.columns else "target_node"
        te_col = "te_i_to_j" if "te_i_to_j" in df.columns else "te_value"

        nodes = sorted(set(df[ni].tolist() + df[nj].tolist()))
        if len(nodes) == 0:
            continue
        mat = pd.DataFrame(np.nan, index=nodes, columns=nodes)
        for _, row in df.iterrows():
            if row[ni] in mat.index and row[nj] in mat.columns:
                mat.loc[row[ni], row[nj]] = row[te_col]

        # Shorten node labels
        short = {n: n.replace("curve_", "crv_")
                      .replace("_binance", "_bin")
                      .replace("_coinbase", "_cb")
                      .replace("_usdt", "_ut")
                      .replace("_usdc", "_uc")
                  for n in nodes}
        mat.index   = [short[n] for n in nodes]
        mat.columns = [short[n] for n in nodes]

        im = ax.imshow(mat.values.astype(float), cmap="Blues",
                       vmin=0, aspect="auto")
        ax.set_xticks(range(len(nodes)))
        ax.set_yticks(range(len(nodes)))
        ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(mat.index, fontsize=7)
        ax.set_title(EVENT_LABELS[ev], fontsize=9, fontweight="bold",
                     color=EVENT_COLORS[ev])
        plt.colorbar(im, ax=ax, shrink=0.7, label="TE (bits)")

    # Remove unused subplot
    if len(ALL_EVENTS) < 6:
        fig.delaxes(fig.add_subplot(gs[1, 2]))

    fig.suptitle("Transfer Entropy Matrices — All Five Stress Events",
                 fontsize=13, fontweight="bold", color=CU_NAVY, y=1.02)
    _watermark(fig)
    _save("C02_transfer_entropy_all_events.png", fig)


def figC03_granger_pvalue_comparison():
    """C03: Granger p-values for all events — dot chart by edge pair."""
    rows = []
    for ev in ALL_EVENTS:
        p = PTBL / f"table_granger_{ev}.csv"
        df = _read_csv(p)
        if df is None or df.empty:
            continue
        df["event_id"] = ev
        rows.append(df)
    if not rows:
        log.warning("C03: no Granger data"); return

    all_df = pd.concat(rows, ignore_index=True)
    ci  = "causing_node" if "causing_node" in all_df.columns else "node_i"
    cj  = "caused_node"  if "caused_node"  in all_df.columns else "node_j"
    all_df["edge"] = (all_df[ci].str[:10] + "→" + all_df[cj].str[:10])
    all_df["log_p"] = -np.log10(all_df["p_value"].clip(lower=1e-6))

    fig, ax = plt.subplots(figsize=(12, 6))
    _style(fig=fig, ax=ax)

    y_pos = 0
    yticks, ylabels = [], []
    ev_handles = {}

    for ev in ALL_EVENTS:
        sub = all_df[all_df["event_id"] == ev].sort_values("log_p", ascending=False)
        for _, row in sub.iterrows():
            sc = ax.scatter(row["log_p"], y_pos, color=EVENT_COLORS[ev],
                            s=70, alpha=0.85, edgecolors="white", lw=0.5, zorder=3)
            if ev not in ev_handles:
                ev_handles[ev] = sc
            yticks.append(y_pos)
            ylabels.append(row["edge"])
            y_pos += 1
        y_pos += 0.5  # gap between events

    ax.axvline(-np.log10(0.05), color=CU_GREY, lw=1.2, ls="--", alpha=0.7,
               label="p = 0.05")
    ax.axvline(-np.log10(0.01), color=CU_RED, lw=1.2, ls="--", alpha=0.7,
               label="p = 0.01 (Bonferroni threshold)")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("−log₁₀(Granger p-value)", fontsize=10, color=CU_SLATE)
    _title(ax, "Granger Causality p-Values — All Events and Pairs")

    legend_patches = [mpatches.Patch(color=EVENT_COLORS[e], label=EVENT_LABELS[e])
                      for e in ALL_EVENTS]
    ax.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color=CU_GREY, ls="--", label="p = 0.05"),
        plt.Line2D([0], [0], color=CU_RED,  ls="--", label="p = 0.01"),
    ], fontsize=8, framealpha=0.8)

    fig.suptitle("Granger Causality: −log₁₀(p) by Event and Edge Pair",
                 fontsize=13, fontweight="bold", color=CU_NAVY)
    fig.tight_layout()
    _watermark(fig)
    _save("C03_granger_pvalue_comparison.png", fig)


def figC04_event_window_calendar():
    """C04: Timeline calendar showing all 5 event windows on a single axis."""
    p = RTBL / "table_event_windows.csv"
    df = _read_csv(p)
    if df is None:
        log.warning("C04: no event_windows table"); return

    fig, ax = plt.subplots(figsize=(13, 4))
    _style(fig=fig, ax=ax)

    mechanism_labels = {
        "usdt_curve_2023":  "DeFi pool imbalance",
        "terra_luna_2022":  "Algorithmic collapse",
        "usdc_svb_2023":    "Fiat-reserve bank shock",
        "ftx_2022":         "Exchange credit shock",
        "busd_2023":        "Issuer wind-down",
    }

    for idx, (ev, col) in enumerate(EVENT_COLORS.items()):
        row = df[df["event_id"] == ev]
        if row.empty:
            continue
        row = row.iloc[0]
        start = pd.to_datetime(row.get("core_start", row.get("analysis_start")))
        end   = pd.to_datetime(row.get("core_end",   row.get("analysis_end")))
        shock = pd.to_datetime(row.get("shock_onset_utc", start))

        y = idx
        ax.barh(y, (end - start).days, left=start,
                height=0.5, color=col, alpha=0.75, edgecolor="white")
        ax.axvline(shock, color=col, lw=1.5, ls="--", alpha=0.8)
        ax.text(start, y + 0.32,
                f"{EVENT_LABELS[ev]}\n{mechanism_labels.get(ev,'')}",
                va="bottom", fontsize=8, color=CU_SLATE, fontweight="bold")

    ax.set_yticks([])
    ax.set_xlabel("Date", fontsize=10, color=CU_SLATE)
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([]))
    try:
        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate(rotation=30)
    except Exception:
        pass

    _title(ax, "Event Window Calendar — Five Stablecoin Stress Episodes (2022–2023)")
    fig.suptitle("", fontsize=1)
    fig.tight_layout()
    _watermark(fig)
    _save("C04_event_window_calendar.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
# GROUP R — Robustness and sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def figR01_robustness_significance_heatmap():
    """R01: Heatmap of significance rate across robustness checks × events."""
    p = RTBL / "table_robustness_summary.csv"
    df = _read_csv(p)
    if df is None or df.empty:
        log.warning("R01: no robustness summary"); return

    pivot = df.pivot_table(index="check", columns="event_id", values="sig_rate",
                           aggfunc="mean")
    # Reorder columns to our event order
    cols = [c for c in ALL_EVENTS if c in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(11, 5))
    _style(fig=fig, ax=ax)

    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([EVENT_LABELS[c].replace(" ", "\n") for c in cols],
                       fontsize=9)
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(cols)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                        fontsize=9, color="white" if v < 0.4 or v > 0.75 else CU_SLATE,
                        fontweight="bold")

    plt.colorbar(im, ax=ax, label="Significance rate", shrink=0.85)
    _title(ax, "Robustness Check: Significance Rate by Method × Event")
    ax.set_xlabel("")
    fig.suptitle("Robustness Sensitivity Matrix\n(Green = more significant; Red = less)",
                 fontsize=12, fontweight="bold", color=CU_NAVY)
    fig.tight_layout()
    _watermark(fig)
    _save("R01_robustness_significance_heatmap.png", fig)


def figR02_robustness_peak_corr_distribution():
    """R02: Distribution of peak_corr across robustness checks for USDT/Curve 2023."""
    p = RTBL / "table_robustness_usdt_curve_2023.csv"
    df = _read_csv(p)
    if df is None or df.empty:
        log.warning("R02: no robustness data"); return

    checks = df["check"].unique()
    fig, ax = plt.subplots(figsize=(10, 5))
    _style(fig=fig, ax=ax)

    colors_r = [CU_NAVY, CU_BLUE, CU_AMBER, CU_GREEN, CU_PURPLE, CU_TEAL]
    bins = np.linspace(-1, 1, 40)

    for i, check in enumerate(checks):
        sub = df[df["check"] == check]["peak_corr"].dropna()
        ax.hist(sub, bins=bins, alpha=0.45, color=colors_r[i % len(colors_r)],
                label=check, edgecolor="white", linewidth=0.3)

    # Mark headline result
    ax.axvline(0.3857, color=CU_AMBER, lw=2.5, ls="--",
               label="Headline: 0.3857 (Bonferroni p≤0.014)")

    ax.set_xlabel("Peak Correlation", fontsize=10, color=CU_SLATE)
    ax.set_ylabel("Count", fontsize=10, color=CU_SLATE)
    _title(ax, "Peak Correlation Distribution Across Robustness Checks",
           "USDT/Curve 2023 — all node pairs, all robustness configurations")
    ax.legend(fontsize=8, framealpha=0.8)
    fig.suptitle("Robustness: Are the Lead-Lag Results Stable Across Grid Sizes?",
                 fontsize=12, fontweight="bold", color=CU_NAVY)
    fig.tight_layout()
    _watermark(fig)
    _save("R02_robustness_peak_corr_distribution.png", fig)


def figR03_grid_sensitivity_headline_pair():
    """R03: Sensitivity of headline pair peak_corr across robustness configurations."""
    p = RTBL / "table_robustness_usdt_curve_2023.csv"
    df = _read_csv(p)
    if df is None or df.empty:
        log.warning("R03: no robustness data"); return

    # Filter to the headline pair (either direction)
    ni, nj = "node_i", "node_j"
    if ni not in df.columns:
        ni, nj = "node_i", "node_j"

    hl = df[
        ((df[ni].str.contains("curve_3pool", na=False) &
          df[nj].str.contains("crvusd", na=False)) |
         (df[ni].str.contains("crvusd", na=False) &
          df[nj].str.contains("curve_3pool", na=False)))
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    if not hl.empty:
        for ax in (ax1, ax2):
            _style(ax=ax)

        checks = hl["check"].unique()
        x = range(len(checks))
        corrs = [hl[hl["check"] == c]["peak_corr"].values for c in checks]
        means = [np.mean(v) if len(v) else np.nan for v in corrs]
        stds  = [np.std(v)  if len(v) else np.nan for v in corrs]

        ax1.bar(x, means, color=CU_NAVY, alpha=0.75, edgecolor="white")
        ax1.errorbar(x, means, yerr=stds, fmt="none", color=CU_AMBER, capsize=4, lw=2)
        ax1.axhline(0.3857, color=CU_RED, lw=1.5, ls="--",
                    label="Baseline (Bonferroni)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(checks, rotation=30, ha="right", fontsize=8)
        ax1.set_ylabel("|Peak Correlation|", fontsize=9, color=CU_SLATE)
        _title(ax1, "Headline Pair: Peak Corr by Robustness Check")
        ax1.legend(fontsize=8)

        # Significance rate
        sig_rates = [np.mean(hl[hl["check"] == c]["significant_p01"].astype(float).values)
                     if len(hl[hl["check"] == c]) else np.nan
                     for c in checks]
        bar_colors = [CU_GREEN if s >= 0.5 else CU_RED for s in sig_rates]
        ax2.bar(x, sig_rates, color=bar_colors, alpha=0.8, edgecolor="white")
        ax2.axhline(0.5, color=CU_GREY, lw=1, ls=":")
        ax2.set_xticks(x)
        ax2.set_xticklabels(checks, rotation=30, ha="right", fontsize=8)
        ax2.set_ylabel("Fraction significant at p < 0.01", fontsize=9, color=CU_SLATE)
        ax2.set_ylim(0, 1.1)
        _title(ax2, "Significance Rate Across Configurations")
    else:
        for ax in (ax1, ax2):
            _style(ax=ax)
            ax.text(0.5, 0.5, "Headline pair not found\nin robustness table",
                    transform=ax.transAxes, ha="center", color=CU_GREY)

    fig.suptitle("Grid-Sensitivity: Headline Pair (curve_3pool ↔ curve_crvusd_usdt)",
                 fontsize=12, fontweight="bold", color=CU_NAVY)
    fig.tight_layout()
    _watermark(fig)
    _save("R03_grid_sensitivity_headline_pair.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
# GROUP M — Methods deep-dive
# ══════════════════════════════════════════════════════════════════════════════

def figM01_leadlag_full_correlation_matrix():
    """M01: Full cross-correlation matrix at lag 0 for USDT/Curve 2023."""
    df = _read_gold("usdt_curve_2023")
    if df is None:
        log.warning("M01: no gold data"); return

    if "usdc_net_sold_1h" not in df.columns:
        log.warning("M01: usdc_net_sold_1h not in columns"); return

    # Pivot: rows=time, cols=node_id, values=usdc_net_sold_1h
    pivot = (df[df["layer"] == "DEX"][["wall_clock_utc", "node_id", "usdc_net_sold_1h"]]
             .dropna()
             .pivot_table(index="wall_clock_utc", columns="node_id",
                          values="usdc_net_sold_1h", aggfunc="mean"))

    if pivot.shape[1] < 2:
        pivot = (df[["wall_clock_utc", "node_id", "usdc_net_sold_1h"]]
                 .dropna()
                 .pivot_table(index="wall_clock_utc", columns="node_id",
                              values="usdc_net_sold_1h", aggfunc="mean"))

    corr = pivot.corr()
    nodes = corr.columns.tolist()
    short = [n.replace("curve_", "crv_")
              .replace("_usdt", "_ut")
              .replace("_usdc", "_uc") for n in nodes]

    fig, ax = plt.subplots(figsize=(9, 7))
    _style(fig=fig, ax=ax)

    mask_diag = np.eye(len(nodes), dtype=bool)
    data = corr.values.copy()
    data[mask_diag] = np.nan

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cu_div", [CU_RED, "white", CU_NAVY])
    im = ax.imshow(data, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(nodes)))
    ax.set_yticks(range(len(nodes)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if not mask_diag[i, j] and not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(data[i,j]) > 0.6 else CU_SLATE)

    plt.colorbar(im, ax=ax, label="Pearson correlation", shrink=0.85)
    _title(ax, "Cross-Correlation Matrix at Lag 0 — USDT/Curve 2023\nTier-A usdc_net_sold_1h, hourly grid")
    fig.suptitle("Full Correlation Structure: All DEX Nodes",
                 fontsize=12, fontweight="bold", color=CU_NAVY)
    fig.tight_layout()
    _watermark(fig)
    _save("M01_leadlag_correlation_matrix.png", fig)


def figM02_tvp_var_spillovers():
    """M02: Time-varying VAR spillovers — heatmap for all 5 events."""
    rows = []
    for ev in ALL_EVENTS:
        p = PTBL / f"table_tvp_var_spillovers_{ev}.csv"
        df = _read_csv(p)
        if df is not None and not df.empty:
            df["event_id"] = ev
            rows.append(df)

    if not rows:
        # TVP-VAR requires sufficient obs; generate informative placeholder
        fig, ax = plt.subplots(figsize=(11, 5))
        _style(fig=fig, ax=ax)
        ax.text(0.5, 0.55,
                "TVP-VAR spillover tables are empty for all events.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=CU_SLATE)
        ax.text(0.5, 0.40,
                "The rolling-window estimator requires more observations than\n"
                "available in the hourly Tier-A panel (≥168h window).\n"
                "This analysis remains a candidate for longer event windows.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color=CU_GREY, style="italic")
        ax.axis("off")
        fig.suptitle("Time-Varying VAR: Spillover Shares — Data Insufficient",
                     fontsize=12, fontweight="bold", color=CU_NAVY)
        _watermark(fig)
        _save("M02_tvp_var_spillovers_all_events.png", fig)
        return

    all_df = pd.concat(rows, ignore_index=True)

    # Summarise: mean fevd_share by event × causing → caused
    ci = "causing_node" if "causing_node" in all_df.columns else "node_i"
    cj = "caused_node"  if "caused_node"  in all_df.columns else "node_j"

    all_df["edge"] = all_df[ci].str[:10] + "→" + all_df[cj].str[:10]

    fig, axes = plt.subplots(1, len(ALL_EVENTS), figsize=(16, 5))
    if len(ALL_EVENTS) == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, ev in zip(axes, ALL_EVENTS):
        _style(ax=ax)
        sub = all_df[all_df["event_id"] == ev]
        if sub.empty:
            ax.text(0.5, 0.5, "No TVP-VAR\ndata", transform=ax.transAxes,
                    ha="center", fontsize=9, color=CU_GREY)
            ax.set_title(EVENT_LABELS[ev][:12], fontsize=8, color=EVENT_COLORS[ev],
                         fontweight="bold")
            continue

        nodes_ev = sorted(set(sub[ci].tolist() + sub[cj].tolist()))
        mat = pd.DataFrame(np.nan, index=nodes_ev, columns=nodes_ev)
        grp = sub.groupby([ci, cj])["fevd_share"].mean()
        for (i, j), v in grp.items():
            if i in mat.index and j in mat.columns:
                mat.loc[i, j] = v

        short = {n: n.replace("curve_", "crv_")
                      .replace("_binance", "_bn")
                      .replace("_coinbase", "_cb")
                  for n in nodes_ev}
        mat.index   = [short[n] for n in nodes_ev]
        mat.columns = [short[n] for n in nodes_ev]

        im = ax.imshow(mat.values.astype(float), cmap="YlOrRd",
                       vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(nodes_ev)))
        ax.set_yticks(range(len(nodes_ev)))
        ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(mat.index, fontsize=7)
        ax.set_title(EVENT_LABELS[ev].replace(" ", "\n"), fontsize=8,
                     color=EVENT_COLORS[ev], fontweight="bold")

    fig.suptitle("Time-Varying VAR: Mean FEVD Spillover Share by Event",
                 fontsize=12, fontweight="bold", color=CU_NAVY)
    fig.tight_layout()
    _watermark(fig)
    _save("M02_tvp_var_spillovers_all_events.png", fig)


def figM03_method_pvalue_comparison():
    """M03: For the headline pair, compare p-values across lead-lag / Granger / TE."""
    ev = "usdt_curve_2023"
    methods, pvals, labels = [], [], []

    # Lead-lag
    p = PTBL / f"table_leadlag_tests_{ev}.csv"
    df = _read_csv(p)
    if df is not None and not df.empty:
        for _, row in df.iterrows():
            pv = row.get("p_bonferroni", row.get("p_value", np.nan))
            ni = row.get("node_i", "")
            nj = row.get("node_j", "")
            if "3pool" in str(ni) or "3pool" in str(nj):
                methods.append("Lead-lag\n(Bonferroni)")
                pvals.append(float(pv) if not pd.isna(pv) else 1.0)
                labels.append(f"{str(ni)[:8]}→{str(nj)[:8]}")

    # Granger
    p2 = PTBL / f"table_granger_{ev}.csv"
    df2 = _read_csv(p2)
    if df2 is not None and not df2.empty:
        ci = "causing_node" if "causing_node" in df2.columns else "node_i"
        cj = "caused_node"  if "caused_node"  in df2.columns else "node_j"
        for _, row in df2.iterrows():
            pv = row.get("p_bonferroni", row.get("p_value", np.nan))
            methods.append("Granger\n(Bonferroni)")
            pvals.append(float(pv) if not pd.isna(pv) else 1.0)
            labels.append(f"{str(row[ci])[:8]}→{str(row[cj])[:8]}")

    # TE
    p3 = PTBL / f"table_transfer_entropy_{ev}.csv"
    df3 = _read_csv(p3)
    if df3 is not None and not df3.empty:
        pc = "p_value_block" if "p_value_block" in df3.columns else "p_value"
        ni = "node_i" if "node_i" in df3.columns else "source_node"
        nj = "node_j" if "node_j" in df3.columns else "target_node"
        for _, row in df3.iterrows():
            pv = row.get(pc, np.nan)
            methods.append("Transfer\nEntropy")
            pvals.append(float(pv) if not pd.isna(pv) else 1.0)
            labels.append(f"{str(row[ni])[:8]}→{str(row[nj])[:8]}")

    if not methods:
        log.warning("M03: no p-value data"); return

    fig, ax = plt.subplots(figsize=(10, 5))
    _style(fig=fig, ax=ax)

    log_p = [-np.log10(max(pv, 1e-8)) for pv in pvals]
    bar_colors = [CU_GREEN if pv <= 0.05 else CU_GREY for pv in pvals]
    x = range(len(methods))
    ax.bar(x, log_p, color=bar_colors, alpha=0.8, edgecolor="white")
    ax.axhline(-np.log10(0.05), color=CU_AMBER, lw=1.5, ls="--", label="p = 0.05")
    ax.axhline(-np.log10(0.01), color=CU_RED,   lw=1.5, ls="--", label="p = 0.01")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n{l}" for m, l in zip(methods, labels)],
                       fontsize=7, rotation=25, ha="right")
    ax.set_ylabel("−log₁₀(p-value)", fontsize=10, color=CU_SLATE)
    _title(ax, "Method Comparison: Significance of Propagation Evidence\nUSDT/Curve 2023 — All Tested Edge Pairs")
    ax.legend(fontsize=9, framealpha=0.8)
    fig.suptitle("", fontsize=1)
    fig.tight_layout()
    _watermark(fig)
    _save("M03_method_pvalue_comparison.png", fig)


def figM04_feature_importance_prediction():
    """M04: Prediction model AUROC/AUPRC comparison across models and events."""
    p = PTBL / "table_prediction_metrics.csv"
    df = _read_csv(p)
    if df is None or df.empty:
        # try raw tables dir
        rows = []
        for ev in ALL_EVENTS:
            p2 = RTBL / f"table_prediction_metrics_{ev}.csv"
            d = _read_csv(p2)
            if d is not None:
                d["event_id"] = ev
                rows.append(d)
        if not rows:
            log.warning("M04: no prediction metrics"); return
        df = pd.concat(rows, ignore_index=True)

    # Use full-ablation, leave-one-event-out rows
    if "split_type" in df.columns:
        df = df[df["split_type"].str.contains("leave_one", na=False)]
    if "ablation" in df.columns:
        df = df[df["ablation"] == "full"]

    models = df["model"].unique() if "model" in df.columns else []
    events_present = [e for e in ALL_EVENTS if e in df["event_id"].unique()]
    if len(models) == 0 or len(events_present) == 0:
        log.warning("M04: filtered df is empty"); return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")

    x = np.arange(len(events_present))
    width = 0.8 / max(len(models), 1)
    model_colors = [CU_NAVY, CU_AMBER, CU_GREEN, CU_TEAL, CU_PURPLE]

    for ax, metric in zip((ax1, ax2), ("AUROC", "AUPRC")):
        _style(ax=ax)
        for mi, model in enumerate(models):
            vals = [df[(df["model"] == model) & (df["event_id"] == ev)][metric].mean()
                    if metric in df.columns else np.nan
                    for ev in events_present]
            offset = (mi - len(models) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width=width * 0.9,
                   color=model_colors[mi % len(model_colors)],
                   alpha=0.8, label=model, edgecolor="white")

        ax.axhline(0.5, color=CU_GREY, lw=1, ls=":", alpha=0.6, label="Random baseline")
        ax.set_xticks(x)
        ax.set_xticklabels([EVENT_LABELS[e].replace(" ", "\n") for e in events_present],
                           fontsize=8)
        ax.set_ylabel(metric, fontsize=10, color=CU_SLATE)
        ax.set_ylim(0, 1.05)
        _title(ax, f"Prediction {metric} by Model × Event")
        ax.legend(fontsize=8, framealpha=0.8)

    fig.suptitle("Leave-One-Event-Out Prediction: AUROC and AUPRC\n(Full feature set, stress label)",
                 fontsize=12, fontweight="bold", color=CU_NAVY)
    fig.tight_layout()
    _watermark(fig)
    _save("M04_prediction_auroc_auprc.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
# GROUP N — Network and centrality
# ══════════════════════════════════════════════════════════════════════════════

def figN01_node_centrality_comparison():
    """N01: Node centrality (in/out degree, eigenvector) across events."""
    rows = []
    for ev in ALL_EVENTS:
        p = RTBL / f"table_node_centrality_{ev}.csv"
        df = _read_csv(p)
        if df is None or df.empty:
            continue
        df["event_id"] = ev
        rows.append(df)

    if not rows:
        # Try combined table
        p2 = PTBL / "table_node_centrality.csv"
        df2 = _read_csv(p2)
        if df2 is not None and not df2.empty:
            rows = [df2]

    if not rows:
        log.warning("N01: no centrality data"); return

    all_df = pd.concat(rows, ignore_index=True)

    metrics = ["out_degree_w", "in_degree_w", "eigenvector", "betweenness"]
    metrics = [m for m in metrics if m in all_df.columns]
    if not metrics:
        log.warning("N01: no centrality metrics"); return

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5))
    fig.patch.set_facecolor("white")
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        _style(ax=ax)
        # For each event, show top-3 nodes by this metric
        for ev in ALL_EVENTS:
            sub = all_df[all_df["event_id"] == ev].sort_values(metric, ascending=False)
            if sub.empty:
                continue
            for rank, (_, row) in enumerate(sub.head(3).iterrows()):
                ax.scatter(ev, row[metric],
                           color=EVENT_COLORS[ev], s=80, alpha=0.8,
                           edgecolors="white", linewidth=0.5, zorder=3)
                ax.text(ev, row[metric], f"  {str(row['node_id'])[:10]}",
                        fontsize=6, va="center", color=CU_SLATE)

        ax.set_xticklabels([EVENT_LABELS[e].replace(" ", "\n") for e in ALL_EVENTS
                            if e in all_df["event_id"].unique()],
                           fontsize=7, rotation=20)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=9, color=CU_SLATE)
        _title(ax, metric.replace("_", " ").title())

    fig.suptitle("Network Centrality — Top Nodes by Event",
                 fontsize=12, fontweight="bold", color=CU_NAVY)
    fig.tight_layout()
    _watermark(fig)
    _save("N01_node_centrality_comparison.png", fig)


def figN02_bipartite_claim_network():
    """N02: Bipartite graph — events × nodes, coloured by claim level."""
    p = PTBL / "table_claim_gate_all_events.csv"
    df = _read_csv(p)
    if df is None:
        p = RTBL / "table_claim_gate_all_events.csv"
        df = _read_csv(p)
    if df is None or df.empty:
        log.warning("N02: no claim gate data"); return

    # Keep paper-claimable + suggestive
    if "paper_claim_allowed" in df.columns:
        df = df[df["paper_claim_allowed"].astype(str).str.lower().isin(["true", "1"])]

    ni = "node_i" if "node_i" in df.columns else "causing_node"
    nj = "node_j" if "node_j" in df.columns else "caused_node"
    cl_col = "claim_level" if "claim_level" in df.columns else None

    fig, ax = plt.subplots(figsize=(12, 8))
    _style(fig=fig, ax=ax)

    claim_colors = {
        "A_A_dex_flow":         CU_GREEN,
        "A_A_onchain_settlement": CU_TEAL,
        "A_B_suggestive_directional": CU_AMBER,
        "B_B_context_only":     CU_GREY,
    }

    plotted = 0
    for _, row in df.iterrows():
        ev   = str(row.get("event_id", ""))
        n_i  = str(row.get(ni, ""))
        n_j  = str(row.get(nj, ""))
        cl   = str(row.get(cl_col, "")) if cl_col else "unknown"

        col  = claim_colors.get(cl, CU_SLATE)
        ev_x = ALL_EVENTS.index(ev) if ev in ALL_EVENTS else 0

        y_i = hash(n_i) % 20 / 20.0
        y_j = hash(n_j) % 20 / 20.0

        ax.annotate("", xy=(ev_x + 0.15, y_j), xytext=(ev_x - 0.15, y_i),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.2, alpha=0.6))
        ax.text(ev_x - 0.2, y_i, n_i[:12], ha="right", fontsize=7, color=CU_SLATE)
        ax.text(ev_x + 0.2, y_j, n_j[:12], ha="left",  fontsize=7, color=CU_SLATE)
        plotted += 1

    ax.set_xticks(range(len(ALL_EVENTS)))
    ax.set_xticklabels([EVENT_LABELS[e].replace(" ", "\n") for e in ALL_EVENTS],
                       fontsize=9)
    ax.set_yticks([])
    ax.set_xlim(-1, len(ALL_EVENTS))

    legend_patches = [mpatches.Patch(color=v, label=k.replace("_", " "))
                      for k, v in claim_colors.items()]
    ax.legend(handles=legend_patches, fontsize=8, framealpha=0.8,
              loc="upper right")
    _title(ax, f"Paper-Claimable Directed Edges by Event ({plotted} edges)")
    fig.suptitle("Claim-Gated Evidence Network — Events × Node Pairs",
                 fontsize=12, fontweight="bold", color=CU_NAVY)
    fig.tight_layout()
    _watermark(fig)
    _save("N02_bipartite_claim_network.png", fig)


def figN03_stress_propagation_sequence():
    """N03: Estimated propagation sequence for USDT/Curve 2023 (event transmission rank)."""
    p = PTBL / "table_event_study_summary.csv"
    df = _read_csv(p)
    if df is None or df.empty:
        p2 = RTBL / "table_event_study_summary_usdt_curve_2023.csv"
        df = _read_csv(p2)
    if df is None or df.empty:
        log.warning("N03: no event study summary"); return

    if "event_id" in df.columns:
        df = df[df["event_id"] == "usdt_curve_2023"]

    if df.empty or "transmission_rank" not in df.columns:
        log.warning("N03: no transmission_rank column"); return

    df = df.dropna(subset=["transmission_rank"]).sort_values("transmission_rank")

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.6)))
    _style(fig=fig, ax=ax)

    colors_seq = [CU_GREEN, CU_AMBER, CU_RED, CU_PURPLE, CU_TEAL, CU_NAVY]
    for idx, (_, row) in enumerate(df.iterrows()):
        col = colors_seq[idx % len(colors_seq)]
        bar = ax.barh(str(row.get("node_id", idx)), row["transmission_rank"],
                      color=col, alpha=0.8, edgecolor="white", height=0.5)
        if "first_deviation_ts" in row and not pd.isna(row["first_deviation_ts"]):
            ax.text(row["transmission_rank"] + 0.02, idx,
                    f"  t={row['first_deviation_ts']:.0f}s",
                    va="center", fontsize=8, color=CU_SLATE)

    ax.set_xlabel("Transmission Rank", fontsize=10, color=CU_SLATE)
    ax.invert_yaxis()
    _title(ax, "Stress Propagation Sequence — USDT/Curve 2023",
           "Lower rank = earlier first significant deviation from baseline")
    fig.suptitle("Node Transmission Timing: Which Node Reacted First?",
                 fontsize=12, fontweight="bold", color=CU_NAVY)
    fig.tight_layout()
    _watermark(fig)
    _save("N03_stress_propagation_sequence.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
# GROUP P — Prediction and forecasting
# ══════════════════════════════════════════════════════════════════════════════

def figP01_prediction_roc_by_event():
    """P01: AUROC bar chart — full model vs. ablation by event."""
    rows = []
    for ev in ALL_EVENTS:
        for suffix in ["", f"_{ev}"]:
            p = RTBL / f"table_prediction_metrics{suffix}.csv"
            d = _read_csv(p)
            if d is not None and not d.empty:
                if "event_id" not in d.columns:
                    d["event_id"] = ev
                rows.append(d)

    # Also check paper tables
    p2 = PTBL / "table_prediction_metrics.csv"
    d2 = _read_csv(p2)
    if d2 is not None and not d2.empty:
        rows.append(d2)

    if not rows:
        log.warning("P01: no prediction data"); return

    df = pd.concat(rows, ignore_index=True).drop_duplicates()
    if "AUROC" not in df.columns:
        log.warning("P01: AUROC column missing"); return

    events_present = [e for e in ALL_EVENTS if e in df["event_id"].unique()]
    models = df["model"].dropna().unique() if "model" in df.columns else ["Model"]

    fig, ax = plt.subplots(figsize=(11, 5))
    _style(fig=fig, ax=ax)

    x = np.arange(len(events_present))
    width = 0.8 / max(len(models), 1)
    model_colors = [CU_NAVY, CU_AMBER, CU_GREEN, CU_TEAL, CU_PURPLE]

    for mi, model in enumerate(models):
        sub = df[df["model"] == model] if "model" in df.columns else df
        vals = []
        for ev in events_present:
            v = sub[sub["event_id"] == ev]["AUROC"]
            vals.append(v.mean() if not v.empty else np.nan)
        offset = (mi - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.9,
               color=model_colors[mi % len(model_colors)],
               alpha=0.8, label=model, edgecolor="white")

    ax.axhline(0.5, color=CU_GREY, lw=1.2, ls="--", alpha=0.7,
               label="Random baseline")
    ax.axhline(0.7, color=CU_AMBER, lw=1, ls=":", alpha=0.5,
               label="AUROC = 0.7")
    ax.set_xticks(x)
    ax.set_xticklabels([EVENT_LABELS[e].replace(" ", "\n") for e in events_present],
                       fontsize=9)
    ax.set_ylabel("AUROC", fontsize=10, color=CU_SLATE)
    ax.set_ylim(0, 1.1)
    _title(ax, "Leave-One-Event-Out AUROC — Stress Label Prediction")
    ax.legend(fontsize=9, framealpha=0.8, ncol=3)
    fig.suptitle("Predictive Performance: Can AMM Flow Predict Downstream Stress?",
                 fontsize=12, fontweight="bold", color=CU_NAVY)
    fig.tight_layout()
    _watermark(fig)
    _save("P01_prediction_roc_by_event.png", fig)


def figP02_claim_strength_summary():
    """P02: Horizontal bar chart — count of edges by claim_strength × event."""
    # Use the edge-level provenance table which has claim_strength per row
    candidate_paths = [
        PTBL / "table_provenance_claimable_edges.csv",
        PTBL / "table_statistically_supported_edges.csv",
        PTBL / "table_leadlag_tests.csv",
    ]
    all_df = None
    for cp in candidate_paths:
        df = _read_csv(cp)
        if df is not None and "claim_strength" in df.columns and not df.empty:
            all_df = df
            break

    if all_df is None or all_df.empty:
        # Fall back: concatenate per-event leadlag tables
        rows = []
        for ev in ALL_EVENTS:
            for tname in [f"table_leadlag_tests_{ev}.csv",
                          f"table_granger_{ev}.csv",
                          f"table_transfer_entropy_{ev}.csv"]:
                df = _read_csv(PTBL / tname)
                if df is not None and "claim_strength" in df.columns and not df.empty:
                    if "event_id" not in df.columns:
                        df["event_id"] = ev
                    rows.append(df)
        if not rows:
            log.warning("P02: no edge table with claim_strength"); return
        all_df = pd.concat(rows, ignore_index=True)

    if "claim_strength" not in all_df.columns:
        log.warning("P02: claim_strength column missing"); return

    # Ensure event_id is present
    if "event_id" not in all_df.columns:
        log.warning("P02: event_id missing"); return

    strength_order = ["robust", "statistically_supported", "suggestive", "descriptive"]
    strength_colors = {
        "robust":                   CU_GREEN,
        "statistically_supported":  CU_AMBER,
        "suggestive":               CU_BLUE,
        "descriptive":              CU_GREY,
    }

    counts = (all_df.groupby(["event_id", "claim_strength"])
              .size().reset_index(name="n"))

    fig, ax = plt.subplots(figsize=(12, 6))
    _style(fig=fig, ax=ax)

    y = np.arange(len(ALL_EVENTS))
    left = np.zeros(len(ALL_EVENTS))

    for strength in strength_order:
        vals = []
        for ev in ALL_EVENTS:
            sub = counts[(counts["event_id"] == ev) & (counts["claim_strength"] == strength)]
            vals.append(int(sub["n"].sum()) if not sub.empty else 0)
        ax.barh(y, vals, left=left, color=strength_colors.get(strength, CU_GREY),
                alpha=0.85, label=strength.replace("_", " ").title(),
                edgecolor="white", height=0.5)
        left += np.array(vals)

    ax.set_yticks(y)
    ax.set_yticklabels([EVENT_LABELS[e] for e in ALL_EVENTS], fontsize=10)
    ax.set_xlabel("Number of Edge Pairs", fontsize=10, color=CU_SLATE)
    _title(ax, "Claim Strength Distribution by Event")
    ax.legend(fontsize=9, framealpha=0.8, loc="lower right")
    fig.suptitle("Evidence Quality: How Strong Is the Claim for Each Edge?",
                 fontsize=12, fontweight="bold", color=CU_NAVY)
    fig.tight_layout()
    _watermark(fig)
    _save("P02_claim_strength_by_event.png", fig)


# ══════════════════════════════════════════════════════════════════════════════
# Registry and main
# ══════════════════════════════════════════════════════════════════════════════

ALL_FIGURES = {
    "E": [figE01_peg_deviation_all_events, figE02_usdt_curve_cumulative_flow,
           figE03_event_study_cab_curves, figE04_basis_bps_distribution],
    "C": [figC01_cross_event_leadlag_dotplot, figC02_transfer_entropy_across_events,
           figC03_granger_pvalue_comparison, figC04_event_window_calendar],
    "R": [figR01_robustness_significance_heatmap, figR02_robustness_peak_corr_distribution,
           figR03_grid_sensitivity_headline_pair],
    "M": [figM01_leadlag_full_correlation_matrix, figM02_tvp_var_spillovers,
           figM03_method_pvalue_comparison, figM04_feature_importance_prediction],
    "N": [figN01_node_centrality_comparison, figN02_bipartite_claim_network,
           figN03_stress_propagation_sequence],
    "P": [figP01_prediction_roc_by_event, figP02_claim_strength_summary],
}

EXTENDED_EXPECTED_FILES = [
    "E01_peg_deviation_all_events.png",
    "E02_usdt_curve_cumulative_flow.png",
    "E03_event_study_cab_curves.png",
    "E04_basis_bps_distribution.png",
    "C01_cross_event_leadlag_dotplot.png",
    "C02_transfer_entropy_all_events.png",
    "C03_granger_pvalue_comparison.png",
    "C04_event_window_calendar.png",
    "R01_robustness_significance_heatmap.png",
    "R02_robustness_peak_corr_distribution.png",
    "R03_grid_sensitivity_headline_pair.png",
    "M01_leadlag_correlation_matrix.png",
    "M02_tvp_var_spillovers_all_events.png",
    "M03_method_pvalue_comparison.png",
    "M04_prediction_auroc_auprc.png",
    "N01_node_centrality_comparison.png",
    "N02_bipartite_claim_network.png",
    "N03_stress_propagation_sequence.png",
    "P01_prediction_roc_by_event.png",
    "P02_claim_strength_by_event.png",
]


def main():
    parser = argparse.ArgumentParser(description="Generate extended figure pack.")
    parser.add_argument("--only", nargs="+", default=list(ALL_FIGURES.keys()),
                        help="Group codes to generate (E C R M N P)")
    args = parser.parse_args()

    groups = [g.upper() for g in args.only]
    fns = []
    for g in groups:
        if g in ALL_FIGURES:
            fns.extend(ALL_FIGURES[g])
        else:
            log.warning("Unknown group %s — skipping", g)

    log.info("Generating %d figures in %s", len(fns), OUT)
    for fn in fns:
        try:
            fn()
        except Exception as exc:
            log.error("Failed %s: %s", fn.__name__, exc, exc_info=True)

    generated = [f for f in EXTENDED_EXPECTED_FILES if (OUT / f).exists()]
    log.info("Done. %d/%d figures generated in %s",
             len(generated), len(EXTENDED_EXPECTED_FILES), OUT)


if __name__ == "__main__":
    main()
