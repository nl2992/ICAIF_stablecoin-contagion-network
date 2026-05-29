"""
scripts/98_make_narrative_figures.py
====================================
Generate three narrative figures for the paper:

  Figure 1  –  Provenance-gated multi-layer network architecture
  Figure 2  –  USDT/Curve 2023 AMM-flow timeline (headline result)
  Figure 3  –  Claim-gate audit bar chart (anti-cherry-pick transparency)

Outputs land in results/paper/figures/.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT  = Path(__file__).resolve().parents[1]
FIG_DIR    = REPO_ROOT / "results" / "paper" / "figures"
TABLE_DIR  = REPO_ROOT / "results" / "paper" / "tables"
GOLD_DIR   = REPO_ROOT / "data" / "gold"

# ── colour palette ─────────────────────────────────────────────────────────────
C_TIER_A     = "#27ae60"   # on-chain Tier A (green)
C_TIER_B     = "#7f8c8d"   # public Tier B  (grey)
C_FIXTURE    = "#bdc3c7"   # fixture / not used (light grey)
C_EDGE_AA    = "#27ae60"
C_EDGE_AB    = "#2980b9"   # A/B suggestive (blue)
C_EDGE_BB    = "#95a5a6"   # B/B contextual (mid-grey)
C_HEADLINE   = "#e67e22"   # headline star (amber)
C_BG_A       = "#d5f5e3"   # light green band
C_BG_B       = "#ebecec"   # light grey band


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Provenance-gated multi-layer network architecture
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure1(out_dir: Path) -> None:
    """Schematic showing the three data layers, node tiers, and claim-gate pipeline."""

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── layer bands ────────────────────────────────────────────────────────────
    bands = [
        # (y_bottom, height, colour, label, tier_tag)
        (0.25, 1.8, C_BG_B, "CEX Market Layer",       "Tier B  –  public OHLCV / BBO / trades"),
        (2.35, 1.9, C_BG_A, "AMM Pool Layer",          "Tier A  –  on-chain TokenExchange logs"),
        (4.45, 2.0, C_BG_A, "Settlement Flow Layer",   "Tier A  –  on-chain Transfer / mint-burn"),
    ]
    for y, h, col, label, sub in bands:
        rect = mpatches.FancyBboxPatch(
            (0.2, y), 8.8, h,
            boxstyle="round,pad=0.05", linewidth=0,
            facecolor=col, zorder=0,
        )
        ax.add_patch(rect)
        ax.text(0.42, y + h - 0.22, label,  fontsize=9.5, fontweight="bold",
                color="#2c3e50", va="top")
        ax.text(0.42, y + h - 0.50, sub,    fontsize=7.5, color="#555", va="top")

    # ── helper: draw a rounded node box ────────────────────────────────────────
    def node_box(cx, cy, text, tier, width=1.52, height=0.52):
        fc = C_TIER_A if tier == "A" else C_FIXTURE if tier == "FIX" else C_TIER_B
        ec = "#27ae60" if tier == "A" else "#aaa"
        lw = 1.6 if tier == "A" else 0.8
        rect = mpatches.FancyBboxPatch(
            (cx - width / 2, cy - height / 2), width, height,
            boxstyle="round,pad=0.06", linewidth=lw,
            facecolor=fc, edgecolor=ec, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=7.2, fontweight="bold", color="#1a1a1a", zorder=4)
        tag = "A" if tier == "A" else ("B" if tier == "B" else "FIX")
        tcol = C_TIER_A if tier == "A" else ("#e74c3c" if tier == "FIX" else C_TIER_B)
        ax.text(cx + width / 2 - 0.03, cy + height / 2 - 0.01,
                tag, ha="right", va="top", fontsize=6, color=tcol,
                fontweight="bold", zorder=5)
        return (cx, cy)

    # ── nodes: CEX layer  (y ≈ 0.25 + 0.9 = 1.15) ─────────────────────────────
    cex_y = 1.15
    cex_nodes = [
        (1.3,  cex_y, "usdc_binance",  "B"),
        (2.95, cex_y, "usdt_binance",  "B"),
        (4.6,  cex_y, "usdc_coinbase", "B"),
        (6.25, cex_y, "busd_binance",  "B"),
        (7.9,  cex_y, "usdt_kraken",   "FIX"),
    ]
    cex_pos = {}
    for x, y, lbl, tier in cex_nodes:
        cex_pos[lbl] = node_box(x, y, lbl.replace("_", "\n"), tier)

    # ── nodes: AMM layer  (y ≈ 2.35 + 0.95 = 3.30) ────────────────────────────
    amm_y = 3.30
    amm_nodes = [
        (2.1,  amm_y, "curve_3pool",        "A"),
        (4.7,  amm_y, "curve_crvusd_usdt",  "A"),
        (7.2,  amm_y, "curve_ust_wormhole", "A"),
    ]
    amm_pos = {}
    for x, y, lbl, tier in amm_nodes:
        amm_pos[lbl] = node_box(x, y, lbl.replace("_", "\n"), tier)

    # ── nodes: Settlement layer  (y ≈ 4.45 + 1.0 = 5.45) ─────────────────────
    sett_y = 5.45
    sett_nodes = [
        (1.6,  sett_y, "usdc_mint_burn",         "A"),
        (3.6,  sett_y, "usdt_mint_burn",          "FIX"),
        (5.7,  sett_y, "eth_usdc_exchange_flows", "B"),
        (7.9,  sett_y, "eth_usdt_exchange_flows", "B"),
    ]
    sett_pos = {}
    for x, y, lbl, tier in sett_nodes:
        sett_pos[lbl] = node_box(x, y, lbl.replace("_", "\n"), tier)

    # ── draw cross-layer edges ─────────────────────────────────────────────────
    def edge(p1, p2, col, lw=1.2, ls="-", zorder=2, alpha=0.75):
        ax.annotate(
            "", xy=p2, xytext=p1,
            arrowprops=dict(
                arrowstyle="-|>", color=col, lw=lw, ls=ls,
                connectionstyle="arc3,rad=0.08",
            ),
            zorder=zorder, alpha=alpha,
        )

    # Headline A/A edge (usdt_curve_2023)
    edge(amm_pos["curve_3pool"], amm_pos["curve_crvusd_usdt"], C_HEADLINE, lw=2.8, alpha=1.0)
    edge(amm_pos["curve_crvusd_usdt"], amm_pos["curve_3pool"], C_HEADLINE, lw=2.8, alpha=1.0)
    # annotate headline
    mid_x = (amm_pos["curve_3pool"][0] + amm_pos["curve_crvusd_usdt"][0]) / 2
    ax.text(mid_x, amm_y + 0.45, "★ A/A DEX-flow (paper-claimable)\np≤0.014, both directions",
            ha="center", va="bottom", fontsize=7.5, color=C_HEADLINE,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fef9e7", edgecolor=C_HEADLINE, lw=1))

    # Terra A/A edge (provenance-valid, not paper-claimable)
    edge(amm_pos["curve_3pool"], amm_pos["curve_ust_wormhole"], C_EDGE_AA, lw=1.5, ls="--", alpha=0.65)
    edge(amm_pos["curve_ust_wormhole"], amm_pos["curve_3pool"], C_EDGE_AA, lw=1.5, ls="--", alpha=0.65)
    ax.text(6.3, amm_y + 0.45, "A/A provenance-valid\n(not sig. hourly)",
            ha="center", va="bottom", fontsize=6.5, color="#888",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#f8f9fa", edgecolor="#ccc", lw=0.7))

    # Settlement → AMM A/A edge (usdc_svb)
    edge(sett_pos["usdc_mint_burn"], amm_pos["curve_3pool"], C_EDGE_AA, lw=1.5, ls="--", alpha=0.65)
    ax.text(1.2, 4.35, "A/A settlement\n(sparse; 4 events)",
            ha="center", va="bottom", fontsize=6.5, color="#888",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#f8f9fa", edgecolor="#ccc", lw=0.7))

    # A/B edges: CEX ↔ AMM (sample)
    edge(amm_pos["curve_3pool"], cex_pos["usdt_binance"], C_EDGE_AB, lw=1.0, alpha=0.55)
    edge(amm_pos["curve_3pool"], cex_pos["usdc_binance"], C_EDGE_AB, lw=1.0, alpha=0.55)

    # ── claim-gate pipeline box (right) ────────────────────────────────────────
    gate_x = 9.3
    gate_stages = [
        (6.1, "#27ae60", "① Provenance\ngate",   "Tier A/B·\nno fixture"),
        (4.7, "#2980b9", "② Statistical\ngate",   "FDR / Bonferroni\nblock-shuffle"),
        (3.3, "#e67e22", "③ Paper\ngate",          "both ① + ②\npaper_claim_allowed"),
    ]
    for gy, gcol, glbl, gsub in gate_stages:
        rect = mpatches.FancyBboxPatch(
            (gate_x - 1.5, gy - 0.55), 3.05, 1.05,
            boxstyle="round,pad=0.06", linewidth=1.4,
            facecolor="white", edgecolor=gcol, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(gate_x + 0.0, gy, glbl, ha="center", va="center",
                fontsize=8, fontweight="bold", color=gcol, zorder=4)
        ax.text(gate_x + 0.0, gy - 0.3, gsub, ha="center", va="center",
                fontsize=6.5, color="#555", zorder=4)

    # arrows between gate stages
    for gy_from, gy_to in [(6.1 - 0.55, 4.7 + 0.55), (4.7 - 0.55, 3.3 + 0.55)]:
        ax.annotate("", xy=(gate_x, gy_to), xytext=(gate_x, gy_from),
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.2), zorder=3)

    ax.text(gate_x, 6.85, "Claim-gate\npipeline", ha="center", va="top",
            fontsize=9, fontweight="bold", color="#2c3e50")
    ax.text(gate_x, 2.55, "claim_strength\nrobust / suggestive\n/ descriptive",
            ha="center", va="top", fontsize=7, color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef9e7",
                      edgecolor=C_HEADLINE, lw=1.2))

    # dotted line separating network from gate panel
    ax.axvline(9.1, color="#ccc", lw=0.8, ls="--", zorder=1)

    # ── legend ─────────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=C_TIER_A, edgecolor="#27ae60", lw=1.5, label="Tier A  – execution-grade on-chain"),
        mpatches.Patch(facecolor=C_TIER_B, edgecolor="#aaa",    lw=0.8, label="Tier B  – public market context"),
        mpatches.Patch(facecolor=C_FIXTURE, edgecolor="#ccc",   lw=0.8, label="Fixture / unavailable"),
        Line2D([0], [0], color=C_HEADLINE, lw=2.5,                      label="A/A DEX-flow  ★ paper-claimable"),
        Line2D([0], [0], color=C_EDGE_AA,  lw=1.5, ls="--",             label="A/A provenance-valid (not sig.)"),
        Line2D([0], [0], color=C_EDGE_AB,  lw=1.2,                      label="A/B suggestive directional"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", bbox_to_anchor=(0.0, -0.03),
              ncol=3, fontsize=7.0, framealpha=0.95,
              edgecolor="#ccc", handlelength=1.6)

    ax.set_title(
        "Figure 1 – Provenance-gated multi-layer stablecoin stress network",
        fontsize=11, fontweight="bold", pad=10, color="#2c3e50",
    )

    out = out_dir / "figure1_provenance_architecture.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", out.name)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 – USDT/Curve 2023 AMM-flow timeline
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure2(out_dir: Path) -> None:
    """Hourly usdc_net_sold_1h for curve_3pool and curve_crvusd_usdt."""

    parquet = GOLD_DIR / "dataset_contagion_features_usdt_curve_2023.parquet"
    if not parquet.exists():
        logger.warning("Gold parquet not found: %s — skipping Figure 2", parquet)
        return

    df = pd.read_parquet(parquet)
    dex = (
        df[df["node_id"].isin(["curve_3pool", "curve_crvusd_usdt"])]
        [["node_id", "wall_clock_utc", "usdc_net_sold_1h", "reserve_imbalance"]]
        .copy()
        .sort_values("wall_clock_utc")
    )

    p3   = dex[dex["node_id"] == "curve_3pool"].set_index("wall_clock_utc")["usdc_net_sold_1h"]
    crvU = dex[dex["node_id"] == "curve_crvusd_usdt"].set_index("wall_clock_utc")["usdc_net_sold_1h"]

    # Convert to thousands (USDC, 6-dec) for readability
    p3_k   = p3   / 1_000
    crvU_k = crvU / 1_000

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08})
    fig.patch.set_facecolor("white")

    # ── Panel 1: curve_3pool ───────────────────────────────────────────────────
    ax1.fill_between(p3_k.index, p3_k, 0,
                     where=(p3_k > 0), color=C_TIER_A, alpha=0.35, label="net USDC sold")
    ax1.fill_between(p3_k.index, p3_k, 0,
                     where=(p3_k < 0), color="#e74c3c", alpha=0.35, label="net USDC bought")
    ax1.plot(p3_k.index, p3_k, color=C_TIER_A, lw=1.1, alpha=0.85)
    ax1.axhline(0, color="#555", lw=0.6, ls="--")
    ax1.set_ylabel("USDC net sold (thousands)\nper hour", fontsize=8.5, color="#2c3e50")
    ax1.set_title("curve_3pool  (Tier A — Etherscan TokenExchange)", fontsize=9,
                  fontweight="bold", color="#2c3e50", loc="left")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}k"))
    ax1.grid(axis="y", color="#e0e0e0", lw=0.5)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Panel 2: curve_crvusd_usdt ─────────────────────────────────────────────
    ax2.fill_between(crvU_k.index, crvU_k, 0,
                     where=(crvU_k > 0), color=C_EDGE_AB, alpha=0.35, label="net USDC sold")
    ax2.fill_between(crvU_k.index, crvU_k, 0,
                     where=(crvU_k < 0), color="#e74c3c", alpha=0.35, label="net USDC bought")
    ax2.plot(crvU_k.index, crvU_k, color=C_EDGE_AB, lw=1.1, alpha=0.85)
    ax2.axhline(0, color="#555", lw=0.6, ls="--")
    ax2.set_ylabel("USDC net sold (thousands)\nper hour", fontsize=8.5, color="#2c3e50")
    ax2.set_title("curve_crvusd_usdt  (Tier A — Etherscan TokenExchange)", fontsize=9,
                  fontweight="bold", color="#2c3e50", loc="left")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}k"))
    ax2.grid(axis="y", color="#e0e0e0", lw=0.5)
    ax2.spines[["top", "right"]].set_visible(False)

    # ── shared date formatting ─────────────────────────────────────────────────
    import matplotlib.dates as mdates
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    # ── headline annotation ────────────────────────────────────────────────────
    peak_date = pd.Timestamp("2023-06-16", tz="UTC")
    ax1.axvline(peak_date, color=C_HEADLINE, lw=1.4, ls=":", alpha=0.8)
    ax2.axvline(peak_date, color=C_HEADLINE, lw=1.4, ls=":", alpha=0.8)
    ax1.text(peak_date, ax1.get_ylim()[1] * 0.92,
             "USDT de-peg\npeak stress",
             ha="left", va="top", fontsize=7.5, color=C_HEADLINE,
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#fef9e7",
                       edgecolor=C_HEADLINE, lw=0.8))

    # result box
    result_txt = (
        "★  Headline result (usdt_curve_2023)\n"
        "curve_3pool  ↔  curve_crvusd_usdt\n"
        "feature = usdc_net_sold_1h  |  grid = 3600 s\n"
        "claim_level = A_A_dex_flow  |  claim_strength = robust\n"
        "p_bonferroni ≤ 0.014 (both directions)"
    )
    ax2.text(0.01, 0.04, result_txt,
             transform=ax2.transAxes,
             fontsize=7.5, va="bottom", ha="left",
             color="#1a1a1a",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef9e7",
                       edgecolor=C_HEADLINE, lw=1.2, alpha=0.95))

    fig.suptitle(
        "Figure 2 – USDT/Curve 2023: hourly Tier-A AMM-flow (usdc_net_sold_1h)\n"
        "Bonferroni-significant bidirectional lead-lag between Curve pools",
        fontsize=10.5, fontweight="bold", color="#2c3e50", y=0.995,
    )

    out = out_dir / "figure2_usdt_curve_amm_flow_timeline.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", out.name)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Claim-gate audit bar chart
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure3(out_dir: Path) -> None:
    """Per-event grouped bar chart of claim-gate audit counts."""

    audit_path = TABLE_DIR / "table_claim_audit_summary.csv"
    if not audit_path.exists():
        logger.warning("table_claim_audit_summary.csv not found — skipping Figure 3")
        return

    df = pd.read_csv(audit_path)
    # drop blank row if it survived
    df = df[df["event_id"].notna() & (df["event_id"] != "")]

    # Pretty event labels
    label_map = {
        "usdt_curve_2023": "USDT/Curve\n2023",
        "terra_luna_2022": "Terra/LUNA\n2022",
        "usdc_svb_2023":   "USDC/SVB\n2023",
        "ftx_2022":        "FTX\n2022",
        "busd_2023":       "BUSD\n2023",
    }
    event_order = ["usdt_curve_2023", "terra_luna_2022", "usdc_svb_2023", "ftx_2022", "busd_2023"]
    df["event_label"] = df["event_id"].map(label_map)
    valid_events = [e for e in event_order if e in df["event_id"].values]
    df = df.set_index("event_id").loc[valid_events]

    x = np.arange(len(df))
    bar_w = 0.18

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")

    # Series to plot
    series = [
        ("n_AA_provenance",       C_TIER_A,    "A/A provenance-valid (may not be paper-claimable)", "//"),
        ("n_AA_paper_claimable",  C_HEADLINE,  "A/A paper-claimable (★ headline)",                  None),
        ("n_AB_paper_claimable",  C_EDGE_AB,   "A/B suggestive (paper-claimable)",                  None),
        ("n_BB_context",          C_EDGE_BB,   "B/B context-only",                                  None),
        ("n_paper_claimable",     "#2c3e50",   "Total paper-claimable (all levels)",                None),
    ]

    offsets = np.linspace(-(len(series) - 1) / 2, (len(series) - 1) / 2, len(series)) * bar_w

    for (col, colour, label, hatch), offset in zip(series, offsets):
        vals = df[col].values.astype(float) if col in df.columns else np.zeros(len(df))
        bars = ax.bar(x + offset, vals, bar_w,
                      color=colour, alpha=0.82, label=label,
                      hatch=hatch, edgecolor="white" if hatch is None else "#555",
                      linewidth=0.5 if hatch is None else 0.8)
        # value labels above non-zero bars
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.08,
                        str(int(v)), ha="center", va="bottom",
                        fontsize=7, color="#2c3e50", fontweight="bold")

    # styling
    ax.set_xticks(x)
    ax.set_xticklabels(df["event_label"].values, fontsize=9.5)
    ax.set_ylabel("Number of edges", fontsize=9.5, color="#2c3e50")
    ax.set_xlabel("Event", fontsize=9.5, color="#2c3e50")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", lw=0.6)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)

    # annotate the one headline event
    if "usdt_curve_2023" in df.index:
        ax.axvspan(-0.5, 0.5, facecolor="#fef9e7", alpha=0.45, zorder=0)
        ax.text(0, ax.get_ylim()[1] * 0.97, "★ headline",
                ha="center", va="top", fontsize=8, color=C_HEADLINE, fontweight="bold")

    ax.legend(fontsize=8, loc="upper right", framealpha=0.95,
              edgecolor="#ccc", handlelength=1.8)

    ax.set_title(
        "Figure 3 – Claim-gate audit: paper-claimable edges per event\n"
        "(anti-cherry-pick transparency; all zero-fixture rows across 867 annotated edges)",
        fontsize=10.5, fontweight="bold", color="#2c3e50", pad=10,
    )

    out = out_dir / "figure3_claim_audit_bar_chart.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", out.name)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper narrative figures 1–3.")
    parser.add_argument("--fig-dir", default=None,
                        help="Output directory (default: results/paper/figures).")
    args = parser.parse_args()

    out_dir = Path(args.fig_dir) if args.fig_dir else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    make_figure1(out_dir)
    make_figure2(out_dir)
    make_figure3(out_dir)
    logger.info("All narrative figures complete.")


if __name__ == "__main__":
    main()
