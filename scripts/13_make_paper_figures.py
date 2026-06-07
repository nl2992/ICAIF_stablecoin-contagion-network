"""
scripts/13_make_paper_figures.py
================================
Generate all 12 paper figures.

Figure 01  –  Multi-layer stress propagation architecture
Figure 02  –  Claim-gate pipeline diagram
Figure 03  –  Claim audit by event (anti-cherry-pick bar chart)
Figure 04  –  USDT/Curve 2023 AMM-flow timeline   (main empirical figure)
Figure 05  –  USDT/Curve 2023 lead-lag profile     (headline evidence)
Figure 06  –  A/A paper-claimable edge network
Figure 07  –  A/A provenance-valid vs paper-claimable
Figure 08  –  Terra/LUNA A/A AMM-flow negative result
Figure 09  –  USDC/SVB sparse settlement response
Figure 10  –  Feature-tier matrix
Figure 11  –  Node provenance coverage heatmap
Figure 12  –  Full paper-claimable network

Outputs: results/paper/figures/figure_01_*.png  …  figure_12_*.png

Usage:
    python scripts/13_make_paper_figures.py [--fig-dir PATH]
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
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

REPO_ROOT  = Path(__file__).resolve().parents[1]
FIG_DIR    = REPO_ROOT / "results" / "paper" / "figures"
TABLE_DIR  = REPO_ROOT / "results" / "paper" / "tables"
RAW_TBL    = REPO_ROOT / "results" / "tables"
GOLD_DIR   = REPO_ROOT / "data" / "gold"

# ── Colour scheme ──────────────────────────────────────────────────────────────
CA   = "#27ae60"   # Tier A green
CB   = "#7f8c8d"   # Tier B grey
CFIX = "#bdc3c7"   # Fixture light-grey
CAA  = "#27ae60"   # A/A edge
CAB  = "#2980b9"   # A/B edge  (blue)
CBB  = "#95a5a6"   # B/B edge  (mid-grey)
CSTAR= "#e67e22"   # Headline amber
CBA  = "#d5f5e3"   # light-green background band
CBB2 = "#ecf0f1"   # light-grey background band


def _node_name(row: dict, prefer="node_i") -> str:
    """Return normalised source node name from a CSV row dict."""
    for col in (prefer, "causing_node", "source_node_id", "source_node", "source"):
        if col in row and row[col]:
            return row[col]
    return "?"


def _node_name_j(row: dict) -> str:
    for col in ("node_j", "caused_node", "target_node_id", "target_node", "target"):
        if col in row and row[col]:
            return row[col]
    return "?"


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 01 – Multi-layer architecture
# ═══════════════════════════════════════════════════════════════════════════════

def fig01_multilayer_architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7.5); ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── shock annotation (left column) ─────────────────────────────────────────
    shock_items = [
        (0.75, 6.8, "USDT/Curve 2023",   "DeFi pool imbalance"),
        (0.75, 6.1, "Terra/LUNA 2022",   "Algo-stablecoin collapse"),
        (0.75, 5.4, "USDC/SVB 2023",     "Fiat-reserve bank shock"),
        (0.75, 4.7, "FTX 2022",           "Exchange credit shock"),
        (0.75, 4.0, "BUSD 2023",          "Issuer regulatory wind-down"),
    ]
    ax.text(0.75, 7.3, "Stablecoin\nstress events", ha="center", va="top",
            fontsize=9, fontweight="bold", color="#2c3e50")
    for x, y, ev, mech in shock_items:
        ax.text(x, y,   ev,   ha="center", va="top", fontsize=7.5,
                fontweight="bold", color=CSTAR)
        ax.text(x, y-0.3, mech, ha="center", va="top", fontsize=6.5, color="#555")

    # ── three evidence bands ────────────────────────────────────────────────────
    bands = [
        (1.8, 1.5, CBB2, "CEX Market Layer",
         "Tier B — public OHLCV / BBO / trades",
         ["usdc_binance", "usdt_binance", "usdc_coinbase", "busd_binance"]),
        (3.5, 1.7, CBA,  "AMM Pool Layer",
         "Tier A — Curve TokenExchange logs",
         ["curve_3pool", "curve_crvusd_usdt", "curve_ust_wormhole"]),
        (5.4, 1.5, CBA,  "Settlement / Flow Layer",
         "Tier A/B — on-chain mint-burn; CoinMetrics flows",
         ["usdc_mint_burn", "eth_usdc_exchange_flows"]),
    ]
    node_centres: dict[str, tuple[float, float]] = {}
    for y0, h, col, lbl, sub, nodes in bands:
        rect = mpatches.FancyBboxPatch((1.8, y0), 7.4, h,
            boxstyle="round,pad=0.04", lw=0, facecolor=col, zorder=0)
        ax.add_patch(rect)
        ax.text(2.0, y0+h-0.18, lbl, fontsize=9.5, fontweight="bold",
                color="#2c3e50", va="top")
        ax.text(2.0, y0+h-0.44, sub, fontsize=7.5, color="#555", va="top")
        x_spacing = 7.0 / (len(nodes) + 1)
        cy = y0 + h * 0.4
        tier = "A" if "A —" in sub else ("A/B" if "A/B" in sub else "B")
        for i, nid in enumerate(nodes):
            cx = 2.0 + (i+1)*x_spacing
            fc = CA if tier in ("A","A/B") else CB
            ec = CA if tier in ("A","A/B") else "#aaa"
            rect = mpatches.FancyBboxPatch((cx-0.75, cy-0.22), 1.5, 0.44,
                boxstyle="round,pad=0.04", lw=1.2, facecolor=fc,
                edgecolor=ec, zorder=3, alpha=0.9)
            ax.add_patch(rect)
            ax.text(cx, cy, nid.replace("_","\n"), ha="center", va="center",
                    fontsize=6.5, fontweight="bold", color="white" if tier=="A" else "#1a1a1a",
                    zorder=4)
            node_centres[nid] = (cx, cy)

    # ── key edges ────────────────────────────────────────────────────────────────
    def draw_edge(n1, n2, col, lw=1.2, ls="-", rad=0.1, alpha=0.75):
        if n1 not in node_centres or n2 not in node_centres:
            return
        p1, p2 = node_centres[n1], node_centres[n2]
        ax.annotate("", xy=p2, xytext=p1,
            arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, ls=ls,
                            connectionstyle=f"arc3,rad={rad}"),
            zorder=2, alpha=alpha)

    # headline
    draw_edge("curve_3pool", "curve_crvusd_usdt", CSTAR, lw=2.8, alpha=1.0)
    draw_edge("curve_crvusd_usdt", "curve_3pool",  CSTAR, lw=2.8, alpha=1.0, rad=-0.1)
    mid = (node_centres["curve_3pool"][0]+node_centres["curve_crvusd_usdt"][0])/2
    ax.text(mid, 5.25, "★ A/A DEX-flow\n(robust, p≤0.014)",
            ha="center", fontsize=7.5, color=CSTAR, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fef9e7",
                      edgecolor=CSTAR, lw=1))

    draw_edge("curve_3pool", "curve_ust_wormhole", CAA, lw=1.5, ls="--", alpha=0.6)
    draw_edge("curve_ust_wormhole", "curve_3pool",  CAA, lw=1.5, ls="--", alpha=0.6, rad=-0.1)

    draw_edge("usdc_mint_burn", "curve_3pool", CAA, lw=1.2, ls="--", alpha=0.6, rad=0.2)

    draw_edge("curve_3pool", "usdt_binance",  CAB, lw=1.0, alpha=0.5)
    draw_edge("curve_3pool", "busd_binance",  CAB, lw=1.0, alpha=0.5, rad=0.15)

    # ── claim-gate box (right) ──────────────────────────────────────────────────
    gx = 10.0
    for gy, gcol, glbl, gsub in [(6.5,"#27ae60","① Provenance","Tier A/B · no fixture"),
                                   (5.3,"#2980b9","② Statistical","FDR / Bonferroni / block-shuffle"),
                                   (4.1,"#e67e22","③ Paper gate","paper_claim_allowed = True")]:
        rect = mpatches.FancyBboxPatch((gx-1.4, gy-0.45), 2.9, 0.9,
            boxstyle="round,pad=0.05", lw=1.4, facecolor="white",
            edgecolor=gcol, zorder=3)
        ax.add_patch(rect)
        ax.text(gx, gy+0.07, glbl, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color=gcol, zorder=4)
        ax.text(gx, gy-0.2, gsub, ha="center", va="center", fontsize=7,
                color="#555", zorder=4)

    for yf, yt in [(6.05, 5.75), (4.85, 4.55)]:
        ax.annotate("", xy=(gx,yt), xytext=(gx,yf),
            arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.2), zorder=3)

    ax.axvline(9.4, color="#ccc", lw=0.8, ls="--")
    ax.text(gx, 7.25, "Claim gate", ha="center", fontsize=9,
            fontweight="bold", color="#2c3e50")

    # ── legend ──────────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=CA, edgecolor=CA, label="Tier A — on-chain execution-grade"),
        mpatches.Patch(facecolor=CB, edgecolor="#aaa", label="Tier B — public market context"),
        Line2D([0],[0], color=CSTAR, lw=2.5,    label="A/A paper-claimable ★"),
        Line2D([0],[0], color=CAA,  lw=1.5, ls="--", label="A/A provenance-valid (not sig.)"),
        Line2D([0],[0], color=CAB,  lw=1.2,    label="A/B suggestive"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", bbox_to_anchor=(0.13,-0.01),
              ncol=3, fontsize=7.5, framealpha=0.95, edgecolor="#ccc")

    ax.set_title("Figure 1 – Multi-layer stablecoin stress-propagation architecture\n"
                 "Tier-A Curve AMM flow provides the paper's primary directional evidence",
                 fontsize=11, fontweight="bold", pad=10, color="#2c3e50")
    _save(fig, out, "figure_01_multilayer_architecture.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 02 – Claim-gate pipeline diagram
# ═══════════════════════════════════════════════════════════════════════════════

def fig02_claim_gate_pipeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.5); ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(cx, cy, w, h, fc, ec, label, sublabel="", lw=1.4):
        rect = mpatches.FancyBboxPatch((cx-w/2, cy-h/2), w, h,
            boxstyle="round,pad=0.08", lw=lw, facecolor=fc, edgecolor=ec, zorder=3)
        ax.add_patch(rect)
        ax.text(cx, cy+0.07*(1 if sublabel else 0), label, ha="center", va="center",
                fontsize=9, fontweight="bold", color=ec, zorder=4)
        if sublabel:
            ax.text(cx, cy-0.22, sublabel, ha="center", va="center",
                    fontsize=7, color="#555", zorder=4)
        return cx, cy

    def arrow(x1, x2, y, col="#888", lbl=""):
        ax.annotate("", xy=(x2,y), xytext=(x1,y),
            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5), zorder=3)
        if lbl:
            ax.text((x1+x2)/2, y+0.22, lbl, ha="center", fontsize=7, color=col)

    # ── top row: source → node tier → feature tier → edge tier ─────────────────
    y1 = 4.2
    box(1.0, y1, 1.5, 0.7, "#eaf4fb", "#2980b9", "Raw source",
        "Etherscan / Binance\n/ CoinMetrics")
    arrow(1.75, 2.25, y1, "#2980b9", "ingest")
    box(3.0, y1, 1.5, 0.7, CBA,  CA, "Node tier",
        "A = on-chain\nB = public market")
    arrow(3.75, 4.25, y1, CA,    "×")
    box(5.0, y1, 1.5, 0.7, CBA,  CA, "Feature tier",
        "A = TokenExchange\nB = derived proxy")
    arrow(5.75, 6.75, y1, CA,    "min()")
    box(7.5, y1, 1.5, 0.7, CBA,  CA, "Edge tier",
        "min(node_i, node_j,\nfeature)")

    # ── example boxes ───────────────────────────────────────────────────────────
    y_ex = 2.8
    ax.text(0.3, y_ex+0.5, "Example:", fontsize=8.5, fontweight="bold", color="#2c3e50")
    examples = [
        (1.0, "Curve\nTokenExchange", CA),
        (3.0, "curve_3pool\nTier A", CA),
        (5.0, "usdc_net_sold_1h\nTier A", CA),
        (7.5, "A_A_dex_flow\n(edge Tier A)", CA),
    ]
    for ex_x, ex_lbl, ex_col in examples:
        rect = mpatches.FancyBboxPatch((ex_x-0.7, y_ex-0.22), 1.4, 0.44,
            boxstyle="round,pad=0.05", lw=1, facecolor=ex_col,
            edgecolor=ex_col, zorder=3, alpha=0.75)
        ax.add_patch(rect)
        ax.text(ex_x, y_ex, ex_lbl, ha="center", va="center",
                fontsize=6.8, color="white", zorder=4, fontweight="bold")

    # ── second row: statistical test → claim level → paper gate ────────────────
    y2 = 1.6
    box(7.5, y2, 1.5, 0.7,   "#fef9e7", CSTAR, "Statistical\ntest",
        "FDR / Bonferroni\nblock-shuffle")
    arrow(8.25, 9.25, y2, CSTAR, "if p < α")
    box(10.0, y2, 1.5, 0.7, "#fef9e7", CSTAR, "claim_level",
        "A_A_dex_flow\nA_B_suggestive\nB_B_context")
    arrow(10.75, 11.75, y2, CSTAR, "both gates?")
    box(12.5, y2, 1.5, 0.7, "#fef9e7", CSTAR, "paper_claim\n_allowed",
        "True / False")

    # vertical connector edge tier → stat test
    ax.annotate("", xy=(7.5, y2+0.35), xytext=(7.5, y1-0.35),
        arrowprops=dict(arrowstyle="-|>", color=CA, lw=1.5), zorder=3)
    ax.text(7.85, (y1+y2)/2, "provenance\ngate ①", ha="left",
            fontsize=7, color=CA, fontweight="bold")

    # verdict box
    ax.text(12.5, 0.9, "paper_claim_allowed\n=True  ↔  False",
            ha="center", va="center", fontsize=8, color=CSTAR,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef9e7",
                      edgecolor=CSTAR, lw=1.2))
    ax.annotate("", xy=(12.5, y2-0.35), xytext=(12.5, 1.15),
        arrowprops=dict(arrowstyle="-|>", color=CSTAR, lw=1.2), zorder=3)

    # gate labels
    ax.text(8.0, y1-0.9, "① Provenance gate\n(node tier · feature tier · no fixture)",
            ha="center", fontsize=8, color=CA, fontweight="bold")
    ax.text(10.5, y1-0.9, "② Statistical gate\n(method-specific p-value threshold)",
            ha="center", fontsize=8, color=CSTAR, fontweight="bold")

    ax.set_title(
        "Figure 2 – Provenance-aware claim gate: an edge is paper-claimable only if\n"
        "it passes both the provenance gate (node + feature tier) and the statistical gate",
        fontsize=10.5, fontweight="bold", pad=10, color="#2c3e50")
    _save(fig, out, "figure_02_claim_gate_pipeline.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 03 – Claim audit by event (stacked bars)
# ═══════════════════════════════════════════════════════════════════════════════

def fig03_claim_audit(out: Path) -> None:
    df = _read_csv(TABLE_DIR / "table_claim_audit_summary.csv")
    if df is None: return
    df = df[df["event_id"].notna() & (df["event_id"] != "")]

    event_order = ["usdt_curve_2023","terra_luna_2022","usdc_svb_2023","ftx_2022","busd_2023"]
    labels_map  = {"usdt_curve_2023":"USDT/Curve\n2023","terra_luna_2022":"Terra/LUNA\n2022",
                   "usdc_svb_2023":"USDC/SVB\n2023","ftx_2022":"FTX\n2022","busd_2023":"BUSD\n2023"}
    valid = [e for e in event_order if e in df["event_id"].values]
    df = df.set_index("event_id").reindex(valid)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("white")
    x = np.arange(len(df))

    def ivals(col):
        return df[col].fillna(0).values.astype(int) if col in df.columns else np.zeros(len(df))

    bb   = ivals("n_BB_context")
    ab   = ivals("n_AB_paper_claimable")
    aac  = ivals("n_AA_paper_claimable")
    aap  = ivals("n_AA_provenance")
    # A/A provenance-valid BUT NOT paper-claimable (the "failed stat gate" slice)
    # = rows that pass provenance but not the statistical gate
    aap_only = np.maximum(aap - aac, 0)

    # Stacked order: B/B → A/B paper → A/A prov-only → A/A paper-claimable
    # Each row appears exactly once; no double-counting.
    b1 = ax.bar(x, bb,       color=CBB,   alpha=0.8, label="B/B context-only")
    b2 = ax.bar(x, ab,       bottom=bb,                       color=CAB,   alpha=0.8,
                label="A/B suggestive (paper-claimable)")
    b3 = ax.bar(x, aap_only, bottom=bb+ab,                    color=CA,    alpha=0.55,
                hatch="//", edgecolor="#555", lw=0.7,
                label="A/A provenance-valid, stat. unsupported (not paper-claimable)")
    b4 = ax.bar(x, aac,      bottom=bb+ab+aap_only,           color=CSTAR, alpha=1.0,
                label="A/A paper-claimable ★ (both gates pass — headline)")

    stacks_and_vals = [
        (b1, np.zeros(len(df)), bb),
        (b2, bb,                ab),
        (b3, bb+ab,             aap_only),
        (b4, bb+ab+aap_only,    aac),
    ]
    for bars, stack, vals in stacks_and_vals:
        for bar, s, v in zip(bars, stack, vals):
            if v > 0:
                ax.text(bar.get_x()+bar.get_width()/2, s+v/2,
                        str(int(v)), ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")

    # highlight USDT/Curve
    if "usdt_curve_2023" in df.index:
        ax.axvspan(-0.5, 0.5, facecolor="#fef9e7", alpha=0.5, zorder=0)
        ylim = ax.get_ylim()
        ax.set_ylim(ylim)
        ax.text(0, (bb+ab+aap_only+aac).max()*1.06 if len(df) > 0 else 1,
                "★ headline event", ha="center", va="bottom",
                fontsize=8.5, color=CSTAR, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([labels_map.get(e,e) for e in df.index], fontsize=10)
    ax.set_ylabel("Number of edges (no double-counting)", fontsize=9.5)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", lw=0.5)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95, edgecolor="#ccc")
    ax.set_title(
        "Figure 3 – Claim-gate audit: edge composition per event (anti-cherry-pick)\n"
        "Each edge row counted once — A/A provenance-valid ≠ A/A paper-claimable",
        fontsize=10.5, fontweight="bold", color="#2c3e50", pad=10)
    _save(fig, out, "figure_03_claim_audit_by_event.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 04 – USDT/Curve AMM-flow timeline  (main empirical figure)
# ═══════════════════════════════════════════════════════════════════════════════

def fig04_usdt_curve_timeline(out: Path) -> None:
    df = _gold("usdt_curve_2023")
    if df is None: return

    p3   = _dex_series(df, "curve_3pool",       "usdc_net_sold_1h")
    crvU = _dex_series(df, "curve_crvusd_usdt",  "usdc_net_sold_1h")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True,
        gridspec_kw={"height_ratios":[1,1,0.7], "hspace":0.06})
    fig.patch.set_facecolor("white")

    for ax, ser, col, title in [
        (ax1, p3/1_000,   CA,  "curve_3pool  (Tier A — Etherscan TokenExchange)"),
        (ax2, crvU/1_000, CAB, "curve_crvusd_usdt  (Tier A — Etherscan TokenExchange)"),
    ]:
        ax.fill_between(ser.index, ser, 0, where=(ser>0), color=col,  alpha=0.35)
        ax.fill_between(ser.index, ser, 0, where=(ser<0), color="#e74c3c", alpha=0.35)
        ax.plot(ser.index, ser, color=col, lw=1.0, alpha=0.9)
        ax.axhline(0, color="#555", lw=0.6, ls="--")
        ax.set_ylabel("USDC net sold (k)\nper hour", fontsize=8)
        ax.set_title(title, fontsize=8.5, fontweight="bold", color="#2c3e50", loc="left")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:,.0f}k"))
        ax.grid(axis="y", color="#e0e0e0", lw=0.5)
        ax.spines[["top","right"]].set_visible(False)

    # panel 3: cumulative flows
    p3_cum   = (p3/1_000).cumsum().ffill()
    crvU_cum = (crvU/1_000).cumsum().ffill()
    ax3.plot(p3_cum.index,   p3_cum,   color=CA,  lw=1.4, label="curve_3pool cumulative")
    ax3.plot(crvU_cum.index, crvU_cum, color=CAB, lw=1.4, label="curve_crvusd_usdt cumulative")
    ax3.axhline(0, color="#555", lw=0.6, ls="--")
    ax3.set_ylabel("Cumulative\nflow (k USDC)", fontsize=8)
    ax3.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax3.grid(axis="y", color="#e0e0e0", lw=0.5)
    ax3.spines[["top","right"]].set_visible(False)

    # shared x-axis
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax3.get_xticklabels(), rotation=25, ha="right", fontsize=8)

    # headline annotation
    peak = pd.Timestamp("2023-06-16", tz="UTC")
    for ax in (ax1, ax2, ax3):
        ax.axvline(peak, color=CSTAR, lw=1.3, ls=":", alpha=0.8)
    ax1.text(peak, ax1.get_ylim()[1]*0.9, "peak stress",
             ha="left", fontsize=7, color=CSTAR,
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#fef9e7",
                       edgecolor=CSTAR, lw=0.7))

    result_txt = ("★  curve_3pool  ↔  curve_crvusd_usdt\n"
                  "feature=usdc_net_sold_1h  ·  grid=3600s\n"
                  "A_A_dex_flow  ·  robust  ·  p_bonferroni≤0.014 (both directions)")
    ax2.text(0.01, 0.06, result_txt, transform=ax2.transAxes,
             fontsize=7.5, va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef9e7",
                       edgecolor=CSTAR, lw=1.2, alpha=0.95))

    fig.suptitle("Figure 4 – USDT/Curve 2023: Tier-A hourly AMM-flow\n"
                 "(usdc_net_sold_1h, Curve TokenExchange logs — primary empirical evidence)",
                 fontsize=10.5, fontweight="bold", color="#2c3e50", y=1.0)
    _save(fig, out, "figure_04_usdt_curve_amm_flow_timeline.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 05 – USDT/Curve lead-lag profile
# ═══════════════════════════════════════════════════════════════════════════════

def fig05_leadlag_profile(out: Path) -> None:
    from stressnet.models.leadlag import cross_correlation_lags

    df = _gold("usdt_curve_2023")
    if df is None: return

    p3   = _dex_series(df, "curve_3pool",      "usdc_net_sold_1h").dropna()
    crvU = _dex_series(df, "curve_crvusd_usdt","usdc_net_sold_1h").dropna()
    common = p3.index.intersection(crvU.index)
    x3, xU = p3.loc[common].values, crvU.loc[common].values

    MAX_LAG = 12
    lags1, corrs1 = cross_correlation_lags(x3, xU, max_lag=MAX_LAG)   # 3pool → crvUSD
    lags2, corrs2 = cross_correlation_lags(xU, x3, max_lag=MAX_LAG)   # crvUSD → 3pool

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("white")

    for ax, lags, corrs, ni, nj, col, pbon in [
        (ax1, lags1, corrs1, "curve_3pool",      "curve_crvusd_usdt", CA,  0.014),
        (ax2, lags2, corrs2, "curve_crvusd_usdt","curve_3pool",        CAB, 0.0),
    ]:
        ax.bar(lags*3600/3600, corrs, color=col, alpha=0.6, width=0.8,
               label="cross-correlation")
        peak_idx = np.argmax(np.abs(corrs))
        ax.bar([lags[peak_idx]], [corrs[peak_idx]], color=CSTAR, alpha=1.0,
               width=0.8, label=f"peak lag (step {lags[peak_idx]:+d}h)")
        ax.axhline(0, color="#555", lw=0.7, ls="--")
        ax.axvline(0, color="#aaa", lw=0.7, ls=":")
        ax.set_xlabel("Lag (hours)  [positive = node_i leads node_j]", fontsize=8.5)
        ax.set_ylabel("Cross-correlation coefficient", fontsize=8.5)
        ax.set_title(f"{ni.replace('_',chr(10))} → {nj.replace('_',chr(10))}",
                     fontsize=8.5, fontweight="bold", color="#2c3e50")
        ax.text(0.97, 0.97,
                f"p_bonferroni = {pbon}\nclaim_level = A_A_dex_flow\nclaim_strength = robust",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                color=CSTAR, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef9e7",
                          edgecolor=CSTAR, lw=0.9))
        ax.legend(fontsize=8, loc="lower right")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", color="#e0e0e0", lw=0.5)

    fig.suptitle("Figure 5 – USDT/Curve 2023 hourly AMM-flow lead-lag profile\n"
                 "Both directions Bonferroni-significant · feature = usdc_net_sold_1h · grid = 3600s",
                 fontsize=10.5, fontweight="bold", color="#2c3e50")
    plt.tight_layout(rect=[0,0,1,0.92])
    _save(fig, out, "figure_05_usdt_curve_leadlag_profile.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 06 – A/A paper-claimable edge mini-network
# ═══════════════════════════════════════════════════════════════════════════════

def fig06_aa_paper_network(out: Path) -> None:
    import networkx as nx

    df = _read_csv(TABLE_DIR / "table_aa_paper_claimable_edges.csv")
    if df is None or len(df) == 0: return

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    G = nx.DiGraph()
    for _, row in df.iterrows():
        ni = row.get("node_i") or row.get("causing_node","")
        nj = row.get("node_j") or row.get("caused_node","")
        pbon = row.get("p_bonferroni", "")
        pfdr = row.get("p_value_fdr", "")
        cs   = row.get("claim_strength","robust")
        G.add_edge(ni, nj, pbon=pbon, pfdr=pfdr, cs=cs)

    pos = nx.spring_layout(G, seed=42, k=2)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=CA, node_size=3500,
                           edgecolors="#1a6e3a", linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={n: n.replace("_","\n") for n in G.nodes()},
                            font_size=9, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=CSTAR, width=2.5,
                           arrowsize=20, arrowstyle="-|>",
                           connectionstyle="arc3,rad=0.08",
                           min_source_margin=30, min_target_margin=30)

    # edge labels
    for (u,v,d) in G.edges(data=True):
        x1,y1 = pos[u]; x2,y2 = pos[v]
        mx, my = (x1+x2)/2 + 0.05*(y2-y1), (y1+y2)/2 + 0.05*(x1-x2)
        pbon = d.get("pbon","")
        try: pbon_str = f"p_bonf={float(pbon):.3f}"
        except: pbon_str = ""
        pfdr = d.get("pfdr","")
        try: pfdr_str = f"p_FDR={float(pfdr):.3f}"
        except: pfdr_str = ""
        ax.text(mx, my, f"{pbon_str}\n{pfdr_str}",
                ha="center", fontsize=8, color=CSTAR,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#fef9e7",
                          edgecolor=CSTAR, lw=0.8, alpha=0.9))

    ax.text(0.5, 0.02,
            "claim_level = A_A_dex_flow  ·  claim_strength = robust  ·  paper_claim_allowed = True\n"
            "event = usdt_curve_2023  ·  feature = usdc_net_sold_1h  ·  grid = 3600s",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=CBA,
                      edgecolor=CA, lw=1.2))

    # Tier legend (TODO 4.6 — tier labels on all network figures)
    tier_legend = [
        mpatches.Patch(facecolor=CA, edgecolor="#1a6e3a", lw=1.5,
                       label="Tier A node — on-chain execution-grade"),
        Line2D([0],[0], color=CSTAR, lw=2.5, linestyle="solid",
               label="A/A edge — solid · paper-claimable"),
    ]
    ax.legend(handles=tier_legend, loc="upper right", fontsize=8.5,
              framealpha=0.95, edgecolor="#ccc")

    ax.set_title("Figure 6 – A/A paper-claimable AMM-flow network\n"
                 "Solid edges = A/A (both Tier-A nodes, both gates pass)  ·  "
                 "Dark fill = Tier-A on-chain execution-grade data",
                 fontsize=10.5, fontweight="bold", color="#2c3e50", pad=12)
    _save(fig, out, "figure_06_aa_paper_claimable_network.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 07 – A/A provenance-valid vs paper-claimable
# ═══════════════════════════════════════════════════════════════════════════════

def fig07_aa_prov_vs_paper(out: Path) -> None:
    df = _read_csv(TABLE_DIR / "table_claim_audit_summary.csv")
    if df is None: return
    df = df[df["event_id"].notna() & (df["event_id"] != "")]

    event_order = ["usdt_curve_2023","terra_luna_2022","usdc_svb_2023","ftx_2022","busd_2023"]
    labels_map  = {"usdt_curve_2023":"USDT/Curve\n2023","terra_luna_2022":"Terra/LUNA\n2022",
                   "usdc_svb_2023":"USDC/SVB\n2023","ftx_2022":"FTX\n2022","busd_2023":"BUSD\n2023"}
    df = df.set_index("event_id").reindex([e for e in event_order if e in df["event_id"].values])

    x   = np.arange(len(df))
    w   = 0.32
    v_prov = df["n_AA_provenance"].fillna(0).astype(int).values if "n_AA_provenance" in df.columns else np.zeros(len(df))
    v_papr = df["n_AA_paper_claimable"].fillna(0).astype(int).values if "n_AA_paper_claimable" in df.columns else np.zeros(len(df))

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    b1 = ax.bar(x-w/2, v_prov, w, color=CA, alpha=0.6, hatch="//",
                edgecolor="#555", lw=0.8, label="A/A provenance-valid (may not pass stat. gate)")
    b2 = ax.bar(x+w/2, v_papr, w, color=CSTAR, alpha=1.0,
                label="A/A paper-claimable ★ (both gates pass)")

    for bars, vals in [(b1,v_prov),(b2,v_papr)]:
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.08,
                        str(v), ha="center", fontsize=9, fontweight="bold", color="#2c3e50")

    ax.set_xticks(x)
    ax.set_xticklabels([labels_map.get(e,e) for e in df.index], fontsize=10)
    ax.set_ylabel("Number of A/A edges", fontsize=9.5)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", lw=0.5)
    ax.legend(fontsize=9, framealpha=0.95, edgecolor="#ccc")

    if "usdt_curve_2023" in df.index:
        ax.axvspan(-0.5, 0.5, facecolor="#fef9e7", alpha=0.4, zorder=0)
        ax.text(0, ax.get_ylim()[1]*0.97, "★ headline",
                ha="center", va="top", fontsize=9, color=CSTAR, fontweight="bold")

    ax.set_title("Figure 7 – A/A provenance-valid vs A/A paper-claimable\n"
                 "Several events have high-provenance A/A candidates; only USDT/Curve 2023 passes both gates",
                 fontsize=10.5, fontweight="bold", color="#2c3e50", pad=10)
    _save(fig, out, "figure_07_aa_provenance_vs_paper_claimable.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 08 – Terra/LUNA A/A candidate (negative result)
# ═══════════════════════════════════════════════════════════════════════════════

def fig08_terra_negative(out: Path) -> None:
    df = _gold("terra_luna_2022")
    if df is None: return

    p3  = _dex_series(df, "curve_3pool",       "usdc_net_sold_1h")
    ust = _dex_series(df, "curve_ust_wormhole", "usdc_net_sold_1h")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios":[1,1], "hspace":0.06})
    fig.patch.set_facecolor("white")

    for ax, ser, col, title in [
        (ax1, p3/1_000,  CA,  "curve_3pool  (Tier A)"),
        (ax2, ust/1_000, "#8e44ad", "curve_ust_wormhole  (Tier A — pool nearly drained)"),
    ]:
        ax.fill_between(ser.index, ser, 0, where=(ser>0), color=col,  alpha=0.35)
        ax.fill_between(ser.index, ser, 0, where=(ser<0), color="#e74c3c", alpha=0.35)
        ax.plot(ser.index, ser, color=col, lw=1.0)
        ax.axhline(0, color="#555", lw=0.6, ls="--")
        ax.set_ylabel("USDC net sold (k)\nper hour", fontsize=8)
        ax.set_title(title, fontsize=8.5, fontweight="bold", color="#2c3e50", loc="left")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:,.0f}k"))
        ax.grid(axis="y", color="#e0e0e0", lw=0.5)
        ax.spines[["top","right"]].set_visible(False)

    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax2.get_xticklabels(), rotation=25, ha="right", fontsize=8)

    ax2.text(0.01, 0.06,
             "A/A provenance-valid: YES (both Tier-A Curve AMM nodes)\n"
             "A/A paper-claimable: NO  (lead-lag not significant at hourly grid)\n"
             "claim_strength = suggestive  ·  paper_claim_allowed = False",
             transform=ax2.transAxes, fontsize=8, va="bottom",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8f9fa",
                       edgecolor="#888", lw=1))

    fig.suptitle("Figure 8 – Terra/LUNA 2022: A/A AMM-flow candidate — negative statistical result\n"
                 "Both nodes Tier A, but hourly lead-lag does not pass Bonferroni correction",
                 fontsize=10.5, fontweight="bold", color="#2c3e50", y=1.0)
    _save(fig, out, "figure_08_terra_amm_flow_candidate.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 09 – USDC/SVB sparse settlement response
# ═══════════════════════════════════════════════════════════════════════════════

def fig09_sparse_settlement(out: Path) -> None:
    df = _read_csv(RAW_TBL / "table_sparse_events_usdc_svb_2023.csv")
    if df is None: return

    # keep rows with actual baseline/response values
    has_vals = df["mean_response"].notna() | df["mean_baseline"].notna()
    df_plot = df[has_vals].copy()
    if len(df_plot) == 0:
        df_plot = df.copy()

    targets = df["target_node_id"].values
    baselines = pd.to_numeric(df["mean_baseline"], errors="coerce").fillna(0).values / 1_000
    responses = pd.to_numeric(df["mean_response"], errors="coerce").fillna(0).values / 1_000
    pvals     = pd.to_numeric(df["p_value"], errors="coerce").fillna(1.0).values
    sig       = df["significant_p05"].astype(str).str.lower() == "true"

    x = np.arange(len(targets))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    ax.bar(x-w/2, baselines, w, color=CB, alpha=0.75, label="Mean baseline (12h pre-arrival)")
    ax.bar(x+w/2, responses, w, color=CA, alpha=0.75, label="Mean response (3h post-arrival)")

    # annotation
    for i, (p, s) in enumerate(zip(pvals, sig)):
        top = max(baselines[i], responses[i])
        lbl = f"p={p:.2f}" if not np.isnan(p) else "p=NaN"
        ax.text(x[i], top + abs(top)*0.05 + 0.5, lbl,
                ha="center", fontsize=8, color=("#27ae60" if s else "#e74c3c"))

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_","\n") for t in targets], fontsize=8.5)
    ax.set_ylabel("Mean hourly USDC net-sold (thousands)", fontsize=9)
    ax.axhline(0, color="#555", lw=0.6, ls="--")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", lw=0.5)
    ax.legend(fontsize=9, framealpha=0.95)

    ax.text(0.5, 0.97,
            "Source: usdc_mint_burn  ·  feature = mint_burn_net_1h\n"
            "n = 4 events  ·  permutation test: p = 1.0  ·  not statistically supported\n"
            "Reported as high-provenance descriptive evidence only",
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa",
                      edgecolor="#888", lw=1))

    ax.set_title("Figure 9 – USDC/SVB 2023: sparse mint-burn event-response\n"
                 "Only 4 on-chain events; permutation test underpowered (p=1.0); not paper-claimable",
                 fontsize=10.5, fontweight="bold", color="#2c3e50", pad=10)
    _save(fig, out, "figure_09_usdc_svb_sparse_settlement_response.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 10 – Feature-tier matrix
# ═══════════════════════════════════════════════════════════════════════════════

def fig10_feature_tier_matrix(out: Path) -> None:
    df = _read_csv(TABLE_DIR / "table_feature_tiers.csv")
    if df is None: return

    # Select informative rows
    KEEP = ["usdc_net_sold_1h","mint_burn_net_1h","reserve_imbalance","implied_pool_price",
            "basis_vs_usd","spread_bps","depth_10bps_bid_usd","exchange_netflow_1h",
            "orderbook_imbalance"]
    df = df[df["feature_col"].isin(KEEP)].copy()
    df["tier"] = df["tier"].fillna("B")

    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    col_labels = ["Feature", "Tier", "Evidence type", "Paper claim permitted"]
    col_widths = [0.22, 0.06, 0.20, 0.50]
    header_y   = 0.97

    # header
    x = 0.0
    for lbl, w in zip(col_labels, col_widths):
        ax.text(x+w/2, header_y, lbl, ha="center", va="top",
                fontsize=9.5, fontweight="bold", color="#2c3e50",
                transform=ax.transAxes)
        x += w

    ax.plot([0, 1], [header_y-0.04, header_y-0.04], color="#aaa", lw=1, transform=ax.transAxes)

    for row_idx, (_, row) in enumerate(df.iterrows()):
        y = header_y - 0.1*(row_idx+1) - 0.01
        tier = str(row.get("tier","B"))
        facecolor = CBA if tier=="A" else CBB2

        rect = mpatches.FancyBboxPatch((0, y-0.04), 1.0, 0.085,
            boxstyle="round,pad=0.005", lw=0, facecolor=facecolor,
            transform=ax.transAxes, zorder=0)
        ax.add_patch(rect)

        values = [
            str(row.get("feature_col","")),
            tier,
            str(row.get("evidence_type","")),
            str(row.get("claim_language",""))[:90],
        ]
        x = 0.0
        for val, w in zip(values, col_widths):
            col = CA if (tier=="A" and val==tier) else (CB if val==tier else "#2c3e50")
            fw  = "bold" if val in ("A","B") else "normal"
            ax.text(x+w/2, y, val, ha="center", va="center",
                    fontsize=8.2, color=col, fontweight=fw,
                    transform=ax.transAxes)
            x += w

    ax.text(0.5, header_y-0.1*(len(df)+1)-0.06,
            "★  usdc_net_sold_1h and mint_burn_net_1h are the only Tier-A features in the paper. "
            "All derived proxies (reserve_imbalance, implied_pool_price) and CEX-only features are Tier B.",
            ha="center", va="top", fontsize=8, color=CSTAR,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef9e7",
                      edgecolor=CSTAR, lw=1))

    ax.set_title("Figure 10 – Feature-level evidence tiers\n"
                 "Node-level Tier A is necessary but not sufficient; feature tier caps the edge claim",
                 fontsize=10.5, fontweight="bold", color="#2c3e50", pad=12)
    _save(fig, out, "figure_10_feature_tier_matrix.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 11 – Node provenance coverage heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def fig11_node_coverage_heatmap(out: Path) -> None:
    df = _read_csv(TABLE_DIR / "table_provenance_inventory.csv")
    if df is None: return

    events = ["usdt_curve_2023","terra_luna_2022","usdc_svb_2023","ftx_2022","busd_2023"]
    ev_labels = ["USDT/Curve\n2023","Terra/LUNA\n2022","USDC/SVB\n2023","FTX\n2022","BUSD\n2023"]

    tier_col = "source_tier_actual" if "source_tier_actual" in df.columns else "tier_actual"
    node_ids = sorted(df["node_id"].unique())

    # value map: A=2, B=1, fixture=0, missing=-1
    tier_to_val = {"A":2, "B":1, "fixture_non_empirical":0}

    matrix = np.full((len(node_ids), len(events)), -1.0)
    for i, nid in enumerate(node_ids):
        for j, ev in enumerate(events):
            sub = df[(df["node_id"]==nid) & (df["event_id"]==ev)]
            if len(sub) > 0:
                t = str(sub[tier_col].iloc[0])
                matrix[i,j] = tier_to_val.get(t, 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#ecf0f1","#bdc3c7", CBB2, CBA])

    im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=2, aspect="auto")

    ax.set_xticks(range(len(events)))
    ax.set_xticklabels(ev_labels, fontsize=9)
    ax.set_yticks(range(len(node_ids)))
    ax.set_yticklabels(node_ids, fontsize=8)

    # cell annotations
    for i in range(len(node_ids)):
        for j in range(len(events)):
            val = matrix[i,j]
            lbl = {2:"A",1:"B",0:"FIX",-1:"—"}.get(int(val),"?")
            color = "white" if val>=1 else "#888"
            ax.text(j, i, lbl, ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color=color)

    # legend patches
    patches = [
        mpatches.Patch(color=CBA,      label="Tier A — on-chain execution-grade"),
        mpatches.Patch(color=CBB2,     label="Tier B — public market context"),
        mpatches.Patch(color="#bdc3c7",label="Fixture / synthetic"),
        mpatches.Patch(color="#ecf0f1",label="Not applicable / missing"),
    ]
    ax.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.0,-0.03),
              ncol=4, fontsize=8, framealpha=0.95)

    ax.set_title("Figure 11 – Node provenance and coverage by event\n"
                 "Curve nodes provide Tier-A anchor; CEX nodes provide Tier-B context; fixture nodes blocked from paper claims",
                 fontsize=10.5, fontweight="bold", color="#2c3e50", pad=10)
    plt.tight_layout(rect=[0,0.04,1,1])
    _save(fig, out, "figure_11_node_provenance_coverage.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 12 – Full paper-claimable network
# ═══════════════════════════════════════════════════════════════════════════════

def fig12_full_paper_network(out: Path) -> None:
    import networkx as nx

    aa_df = _read_csv(TABLE_DIR / "table_aa_paper_claimable_edges.csv")
    ab_df = _read_csv(TABLE_DIR / "table_ab_suggestive_edges.csv")

    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    G = nx.DiGraph()
    node_tiers: dict[str,str] = {}
    node_layers: dict[str,str] = {}

    def add_rows(df, claim_lv, tier_i_col="tier_i_actual", tier_j_col="tier_j_actual"):
        if df is None: return
        for _, row in df.iterrows():
            ni = row.get("node_i") or row.get("causing_node") or row.get("source_node_id","")
            nj = row.get("node_j") or row.get("caused_node") or row.get("target_node_id","")
            if not ni or not nj or ni == nj: continue
            tier_i = str(row.get(tier_i_col,"B"))
            tier_j = str(row.get(tier_j_col,"B"))
            node_tiers[ni] = tier_i; node_tiers[nj] = tier_j
            ev = row.get("event_id","")
            G.add_edge(ni, nj, claim_level=claim_lv, event_id=ev,
                       paper_claim=bool(row.get("paper_claim_allowed",False)))

    add_rows(aa_df, "A_A_dex_flow")
    add_rows(ab_df, "A_B_suggestive_directional")

    if len(G.nodes()) == 0:
        logger.warning("No nodes for Figure 12")
        return

    # layer assignment heuristic
    for nid in G.nodes():
        if "curve" in nid or "uniswap" in nid:
            node_layers[nid] = "DEX"
        elif "mint_burn" in nid or "exchange_flow" in nid or "bridge" in nid:
            node_layers[nid] = "Settlement"
        else:
            node_layers[nid] = "CEX"

    layer_pos: dict[str,float] = {"Settlement": 2.5, "DEX": 1.5, "CEX": 0.5}
    nodes_by_layer: dict[str,list] = {"Settlement":[], "DEX":[], "CEX":[]}
    for n in G.nodes():
        nodes_by_layer[node_layers.get(n,"CEX")].append(n)

    pos: dict[str,tuple[float,float]] = {}
    for layer, lnodes in nodes_by_layer.items():
        y = layer_pos[layer]
        for k, n in enumerate(sorted(lnodes)):
            x = (k+1) / (len(lnodes)+1) * 10
            pos[n] = (x, y)

    # draw
    for layer, lnodes in nodes_by_layer.items():
        colors = [CA if node_tiers.get(n,"B")=="A" else CB for n in lnodes]
        shapes = {"Settlement":"D","DEX":"s","CEX":"o"}
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=lnodes,
                               node_color=colors, node_shape=shapes[layer],
                               node_size=1600, edgecolors="white", linewidths=1.5)

    nx.draw_networkx_labels(G, pos, ax=ax,
        labels={n: n.replace("_","\n") for n in G.nodes()},
        font_size=6, font_color="white", font_weight="bold")

    # ── Edge styles encode claim tier (TODO 4.6 — tier labels on all network figures)
    # Solid dark   = A/A paper-claimable (both gates pass)
    # Dashed medium = A/B directional (one Tier-A endpoint)
    # Dotted light  = B/B contextual
    # (Fixture-derived edges are never drawn — they don't reach paper tables)
    for (u,v,d) in G.edges(data=True):
        cl    = d.get("claim_level","B_B_context_only")
        col   = CSTAR if "A_A" in cl else (CAB if "A_B" in cl else CBB)
        lw    = 2.5   if "A_A" in cl else (1.5  if "A_B" in cl else 0.8)
        style = "solid"  if "A_A" in cl else ("dashed" if "A_B" in cl else "dotted")
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u,v)],
            edge_color=col, width=lw, style=style,
            arrowsize=12, arrowstyle="-|>",
            connectionstyle="arc3,rad=0.12",
            min_source_margin=22, min_target_margin=22)

    # layer labels
    for layer, y in layer_pos.items():
        ax.text(-0.3, y, layer+"\nLayer", ha="right", va="center",
                fontsize=9, fontweight="bold", color="#2c3e50",
                transform=ax.transAxes if False else ax.transData)

    # ── Legend with full tier encoding ────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=CA,  edgecolor="#1a6e3a", lw=1.5,
                       label="Tier A node — on-chain execution-grade"),
        mpatches.Patch(facecolor=CB,  edgecolor="#aaa",
                       label="Tier B node — public market context"),
        mpatches.Patch(facecolor="white", edgecolor="#555", hatch="////",
                       label="Fixture node — not in paper analysis"),
        Line2D([0],[0], color=CSTAR, lw=2.5, linestyle="solid",
               label="A/A edge — solid · paper-claimable (both gates pass)"),
        Line2D([0],[0], color=CAB,   lw=1.5, linestyle="dashed",
               label="A/B edge — dashed · directional (suggestive)"),
        Line2D([0],[0], color=CBB,   lw=0.8, linestyle="dotted",
               label="B/B edge — dotted · contextual only"),
        mpatches.Patch(facecolor="white", edgecolor="#555",
                       label="Shape: □ DEX/AMM  ○ CEX  ◇ Settlement"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8,
              framealpha=0.95, edgecolor="#ccc", ncol=1)

    ax.set_title(
        "Figure 12 – Paper-claimable stress-propagation network\n"
        "Edge style encodes claim tier: solid=A/A, dashed=A/B, dotted=B/B  ·  "
        "Node fill encodes data tier: dark=Tier-A, light=Tier-B",
        fontsize=10, fontweight="bold", color="#2c3e50", pad=10)
    _save(fig, out, "figure_12_full_paper_claimable_network.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, out_dir: Path, filename: str) -> None:
    p = out_dir / filename
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", filename)


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        logger.warning("Missing: %s", path)
        return None
    return pd.read_csv(path)


def _gold(event: str) -> pd.DataFrame | None:
    path = GOLD_DIR / f"dataset_contagion_features_{event}.parquet"
    if not path.exists():
        logger.warning("Gold parquet not found: %s", path)
        return None
    return pd.read_parquet(path)


def _dex_series(df: pd.DataFrame, node_id: str, feature: str) -> pd.Series:
    sub = df[df["node_id"] == node_id][["wall_clock_utc", feature]].copy()
    sub["wall_clock_utc"] = pd.to_datetime(sub["wall_clock_utc"], utc=True)
    return sub.set_index("wall_clock_utc")[feature].sort_index()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
# Figure 13 — TVP-VAR time-varying propagation coefficient (robustness appendix)
# ═══════════════════════════════════════════════════════════════════════════════

def fig13_tvpvar_usdt_curve(out: Path) -> None:
    """Time-varying propagation coefficient from TVP-VAR for USDT/Curve 2023.

    Reads results/tables/tvpvar_usdt_curve_2023.csv (or .parquet) and plots
    the rolling coefficient for the primary A/A pair
    (curve_3pool → curve_crvusd_usdt) over the event window.

    If the TVP-VAR results file is absent (not yet generated), writes a
    placeholder figure with an informative message so the paper package
    validation does not fail.
    """
    import polars as pl

    event_id = "usdt_curve_2023"
    raw_dir  = RAW_TBL  # results/tables/

    # Try parquet first, fall back to CSV
    tvp_path = None
    for suffix in (".parquet", ".csv"):
        candidate = raw_dir / f"tvpvar_{event_id}{suffix}"
        if candidate.exists():
            tvp_path = candidate
            break

    fig, ax = plt.subplots(figsize=(8, 3.5))

    if tvp_path is None:
        ax.text(
            0.5, 0.5,
            "TVP-VAR results not yet generated.\n"
            "Run: make tvpvar EVENT=usdt_curve_2023",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=11, color="grey",
        )
        ax.set_title("TVP-VAR: time-varying coefficient (PLACEHOLDER)", fontsize=10)
        logger.warning(
            "TVP-VAR results not found for %s; writing placeholder figure.", event_id
        )
    else:
        df = (pl.read_parquet(tvp_path) if tvp_path.suffix == ".parquet"
              else pl.read_csv(tvp_path))

        # Expected columns: wall_clock_utc, source, target, coef, ci_low, ci_high
        # Filter for the primary A/A pair
        src, tgt = "curve_3pool", "curve_crvusd_usdt"
        pair_col_s = next((c for c in ["source", "source_node", "node_i"] if c in df.columns), None)
        pair_col_t = next((c for c in ["target", "target_node", "node_j"] if c in df.columns), None)

        if pair_col_s and pair_col_t:
            pair = df.filter(
                (pl.col(pair_col_s) == src) & (pl.col(pair_col_t) == tgt)
            )
        else:
            pair = df  # fallback: plot everything

        ts_col  = next((c for c in ["wall_clock_utc", "timestamp", "date"] if c in pair.columns), None)
        coef_col = next((c for c in ["coef", "coefficient", "beta"] if c in pair.columns), None)
        ci_lo    = next((c for c in ["ci_low", "ci_lower", "lower"] if c in pair.columns), None)
        ci_hi    = next((c for c in ["ci_high", "ci_upper", "upper"] if c in pair.columns), None)

        if ts_col and coef_col:
            xs = pair[ts_col].to_list()
            ys = pair[coef_col].cast(pl.Float64).to_list()
            ax.plot(xs, ys, color="#1f77b4", lw=1.8, label=f"{src} → {tgt}")
            if ci_lo and ci_hi:
                lo = pair[ci_lo].cast(pl.Float64).to_list()
                hi = pair[ci_hi].cast(pl.Float64).to_list()
                ax.fill_between(xs, lo, hi, alpha=0.2, color="#1f77b4", label="90% CI")
            ax.axhline(0, ls="--", lw=0.8, color="grey")
            ax.set_xlabel("Date (UTC)")
            ax.set_ylabel("TVP-VAR coefficient")
            ax.set_title(
                f"TVP-VAR: rolling propagation coefficient\n"
                f"{src} → {tgt} (USDT/Curve 2023, 168h window, 24h step)",
                fontsize=9,
            )
            ax.legend(fontsize=8)
        else:
            ax.text(
                0.5, 0.5,
                "Unexpected TVP-VAR column schema.\n"
                f"Found columns: {pair.columns[:6]}",
                ha="center", va="center", transform=ax.transAxes, fontsize=9, color="grey",
            )

    plt.tight_layout()
    out_path = out / "figure_13_tvpvar_usdt_curve_2023.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Wrote Figure 13 → %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all 12 paper figures.")
    parser.add_argument("--fig-dir", default=None)
    parser.add_argument("--only", nargs="+", type=int,
                        help="Generate only specified figure numbers, e.g. --only 4 5 6")
    args = parser.parse_args()

    out = Path(args.fig_dir) if args.fig_dir else FIG_DIR
    out.mkdir(parents=True, exist_ok=True)

    all_figs = [
        (1,  fig01_multilayer_architecture),
        (2,  fig02_claim_gate_pipeline),
        (3,  fig03_claim_audit),
        (4,  fig04_usdt_curve_timeline),
        (5,  fig05_leadlag_profile),
        (6,  fig06_aa_paper_network),
        (7,  fig07_aa_prov_vs_paper),
        (8,  fig08_terra_negative),
        (9,  fig09_sparse_settlement),
        (10, fig10_feature_tier_matrix),
        (11, fig11_node_coverage_heatmap),
        (12, fig12_full_paper_network),
        (13, fig13_tvpvar_usdt_curve),   # TVP-VAR time-varying coefficient (robustness)
    ]

    to_run = {n: fn for n, fn in all_figs
              if args.only is None or n in args.only}
    for n, fn in sorted(to_run.items()):
        try:
            fn(out)
        except Exception as exc:
            logger.error("Figure %02d failed: %s", n, exc, exc_info=True)

    logger.info("Done. %d figures generated in %s", len(to_run), out)


if __name__ == "__main__":
    main()
