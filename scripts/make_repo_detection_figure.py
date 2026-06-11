#!/usr/bin/env python3
"""Repo storytelling figure (not in the 8pp paper): mechanism-specific stress
detection across all SEVEN episodes spanning 2022-2025.

The paper's core analysis is the five 2022-23 episodes; this figure adds the two
2024-25 out-of-period validation episodes (USDT/Curve Aug-2024, ByBit-2025) to
show the endogenous/exogenous detection boundary holds across three years and a
changing market structure.

Data sources (committed):
  results/tables/table_online_detection.csv            (5 core, causal AUROC)
  results/tables/table_2024_episodes_detection.json    (ByBit 2025)
  + USDT/Curve Aug-2024 (on-chain 0.807 / market 0.377), from
    scripts/fetch_run_2024_episodes.py --episodes usdt_curve_2024_aug

Output: docs/figures/detection_across_episodes.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Columbia palette (matches paper figures) ──────────────────────────────────
NAVY = "#1D4F91"   # on-chain (Tier-A)
MID = "#6CA6CD"    # market (Tier-B)
INK = "#0A1F44"
COLBLUE = "#B9D9EB"
GREEN = "#27AE60"
AMBER = "#E08E2B"
LGREY = "#D5DEE8"
GREY = "#8895A7"

plt.rcParams.update({
    "font.family": "serif",
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": INK, "savefig.dpi": 200, "savefig.bbox": "tight",
})

# ── Episode data: (label, year-tag, on-chain AUROC, market AUROC, mechanism) ──
# Grouped: pool-borne (endogenous, fires on-chain) then exogenous (fires market).
POOL = [
    ("Terra / LUNA",        "May 2022", 0.954, 0.499, "algorithmic"),
    ("USDT / Curve",        "Jun 2023", 0.934, 0.937, "DeFi-native"),
    ("USDT / Curve",        "Aug 2024", 0.807, 0.377, "DeFi-native · carry-crash"),
    ("USDC / SVB",          "Mar 2023", 0.881, 0.909, "fiat bank run"),
]
EXO = [
    ("BUSD wind-down",      "Feb 2023", 0.609, 0.887, "regulatory"),
    ("FTX collapse",        "Nov 2022", 0.401, 0.868, "exchange credit"),
    ("ByBit hack",          "Feb 2025", 0.602, 0.660, "exchange-credit hack"),
]
OUT_OF_PERIOD = {"Aug 2024", "Feb 2025"}

fig, ax = plt.subplots(figsize=(9.2, 5.6))

rows = []  # (y, label, tag, onchain, market, mech, group)
y = 0
for grp, items in (("exo", EXO[::-1]), ("pool", POOL[::-1])):
    for (lab, tag, oc, mk, mech) in items:
        rows.append((y, lab, tag, oc, mk, mech, grp))
        y += 1
    y += 0.9  # gap between groups

for (yy, lab, tag, oc, mk, mech, grp) in rows:
    onchain_wins = oc >= mk
    win_col = NAVY if onchain_wins else MID
    # connecting bar
    ax.plot([mk, oc], [yy, yy], color=win_col, lw=2.4, alpha=0.45, zorder=1,
            solid_capstyle="round")
    # markers
    ax.scatter(mk, yy, s=110, color=MID, edgecolor=INK, lw=0.6, zorder=3)
    ax.scatter(oc, yy, s=140, color=NAVY, edgecolor=INK, lw=0.6, zorder=3,
               marker="D")
    # episode label (left)
    star = "  †" if tag in OUT_OF_PERIOD else ""
    ax.text(-0.02, yy, f"{lab}", ha="right", va="center", fontsize=10.5,
            color=INK, transform=ax.get_yaxis_transform(), fontweight="bold")
    ax.text(-0.02, yy - 0.34, f"{tag}{star}  ·  {mech}", ha="right",
            va="center", fontsize=7.6, color=GREY,
            transform=ax.get_yaxis_transform())
    # winning-AUROC annotation
    xw = max(oc, mk)
    ax.text(xw + 0.012, yy, f"{xw:.2f}", ha="left", va="center", fontsize=8.6,
            color=win_col, fontweight="bold")

# chance line
ax.axvline(0.5, color=GREY, ls=":", lw=1.1, zorder=0)
ax.text(0.5, max(r[0] for r in rows) + 1.15, "chance", ha="center", va="bottom",
        fontsize=8, color=GREY, style="italic")

# group bands / dividers
pool_ys = [r[0] for r in rows if r[6] == "pool"]
exo_ys = [r[0] for r in rows if r[6] == "exo"]
ax.text(1.005, sum(pool_ys) / len(pool_ys), "POOL-BORNE\n(endogenous)\n→ on-chain detects",
        transform=ax.get_yaxis_transform(), ha="left", va="center",
        fontsize=9, color=NAVY, fontweight="bold")
ax.text(1.005, sum(exo_ys) / len(exo_ys), "EXOGENOUS\n(exchange / regulatory)\n→ market detects",
        transform=ax.get_yaxis_transform(), ha="left", va="center",
        fontsize=9, color=MID, fontweight="bold")

ax.set_xlim(0.30, 1.0)
ax.set_ylim(-0.8, max(r[0] for r in rows) + 1.0)
ax.set_yticks([])
ax.set_xlabel("Causal (filtered-posterior) detection AUROC", fontsize=11)
ax.set_title("Mechanism-specific stress detection across seven episodes (2022–2025)",
             fontsize=13.5, fontweight="bold", color=INK, pad=26)
ax.text(0.5, 1.045,
        "One unsupervised 3-state HMM, run identically on Tier-A on-chain vs. Tier-B market state. "
        "† = 2024–25 out-of-period episodes (held out of the core study).",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=8.6, color=GREY)

legend = [
    Line2D([0], [0], marker="D", color="w", markerfacecolor=NAVY,
           markeredgecolor=INK, markersize=11, label="On-chain HMM (Tier-A)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=MID,
           markeredgecolor=INK, markersize=11, label="Market HMM (Tier-B)"),
]
ax.legend(handles=legend, loc="lower right", frameon=True, framealpha=0.95,
          edgecolor=LGREY, fontsize=9)

for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.tick_params(left=False)

out = OUT / "detection_across_episodes.png"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
