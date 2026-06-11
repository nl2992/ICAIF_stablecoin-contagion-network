#!/usr/bin/env python3
"""Rolling cross-pool coupling timeline for the USDT/Curve 2023 episode.

Visualises the time profile that the Forbes--Rigobon scalar contrast summarises:
the lag-0 coupling between the two Tier-A Curve A/A pools (3pool and crvUSD/USDT)
is flat near zero through the calm regime, rises sharply at onset, sustains through
the acute window, then partially relaxes. Computed as a 36-hour rolling Pearson
correlation of the hourly ``usdc_net_sold_1h'' flow, the same feature and pair used
for the scalar Forbes--Rigobon test (calm rho=0.09, acute rho=0.53).

No new experiment: re-plots the committed gold flow panel.
Output: results/paper/figures/figure_coupling_rho_timeline.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).parent.parent
GOLD = REPO / "data" / "gold" / "dataset_contagion_features_usdt_curve_2023.parquet"
OUT = REPO / "results" / "paper" / "figures" / "figure_coupling_rho_timeline.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

NAVY, CALM, PANIC = "#003057", "#5B8DEF", "#E0533D"
WINDOW = 36  # hours; acute-window rolling mean ~0.52 matches the FR scalar 0.53

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "axes.spines.top": False, "axes.spines.right": False})

df = pd.read_parquet(GOLD)
pair = ["curve_3pool", "curve_crvusd_usdt"]
sub = df[df.node_id.isin(pair)]
wide = sub.pivot_table(index="wall_clock_utc", columns="node_id",
                       values="usdc_net_sold_1h").sort_index()
# hours since onset (event_time_seconds == 0 at onset)
t = (df.drop_duplicates("wall_clock_utc").set_index("wall_clock_utc")
     ["event_time_seconds"].reindex(wide.index) / 3600.0)
rho = wide[pair[0]].rolling(WINDOW, min_periods=WINDOW // 3).corr(wide[pair[1]])

fig, ax = plt.subplots(figsize=(3.34, 2.15))
# regime shading
ax.axvspan(t.min(), 0, color=CALM, alpha=0.10)
ax.axvspan(0, t.max(), color=PANIC, alpha=0.10)
ax.plot(t.values, rho.values, color=NAVY, lw=1.3, zorder=3)
# Forbes--Rigobon scalar references
ax.axhline(0.0905, xmin=0, xmax=0.5, color=CALM, lw=1.0, ls=":")
ax.axhline(0.5273, color=PANIC, lw=1.0, ls=":")
ax.axvline(0, color="#555", lw=0.8, ls="--")
ax.text(t.min() + 2, -0.78, "calm", color=CALM, fontsize=7.5, fontweight="bold")
ax.text(4, -0.78, "acute", color=PANIC, fontsize=7.5, fontweight="bold")
ax.text(2, 0.55, r"FR $\hat{\rho}=0.53$", color=PANIC, fontsize=6.8, va="bottom")
ax.text(t.min() + 2, 0.12, r"FR $\hat{\rho}=0.09$", color=CALM, fontsize=6.8, va="bottom")
ax.annotate("onset", (0, -0.95), fontsize=6.8, color="#555", ha="center")

ax.set_xlabel("hours since shock onset", fontsize=8)
ax.set_ylabel(r"rolling coupling $\hat{\rho}(\tau)$", fontsize=8)
ax.set_ylim(-1.0, 1.0)
ax.tick_params(labelsize=7)
fig.tight_layout(pad=0.3)
fig.savefig(OUT, dpi=240, bbox_inches="tight")
print(f"Saved {OUT}")
