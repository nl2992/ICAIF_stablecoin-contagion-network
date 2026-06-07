"""Figure 1: schematic of the three-layer node taxonomy and the three-gate
provenance pipeline.  Pure-schematic (no data); matches house style.
Writes results/paper/figures/fig_method.{pdf,png}.
"""
from __future__ import annotations
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from stressnet.config import results_root
from stressnet.utils.logging import get_logger
logger = get_logger(__name__)

CA, CB, CFX = "#27AE60", "#B9D9EB", "#E0533D"   # Tier-A, Tier-B, blocked/fixture
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5})

def box(ax, x, y, w, h, text, fc, ec="#333", fs=7.5, lw=1.0, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=tc)

def arrow(ax, x1, y1, x2, y2, lw=1.2, color="#444", style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=10, lw=lw, color=color))

def main():
    fig, ax = plt.subplots(figsize=(7.0, 2.35))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.4); ax.axis("off")

    # ---- (a) three-layer node taxonomy (left) ----
    ax.text(2.0, 3.25, r"(a) Three-layer evidence model", ha="center", fontsize=8.5, weight="bold")
    box(ax, 0.3, 2.30, 3.4, 0.55, "CEX layer (Binance, Coinbase)\nTier B", CB, fs=7)
    box(ax, 0.3, 1.45, 3.4, 0.55, "AMM/DEX layer (Curve pools)\nTier A", CA, fs=7, tc="white")
    box(ax, 0.3, 0.60, 3.4, 0.55, "Settlement layer (mint / burn)\nTier A", CA, fs=7, tc="white")
    # inter-layer edges
    arrow(ax, 2.0, 2.30, 2.0, 2.00, color="#888")
    arrow(ax, 2.0, 1.45, 2.0, 1.15, color="#888")
    ax.text(3.95, 1.72, "edge tier =\n$\\min$(node, node,\nfeature)", fontsize=6.2, va="center", color="#333")

    # ---- (b) three-gate pipeline (right) ----
    ax.text(7.4, 3.25, r"(b) Provenance-gated claim pipeline", ha="center", fontsize=8.5, weight="bold")
    gx = 5.6
    box(ax, gx, 2.55, 1.25, 0.55, "Gate 1\nProvenance", "#F3F6FA")
    box(ax, gx+1.55, 2.55, 1.25, 0.55, "Gate 2\nStatistical", "#F3F6FA")
    box(ax, gx+3.10, 2.55, 1.25, 0.55, "Gate 3\nPaper claim", "#F3F6FA")
    arrow(ax, gx+1.25, 2.82, gx+1.55, 2.82)
    arrow(ax, gx+2.80, 2.82, gx+3.10, 2.82)
    # funnel counts
    box(ax, gx, 1.55, 1.25, 0.55, "candidate\nedges", CB, fs=6.8)
    box(ax, gx+1.55, 1.55, 1.25, 0.55, "provenance-\nvalid (A/A)", CA, fs=6.5, tc="white")
    box(ax, gx+3.10, 1.55, 1.25, 0.55, "paper-\nclaimable", CA, fs=6.8, tc="white")
    arrow(ax, gx+1.25, 1.82, gx+1.55, 1.82)
    arrow(ax, gx+2.80, 1.82, gx+3.10, 1.82)
    for cx in (gx+0.625, gx+2.175, gx+3.725):
        arrow(ax, cx, 2.55, cx, 2.10, color="#999", lw=0.9)
    # blocked branch
    box(ax, gx+0.30, 0.55, 2.0, 0.5, "fixture / Tier-B  →  blocked", CFX, fs=6.8, tc="white")
    arrow(ax, gx+0.95, 1.55, gx+1.0, 1.05, color=CFX, lw=1.0, style="-|>")

    # legend
    ax.text(0.3, 0.12, r"Tier A (execution-grade on-chain)", fontsize=6, color=CA)
    ax.text(4.3, 0.12, r"Tier B (market proxy)", fontsize=6, color="#3A6B8F")
    ax.text(7.1, 0.12, r"blocked (fixture / capped)", fontsize=6, color=CFX)

    fig.tight_layout()
    out = results_root()/"paper"/"figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/"fig_method.pdf", bbox_inches="tight")
    fig.savefig(out/"fig_method.png", bbox_inches="tight", dpi=150)
    plt.close(fig); logger.info("wrote fig_method")

if __name__ == "__main__":
    main()
