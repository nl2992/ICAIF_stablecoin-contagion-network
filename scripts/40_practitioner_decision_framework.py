"""Practitioner Decision Framework: When to use on-chain vs market data for DeFi stress.
Generates a comprehensive visual showing:
  - Event mechanism classification (endogenous vs exogenous)
  - Best evidence layer per mechanism
  - Detection latency and AUROC per layer
  - Recommended monitoring architecture

This figure bridges research findings to operational surveillance decisions.
Writes results/paper/figures/fig_decision_framework.{pdf,png}
"""
from __future__ import annotations
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

FIG = "results/paper/figures"

plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

C_OC = "#27AE60"      # on-chain green
C_MK = "#5B8DEF"      # market blue
C_ENDOGENOUS = "#E67E22"  # orange
C_EXOGENOUS = "#3498DB"   # light blue
C_HIGHLIGHT = "#E74C3C"   # red for key findings

def main():
    fig = plt.figure(figsize=(7.5, 5.2))

    # Create grid layout: left side (decision tree), right side (results table)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1.0], height_ratios=[1, 1.2, 0.8],
                          hspace=0.35, wspace=0.30)

    # =========================================================================
    # TITLE & MECHANISM CLASSIFIER (Top)
    # =========================================================================
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.7, 'DeFi Stress Surveillance: Mechanism-Based Detector Selection',
                  ha='center', va='center', fontsize=10.5, weight='bold',
                  transform=ax_title.transAxes)
    ax_title.text(0.5, 0.2,
                  'Choose on-chain (A) or market (B) data based on shock origin; both needed for complete coverage',
                  ha='center', va='center', fontsize=7.5, style='italic',
                  transform=ax_title.transAxes, color='#555')

    # =========================================================================
    # LEFT: DECISION TREE / FLOWCHART
    # =========================================================================
    ax_tree = fig.add_subplot(gs[1:, 0])
    ax_tree.set_xlim(0, 10)
    ax_tree.set_ylim(0, 10)
    ax_tree.axis('off')

    # Root: Event type classifier
    root_box = FancyBboxPatch((1, 7.5), 8, 1.2, boxstyle="round,pad=0.1",
                              edgecolor='#333', facecolor='#F0F0F0', linewidth=1.5)
    ax_tree.add_patch(root_box)
    ax_tree.text(5, 8.1, 'SHOCK ORIGIN', ha='center', va='center', fontsize=8, weight='bold')

    # Left branch: Endogenous
    endo_box = FancyBboxPatch((0.3, 5), 3.5, 1.0, boxstyle="round,pad=0.08",
                              edgecolor=C_ENDOGENOUS, facecolor=C_ENDOGENOUS, alpha=0.15, linewidth=1.2)
    ax_tree.add_patch(endo_box)
    ax_tree.text(2.05, 5.5, 'ENDOGENOUS\n(Pool imbalance)',
                ha='center', va='center', fontsize=7.5, weight='bold')
    ax_tree.arrow(3.5, 7.5, -0.5, -1.3, head_width=0.25, head_length=0.15, fc='#666', ec='#666')

    # Right branch: Exogenous
    exo_box = FancyBboxPatch((6.2, 5), 3.5, 1.0, boxstyle="round,pad=0.08",
                             edgecolor=C_EXOGENOUS, facecolor=C_EXOGENOUS, alpha=0.15, linewidth=1.2)
    ax_tree.add_patch(exo_box)
    ax_tree.text(7.95, 5.5, 'EXOGENOUS\n(CEX, regulatory, banking)',
                ha='center', va='center', fontsize=7.5, weight='bold')
    ax_tree.arrow(6.5, 7.5, 0.5, -1.3, head_width=0.25, head_length=0.15, fc='#666', ec='#666')

    # Endogenous → On-chain detector
    oc_rec = Rectangle((0.1, 2.8), 4.0, 1.3, edgecolor=C_OC, facecolor=C_OC, alpha=0.2, linewidth=1.5)
    ax_tree.add_patch(oc_rec)
    ax_tree.text(2.1, 3.7, 'USE: On-Chain Detector (A)', ha='center', va='center',
                fontsize=8, weight='bold', color=C_OC)
    ax_tree.text(2.1, 3.2, 'HMM on pool deviation,\nflow, imbalance',
                ha='center', va='center', fontsize=6.5)
    ax_tree.arrow(2.05, 5, -0.1, -1.0, head_width=0.2, head_length=0.12, fc=C_OC, ec=C_OC, linewidth=1.5)

    # Exogenous → Market detector
    mk_rec = Rectangle((5.9, 2.8), 4.0, 1.3, edgecolor=C_MK, facecolor=C_MK, alpha=0.2, linewidth=1.5)
    ax_tree.add_patch(mk_rec)
    ax_tree.text(7.9, 3.7, 'USE: Market Detector (B)', ha='center', va='center',
                fontsize=8, weight='bold', color=C_MK)
    ax_tree.text(7.9, 3.2, 'HMM on basis, spread,\norder-book imbalance',
                ha='center', va='center', fontsize=6.5)
    ax_tree.arrow(7.95, 5, 0.1, -1.0, head_width=0.2, head_length=0.12, fc=C_MK, ec=C_MK, linewidth=1.5)

    # Bottom: Complementarity note
    combo_box = FancyBboxPatch((0.5, 0.2), 9, 1.8, boxstyle="round,pad=0.1",
                               edgecolor='#555', facecolor='#FFF9E6', linewidth=1.2, linestyle='--')
    ax_tree.add_patch(combo_box)
    ax_tree.text(5, 1.6, '⚠ CRITICAL: Use BOTH layers', ha='center', va='center',
                fontsize=7.5, weight='bold', color='#C0392B')
    ax_tree.text(5, 0.9,
                'Supervised cross-event prediction fails (0.50 AUROC); unsupervised per-event detection succeeds.\n' +
                'Each layer is blind to the shock type it cannot observe.',
                ha='center', va='center', fontsize=6.5, style='italic', color='#555')

    # =========================================================================
    # RIGHT: PERFORMANCE MATRIX
    # =========================================================================
    ax_perf = fig.add_subplot(gs[1, 1])
    ax_perf.axis('off')

    events = ['Terra', 'USDT/\nCurve', 'USDC/\nSVB', 'FTX', 'BUSD']
    mechanisms = ['Algo.', 'Pool Imb.', 'Bank', 'CEX', 'Reg.']
    on_chain_auroc = [0.954, 0.934, 0.881, 0.401, 0.609]
    market_auroc = [0.499, 0.937, 0.909, 0.868, 0.887]
    winner = ['A', 'Tie', 'Tie', 'B', 'B']

    # Draw header
    header_y = 0.95
    ax_perf.text(0.05, header_y, 'EVENT', fontsize=7, weight='bold', transform=ax_perf.transAxes)
    ax_perf.text(0.35, header_y, 'On-Chain', fontsize=7, weight='bold', color=C_OC, transform=ax_perf.transAxes)
    ax_perf.text(0.60, header_y, 'Market', fontsize=7, weight='bold', color=C_MK, transform=ax_perf.transAxes)
    ax_perf.text(0.82, header_y, 'Winner', fontsize=7, weight='bold', transform=ax_perf.transAxes)

    # Draw rows
    y_step = 0.17
    for i, (evt, mech, oc, mk, w) in enumerate(zip(events, mechanisms, on_chain_auroc, market_auroc, winner)):
        y_pos = header_y - 0.12 - i * y_step

        # Event name + mechanism
        ax_perf.text(0.02, y_pos, f'{evt}\n({mech})', fontsize=6.5, transform=ax_perf.transAxes, va='center')

        # On-chain AUROC (highlight if winner)
        oc_color = C_OC if w in ['A', 'Tie'] else '#CCC'
        ax_perf.text(0.35, y_pos, f'{oc:.2f}', fontsize=6.5, color=oc_color, weight='bold' if w == 'A' else 'normal',
                    transform=ax_perf.transAxes, va='center')

        # Market AUROC (highlight if winner)
        mk_color = C_MK if w in ['B', 'Tie'] else '#CCC'
        ax_perf.text(0.60, y_pos, f'{mk:.2f}', fontsize=6.5, color=mk_color, weight='bold' if w == 'B' else 'normal',
                    transform=ax_perf.transAxes, va='center')

        # Winner
        if w == 'A':
            ax_perf.text(0.82, y_pos, '🟢 A', fontsize=7, weight='bold', color=C_OC, transform=ax_perf.transAxes, va='center')
        elif w == 'B':
            ax_perf.text(0.82, y_pos, '🔵 B', fontsize=7, weight='bold', color=C_MK, transform=ax_perf.transAxes, va='center')
        else:
            ax_perf.text(0.82, y_pos, '⚖ Both', fontsize=6, weight='bold', color='#999', transform=ax_perf.transAxes, va='center')

    # =========================================================================
    # BOTTOM: LATENCY & ACTIONABILITY
    # =========================================================================
    ax_action = fig.add_subplot(gs[2, :])
    ax_action.axis('off')

    # Latency comparison for Terra (headline result)
    ax_action.text(0.5, 0.9, 'Terra/LUNA Case Study: Time-to-Detection Advantage',
                  ha='center', va='top', fontsize=8.5, weight='bold', transform=ax_action.transAxes)

    # Timeline visualization
    timeline_y = 0.6
    ax_action.plot([0.1, 0.9], [timeline_y, timeline_y], 'k-', linewidth=2, transform=ax_action.transAxes)

    # On-chain detection (earlier)
    oc_time = 0.25
    ax_action.scatter([oc_time], [timeline_y], s=150, color=C_OC, marker='o', zorder=10, transform=ax_action.transAxes)
    ax_action.text(oc_time, timeline_y + 0.15, 'On-Chain\nAlarm Fires\nAUROC 0.954',
                  ha='center', fontsize=6.5, weight='bold', color=C_OC, transform=ax_action.transAxes)

    # Market detection (later)
    mk_time = 0.75
    ax_action.scatter([mk_time], [timeline_y], s=150, color=C_MK, marker='s', zorder=10, transform=ax_action.transAxes)
    ax_action.text(mk_time, timeline_y + 0.15, 'Market\nAlarm Fires\nAUROC 0.499',
                  ha='center', fontsize=6.5, weight='bold', color=C_MK, transform=ax_action.transAxes)

    # Arrow showing latency advantage
    ax_action.annotate('', xy=(mk_time - 0.05, timeline_y - 0.15), xytext=(oc_time + 0.05, timeline_y - 0.15),
                      arrowprops=dict(arrowstyle='<->', color=C_HIGHLIGHT, lw=2),
                      transform=ax_action.transAxes)
    ax_action.text(0.5, timeline_y - 0.25, '116 hours earlier detection\nwith on-chain data',
                  ha='center', fontsize=7, weight='bold', color=C_HIGHLIGHT, transform=ax_action.transAxes,
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE6E6', edgecolor=C_HIGHLIGHT, linewidth=1))

    plt.savefig(f'{FIG}/fig_decision_framework.pdf', dpi=150)
    plt.savefig(f'{FIG}/fig_decision_framework.png', dpi=150)
    plt.close(fig)
    print("[OK] Wrote fig_decision_framework.{pdf,png}")


if __name__ == "__main__":
    main()
