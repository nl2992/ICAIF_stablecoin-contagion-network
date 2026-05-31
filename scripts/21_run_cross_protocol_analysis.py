"""Cross-protocol lead-lag: Curve pools ↔ Uniswap v3 pools.

Compares whether stress propagates within a single protocol (Curve↔Curve)
or across protocols (Curve↔Uniswap).  Both node types are Tier A (on-chain
Swap event logs), so any significant cross-protocol edge is A/A paper-claimable.

This directly addresses the reviewer observation that the paper needs "cross-
protocol comparison" and "depth of results beyond one ρ̂ = 0.386."

Output
------
    results/paper/tables/table_cross_protocol_leadlag_{event}.csv
    results/paper/figures_cross_protocol/A_cross_protocol_network.png

Usage
-----
    python scripts/21_run_cross_protocol_analysis.py --event usdt_curve_2023
    python scripts/21_run_cross_protocol_analysis.py --all-events
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

from stressnet.config import bronze_root, results_root
from stressnet.models.leadlag import compute_leadlag_table
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

# Columbia palette
CNV  = "#003865"
CTA  = "#27AE60"
CTB  = "#7F8C8D"
CAMB = "#E67E22"
CBLU = "#2980B9"
CWH  = "#FFFFFF"
CBKG = "#F8FBFD"
CSL  = "#2C3E50"

# Cross-protocol node pairs per event:
# (source_node_id, target_node_id, protocol_src, protocol_tgt)
_CROSS_PROTOCOL_PAIRS: dict[str, list[tuple[str, str, str, str]]] = {
    "usdt_curve_2023": [
        # Within-Curve (headline baseline)
        ("curve_3pool",           "curve_crvusd_usdt",       "Curve", "Curve"),
        ("curve_crvusd_usdt",     "curve_3pool",             "Curve", "Curve"),
        # Cross-protocol
        ("curve_3pool",           "uniswap_usdc_usdt_005",   "Curve", "Uniswap"),
        ("uniswap_usdc_usdt_005", "curve_3pool",             "Uniswap", "Curve"),
        ("curve_crvusd_usdt",     "uniswap_usdc_usdt_005",   "Curve", "Uniswap"),
        ("uniswap_usdc_usdt_005", "curve_crvusd_usdt",       "Uniswap", "Curve"),
    ],
    "usdc_svb_2023": [
        ("curve_3pool",           "uniswap_usdc_usdt_005",   "Curve", "Uniswap"),
        ("uniswap_usdc_usdt_005", "curve_3pool",             "Uniswap", "Curve"),
    ],
}


def _load_node_series(event_id: str, node_id: str) -> pl.DataFrame | None:
    """Load bronze pool-events for a node; returns None if not available."""
    bronze_ev = bronze_root() / event_id
    # Try real data first, fall back to existing fixture
    for stem in [f"{node_id}_pool_events.parquet",
                 f"{node_id}_3600s_pool_events.parquet"]:
        p = bronze_ev / stem
        if p.exists():
            df = pl.read_parquet(p)
            if "usdc_net_sold_1h" in df.columns and df.height > 0:
                return df.with_columns(pl.lit(node_id).alias("node_id"),
                                       pl.lit("A").alias("tier_actual"))
    return None


def run_cross_protocol(event_id: str) -> pl.DataFrame | None:
    """Run cross-protocol lead-lag for one event. Returns results DataFrame."""
    pairs_cfg = _CROSS_PROTOCOL_PAIRS.get(event_id, [])
    if not pairs_cfg:
        logger.warning("No cross-protocol pairs configured for %s", event_id)
        return None

    # Load node data
    node_ids_needed = {n for pair in pairs_cfg for n in (pair[0], pair[1])}
    frames: dict[str, pl.DataFrame] = {}
    for nid in node_ids_needed:
        df = _load_node_series(event_id, nid)
        if df is None:
            logger.warning("No data for %s in %s — skipping", nid, event_id)
        else:
            logger.info("Loaded %s: %d rows", nid, df.height)
            frames[nid] = df

    if len(frames) < 2:
        logger.warning("Need ≥2 nodes; only %d available for %s", len(frames), event_id)
        return None

    # Build panel
    panel = pl.concat(list(frames.values()), how="diagonal_relaxed")
    min_ts = panel["wall_clock_utc"].min()
    panel = panel.with_columns(
        (pl.col("wall_clock_utc") - pl.lit(min_ts))
        .dt.total_seconds()
        .alias("event_time_seconds")
    )

    available_nodes = list(frames.keys())
    node_pairs = [
        (s, t) for s, t, _, _ in pairs_cfg
        if s in available_nodes and t in available_nodes
    ]

    if not node_pairs:
        logger.warning("No valid pairs after filtering for %s", event_id)
        return None

    try:
        ll_df = compute_leadlag_table(
            panel=panel,
            node_pairs=node_pairs,
            feature_col="usdc_net_sold_1h",
            grid_seconds=3600,
            max_lag=12,
            n_reps=500,
            ts_col="event_time_seconds",
            max_staleness_seconds=3600,
        )
    except Exception as exc:
        logger.error("Lead-lag failed for %s: %s", event_id, exc)
        return None

    # compute_leadlag_table returns node_i / node_j (directed i→j)
    proto_map = {(s, t): (ps, pt) for s, t, ps, pt in pairs_cfg}

    def _proto(row: dict, idx: int) -> str:
        return proto_map.get((row["node_i"], row["node_j"]), ("?", "?"))[idx]

    ll_df = ll_df.with_columns(pl.lit(event_id).alias("event_id"))
    ll_df = ll_df.with_columns(
        pl.struct(["node_i", "node_j"]).map_elements(
            lambda r: proto_map.get((r["node_i"], r["node_j"]), ("?", "?"))[0],
            return_dtype=pl.Utf8
        ).alias("protocol_src"),
        pl.struct(["node_i", "node_j"]).map_elements(
            lambda r: proto_map.get((r["node_i"], r["node_j"]), ("?", "?"))[1],
            return_dtype=pl.Utf8
        ).alias("protocol_tgt"),
    )
    ll_df = ll_df.with_columns(
        pl.when(pl.col("protocol_src") == pl.col("protocol_tgt"))
          .then(pl.lit("within_protocol"))
          .otherwise(pl.lit("cross_protocol"))
          .alias("edge_type")
    )
    return ll_df


def plot_cross_protocol(results: dict[str, pl.DataFrame], out_dir: Path) -> Path:
    """Side-by-side within vs cross-protocol significance."""
    events = [e for e in _CROSS_PROTOCOL_PAIRS if e in results]
    n = len(events)
    if n == 0:
        logger.warning("No results to plot.")
        return out_dir / "A_cross_protocol_network.png"

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    fig.patch.set_facecolor(CWH)

    for ax, ev in zip(axes[0], events):
        df = results[ev]
        ax.set_facecolor(CBKG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        within = df.filter(pl.col("edge_type") == "within_protocol")
        cross  = df.filter(pl.col("edge_type") == "cross_protocol")

        def _sig_frac(sub: pl.DataFrame) -> float:
            if sub.height == 0 or "p_bonferroni" not in sub.columns:
                return 0.0
            return float((sub["p_bonferroni"] < 0.05).mean() or 0)

        def _mean_rho(sub: pl.DataFrame) -> float:
            if sub.height == 0 or "peak_corr" not in sub.columns:
                return 0.0
            return float(sub["peak_corr"].abs().mean() or 0)

        cats   = ["Within-protocol\n(Curve↔Curve)",
                  "Cross-protocol\n(Curve↔Uniswap)"]
        sig_f  = [_sig_frac(within), _sig_frac(cross)]
        rho_m  = [_mean_rho(within), _mean_rho(cross)]
        colors = [CAMB, CBLU]

        x = np.arange(2)
        bar1 = ax.bar(x - 0.18, sig_f, width=0.32, label="Frac. Bonferroni sig.",
                      color=colors, edgecolor="white", alpha=0.9)
        bar2 = ax.bar(x + 0.18, rho_m, width=0.32, label="Mean |peak ρ|",
                      color=colors, edgecolor="white", alpha=0.55, hatch="//")

        for bar, v in zip(list(bar1) + list(bar2), sig_f + rho_m):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=CSL)

        ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8.5)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Value", fontsize=8.5, color=CSL)
        ax.set_title(ev.replace("_", "/"), fontsize=9, fontweight="bold", color=CNV)
        ax.tick_params(colors=CSL)

        # Significance annotation
        if sig_f[1] > 0 and sig_f[1] >= sig_f[0]:
            ax.text(1, sig_f[1] + 0.08, "Cross-protocol\nA/A ✓",
                    ha="center", va="bottom", fontsize=7.5, color=CTA, fontweight="bold")
    handles = [
        mpatches.Patch(fc=CAMB, label="Within-protocol"),
        mpatches.Patch(fc=CBLU, label="Cross-protocol"),
        mpatches.Patch(fc="white", hatch="//", ec="grey", label="Mean |peak ρ|"),
        mpatches.Patch(fc="white", ec="grey", label="Frac. Bonferroni sig."),
    ]
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.08), framealpha=0.7)
    fig.suptitle("Cross-Protocol vs Within-Protocol Stress Propagation",
                 fontsize=12, fontweight="bold", color=CNV, y=1.02)

    out_path = out_dir / "A_cross_protocol_network.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=CWH)
    plt.close(fig)
    logger.info("Saved cross-protocol figure: %s", out_path.name)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-protocol lead-lag analysis.")
    parser.add_argument("--event", default=None)
    parser.add_argument("--all-events", action="store_true")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    if not args.event and not args.all_events:
        raise SystemExit("Specify --event <id> or --all-events")

    events = list(_CROSS_PROTOCOL_PAIRS.keys()) if args.all_events else [args.event]
    tables_dir = results_root() / "paper" / "tables"
    fig_dir    = results_root() / "paper" / "figures_cross_protocol"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, pl.DataFrame] = {}
    for event_id in events:
        logger.info("=== Cross-protocol: %s ===", event_id)
        ll = run_cross_protocol(event_id)
        if ll is not None:
            results[event_id] = ll
            out_path = tables_dir / f"table_cross_protocol_leadlag_{event_id}.csv"
            ll.write_csv(out_path)
            logger.info("Saved: %s  (%d rows)", out_path.name, ll.height)
            n_sig = ll.filter(pl.col("p_bonferroni") < 0.05).height if "p_bonferroni" in ll.columns else 0
            n_cross_sig = (
                ll.filter(
                    (pl.col("edge_type") == "cross_protocol") &
                    (pl.col("p_bonferroni") < 0.05)
                ).height if ("p_bonferroni" in ll.columns and "edge_type" in ll.columns) else 0
            )
            logger.info("  Total sig: %d  |  Cross-protocol sig: %d", n_sig, n_cross_sig)
            if n_cross_sig > 0:
                logger.info("  *** NEW A/A cross-protocol paper-claimable result! ***")
                cross_rows = ll.filter(
                    (pl.col("edge_type") == "cross_protocol") & (pl.col("p_bonferroni") < 0.05)
                )
                logger.info("\n%s", cross_rows.select(["node_i", "node_j", "peak_corr", "p_bonferroni"]))

    if results and not args.no_figure:
        plot_cross_protocol(results, fig_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
