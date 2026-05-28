"""Generate all paper-ready tables and figures.

In default mode reads from results/tables/ (all annotated edges).
In --strict mode reads exclusively from results/paper/tables/ (claim_allowed only)
and aborts if any fixture-derived row leaks through.

Usage:
    python scripts/99_make_paper_outputs.py              # diagnostic mode
    python scripts/99_make_paper_outputs.py --strict     # paper-safe mode
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from stressnet.config import results_root
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_EVENTS = ["usdc_svb_2023", "terra_luna_2022", "usdt_curve_2023", "ftx_2022", "busd_2023"]

# (output_stem, per-event filename pattern)
_TABLE_SPECS = [
    ("table_leadlag_tests",       "table_leadlag_tests_{event}.csv"),
    ("table_var_spillovers",      "table_var_spillovers_{event}.csv"),
    ("table_hawkes_params",       "table_hawkes_params_{event}.csv"),
    ("table_transfer_entropy",    "table_transfer_entropy_{event}.csv"),
    ("table_tvp_var_summary",     "table_tvp_var_summary_{event}.csv"),
    ("table_event_study_summary", "table_event_study_summary_{event}.csv"),
    ("table_node_centrality",     "table_node_centrality_{event}.csv"),
    ("table_prediction_metrics",  "table_prediction_metrics_{event}.csv"),
]


# ---------------------------------------------------------------------------
# Strict-mode enforcement helpers
# ---------------------------------------------------------------------------

def _enforce_clean(df: pl.DataFrame, table_name: str, strict: bool) -> pl.DataFrame:
    """Filter to claim_allowed rows and fail on fixture leakage in strict mode."""
    if "claim_allowed" in df.columns:
        n_before = df.height
        df = df.filter(pl.col("claim_allowed"))
        dropped = n_before - df.height
        if dropped:
            logger.info("  %s: dropped %d non-claimable rows", table_name, dropped)

    if strict:
        if "uses_fixture" in df.columns and df.filter(pl.col("uses_fixture")).height > 0:
            raise SystemExit(
                f"--strict: fixture-derived rows found in {table_name} after filtering. "
                "Run 00c_claim_gate.py --all-events --strict first."
            )
        if "edge_tier_actual" in df.columns:
            bad = df.filter(pl.col("edge_tier_actual").is_in(["fixture_non_empirical", "missing"]))
            if bad.height > 0:
                raise SystemExit(
                    f"--strict: {bad.height} fixture/missing-tier edges remain in {table_name}."
                )
    return df


# ---------------------------------------------------------------------------
# Table consolidation
# ---------------------------------------------------------------------------

def consolidate_table(
    table_name: str,
    pattern: str,
    tables_dir: Path,
    out_dir: Path,
    events: list[str],
    strict: bool,
) -> pl.DataFrame | None:
    """Merge per-event tables into one consolidated CSV."""
    frames = []
    for event_id in events:
        path = tables_dir / pattern.format(event=event_id)
        if not path.exists():
            logger.debug("Missing: %s", path.name)
            continue
        df = pl.read_csv(path)
        if "event_id" not in df.columns:
            df = df.with_columns(pl.lit(event_id).alias("event_id"))
        df = _enforce_clean(df, path.name, strict)
        if df.height > 0:
            frames.append(df)

    if not frames:
        logger.warning("No data for %s (all events missing or all rows blocked)", table_name)
        return None

    combined = pl.concat(frames, how="diagonal")
    out_path = out_dir / f"{table_name}.csv"
    combined.write_csv(out_path)
    logger.info("Wrote %s (%d rows)", out_path.name, combined.height)
    return combined


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def _require_mpl():
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
        return plt
    except ImportError:
        logger.warning("matplotlib not available; skipping figures.")
        return None


def plot_auc_by_event(tables_dir: Path, figures_dir: Path) -> None:
    plt = _require_mpl()
    if plt is None:
        return
    path = tables_dir / "table_prediction_metrics.csv"
    if not path.exists():
        logger.warning("table_prediction_metrics.csv not found; skipping AUC figure.")
        return

    df = pl.read_csv(path)
    if "AUROC" not in df.columns:
        return

    events = df["event_id"].unique().sort().to_list() if "event_id" in df.columns else ["unknown"]
    fig, axes = plt.subplots(1, len(events), figsize=(len(events) * 4, 5), sharey=True)
    if len(events) == 1:
        axes = [axes]

    for ax, event_id in zip(axes, events):
        sub = df.filter(pl.col("event_id") == event_id) if "event_id" in df.columns else df
        ax.barh(sub["model"].to_list(), sub["AUROC"].to_list(), color="#2980b9")
        ax.axvline(0.5, color="red", linestyle="--", linewidth=1)
        ax.set_xlim(0.4, 1.0)
        ax.set_title(event_id, fontsize=9)
        ax.set_xlabel("AUROC")

    axes[0].set_ylabel("Model")
    fig.suptitle("AUROC by Event", fontsize=12)
    plt.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / "figure_auc_by_event.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure_auc_by_event.png")


def plot_leadlag_heatmap(tables_dir: Path, figures_dir: Path) -> None:
    """Heatmap of peak cross-correlation by node-pair, one panel per event."""
    plt = _require_mpl()
    if plt is None:
        return
    import numpy as np

    path = tables_dir / "table_leadlag_tests.csv"
    if not path.exists():
        return

    df = pl.read_csv(path)
    if not {"node_i", "node_j", "peak_corr", "event_id"}.issubset(df.columns):
        return

    events = sorted(df["event_id"].unique().to_list())
    n_events = len(events)
    if n_events == 0:
        return

    fig, axes = plt.subplots(1, n_events, figsize=(n_events * 5, 5), squeeze=False)
    for col_idx, event_id in enumerate(events):
        sub = df.filter(pl.col("event_id") == event_id)
        nodes = sorted(set(sub["node_i"].to_list()) | set(sub["node_j"].to_list()))
        n = len(nodes)
        mat = np.zeros((n, n))
        idx = {nd: i for i, nd in enumerate(nodes)}
        for row in sub.iter_rows(named=True):
            i, j = idx.get(row["node_i"]), idx.get(row["node_j"])
            if i is not None and j is not None:
                mat[i, j] = row.get("peak_corr", 0) or 0
        ax = axes[0][col_idx]
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels([nd.replace("_", "\n") for nd in nodes], fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels([nd.replace("_", "\n") for nd in nodes], fontsize=7)
        ax.set_title(event_id.replace("_", "\n"), fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Lead-lag peak cross-correlation (row leads column)", fontsize=11)
    plt.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / "figure_leadlag_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure_leadlag_heatmap.png")


def plot_te_heatmap(tables_dir: Path, figures_dir: Path) -> None:
    """Heatmap of TE(i→j) by node-pair, one panel per event."""
    plt = _require_mpl()
    if plt is None:
        return
    import numpy as np

    path = tables_dir / "table_transfer_entropy.csv"
    if not path.exists():
        return

    df = pl.read_csv(path)
    if not {"node_i", "node_j", "te_i_to_j", "event_id"}.issubset(df.columns):
        return

    events = sorted(df["event_id"].unique().to_list())
    n_events = len(events)
    if n_events == 0:
        return

    fig, axes = plt.subplots(1, n_events, figsize=(n_events * 5, 5), squeeze=False)
    for col_idx, event_id in enumerate(events):
        sub = df.filter(pl.col("event_id") == event_id)
        nodes = sorted(set(sub["node_i"].to_list()) | set(sub["node_j"].to_list()))
        n = len(nodes)
        mat = np.zeros((n, n))
        idx = {nd: i for i, nd in enumerate(nodes)}
        sig_col = "significant_block_fdr" if "significant_block_fdr" in df.columns else "significant_p05"
        for row in sub.iter_rows(named=True):
            i, j = idx.get(row["node_i"]), idx.get(row["node_j"])
            if i is not None and j is not None:
                val = row.get("te_i_to_j", 0) or 0
                # grey out non-significant
                if not row.get(sig_col, True):
                    val = 0.0
                mat[i, j] = val
        ax = axes[0][col_idx]
        im = ax.imshow(mat, vmin=0, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels([nd.replace("_", "\n") for nd in nodes], fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels([nd.replace("_", "\n") for nd in nodes], fontsize=7)
        ax.set_title(event_id.replace("_", "\n"), fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Transfer entropy TE(row → col) — block-FDR significant only", fontsize=11)
    plt.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / "figure_te_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure_te_heatmap.png")


def plot_event_study_cab(tables_dir: Path, figures_dir: Path) -> None:
    """Cumulative abnormal basis (CAB) time series per event."""
    plt = _require_mpl()
    if plt is None:
        return
    import numpy as np
    import matplotlib.ticker as mticker

    events = _EVENTS
    fig, axes = plt.subplots(len(events), 1, figsize=(10, 3 * len(events)), sharex=False)
    if len(events) == 1:
        axes = [axes]

    plotted_any = False
    for ax, event_id in zip(axes, events):
        ts_path = tables_dir / f"table_event_study_timeseries_{event_id}.csv"
        if not ts_path.exists():
            ax.set_visible(False)
            continue
        df = pl.read_csv(ts_path)
        if not {"node_id", "event_time_seconds", "cab"}.issubset(df.columns):
            ax.set_visible(False)
            continue

        nodes = df.filter(
            pl.col("event_time_seconds") >= 0
        )["node_id"].unique().to_list()

        has_data = False
        for node_id in sorted(nodes):
            nd = df.filter(pl.col("node_id") == node_id).sort("event_time_seconds")
            t = nd["event_time_seconds"].to_numpy() / 3600  # hours
            cab = nd["cab"].to_numpy().astype(float)
            if np.all(np.isnan(cab)):
                continue
            ax.plot(t, cab, label=node_id.replace("_", " "), linewidth=1.2)
            has_data = True

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, label="onset")
        ax.axhline(0, color="grey", linestyle=":", linewidth=0.6)
        ax.set_title(event_id.replace("_", " "), fontsize=9)
        ax.set_xlabel("Hours since onset")
        ax.set_ylabel("CAB")
        ax.legend(fontsize=7, loc="upper left")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
        if has_data:
            plotted_any = True

    if plotted_any:
        fig.suptitle("Cumulative Abnormal Basis (CAB) — real nodes only", fontsize=12)
        plt.tight_layout()
        figures_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(figures_dir / "figure_event_study_cab.png", dpi=150, bbox_inches="tight")
        logger.info("Saved figure_event_study_cab.png")
    plt.close()


def plot_tvp_var_spillovers(tables_dir: Path, figures_dir: Path) -> None:
    """Rolling FEVD spillover share over time, one panel per event."""
    plt = _require_mpl()
    if plt is None:
        return
    import numpy as np

    events = _EVENTS
    rows_with_data = [(e, tables_dir / f"table_tvp_var_spillovers_{e}.csv")
                      for e in events if (tables_dir / f"table_tvp_var_spillovers_{e}.csv").exists()]
    if not rows_with_data:
        return

    n = len(rows_with_data)
    fig, axes = plt.subplots(1, n, figsize=(n * 5, 4), squeeze=False)

    for col_idx, (event_id, path) in enumerate(rows_with_data):
        df = pl.read_csv(path)
        if not {"caused_node", "causing_node", "window_start_ts", "fevd_share"}.issubset(df.columns):
            continue
        ax = axes[0][col_idx]
        pairs = df.select(["caused_node", "causing_node"]).unique().to_dicts()
        for pair in pairs:
            sub = df.filter(
                (pl.col("caused_node") == pair["caused_node"]) &
                (pl.col("causing_node") == pair["causing_node"])
            ).sort("window_start_ts")
            t = sub["window_start_ts"].to_numpy() / 3600
            v = sub["fevd_share"].to_numpy().astype(float)
            label = f"{pair['causing_node'].split('_')[0]}→{pair['caused_node'].split('_')[0]}"
            ax.plot(t, v, label=label, linewidth=1.2)
        ax.set_title(event_id.replace("_", "\n"), fontsize=9)
        ax.set_xlabel("Hours since onset")
        ax.set_ylabel("FEVD share")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6)

    fig.suptitle("TVP-VAR Rolling FEVD Spillover Shares", fontsize=11)
    plt.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / "figure_tvp_var_spillovers.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure_tvp_var_spillovers.png")


def plot_claim_gate_summary(tables_dir: Path, figures_dir: Path) -> None:
    """Bar chart: claimable vs blocked rows per table per event."""
    plt = _require_mpl()
    if plt is None:
        return
    import numpy as np

    path = tables_dir / "table_claim_gate_all_events.csv"
    if not path.exists():
        path = tables_dir / "table_claim_gate_paper.csv"
    if not path.exists():
        return

    df = pl.read_csv(path).filter(pl.col("status") == "annotated")
    if df.is_empty() or not {"table", "claimable_rows", "rows"}.issubset(df.columns):
        return

    tables = df["table"].to_list()
    claimable = df["claimable_rows"].to_numpy()
    total = df["rows"].to_numpy()
    blocked = total - claimable
    x = np.arange(len(tables))

    fig, ax = plt.subplots(figsize=(max(8, len(tables) * 0.7), 5))
    ax.bar(x, claimable, label="Claimable", color="#27ae60")
    ax.bar(x, blocked, bottom=claimable, label="Blocked/Fixture", color="#e74c3c", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("table_", "").replace("_", "\n") for t in tables],
                       fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Edge rows")
    ax.set_title("Claim gate: paper-claimable vs blocked edge rows", fontsize=11)
    ax.legend()
    plt.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / "figure_claim_gate_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure_claim_gate_summary.png")


def write_paper_summary_table(tables_dir: Path, out_dir: Path, strict: bool) -> None:
    """Write a compact cross-event summary table for the paper."""
    rows = []
    for event_id in _EVENTS:
        row: dict = {"event_id": event_id}

        # Lead-lag significant pairs
        ll_path = tables_dir / f"table_leadlag_tests_{event_id}.csv"
        if ll_path.exists():
            ll = pl.read_csv(ll_path)
            ll = _enforce_clean(ll, ll_path.name, strict)
            sig_col = "significant_p01" if "significant_p01" in ll.columns else "significant"
            row["ll_sig"] = int(ll.filter(pl.col(sig_col)).height) if sig_col in ll.columns else 0
            row["ll_total"] = ll.height
        else:
            row["ll_sig"] = row["ll_total"] = 0

        # TE significant pairs (block-FDR)
        te_path = tables_dir / f"table_transfer_entropy_{event_id}.csv"
        if te_path.exists():
            te = pl.read_csv(te_path)
            te = _enforce_clean(te, te_path.name, strict)
            sig_col = "significant_block_fdr" if "significant_block_fdr" in te.columns else "significant_p05"
            row["te_sig"] = int(te.filter(pl.col(sig_col)).height) if sig_col in te.columns else 0
            row["te_total"] = te.height
        else:
            row["te_sig"] = row["te_total"] = 0

        # Granger significant (FDR)
        gr_path = tables_dir / f"table_granger_{event_id}.csv"
        if gr_path.exists():
            gr = pl.read_csv(gr_path)
            gr = _enforce_clean(gr, gr_path.name, strict)
            sig_col = "significant_fdr" if "significant_fdr" in gr.columns else "significant_p05"
            row["granger_sig"] = int(gr.filter(pl.col(sig_col)).height) if sig_col in gr.columns else 0
            row["granger_total"] = gr.height
        else:
            row["granger_sig"] = row["granger_total"] = 0

        # Event study — nodes with significant CAB
        es_path = tables_dir / f"table_event_study_summary_{event_id}.csv"
        if es_path.exists():
            es = pl.read_csv(es_path)
            row["event_study_sig"] = int(es.filter(pl.col("significant_p05")).height) if "significant_p05" in es.columns else 0
            row["event_study_total"] = es.height
        else:
            row["event_study_sig"] = row["event_study_total"] = 0

        rows.append(row)

    if not rows:
        return

    summary = pl.DataFrame(rows)
    out_path = out_dir / "table_paper_summary.csv"
    summary.write_csv(out_path)
    logger.info("Wrote %s", out_path.name)

    # Pretty-print
    print("\n=== Paper summary (real nodes only) ===")
    print(summary)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate paper outputs.")
    parser.add_argument("--events", nargs="+", default=_EVENTS)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Read from results/paper/tables/, enforce claim_allowed, fail on fixture.",
    )
    parser.add_argument(
        "--paper-dir",
        default=None,
        help="Override paper tables directory (default: results/paper/tables in strict mode).",
    )
    args = parser.parse_args()

    root = results_root()

    if args.strict:
        tables_dir  = Path(args.paper_dir) if args.paper_dir else root / "paper" / "tables"
        out_dir     = tables_dir
        figures_dir = root / "paper" / "figures"
        logger.info("STRICT mode: reading from %s", tables_dir)
    else:
        tables_dir  = root / "tables"
        out_dir     = tables_dir
        figures_dir = root / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Consolidate per-event tables
    for table_name, pattern in _TABLE_SPECS:
        consolidate_table(table_name, pattern, tables_dir, out_dir, args.events, args.strict)

    # Cross-event summary
    write_paper_summary_table(tables_dir, out_dir, args.strict)

    # Figures
    plot_auc_by_event(out_dir, figures_dir)
    plot_leadlag_heatmap(out_dir, figures_dir)
    plot_te_heatmap(out_dir, figures_dir)
    plot_event_study_cab(tables_dir, figures_dir)   # reads per-event timeseries, always from main tables
    plot_tvp_var_spillovers(tables_dir, figures_dir)
    plot_claim_gate_summary(root / "tables", figures_dir)  # always from full tables for audit fig

    logger.info("Paper outputs complete (%s mode).", "strict" if args.strict else "diagnostic")


if __name__ == "__main__":
    main()
