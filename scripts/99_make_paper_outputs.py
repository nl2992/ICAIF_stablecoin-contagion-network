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

# (output_stem, per-event filename pattern, is_edge_table)
# is_edge_table=True  → claim-gated in strict mode (read from paper/tables)
# is_edge_table=False → no claim columns; read from results/tables even in strict mode
_TABLE_SPECS: list[tuple[str, str, bool]] = [
    ("table_leadlag_tests",       "table_leadlag_tests_{event}.csv",       True),
    ("table_hayashi_yoshida",     "table_hayashi_yoshida_{event}.csv",     True),
    ("table_var_spillovers",      "table_var_spillovers_{event}.csv",       True),
    ("table_hawkes_params",       "table_hawkes_params_{event}.csv",        True),
    ("table_transfer_entropy",    "table_transfer_entropy_{event}.csv",     True),
    ("table_tvp_var_summary",     "table_tvp_var_summary_{event}.csv",      True),
    ("table_event_study_summary", "table_event_study_summary_{event}.csv",  False),
    ("table_node_centrality",     "table_node_centrality_{event}.csv",      False),
    ("table_prediction_metrics",  "table_prediction_metrics_{event}.csv",   False),
]


# ---------------------------------------------------------------------------
# Strict-mode enforcement helpers
# ---------------------------------------------------------------------------

def _enforce_clean(df: pl.DataFrame, table_name: str, strict: bool) -> pl.DataFrame:
    """Filter to claim_allowed rows and fail on fixture leakage in strict mode."""
    if "claim_allowed" in df.columns:
        n_before = df.height
        # CSV round-trip may store booleans as "true"/"false" strings
        col = df["claim_allowed"]
        if col.dtype == pl.Boolean:
            df = df.filter(pl.col("claim_allowed"))
        else:
            df = df.filter(pl.col("claim_allowed").cast(pl.String).str.to_lowercase() == "true")
        dropped = n_before - df.height
        if dropped:
            logger.info("  %s: dropped %d non-claimable rows", table_name, dropped)

    if strict:
        if "uses_fixture" in df.columns:
            fix_col = df["uses_fixture"]
            if fix_col.dtype == pl.Boolean:
                n_fix = df.filter(pl.col("uses_fixture")).height
            else:
                n_fix = df.filter(
                    pl.col("uses_fixture").cast(pl.String).str.to_lowercase() == "true"
                ).height
            if n_fix > 0:
                raise SystemExit(
                    f"--strict: {n_fix} fixture-derived rows found in {table_name} after filtering. "
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
        # Remove any stale paper-table file so acceptance tests don't see old columns
        stale = out_dir / f"{table_name}.csv"
        if stale.exists():
            stale.unlink()
            logger.info("Removed stale paper table %s (no paper-claimable rows)", stale.name)
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


def _count_sig(df: pl.DataFrame, sig_col: str) -> int:
    """Count rows where a boolean (or boolean-string) column is True."""
    if sig_col not in df.columns:
        return 0
    col = df[sig_col]
    if col.dtype == pl.Boolean:
        return int(df.filter(pl.col(sig_col)).height)
    return int(df.filter(pl.col(sig_col).cast(pl.String).str.to_lowercase() == "true").height)


def write_paper_summary_table(tables_dir: Path, out_dir: Path, strict: bool) -> None:
    """Write a compact cross-event summary table for the paper."""
    # Granger tables only live in results/tables (not paper-gated)
    raw_dir = tables_dir.parent.parent / "tables" if strict else tables_dir
    rows = []
    for event_id in _EVENTS:
        row: dict = {"event_id": event_id}

        # Lead-lag significant pairs
        ll_path = tables_dir / f"table_leadlag_tests_{event_id}.csv"
        if ll_path.exists():
            ll = _enforce_clean(pl.read_csv(ll_path), ll_path.name, strict)
            sig_col = "significant_p01" if "significant_p01" in ll.columns else "significant"
            row["ll_sig"]   = _count_sig(ll, sig_col)
            row["ll_total"] = ll.height
        else:
            row["ll_sig"] = row["ll_total"] = 0

        # TE significant pairs (block-FDR)
        te_path = tables_dir / f"table_transfer_entropy_{event_id}.csv"
        if te_path.exists():
            te = _enforce_clean(pl.read_csv(te_path), te_path.name, strict)
            sig_col = "significant_block_fdr" if "significant_block_fdr" in te.columns else "significant_p05"
            row["te_sig"]   = _count_sig(te, sig_col)
            row["te_total"] = te.height
        else:
            row["te_sig"] = row["te_total"] = 0

        # Granger significant (FDR) — read from raw tables dir
        gr_path = (results_root() / "tables") / f"table_granger_{event_id}.csv"
        if gr_path.exists():
            gr = pl.read_csv(gr_path)
            gr = _enforce_clean(gr, gr_path.name, False)  # no strict on granger (not in paper dir)
            sig_col = "significant_fdr" if "significant_fdr" in gr.columns else "significant_p05"
            row["granger_sig"]   = _count_sig(gr, sig_col)
            row["granger_total"] = gr.height
        else:
            row["granger_sig"] = row["granger_total"] = 0

        # Event study — nodes with significant CAB (always from results/tables)
        es_path = (results_root() / "tables") / f"table_event_study_summary_{event_id}.csv"
        if es_path.exists():
            es = pl.read_csv(es_path)
            row["event_study_sig"]   = _count_sig(es, "significant_p05")
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


def write_empirical_coverage_table(out_dir: Path) -> None:
    """Write a paper-facing coverage table with fixture rows clearly excluded."""
    coverage_path = results_root() / "tables" / "table_node_coverage.csv"
    if not coverage_path.exists():
        logger.warning("table_node_coverage.csv not found; skipping empirical coverage table.")
        return
    coverage = pl.read_csv(coverage_path)
    tier_col = "tier_actual" if "tier_actual" in coverage.columns else "source_tier_actual"
    if tier_col not in coverage.columns:
        logger.warning("Coverage table lacks tier column; skipping empirical coverage table.")
        return
    empirical = coverage.with_columns(
        (~pl.col(tier_col).is_in(["fixture_non_empirical", "missing"])).alias("empirical_node")
    )
    out_path = out_dir / "table_node_coverage_empirical.csv"
    empirical.write_csv(out_path)
    logger.info("Wrote %s", out_path.name)


def write_tiered_edge_tables(
    tables_dir: Path,
    out_dir: Path,
    events: list[str],
) -> None:
    """Write two provenance-stratified edge tables for the paper.

    table_provenance_claimable_edges.csv
        All edge rows that pass the provenance gate (claim_allowed=True)
        across all result tables and events.  Use this to show the full set
        of real-data edges the analysis is based on.

    table_statistically_supported_edges.csv
        Subset that also passes the statistical gate (paper_claim_allowed=True).
        Use this for headline directional claims in the paper.
    """
    prov_rows: list[pl.DataFrame] = []
    paper_rows: list[pl.DataFrame] = []

    source_dir = results_root() / "tables"  # always read annotated tables from full dir

    for table_name, pattern, is_edge in _TABLE_SPECS:
        if not is_edge:
            continue
        for event_id in events:
            path = source_dir / pattern.format(event=event_id)
            if not path.exists():
                continue
            try:
                df = pl.read_csv(path)
            except Exception:
                continue
            if "claim_allowed" not in df.columns:
                continue
            if "event_id" not in df.columns:
                df = df.with_columns(pl.lit(event_id).alias("event_id"))
            df = df.with_columns(pl.lit(table_name).alias("method"))

            # Parse boolean columns (CSV round-trip stores as strings)
            for bool_col in ("claim_allowed", "paper_claim_allowed", "provenance_claim_allowed"):
                if bool_col in df.columns and df[bool_col].dtype != pl.Boolean:
                    df = df.with_columns(
                        (pl.col(bool_col).cast(pl.String).str.to_lowercase() == "true")
                        .alias(bool_col)
                    )

            prov = df.filter(pl.col("claim_allowed"))
            if prov.height > 0:
                prov_rows.append(prov)

            if "paper_claim_allowed" in df.columns:
                paper = df.filter(pl.col("paper_claim_allowed"))
            else:
                paper = prov.filter(pl.lit(False))  # empty — stat gate column not present
            if paper.height > 0:
                paper_rows.append(paper)

    out_dir.mkdir(parents=True, exist_ok=True)

    if prov_rows:
        prov_df = pl.concat(prov_rows, how="diagonal")
        prov_path = out_dir / "table_provenance_claimable_edges.csv"
        prov_df.write_csv(prov_path)
        logger.info(
            "Wrote %s (%d rows, %d events, %d claim levels)",
            prov_path.name, prov_df.height,
            prov_df["event_id"].n_unique() if "event_id" in prov_df.columns else 0,
            prov_df["claim_level"].n_unique() if "claim_level" in prov_df.columns else 0,
        )
        if "claim_level" in prov_df.columns:
            print("\n=== Provenance-claimable edges by claim level ===")
            print(prov_df.group_by("claim_level").len().sort("len", descending=True))
    else:
        logger.warning("No provenance-claimable edges found.")

    if paper_rows:
        paper_df = pl.concat(paper_rows, how="diagonal")
        paper_path = out_dir / "table_statistically_supported_edges.csv"
        paper_df.write_csv(paper_path)
        logger.info(
            "Wrote %s (%d rows, %d A/A rows)",
            paper_path.name, paper_df.height,
            paper_df.filter(
                pl.col("claim_level").str.starts_with("A_A_")
            ).height if "claim_level" in paper_df.columns else 0,
        )
        if "claim_level" in paper_df.columns:
            print("\n=== Statistically supported edges by claim level ===")
            print(paper_df.group_by("claim_level").len().sort("len", descending=True))
    else:
        logger.warning(
            "No statistically supported edges found. "
            "Run analysis scripts then re-run 00c_claim_gate.py --all-events."
        )


def write_aa_amm_edge_table(tables_dir: Path, out_dir: Path, events: list[str]) -> None:
    """Write the focused A/A AMM-flow evidence table for the paper.

    This is Table 4 in the paper: only ``A_A_dex_flow`` claim-level edges on the
    ``usdc_net_sold_1h`` Tier-A feature.  Two files are written:

    table_aa_amm_provenance_edges.csv
        All rows where ``claim_level == "A_A_dex_flow"`` (provenance gate passed).

    table_aa_amm_paper_edges.csv
        Subset that also has ``paper_claim_allowed == True`` (both gates passed).
        These are the headline directional AMM-flow claims in the paper.
    """
    _AA_DEX_LEVEL  = "A_A_dex_flow"
    _AMM_FEATURE   = "usdc_net_sold_1h"
    _KEEP_COLS     = [
        "event_id", "node_i", "node_j", "feature_col",
        "method",
        "peak_lag_seconds", "peak_corr",           # leadlag
        "te_i_to_j",                                # TE
        "fevd_share",                               # VAR spillover
        "p_value", "p_value_fdr",                   # generic p-value columns
        "TE_p",                                     # TE p-value
        "significant_block_fdr", "significant_fdr",
        "significant_bonferroni", "significant_p01",
        "claim_level", "claim_language", "claim_sentence",
        "provenance_claim_allowed", "statistical_claim_allowed",
        "paper_claim_allowed", "claim_strength",
    ]

    prov_rows: list[pl.DataFrame] = []
    paper_rows: list[pl.DataFrame] = []

    for table_name, pattern, is_edge in _TABLE_SPECS:
        if not is_edge:
            continue
        for event_id in events:
            path = tables_dir / pattern.format(event=event_id)
            if not path.exists():
                continue
            try:
                df = pl.read_csv(path)
            except Exception:
                continue
            if "claim_level" not in df.columns:
                continue
            if "event_id" not in df.columns:
                df = df.with_columns(pl.lit(event_id).alias("event_id"))
            df = df.with_columns(pl.lit(table_name).alias("method"))

            # Parse boolean columns
            for bool_col in ("claim_allowed", "paper_claim_allowed", "provenance_claim_allowed"):
                if bool_col in df.columns and df[bool_col].dtype != pl.Boolean:
                    df = df.with_columns(
                        (pl.col(bool_col).cast(pl.String).str.to_lowercase() == "true")
                        .alias(bool_col)
                    )

            # Normalise edge-column naming: some tables use causing_node/caused_node
            if "causing_node" in df.columns and "node_i" not in df.columns:
                df = df.rename({"causing_node": "node_i"})
            if "caused_node" in df.columns and "node_j" not in df.columns:
                df = df.rename({"caused_node": "node_j"})
            if "source" in df.columns and "node_i" not in df.columns:
                df = df.rename({"source": "node_i"})
            if "target" in df.columns and "node_j" not in df.columns:
                df = df.rename({"target": "node_j"})

            aa_dex = df.filter(
                (pl.col("claim_level") == _AA_DEX_LEVEL)
            )
            # Optionally narrow to AMM feature only when feature_col is present
            if "feature_col" in aa_dex.columns:
                aa_dex = aa_dex.filter(pl.col("feature_col") == _AMM_FEATURE)

            if aa_dex.height == 0:
                continue

            # Keep only columns that exist
            keep = [c for c in _KEEP_COLS if c in aa_dex.columns]
            aa_dex = aa_dex.select(keep)
            prov_rows.append(aa_dex)

            if "paper_claim_allowed" in aa_dex.columns:
                paper = aa_dex.filter(pl.col("paper_claim_allowed"))
                if paper.height > 0:
                    paper_rows.append(paper)

    out_dir.mkdir(parents=True, exist_ok=True)

    if prov_rows:
        prov_df = pl.concat(prov_rows, how="diagonal")
        prov_path = out_dir / "table_aa_amm_provenance_edges.csv"
        prov_df.write_csv(prov_path)
        logger.info(
            "Wrote %s (%d A/A DEX-flow provenance-valid rows across %d events)",
            prov_path.name, prov_df.height,
            prov_df["event_id"].n_unique() if "event_id" in prov_df.columns else 0,
        )
        if "node_i" in prov_df.columns and "node_j" in prov_df.columns:
            print("\n=== A/A AMM provenance-valid edges ===")
            pair_cols = [c for c in ["event_id", "node_i", "node_j", "method",
                                      "claim_strength", "paper_claim_allowed"]
                         if c in prov_df.columns]
            print(prov_df.select(pair_cols))
    else:
        logger.warning(
            "No A/A DEX-flow edges found on %s. "
            "Run amm_leadlag targets then re-run 00c_claim_gate.py --all-events.",
            _AMM_FEATURE,
        )

    if paper_rows:
        paper_df = pl.concat(paper_rows, how="diagonal")
        paper_path = out_dir / "table_aa_amm_paper_edges.csv"
        paper_df.write_csv(paper_path)
        logger.info(
            "Wrote %s (%d headline A/A AMM-flow paper-claimable rows)",
            paper_path.name, paper_df.height,
        )
    else:
        logger.warning(
            "No statistically supported A/A AMM-flow edges found. "
            "High-provenance descriptive edges exist (see table_aa_amm_provenance_edges.csv) "
            "but none pass the statistical gate. "
            "Report as high-provenance descriptive evidence, not directional propagation."
        )


def write_claim_language_summary(tables_dir: Path, out_dir: Path) -> None:
    """Summarise claim-language levels across claim-gated paper edge tables."""
    rows = []
    for path in sorted(tables_dir.glob("table_*.csv")):
        if path.name.startswith("table_claim_gate"):
            continue
        try:
            df = pl.read_csv(path)
        except Exception:
            continue
        if "claim_level" not in df.columns:
            continue
        for level in df["claim_level"].drop_nulls().unique().to_list():
            rows.append(
                {
                    "table": path.name,
                    "claim_level": level,
                    "n_rows": df.filter(pl.col("claim_level") == level).height,
                }
            )
    if rows:
        out_path = out_dir / "table_claim_language_summary.csv"
        pl.DataFrame(rows).sort(["claim_level", "table"]).write_csv(out_path)
        logger.info("Wrote %s", out_path.name)


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
    # Edge tables: in strict mode read from paper/tables (claim-gated)
    # Non-edge tables: always read from results/tables (no claim columns)
    raw_tables_dir = results_root() / "tables"
    for table_name, pattern, is_edge in _TABLE_SPECS:
        src = tables_dir if is_edge else raw_tables_dir
        consolidate_table(table_name, pattern, src, out_dir, args.events,
                          strict=args.strict if is_edge else False)

    # Cross-event summary
    write_paper_summary_table(tables_dir, out_dir, args.strict)
    write_empirical_coverage_table(out_dir)
    write_claim_language_summary(tables_dir, out_dir)

    # Provenance-stratified paper edge tables (key paper output)
    write_tiered_edge_tables(raw_tables_dir, out_dir, args.events)

    # Focused A/A AMM-flow evidence table (primary headline table in the paper)
    write_aa_amm_edge_table(raw_tables_dir, out_dir, args.events)

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
