"""Compute a Stress Propagation Score (SPS) for each event.

The SPS is a composite index that quantifies how strongly evidence of
stress propagation is supported across methods and tiers.  It is designed
to be:

  * Monotonically increasing in paper-claimable evidence
  * Penalised for having only fixture or B/B evidence
  * Comparable across events (normalised to [0, 1])

Formula
-------
    raw_SPS(e) =   w_AA   × n_AA_paper_claimable(e)
                 + w_AB   × n_AB_paper_claimable(e)
                 + w_eff  × mean_peak_corr_AA(e)       (0 if no A/A edges)
                 + w_stat × (1 − mean_p_bonferroni_AA(e))  (0 if no A/A edges)

Weights (default):
    w_AA   = 4.0   # A/A paper-claimable edges are the headline
    w_AB   = 1.0   # A/B suggestive edges add minor signal
    w_eff  = 3.0   # effect size (peak cross-correlation)
    w_stat = 2.0   # statistical confidence

The normalised SPS divides by the maximum raw score across all events so
scores lie in [0, 1].

Output
------
    results/paper/tables/table_stress_propagation_score.csv
    results/paper/figures_cross_protocol/A_sps_event_ranking.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import polars as pl

from stressnet.config import results_root
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

W_AA   = 4.0
W_AB   = 1.0
W_EFF  = 3.0
W_STAT = 2.0

EVENTS_ORDER = [
    "usdt_curve_2023",
    "terra_luna_2022",
    "usdc_svb_2023",
    "ftx_2022",
    "busd_2023",
]

# Columbia palette
CNV  = "#003865"
CTA  = "#27AE60"
CTB  = "#7F8C8D"
CAMB = "#E67E22"
CBLU = "#2980B9"
CRED = "#C0392B"
CWH  = "#FFFFFF"
CLT  = "#B9D9EB"
CBKG = "#F8FBFD"
CSL  = "#2C3E50"

EVENT_COLORS = {
    "usdt_curve_2023": CAMB,
    "terra_luna_2022": CTB,
    "usdc_svb_2023":   CBLU,
    "ftx_2022":        CSL,
    "busd_2023":       CTB,
}


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def load_audit(tables_dir: Path) -> pl.DataFrame | None:
    p = tables_dir / "table_claim_audit_summary.csv"
    if not p.exists():
        logger.warning("Audit summary not found: %s", p)
        return None
    return pl.read_csv(p)


def load_leadlag(tables_dir: Path, event_id: str) -> pl.DataFrame | None:
    p = tables_dir / f"table_leadlag_tests_{event_id}.csv"
    if not p.exists():
        return None
    return pl.read_csv(p)


def load_cross_protocol(tables_dir: Path, event_id: str) -> pl.DataFrame | None:
    p = tables_dir / f"table_cross_protocol_leadlag_{event_id}.csv"
    if not p.exists():
        return None
    return pl.read_csv(p)


def cross_protocol_claimable_summary(tables_dir: Path, event_id: str) -> tuple[int, float, float] | None:
    """Return count/effect/p-value summary for Tier-A cross-protocol rows.

    The main claim-gate tables predate the Uniswap v3 extension, so the
    cross-protocol lead-lag output is folded into SPS explicitly.  Rows are
    execution-grade AMM-flow edges by construction; the statistical gate uses
    Bonferroni significance when available, then FDR as a fallback.
    """
    df = load_cross_protocol(tables_dir, event_id)
    if df is None or df.is_empty():
        return None

    if "p_bonferroni" in df.columns:
        sig = df.filter(pl.col("p_bonferroni") < 0.05)
        p_col = "p_bonferroni"
    elif "p_value_fdr" in df.columns:
        sig = df.filter(pl.col("p_value_fdr") < 0.05)
        p_col = "p_value_fdr"
    elif "significant_fdr" in df.columns:
        sig = df.filter(pl.col("significant_fdr").fill_null(False))
        p_col = "p_value_fdr" if "p_value_fdr" in df.columns else None
    else:
        sig = pl.DataFrame()
        p_col = None

    if sig.is_empty():
        return 0, 0.0, 1.0

    mean_peak_corr = 0.0
    if "peak_corr" in sig.columns:
        vals = [abs(float(v)) for v in sig["peak_corr"].drop_nulls().to_list()]
        mean_peak_corr = float(np.mean(vals)) if vals else 0.0

    mean_p = 1.0
    if p_col and p_col in sig.columns:
        pvals = [float(v) for v in sig[p_col].drop_nulls().to_list()]
        mean_p = float(np.mean(pvals)) if pvals else 1.0

    return sig.height, mean_peak_corr, mean_p


def compute_scores(tables_dir: Path) -> pl.DataFrame:
    """Return a DataFrame with SPS per event."""
    audit = load_audit(tables_dir)

    rows = []
    for ev in EVENTS_ORDER:
        # Pull counts from audit summary
        n_aa_pc = n_ab_pc = 0
        if audit is not None and "event_id" in audit.columns:
            ev_row = audit.filter(pl.col("event_id") == ev)
            if ev_row.height > 0:
                n_aa_pc = int(ev_row["n_AA_paper_claimable"][0]) if "n_AA_paper_claimable" in ev_row.columns else 0
                n_ab_pc = int(ev_row["n_AB_paper_claimable"][0]) if "n_AB_paper_claimable" in ev_row.columns else 0

        # Pull effect size from lead-lag table
        mean_peak_corr = 0.0
        mean_p_bon     = 1.0
        ll = load_leadlag(tables_dir, ev)
        if ll is not None and n_aa_pc > 0:
            aa_rows = ll
            # Filter to A/A rows if claim_level column exists
            if "claim_level" in ll.columns:
                aa_rows = ll.filter(pl.col("claim_level").str.starts_with("A_A"))
            if aa_rows.height > 0:
                if "peak_corr" in aa_rows.columns:
                    mean_peak_corr = float(aa_rows["peak_corr"].drop_nulls().mean() or 0)
                if "p_bonferroni" in aa_rows.columns:
                    mean_p_bon = float(aa_rows["p_bonferroni"].drop_nulls().mean() or 1.0)

        cross_summary = cross_protocol_claimable_summary(tables_dir, ev)
        if cross_summary is not None:
            cross_n, cross_effect, cross_p = cross_summary
            if cross_n > n_aa_pc:
                logger.info(
                    "%s: replacing audit A/A count %d with cross-protocol count %d",
                    ev, n_aa_pc, cross_n,
                )
                n_aa_pc = cross_n
                mean_peak_corr = cross_effect
                mean_p_bon = cross_p

        raw = (
            W_AA   * n_aa_pc
          + W_AB   * n_ab_pc
          + W_EFF  * mean_peak_corr
          + W_STAT * max(0.0, 1.0 - mean_p_bon)
        )

        rows.append({
            "event_id":            ev,
            "n_AA_paper_claimable": n_aa_pc,
            "n_AB_paper_claimable": n_ab_pc,
            "mean_peak_corr_AA":   mean_peak_corr,
            "mean_p_bonferroni_AA": mean_p_bon,
            "raw_SPS":             raw,
        })

    df = pl.DataFrame(rows)

    # Normalise to [0, 1]
    max_raw = df["raw_SPS"].max()
    if max_raw and max_raw > 0:
        df = df.with_columns((pl.col("raw_SPS") / max_raw).alias("SPS"))
    else:
        df = df.with_columns(pl.lit(0.0).alias("SPS"))

    return df.sort("SPS", descending=True)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_sps(df: pl.DataFrame, out_dir: Path) -> Path:
    """Horizontal bar chart of SPS per event, colour-coded by claim tier."""
    events   = df["event_id"].to_list()
    sps      = df["SPS"].to_list()
    n_aa     = df["n_AA_paper_claimable"].to_list()

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(CWH)
    ax.set_facecolor(CBKG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    bar_colors = [CAMB if n > 0 else CBLU if "svb" in ev else CTB
                  for ev, n in zip(events, n_aa)]
    bars = ax.barh(events[::-1], sps[::-1], color=bar_colors[::-1],
                   edgecolor="white", linewidth=0.8, height=0.55)

    # Annotate SPS value
    for bar, score, n_aa_val in zip(bars, sps[::-1], n_aa[::-1]):
        x = bar.get_width()
        label = f"SPS={score:.2f}"
        if n_aa_val > 0:
            label += f"  ✓ {n_aa_val} A/A paper-claimable"
        ax.text(x + 0.01, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left", fontsize=8.5, color=CSL)

    ax.set_xlim(0, 1.45)
    ax.set_xlabel("Stress Propagation Score (normalised)", fontsize=9, color=CSL)
    ax.tick_params(colors=CSL, labelsize=9)

    patches = [
        mpatches.Patch(fc=CAMB, label="A/A paper-claimable"),
        mpatches.Patch(fc=CBLU, label="Sparse / settlement"),
        mpatches.Patch(fc=CTB,  label="A/B or no paper-claim"),
    ]
    ax.legend(handles=patches, fontsize=8, framealpha=0.6, loc="lower right")

    ax.set_title("Stress Propagation Score by Event",
                 fontsize=12, fontweight="bold", color=CNV, pad=10, loc="left")
    ax.text(0, 1.02, "SPS = weighted composite of A/A edges, A/B edges, effect size, and statistical confidence",
            transform=ax.transAxes, fontsize=8, color=CSL, va="bottom")
    out_path = out_dir / "A_sps_event_ranking.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=CWH)
    plt.close(fig)
    logger.info("Saved SPS figure: %s", out_path.name)
    return out_path


# ---------------------------------------------------------------------------
# Multi-resolution SPS table (if sub-hourly results exist)
# ---------------------------------------------------------------------------

def augment_with_subhourly(df: pl.DataFrame, tables_dir: Path) -> pl.DataFrame:
    """Add SPS contributions from sub-hourly Terra/LUNA analysis if available."""
    subhourly_file = tables_dir / "table_subhourly_leadlag_terra_luna_2022.csv"
    if not subhourly_file.exists():
        return df

    sh = pl.read_csv(subhourly_file)
    if "p_bonferroni" not in sh.columns or "grid_seconds" not in sh.columns:
        return df

    # Check if sub-hourly Terra result is significant
    sig_grids = (
        sh.filter(pl.col("p_bonferroni") < 0.05)
          .select("grid_seconds")
          .unique()
          .to_series()
          .to_list()
    )
    if not sig_grids:
        logger.info("Sub-hourly Terra: no Bonferroni-significant rows found.")
        return df

    finest_sig = min(sig_grids)
    logger.info("Sub-hourly Terra: significant at grid=%ds — upgrading terra_luna_2022 SPS", finest_sig)
    sh_sub = sh.filter(
        (pl.col("p_bonferroni") < 0.05) & (pl.col("grid_seconds") == finest_sig)
    )
    n_aa_sub = sh_sub.height

    # Bump terra_luna_2022 SPS
    terra_idx = df["event_id"].to_list().index("terra_luna_2022")
    old_n_aa = df[terra_idx]["n_AA_paper_claimable"][0]
    # Only add if genuinely new rows
    new_n_aa = max(old_n_aa, n_aa_sub)
    df = df.with_columns(
        pl.when(pl.col("event_id") == "terra_luna_2022")
          .then(pl.lit(new_n_aa))
          .otherwise(pl.col("n_AA_paper_claimable"))
          .alias("n_AA_paper_claimable")
    )
    df = df.with_columns(
        pl.when(pl.col("event_id") == "terra_luna_2022")
          .then(pl.lit(f"sub-hourly ({finest_sig}s grid)"))
          .otherwise(pl.lit("hourly"))
          .alias("resolution_note")
    )
    logger.info("terra_luna_2022: n_AA_paper_claimable updated %d → %d", old_n_aa, new_n_aa)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Compute Stress Propagation Score per event.")
    parser.add_argument("--tables-dir", default=None,
                        help="Override results/paper/tables directory.")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    tables_dir = Path(args.tables_dir) if args.tables_dir else (
        results_root() / "paper" / "tables"
    )
    fig_dir = results_root() / "paper" / "figures_cross_protocol"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = compute_scores(tables_dir)
    df = augment_with_subhourly(df, tables_dir)

    # Re-normalise after potential sub-hourly augmentation
    max_raw = df["raw_SPS"].max() if "raw_SPS" in df.columns else None
    if max_raw and float(max_raw) > 0:
        df = df.with_columns((pl.col("raw_SPS") / float(max_raw)).alias("SPS"))

    out_path = tables_dir / "table_stress_propagation_score.csv"
    df.write_csv(out_path)
    logger.info("Saved SPS table: %s", out_path)
    logger.info("\n%s", df)

    if not args.no_figure:
        plot_sps(df, fig_dir)


if __name__ == "__main__":
    main()
