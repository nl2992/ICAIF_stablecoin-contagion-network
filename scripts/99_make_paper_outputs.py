"""Generate all paper-ready tables and figures.

Consolidates per-event outputs into final paper-ready artefacts:
    results/tables/table_leadlag_tests.csv
    results/tables/table_var_spillovers.csv
    results/tables/table_hawkes_params.csv
    results/tables/table_transfer_entropy.csv
    results/tables/table_node_centrality.csv
    results/tables/table_prediction_metrics.csv
    results/figures/figure_auc_by_event.png
"""

import argparse
from pathlib import Path

import polars as pl

from stressnet.config import load_events, results_root
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_EVENTS = ["usdc_svb_2023", "terra_luna_2022", "usdt_curve_2023", "ftx_2022", "busd_2023"]

_TABLE_SPECS = [
    ("table_leadlag_tests", "table_leadlag_tests_{event}.csv"),
    ("table_var_spillovers", "table_var_spillovers_{event}.csv"),
    ("table_hawkes_params", "table_hawkes_params_{event}.csv"),
    ("table_transfer_entropy", "table_transfer_entropy_{event}.csv"),
    ("table_node_centrality", "table_node_centrality_{event}.csv"),
    ("table_prediction_metrics", "table_prediction_metrics_{event}.csv"),
]


def consolidate_table(table_name: str, pattern: str, tables_dir: Path, events: list[str]) -> None:
    """Merge per-event tables into a single consolidated CSV."""
    frames = []
    for event_id in events:
        path = tables_dir / pattern.format(event=event_id)
        if path.exists():
            df = pl.read_csv(path)
            if "event_id" not in df.columns:
                df = df.with_columns(pl.lit(event_id).alias("event_id"))
            frames.append(df)
        else:
            logger.debug("Missing: %s", path.name)

    if not frames:
        logger.warning("No data for %s", table_name)
        return

    combined = pl.concat(frames, how="diagonal")
    out_path = tables_dir / f"{table_name}.csv"
    combined.write_csv(out_path)
    logger.info("Wrote %s (%d rows)", out_path.name, len(combined))


def plot_auc_by_event(tables_dir: Path, figures_dir: Path) -> None:
    """Plot AUROC and AUPRC for each model across events."""
    path = tables_dir / "table_prediction_metrics.csv"
    if not path.exists():
        logger.warning("table_prediction_metrics.csv not found; skipping AUC figure.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    df = pl.read_csv(path)
    events = df["event_id"].unique().sort().to_list() if "event_id" in df.columns else ["unknown"]
    models = df["model"].unique().sort().to_list() if "model" in df.columns else []

    if "AUROC" not in df.columns:
        return

    fig, axes = plt.subplots(1, len(events), figsize=(len(events) * 4, 5), sharey=True)
    if len(events) == 1:
        axes = [axes]

    for ax, event_id in zip(axes, events):
        sub = df.filter(pl.col("event_id") == event_id) if "event_id" in df.columns else df
        models_here = sub["model"].to_list()
        aurocs = sub["AUROC"].to_list()
        ax.barh(models_here, aurocs, color="#2980b9")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate paper outputs.")
    parser.add_argument("--events", nargs="+", default=_EVENTS)
    args = parser.parse_args()

    tables_dir = results_root() / "tables"
    figures_dir = results_root() / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for table_name, pattern in _TABLE_SPECS:
        consolidate_table(table_name, pattern, tables_dir, args.events)

    plot_auc_by_event(tables_dir, figures_dir)
    logger.info("Paper outputs complete.")


if __name__ == "__main__":
    main()
