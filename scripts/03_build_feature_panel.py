"""Build the gold-layer event-time feature panel.

Reads data/silver/ and writes data/gold/dataset_contagion_features_{event}.parquet.
"""

import argparse
from pathlib import Path

import polars as pl

from stressnet.config import load_events, silver_root, gold_root
from stressnet.features.basis import label_basis_exceedance
from stressnet.features.panels import save_panel
from stressnet.utils.logging import get_logger
from stressnet.utils.validation import check_no_lookahead

logger = get_logger(__name__)


def load_silver_node(event_id: str, node_id: str) -> pl.DataFrame | None:
    """Load a silver-layer node DataFrame if it exists."""
    for suffix in ["_books.parquet", "_pool_states.parquet", "_flows.parquet"]:
        path = silver_root() / event_id / f"{node_id}{suffix}"
        if path.exists():
            return pl.read_parquet(path)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gold-layer feature panel.")
    parser.add_argument("--event", required=True, help="Event ID from configs/events.yaml")
    parser.add_argument("--grid", type=int, default=60, help="Sampling grid in seconds (default: 60)")
    args = parser.parse_args()

    events = load_events()
    if args.event not in events:
        raise SystemExit(f"Unknown event '{args.event}'")

    from stressnet.graph.nodes import nodes_for_event
    nodes = nodes_for_event(args.event)
    logger.info("Building panel for event '%s' with %d configured nodes", args.event, len(nodes))

    node_frames = {}
    for node in nodes:
        df = load_silver_node(args.event, node.id)
        if df is None:
            logger.warning("No silver data found for node %s; skipping.", node.id)
            continue
        node_frames[node.id] = df
        logger.info("  Loaded node %s: %d rows", node.id, len(df))

    if not node_frames:
        logger.error(
            "No silver data found for any node. Run script 02 first to build books and pools."
        )
        return

    # TODO: compute features per node using stressnet.features.market, .dex, .onchain
    # For now, assemble what we have and add labels
    frames = []
    for node_id, df in node_frames.items():
        df = df.with_columns([
            pl.lit(args.event).alias("event_id"),
            pl.lit(node_id).alias("node_id"),
        ])
        if "basis_vs_usd" in df.columns:
            df = label_basis_exceedance(df)
        frames.append(df)

    panel = pl.concat(frames, how="diagonal")
    logger.info("Panel shape: %d rows × %d cols", panel.height, panel.width)

    # Validation
    feature_cols = [c for c in panel.columns if not c.startswith("label_")]
    label_cols = [c for c in panel.columns if c.startswith("label_")]
    if label_cols:
        check_no_lookahead(panel, feature_cols, label_cols)

    save_panel(panel, args.event)


if __name__ == "__main__":
    main()
