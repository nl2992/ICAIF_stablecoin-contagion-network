"""Build the gold-layer event-time feature panel.

Reads data/silver/ and writes data/gold/dataset_contagion_features_{event}.parquet.
"""

import argparse

import polars as pl

from stressnet.config import load_events, manifests_root, silver_root, gold_root
from stressnet.features.basis import label_basis_exceedance
from stressnet.features.panels import save_panel
from stressnet.graph.nodes import Node
from stressnet.utils.logging import get_logger
from stressnet.utils.manifest import write_manifest_row
from stressnet.utils.time import parse_iso_utc
from stressnet.utils.validation import check_no_lookahead

logger = get_logger(__name__)

GOLD_COLUMNS = [
    "event_id",
    "node_id",
    "layer",
    "asset",
    "venue",
    "tier_nominal",
    "tier_actual",
    "wall_clock_utc",
    "event_time_seconds",
    "mid_price",
    "spread_bps",
    "depth_10bps_bid_usd",
    "depth_10bps_ask_usd",
    "orderbook_imbalance",
    "executable_price_10k_buy",
    "executable_price_10k_sell",
    "basis_vs_usd",
    "reserve_imbalance",
    "implied_pool_price",
    "pool_slippage_10k",
    "exchange_inflow_1h",
    "exchange_outflow_1h",
    "exchange_netflow_1h",
    "mint_burn_net_1h",
    "gas_base_fee_gwei",
    "label_basis_gt10bps",
    "label_basis_gt50bps",
    "label_downstream_gt10bps_1m",
    "label_downstream_gt50bps_5m",
]


def load_silver_node(event_id: str, node_id: str) -> pl.DataFrame | None:
    """Load a silver-layer node DataFrame if it exists."""
    for suffix in ["_books.parquet", "_pool_states.parquet", "_flows.parquet"]:
        path = silver_root() / event_id / f"{node_id}{suffix}"
        if path.exists():
            return pl.read_parquet(path)
    return None


def _add_missing_columns(df: pl.DataFrame) -> pl.DataFrame:
    numeric_cols = {
        "mid_price",
        "spread_bps",
        "depth_10bps_bid_usd",
        "depth_10bps_ask_usd",
        "orderbook_imbalance",
        "executable_price_10k_buy",
        "executable_price_10k_sell",
        "basis_vs_usd",
        "reserve_imbalance",
        "implied_pool_price",
        "pool_slippage_10k",
        "exchange_inflow_1h",
        "exchange_outflow_1h",
        "exchange_netflow_1h",
        "mint_burn_net_1h",
        "gas_base_fee_gwei",
    }
    for col in numeric_cols:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
    return df


def _actual_tier(event_id: str, node_id: str, fallback: str) -> str:
    manifest_path = manifests_root() / f"manifest_{event_id}.csv"
    if not manifest_path.exists():
        return fallback
    manifest = pl.read_csv(manifest_path)
    match = manifest.filter(pl.col("node_id") == node_id)
    if match.height == 0:
        return fallback
    return match["source_tier_actual"][-1]


def _node_features(event_id: str, node: Node, df: pl.DataFrame, shock_onset) -> pl.DataFrame:
    df = _add_missing_columns(df)
    if "basis_vs_usd" not in df.columns or df["basis_vs_usd"].null_count() == df.height:
        price_col = "mid_price" if node.layer == "CEX" else "implied_pool_price"
        if price_col in df.columns:
            df = df.with_columns(
                pl.when(pl.col(price_col) > 0)
                .then(pl.col(price_col).log())
                .otherwise(None)
                .alias("basis_vs_usd")
            )

    return df.with_columns(
        pl.lit(event_id).alias("event_id"),
        pl.lit(node.id).alias("node_id"),
        pl.lit(node.layer).alias("layer"),
        pl.lit(node.asset).alias("asset"),
        pl.lit(node.venue).alias("venue"),
        pl.lit(node.tier).alias("tier_nominal"),
        pl.lit(_actual_tier(event_id, node.id, node.tier)).alias("tier_actual"),
        (pl.col("wall_clock_utc") - pl.lit(shock_onset)).dt.total_seconds().alias(
            "event_time_seconds"
        ),
    )


def _add_downstream_labels(panel: pl.DataFrame, grid_seconds: int = 60) -> pl.DataFrame:
    """Add forward-looking downstream stress labels.

    Args:
        panel: Gold feature panel with wall_clock_utc and basis_vs_usd columns.
        grid_seconds: Sampling grid interval in seconds (from --grid argument).
            Used to convert horizon_seconds to a row-count shift so the label
            is correct regardless of whether the panel was built at 1s, 5s, or 60s.
    """
    if "basis_vs_usd" not in panel.columns:
        return panel

    base = panel.sort(["node_id", "wall_clock_utc"])
    threshold_10 = 10.0 / 10_000
    threshold_50 = 50.0 / 10_000

    labels = []
    for horizon_seconds, threshold, name in [
        (60, threshold_10, "label_downstream_gt10bps_1m"),
        (300, threshold_50, "label_downstream_gt50bps_5m"),
    ]:
        shift_steps = max(1, int(round(horizon_seconds / grid_seconds)))
        node_frames = []
        for node_id in base["node_id"].unique().to_list():
            this_node = base.filter(pl.col("node_id") == node_id)
            other = (
                base.filter(pl.col("node_id") != node_id)
                .group_by("wall_clock_utc")
                .agg((pl.col("basis_vs_usd").abs().max() > threshold).alias("_any_stress"))
                .sort("wall_clock_utc")
                .with_columns(
                    pl.col("_any_stress")
                    .shift(-shift_steps)
                    .fill_null(False)
                    .alias(name)
                )
                .select(["wall_clock_utc", name])
            )
            node_frames.append(
                this_node.select(["event_id", "node_id", "wall_clock_utc"]).join(
                    other, on="wall_clock_utc", how="left"
                )
            )
        labels.append(pl.concat(node_frames, how="vertical"))

    result = base
    for label_df in labels:
        result = result.join(label_df, on=["event_id", "node_id", "wall_clock_utc"], how="left")
    return result.with_columns(
        pl.col("label_downstream_gt10bps_1m").fill_null(False),
        pl.col("label_downstream_gt50bps_5m").fill_null(False),
    )


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

    event_cfg = events[args.event]
    shock_onset = parse_iso_utc(event_cfg["shock_onset_utc"])

    node_frames = {}
    for node in nodes:
        df = load_silver_node(args.event, node.id)
        if df is None:
            logger.warning("No silver data found for node %s; skipping.", node.id)
            continue
        node_frames[node.id] = _node_features(args.event, node, df, shock_onset)
        logger.info("  Loaded node %s: %d rows", node.id, len(df))

    if not node_frames:
        logger.error(
            "No silver data found for any node. Run script 02 first to build books and pools."
        )
        return

    panel = pl.concat(list(node_frames.values()), how="diagonal")
    if "basis_vs_usd" in panel.columns:
        panel = label_basis_exceedance(panel)
    panel = _add_downstream_labels(panel, grid_seconds=args.grid)
    for col in GOLD_COLUMNS:
        if col not in panel.columns:
            panel = panel.with_columns(pl.lit(None).alias(col))
    panel = panel.select(GOLD_COLUMNS + [c for c in panel.columns if c not in GOLD_COLUMNS])
    logger.info("Panel shape: %d rows × %d cols", panel.height, panel.width)

    # Validation
    feature_cols = [c for c in panel.columns if not c.startswith("label_")]
    label_cols = [c for c in panel.columns if c.startswith("label_")]
    if label_cols:
        check_no_lookahead(panel, feature_cols, label_cols)

    save_panel(panel, args.event)
    write_manifest_row(
        event_id=args.event,
        node_id="__event_panel__",
        source_name="gold_panel_builder",
        source_tier_nominal="mixed",
        source_tier_actual="fixture_non_empirical"
        if "fixture_non_empirical" in panel["tier_actual"].unique().to_list()
        else "mixed",
        start_utc=f"{event_cfg['analysis_window_utc'][0]}T00:00:00Z",
        end_utc=f"{event_cfg['analysis_window_utc'][1]}T23:59:59Z",
        file_path=gold_root() / f"dataset_contagion_features_{args.event}.parquet",
        row_count=panel.height,
        notes="Gold event feature panel assembled from silver node states.",
        layer="gold_panel",
        file_stage="gold",
        url_or_query="silver://event_nodes",
    )


if __name__ == "__main__":
    main()
