"""Reconstruct standardized silver node states from bronze artefacts.

Calls stressnet.reconstruct.silver.standardize_bronze() to detect the bronze
format (klines, bookTicker, pool_events, flows, fixture) and apply the
appropriate feature transform.  Then fills any remaining standard columns with
null so that the gold panel builder always sees a consistent schema.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from stressnet.config import bronze_root, load_events, manifests_root, silver_root
from stressnet.graph.nodes import Node, nodes_for_event
from stressnet.reconstruct.silver import standardize_bronze
from stressnet.utils.logging import get_logger
from stressnet.utils.manifest import build_node_coverage_table, write_manifest_row
from stressnet.utils.time import parse_iso_utc

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _event_bounds(event_id: str) -> tuple[str, str]:
    cfg = load_events()[event_id]
    return f"{cfg['analysis_window_utc'][0]}T00:00:00Z", f"{cfg['analysis_window_utc'][1]}T23:59:59Z"


def _bronze_input(event_id: str, node: Node) -> tuple[Path, str] | None:
    """Find the bronze parquet for this node; return (path, kind) or None."""
    candidates = [
        (bronze_root() / event_id / f"{node.id}_books.parquet",       "books"),
        (bronze_root() / event_id / f"{node.id}_pool_events.parquet", "pool_states"),
        (bronze_root() / event_id / f"{node.id}_flows.parquet",       "flows"),
    ]
    for path, kind in candidates:
        if path.exists():
            return path, kind
    return None


def _silver_name(node: Node) -> str:
    if node.layer == "CEX":
        return f"{node.id}_books.parquet"
    if node.layer == "DEX":
        return f"{node.id}_pool_states.parquet"
    return f"{node.id}_flows.parquet"


def _standardize(df: pl.DataFrame, node: Node) -> pl.DataFrame:
    """Apply silver transforms then fill missing standard columns with NULL.

    1. Validates wall_clock_utc presence.
    2. Calls standardize_bronze() to detect format and apply transforms.
    3. Ensures all standard silver columns exist (null-fills if absent).
    4. Sorts by wall_clock_utc.
    """
    if "wall_clock_utc" not in df.columns:
        raise ValueError(f"Bronze data for {node.id} lacks wall_clock_utc")

    result = standardize_bronze(df)
    result = result.sort("wall_clock_utc")

    required_nulls = {
        "mid_price":                  pl.Float64,
        "spread_bps":                 pl.Float64,
        "depth_10bps_bid_usd":        pl.Float64,
        "depth_10bps_ask_usd":        pl.Float64,
        "orderbook_imbalance":        pl.Float64,
        "executable_price_10k_buy":   pl.Float64,
        "executable_price_10k_sell":  pl.Float64,
        "basis_vs_usd":               pl.Float64,
        "reserve_imbalance":          pl.Float64,
        "implied_pool_price":         pl.Float64,
        "pool_slippage_10k":          pl.Float64,
        "exchange_inflow_1h":         pl.Float64,
        "exchange_outflow_1h":        pl.Float64,
        "exchange_netflow_1h":        pl.Float64,
        "mint_burn_net_1h":           pl.Float64,
        "gas_base_fee_gwei":          pl.Float64,
    }
    for col, dtype in required_nulls.items():
        if col not in result.columns:
            result = result.with_columns(pl.lit(None, dtype=dtype).alias(col))
    return result


def _actual_tier_from_manifest(event_id: str, path: Path, fallback: str) -> str:
    manifest_path = manifests_root() / f"manifest_{event_id}.csv"
    if not manifest_path.exists():
        return fallback
    manifest = pl.read_csv(manifest_path)
    match = manifest.filter(pl.col("file_path") == str(path))
    if match.height == 0:
        return fallback
    return match["source_tier_actual"][0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct silver data for one event.")
    parser.add_argument("--event",  required=True)
    parser.add_argument("--nodes",  nargs="+", default=None)
    args = parser.parse_args()

    events = load_events()
    if args.event not in events:
        raise SystemExit(f"Unknown event '{args.event}'")

    nodes = nodes_for_event(args.event)
    if args.nodes:
        requested = set(args.nodes)
        nodes = [n for n in nodes if n.id in requested]

    out_dir = silver_root() / args.event
    out_dir.mkdir(parents=True, exist_ok=True)
    start_str, end_str = _event_bounds(args.event)

    wrote = 0
    for node in nodes:
        located = _bronze_input(args.event, node)
        if located is None:
            logger.warning("No bronze input found for %s; skipping.", node.id)
            continue
        in_path, kind = located

        try:
            bronze_df = pl.read_parquet(in_path)
            silver = _standardize(bronze_df, node)
        except Exception as exc:
            logger.error("Silver reconstruction failed for %s: %s", node.id, exc)
            continue

        out_path = out_dir / _silver_name(node)
        silver.write_parquet(out_path)

        actual_tier = _actual_tier_from_manifest(args.event, in_path, node.tier)
        write_manifest_row(
            event_id=args.event,
            node_id=node.id,
            source_name=f"silver_reconstruction_from_{kind}",
            source_tier_nominal=node.tier,
            source_tier_actual=actual_tier,
            start_utc=start_str,
            end_utc=end_str,
            file_path=out_path,
            row_count=silver.height,
            notes=f"Standardized silver output from {in_path.name}.",
            layer=node.layer,
            file_stage="silver",
            url_or_query=str(in_path),
        )
        logger.info(
            "Silver %-35s  rows=%d  tier=%s",
            node.id, silver.height, actual_tier,
        )
        wrote += 1

    if wrote == 0:
        raise SystemExit("No silver files written. Run scripts/01_ingest_raw_data.py first.")

    coverage_path = build_node_coverage_table()
    if coverage_path:
        logger.info("Updated coverage table: %s", coverage_path)


if __name__ == "__main__":
    main()
