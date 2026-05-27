"""Reconstruct standardized silver node states from bronze artefacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from stressnet.config import bronze_root, load_events, silver_root
from stressnet.config import manifests_root
from stressnet.graph.nodes import Node, nodes_for_event
from stressnet.utils.logging import get_logger
from stressnet.utils.manifest import build_node_coverage_table, write_manifest_row
from stressnet.utils.time import parse_iso_utc

logger = get_logger(__name__)


def _event_bounds(event_id: str) -> tuple[str, str]:
    cfg = load_events()[event_id]
    return f"{cfg['analysis_window_utc'][0]}T00:00:00Z", f"{cfg['analysis_window_utc'][1]}T23:59:59Z"


def _bronze_input(event_id: str, node: Node) -> tuple[Path, str] | None:
    candidates = [
        (bronze_root() / event_id / f"{node.id}_books.parquet", "books"),
        (bronze_root() / event_id / f"{node.id}_pool_events.parquet", "pool_states"),
        (bronze_root() / event_id / f"{node.id}_flows.parquet", "flows"),
    ]
    for path, kind in candidates:
        if path.exists():
            return path, kind
    return None


def _silver_name(node: Node, kind: str) -> str:
    if node.layer == "CEX":
        return f"{node.id}_books.parquet"
    if node.layer == "DEX":
        return f"{node.id}_pool_states.parquet"
    return f"{node.id}_flows.parquet"


def _standardize(df: pl.DataFrame, node: Node) -> pl.DataFrame:
    if "wall_clock_utc" not in df.columns:
        raise ValueError(f"Bronze data for {node.id} lacks wall_clock_utc")
    result = df.sort("wall_clock_utc")

    required_defaults = {
        "mid_price": None,
        "spread_bps": None,
        "depth_10bps_bid_usd": None,
        "depth_10bps_ask_usd": None,
        "orderbook_imbalance": None,
        "executable_price_10k_buy": None,
        "executable_price_10k_sell": None,
        "basis_vs_usd": None,
        "reserve_imbalance": None,
        "implied_pool_price": None,
        "pool_slippage_10k": None,
        "exchange_inflow_1h": None,
        "exchange_outflow_1h": None,
        "exchange_netflow_1h": None,
        "mint_burn_net_1h": None,
        "gas_base_fee_gwei": None,
    }
    for col, default in required_defaults.items():
        if col not in result.columns:
            result = result.with_columns(pl.lit(default, dtype=pl.Float64).alias(col))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct silver data for one event.")
    parser.add_argument("--event", required=True, help="Event ID from configs/events.yaml")
    parser.add_argument("--nodes", nargs="+", default=None)
    args = parser.parse_args()

    events = load_events()
    if args.event not in events:
        raise SystemExit(f"Unknown event '{args.event}'")

    nodes = nodes_for_event(args.event)
    if args.nodes:
        requested = set(args.nodes)
        nodes = [node for node in nodes if node.id in requested]

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
        silver = _standardize(pl.read_parquet(in_path), node)
        out_path = out_dir / _silver_name(node, kind)
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
            notes=f"Standardized silver output from {in_path}.",
            layer=node.layer,
            file_stage="silver",
            url_or_query=str(in_path),
        )
        logger.info("Wrote silver %s (%d rows)", out_path, silver.height)
        wrote += 1

    if wrote == 0:
        raise SystemExit("No silver files written. Run scripts/01_ingest_raw_data.py first.")

    coverage_path = build_node_coverage_table()
    if coverage_path:
        logger.info("Updated coverage table: %s", coverage_path)


if __name__ == "__main__":
    main()
