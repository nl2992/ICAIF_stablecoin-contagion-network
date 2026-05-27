"""Ingest raw/bronze data for one event.

This script is intentionally provenance-first. If no local raw source files are
present, it can create deterministic fixture bronze files so the pipeline can be
tested end-to-end; those files are marked ``fixture_non_empirical`` in the
manifest and must not be used as paper evidence.
"""

from __future__ import annotations

import argparse
import math
from datetime import timedelta
from pathlib import Path

import polars as pl

from stressnet.config import bronze_root, load_events
from stressnet.graph.nodes import Node, nodes_for_event
from stressnet.utils.logging import get_logger
from stressnet.utils.manifest import build_node_coverage_table, write_manifest_row
from stressnet.utils.time import parse_iso_utc

logger = get_logger(__name__)


def _event_bounds(event_id: str) -> tuple[str, str, object, object]:
    cfg = load_events()[event_id]
    start_str = f"{cfg['analysis_window_utc'][0]}T00:00:00Z"
    end_str = f"{cfg['analysis_window_utc'][1]}T23:59:59Z"
    return start_str, end_str, parse_iso_utc(start_str), parse_iso_utc(end_str)


def _fixture_times(start, hours: int = 72, step_minutes: int = 5) -> list:
    n = int(hours * 60 / step_minutes)
    return [start + timedelta(minutes=step_minutes * i) for i in range(n)]


def _stress_shape(i: int, n: int, node_shift: int) -> float:
    center = int(n * 0.38) + node_shift
    width = max(n / 16, 1)
    return math.exp(-((i - center) ** 2) / (2 * width**2))


def _market_fixture(event_id: str, node: Node, start) -> pl.DataFrame:
    ts = _fixture_times(start)
    n = len(ts)
    shift = sum(ord(c) for c in node.id) % 18
    severity = 0.006 if node.asset == "USDC" else 0.0015
    rows = []
    for i, t in enumerate(ts):
        pulse = _stress_shape(i, n, shift)
        wobble = math.sin(i / 11 + shift) * 0.00008
        mid = 1.0 - severity * pulse + wobble
        spread = 1.5 + 18 * pulse + (shift % 4)
        depth = max(50_000, 1_500_000 * (1 - 0.72 * pulse))
        rows.append(
            {
                "wall_clock_utc": t,
                "mid_price": mid,
                "spread_bps": spread,
                "depth_10bps_bid_usd": depth * (1 - 0.08 * pulse),
                "depth_10bps_ask_usd": depth * (1 + 0.05 * pulse),
                "orderbook_imbalance": -0.45 * pulse + math.sin(i / 17) * 0.03,
                "executable_price_10k_buy": mid * (1 + spread / 20_000),
                "executable_price_10k_sell": mid * (1 - spread / 20_000),
                "basis_vs_usd": math.log(mid),
            }
        )
    return pl.DataFrame(rows)


def _pool_fixture(event_id: str, node: Node, start) -> pl.DataFrame:
    ts = _fixture_times(start)
    n = len(ts)
    shift = 8 if "uniswap" in node.id else 14
    rows = []
    for i, t in enumerate(ts):
        pulse = _stress_shape(i, n, shift)
        implied = 1.0 - 0.0045 * pulse + math.sin(i / 19) * 0.00005
        rows.append(
            {
                "wall_clock_utc": t,
                "reserve_imbalance": 0.55 * pulse,
                "implied_pool_price": implied,
                "pool_slippage_10k": 0.8 + 28 * pulse,
                "virtual_price": 1.0 + i * 1e-8,
                "lp_supply": 400_000_000 * (1 - 0.05 * pulse),
                "basis_vs_usd": math.log(implied),
            }
        )
    return pl.DataFrame(rows)


def _flow_fixture(event_id: str, node: Node, start) -> pl.DataFrame:
    ts = _fixture_times(start, step_minutes=60)
    n = len(ts)
    shift = sum(ord(c) for c in node.id) % 8
    rows = []
    for i, t in enumerate(ts):
        pulse = _stress_shape(i, n, shift)
        inflow = 3_000_000 + 50_000_000 * pulse
        outflow = 2_500_000 + 25_000_000 * max(0.0, pulse - 0.15)
        mint_burn = -35_000_000 * pulse if "mint_burn" in node.layer else None
        rows.append(
            {
                "wall_clock_utc": t,
                "exchange_inflow_1h": inflow if "flow" in node.layer else None,
                "exchange_outflow_1h": outflow if "flow" in node.layer else None,
                "exchange_netflow_1h": inflow - outflow if "flow" in node.layer else None,
                "mint_burn_net_1h": mint_burn,
                "gas_base_fee_gwei": 22 + 70 * pulse,
                "basis_vs_usd": -0.0008 * pulse,
            }
        )
    return pl.DataFrame(rows)


def _fixture_for_node(event_id: str, node: Node, start) -> tuple[str, pl.DataFrame]:
    if node.layer == "CEX":
        return "books", _market_fixture(event_id, node, start)
    if node.layer == "DEX":
        return "pool_events", _pool_fixture(event_id, node, start)
    return "flows", _flow_fixture(event_id, node, start)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest raw/bronze data for one event.")
    parser.add_argument("--event", required=True, help="Event ID from configs/events.yaml")
    parser.add_argument("--nodes", nargs="+", default=None, help="Optional node subset")
    parser.add_argument(
        "--no-fixture",
        action="store_true",
        help="Fail instead of generating fixture bronze files when raw data is absent.",
    )
    args = parser.parse_args()

    events = load_events()
    if args.event not in events:
        raise SystemExit(f"Unknown event '{args.event}'. Available: {list(events)}")

    start_str, end_str, start_utc, _ = _event_bounds(args.event)
    nodes = nodes_for_event(args.event)
    if args.nodes:
        requested = set(args.nodes)
        nodes = [node for node in nodes if node.id in requested]
    if not nodes:
        raise SystemExit("No configured nodes selected.")

    out_dir = bronze_root() / args.event
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.warning(
        "No live vendor/API ingestion is configured. Writing fixture bronze files marked "
        "fixture_non_empirical unless --no-fixture is supplied."
    )
    if args.no_fixture:
        raise SystemExit("No raw inputs found and fixture fallback is disabled.")

    for node in nodes:
        artefact_type, df = _fixture_for_node(args.event, node, start_utc)
        out_path = out_dir / f"{node.id}_{artefact_type}.parquet"
        df.write_parquet(out_path)
        write_manifest_row(
            event_id=args.event,
            node_id=node.id,
            source_name="deterministic_pipeline_fixture",
            source_tier_nominal=node.tier,
            source_tier_actual="fixture_non_empirical",
            start_utc=start_str,
            end_utc=end_str,
            file_path=out_path,
            row_count=df.height,
            notes="Generated fixture for pipeline validation only; not empirical evidence.",
        )
        logger.info("Wrote bronze fixture %s (%d rows)", out_path, df.height)

    coverage_path = build_node_coverage_table()
    if coverage_path:
        logger.info("Updated coverage table: %s", coverage_path)


if __name__ == "__main__":
    main()
