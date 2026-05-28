"""Reconstruct standardized silver node states from bronze artefacts.

Calls stressnet.reconstruct.silver.standardize_bronze() to detect the bronze
format (klines, bookTicker, pool_events, flows, fixture) and apply the
appropriate feature transform.  Then fills any remaining standard columns with
null so that the gold panel builder always sees a consistent schema.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median

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

    required_quality = {
        "depth_source": "precomputed_or_unknown",
        "executable_price_source": "precomputed_or_unknown",
        "microstructure_quality": "precomputed_or_unknown",
    }
    for col, value in required_quality.items():
        if col not in result.columns:
            result = result.with_columns(pl.lit(value).alias(col))
    if "is_executable_bookwalk" not in result.columns:
        result = result.with_columns(pl.lit(False).alias("is_executable_bookwalk"))
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


def _coverage_diagnostics(
    df: pl.DataFrame,
    start_utc: str,
    end_utc: str,
) -> dict[str, float | int | None]:
    """Compute simple manifest diagnostics from a silver node state table."""
    diagnostics: dict[str, float | int | None] = {
        "coverage_pct": None,
        "sequence_gap_count": None,
        "gap_rate": None,
        "resync_count": None,
        "clock_offset_ms": None,
    }
    if df.height == 0 or "wall_clock_utc" not in df.columns:
        diagnostics["coverage_pct"] = 0.0
        return diagnostics

    times = (
        df.select(pl.col("wall_clock_utc").cast(pl.Datetime("us", "UTC")))
        .sort("wall_clock_utc")["wall_clock_utc"]
        .to_list()
    )
    deltas = [
        (right - left).total_seconds()
        for left, right in zip(times, times[1:])
        if (right - left).total_seconds() > 0
    ]
    if deltas:
        cadence = max(1.0, float(median(deltas)))
        window_seconds = max(0.0, (parse_iso_utc(end_utc) - parse_iso_utc(start_utc)).total_seconds())
        expected_rows = max(1.0, window_seconds / cadence + 1.0)
        diagnostics["coverage_pct"] = min(100.0, 100.0 * df.height / expected_rows)

    if "sequence_gap_flag" in df.columns:
        gaps = int(df["sequence_gap_flag"].fill_null(False).sum())
        diagnostics["sequence_gap_count"] = gaps
        diagnostics["gap_rate"] = gaps / max(df.height, 1)
    elif "source_sequence" in df.columns and df.height > 1:
        seq = [value for value in df["source_sequence"].to_list() if value is not None]
        if len(seq) > 1:
            gaps = sum(1 for left, right in zip(seq, seq[1:]) if int(right) - int(left) > 1)
            diagnostics["sequence_gap_count"] = gaps
            diagnostics["gap_rate"] = gaps / max(len(seq) - 1, 1)

    if "resync_flag" in df.columns:
        diagnostics["resync_count"] = int(df["resync_flag"].fill_null(False).sum())
    if "clock_offset_ms" in df.columns:
        values = [abs(float(value)) for value in df["clock_offset_ms"].drop_nulls().to_list()]
        if values:
            diagnostics["clock_offset_ms"] = max(values)
    return diagnostics


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
        diagnostics = _coverage_diagnostics(silver, start_str, end_str)
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
            **diagnostics,
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
