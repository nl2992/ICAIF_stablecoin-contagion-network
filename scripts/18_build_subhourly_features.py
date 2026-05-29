"""Re-fetch Curve pool events at finer temporal resolution and rebuild feature panels.

Tests the resolution-dependence hypothesis: if Terra/LUNA fails the hourly
lead-lag gate, does it pass at 5-minute or 15-minute resolution?

Either outcome is scientifically useful:
  - Significance at 300s → second paper-claimable A/A result; richer story
  - Still non-significant → Terra collapse was too fast for AMM-level detection

Output
------
For each (event, pool_node, grid_seconds):
    data/bronze/{event}/{node_id}_pool_events_{grid_seconds}s.parquet
    results/paper/tables/table_subhourly_leadlag_{event}.csv

Usage
-----
    python scripts/18_build_subhourly_features.py --event terra_luna_2022
    python scripts/18_build_subhourly_features.py --all-events
    python scripts/18_build_subhourly_features.py --event terra_luna_2022 --grids 300 900 3600
    python scripts/18_build_subhourly_features.py --dry-run
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import polars as pl

from stressnet.config import bronze_root, load_events, results_root
from stressnet.data.curve import CURVE_3POOL_ADDRESS, CURVE_CRVUSD_USDT, CURVE_UST_WORMHOLE
from stressnet.data.curve import ingest_curve_pool_events
from stressnet.data.etherscan import get_block_number_by_timestamp
from stressnet.models.leadlag import compute_leadlag_table
from stressnet.utils.logging import get_logger
from stressnet.utils.time import parse_iso_utc

logger = get_logger(__name__)

# Pool nodes to re-ingest at sub-hourly resolution, per event
_SUBHOURLY_NODES: dict[str, list[tuple[str, str]]] = {
    "terra_luna_2022": [
        ("curve_3pool",        CURVE_3POOL_ADDRESS),
        ("curve_ust_wormhole", CURVE_UST_WORMHOLE),
    ],
    "usdt_curve_2023": [
        ("curve_3pool",        CURVE_3POOL_ADDRESS),
        ("curve_crvusd_usdt",  CURVE_CRVUSD_USDT),
    ],
    "usdc_svb_2023": [
        ("curve_3pool",        CURVE_3POOL_ADDRESS),
    ],
}

# Default grids to test (seconds)
DEFAULT_GRIDS = [300, 900, 3600]


def _fetch_block_range(event_id: str, events_cfg: dict) -> tuple[int, int]:
    """Return (start_block, end_block) for an event's analysis window."""
    cfg = events_cfg[event_id]
    start_str = f"{cfg['analysis_window_utc'][0]}T00:00:00Z"
    end_str   = f"{cfg['analysis_window_utc'][1]}T23:59:59Z"
    start_ts  = int(parse_iso_utc(start_str).timestamp())
    end_ts    = int(parse_iso_utc(end_str).timestamp())

    logger.info("Resolving block numbers for %s…", event_id)
    start_block = get_block_number_by_timestamp(start_ts, "before")
    end_block   = get_block_number_by_timestamp(end_ts, "after")

    if start_block is None or end_block is None:
        raise RuntimeError(
            f"Could not resolve blocks for {event_id}. "
            "Check ETHERSCAN_API_KEY and network."
        )
    logger.info("Block range: %d → %d", start_block, end_block)
    return start_block, end_block


def build_subhourly_for_event(
    event_id: str,
    grids: list[int],
    dry_run: bool = False,
) -> dict[int, pl.DataFrame]:
    """Re-ingest pool events and run lead-lag at multiple resolutions.

    Returns:
        Mapping of grid_seconds → lead-lag result DataFrame.
    """
    if not os.environ.get("ETHERSCAN_API_KEY"):
        logger.error("ETHERSCAN_API_KEY not set. Cannot fetch raw events.")
        return {}

    nodes = _SUBHOURLY_NODES.get(event_id)
    if not nodes:
        logger.warning("No sub-hourly nodes configured for %s", event_id)
        return {}

    events_cfg = load_events()
    if event_id not in events_cfg:
        raise SystemExit(f"Unknown event '{event_id}'")

    if dry_run:
        logger.info("[DRY RUN] Would re-ingest %d nodes for %s at grids %s",
                    len(nodes), event_id, grids)
        return {}

    start_block, end_block = _fetch_block_range(event_id, events_cfg)

    results: dict[int, pl.DataFrame] = {}

    for grid_seconds in grids:
        label = f"{grid_seconds}s"
        logger.info("=== %s grid %s ===", event_id, label)
        bronze_event = bronze_root() / event_id

        frames: dict[str, pl.DataFrame] = {}
        for node_id, contract_address in nodes:
            out_path, tier = ingest_curve_pool_events(
                contract_address=contract_address,
                start_block=start_block,
                end_block=end_block,
                out_dir=bronze_event,
                event_id=event_id,
                node_id=f"{node_id}_{label}",
                grid_seconds=grid_seconds,
                save_raw=(grid_seconds == min(grids)),   # save raw once at finest grid
            )
            if out_path is None:
                logger.warning("Skipped %s at %s (fetch failed or API unavailable)", node_id, label)
                continue

            df = pl.read_parquet(out_path)
            frames[node_id] = df

        if len(frames) < 2:
            logger.warning("Need ≥2 nodes for lead-lag; only %d available", len(frames))
            continue

        # Join node frames into a panel
        node_ids = list(frames.keys())
        panel_rows = []
        for nid, df in frames.items():
            panel_rows.append(df.with_columns(
                pl.lit(nid).alias("node_id"),
                pl.lit(tier if tier else "A").alias("tier_actual"),
            ))
        panel = pl.concat(panel_rows, how="diagonal_relaxed")
        min_ts = panel["wall_clock_utc"].min()
        panel = panel.with_columns(
            (pl.col("wall_clock_utc") - pl.lit(min_ts)).dt.total_seconds().alias("event_time_seconds")
        )
        node_pairs = [(i, j) for i in node_ids for j in node_ids if i != j]

        # Run lead-lag between all pairs
        try:
            ll_df = compute_leadlag_table(
                panel=panel,
                node_pairs=node_pairs,
                feature_col="usdc_net_sold_1h",
                grid_seconds=grid_seconds,
                max_lag=12,         # 12 grid steps (1h at 5-min, 12h at 1h)
                n_reps=500,
                ts_col="event_time_seconds",
                max_staleness_seconds=grid_seconds,
            )
        except Exception as exc:
            logger.error("Lead-lag failed at grid=%s: %s", label, exc)
            continue

        # Annotate with resolution
        ll_df = ll_df.with_columns(
            pl.lit(grid_seconds).alias("grid_seconds"),
            pl.lit(event_id).alias("event_id"),
        )
        results[grid_seconds] = ll_df
        logger.info(
            "grid=%s: %d pairs, %d statistically significant (Bonferroni p<0.05)",
            label,
            ll_df.height,
            ll_df.filter(pl.col("p_bonferroni") < 0.05).height
            if "p_bonferroni" in ll_df.columns else 0,
        )

    return results


def save_results(
    results: dict[int, pl.DataFrame],
    event_id: str,
) -> Path | None:
    """Write per-event multi-resolution lead-lag table."""
    if not results:
        return None

    combined = pl.concat(list(results.values()), how="diagonal_relaxed")

    out_dir = results_root() / "paper" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"table_subhourly_leadlag_{event_id}.csv"
    combined.write_csv(out_path)
    logger.info("Saved sub-hourly lead-lag table: %s  (%d rows)", out_path.name, combined.height)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild Curve feature panels at sub-hourly resolution."
    )
    parser.add_argument("--event", default=None, help="Single event ID.")
    parser.add_argument("--all-events", action="store_true",
                        help="Run for all configured events.")
    parser.add_argument("--grids", nargs="+", type=int, default=DEFAULT_GRIDS,
                        help="Grid sizes in seconds (default: 300 900 3600).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned fetches without executing.")
    args = parser.parse_args()

    if not args.event and not args.all_events:
        raise SystemExit("Specify --event <id> or --all-events")

    events = list(_SUBHOURLY_NODES.keys()) if args.all_events else [args.event]

    for event_id in events:
        logger.info("Processing %s…", event_id)
        results = build_subhourly_for_event(event_id, args.grids, dry_run=args.dry_run)
        if results and not args.dry_run:
            save_results(results, event_id)

    logger.info("Done.")


if __name__ == "__main__":
    main()
