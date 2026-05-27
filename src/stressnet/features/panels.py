"""Build the final event-time aligned feature panel (gold layer)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from stressnet.config import gold_root, load_events
from stressnet.features.basis import label_basis_exceedance
from stressnet.utils.logging import get_logger
from stressnet.utils.time import event_time_seconds, parse_iso_utc
from stressnet.utils.validation import check_no_lookahead

logger = get_logger(__name__)


def build_event_panel(
    event_id: str,
    node_features: dict[str, pl.DataFrame],
    label_horizon_minutes: int = 1,
) -> pl.DataFrame:
    """Concatenate per-node feature DataFrames into an event-time panel.

    Args:
        event_id: Event ID from configs/events.yaml (e.g. 'usdc_svb_2023').
        node_features: Dict mapping node_id → DataFrame with feature columns.
        label_horizon_minutes: Forward-look horizon for downstream labels.

    Returns:
        Gold-layer DataFrame with all nodes, aligned on wall_clock_utc.
    """
    events = load_events()
    if event_id not in events:
        raise ValueError(f"Unknown event_id '{event_id}'")

    onset_str = events[event_id].get("shock_onset_utc")
    onset_utc = parse_iso_utc(onset_str) if onset_str else None

    frames = []
    for node_id, df in node_features.items():
        df = df.with_columns([
            pl.lit(event_id).alias("event_id"),
            pl.lit(node_id).alias("node_id"),
        ])
        if onset_utc is not None and "wall_clock_utc" in df.columns:
            df = df.with_columns(
                pl.col("wall_clock_utc")
                .map_elements(
                    lambda ts: (ts - onset_utc).total_seconds()
                    if isinstance(ts, datetime) else None,
                    return_dtype=pl.Float64,
                )
                .alias("event_time_seconds")
            )
        frames.append(df)

    panel = pl.concat(frames, how="diagonal")

    # Add downstream stress labels if basis_vs_usd is present
    if "basis_vs_usd" in panel.columns:
        panel = label_basis_exceedance(panel)

    return panel


def save_panel(panel: pl.DataFrame, event_id: str) -> Path:
    """Write the gold-layer panel to Parquet."""
    out_path = gold_root() / f"dataset_contagion_features_{event_id}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(out_path)
    logger.info("Saved panel: %d rows × %d cols → %s", len(panel), len(panel.columns), out_path)
    return out_path
