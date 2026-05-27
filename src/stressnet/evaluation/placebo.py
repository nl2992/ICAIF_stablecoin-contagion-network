"""Placebo event construction for robustness testing.

Placebo events are matched high-volatility, non-stress-event periods used to
verify that contagion estimates are not driven by generic market dynamics.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def build_placebo_windows(
    event_window_start: datetime,
    event_window_end: datetime,
    n_placebos: int = 5,
    offset_days: list[int] | None = None,
) -> list[tuple[datetime, datetime]]:
    """Return placebo windows of the same duration, offset by fixed day counts.

    Args:
        event_window_start: Start of the real event window.
        event_window_end: End of the real event window.
        n_placebos: Number of placebo windows to generate.
        offset_days: Day offsets relative to the event start.
                     Defaults to [-28, -21, -14, -7, +14] (4-week pre-event + 2-week post).

    Returns:
        List of (start, end) tuples for placebo windows.
    """
    if offset_days is None:
        offset_days = [-28, -21, -14, -7, 14]

    duration = event_window_end - event_window_start
    placebos = []
    for offset in offset_days[:n_placebos]:
        start = event_window_start + timedelta(days=offset)
        end = start + duration
        placebos.append((start, end))

    return placebos


def tag_placebo_rows(
    panel: pl.DataFrame,
    placebo_windows: list[tuple[datetime, datetime]],
    ts_col: str = "wall_clock_utc",
) -> pl.DataFrame:
    """Add a 'placebo_id' column indicating which placebo window each row falls in.

    Rows not in any placebo window get placebo_id = None.
    """
    result = panel.with_columns(pl.lit(None).cast(pl.Utf8).alias("placebo_id"))
    for i, (start, end) in enumerate(placebo_windows):
        result = result.with_columns(
            pl.when(
                (pl.col(ts_col) >= start) & (pl.col(ts_col) < end)
            )
            .then(pl.lit(f"placebo_{i+1}"))
            .otherwise(pl.col("placebo_id"))
            .alias("placebo_id")
        )
    return result
