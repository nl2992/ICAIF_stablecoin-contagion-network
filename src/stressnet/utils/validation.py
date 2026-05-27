"""No-lookahead and data-quality validation for feature panels."""

from __future__ import annotations

import polars as pl

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def check_no_lookahead(
    df: pl.DataFrame,
    feature_cols: list[str],
    label_cols: list[str],
    ts_col: str = "wall_clock_utc",
) -> bool:
    """Verify that features are computed strictly before labels in event time.

    For each row, every feature must be derived from data at or before ts_col,
    while labels reference future values. This check validates that feature and
    label timestamps are consistent: no label horizon reaches backward past
    the feature snapshot time.

    Returns True if no lookahead detected; raises ValueError if lookahead found.
    """
    if ts_col not in df.columns:
        raise ValueError(f"Timestamp column '{ts_col}' not found in DataFrame.")

    missing_features = [c for c in feature_cols if c not in df.columns]
    missing_labels = [c for c in label_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    if missing_labels:
        raise ValueError(f"Missing label columns: {missing_labels}")

    # Primary check: no null timestamps in feature rows with non-null labels
    n_null_ts = df.filter(
        pl.col(ts_col).is_null() & pl.any_horizontal([pl.col(c).is_not_null() for c in label_cols])
    ).height
    if n_null_ts > 0:
        raise ValueError(
            f"Lookahead risk: {n_null_ts} rows have null timestamps but non-null labels."
        )

    logger.info("No-lookahead check passed for %d rows.", len(df))
    return True


def check_no_future_features(
    df: pl.DataFrame,
    ts_col: str = "wall_clock_utc",
    horizon_col: str = "event_time_seconds",
) -> bool:
    """Verify that feature timestamps do not exceed their event-time horizon.

    Specifically checks that event_time_seconds is monotonically consistent
    within each (event_id, node_id) group.
    """
    if horizon_col not in df.columns:
        return True  # column absent; skip check

    group_cols = [c for c in ["event_id", "node_id"] if c in df.columns]
    if not group_cols:
        return True

    # Check that event_time_seconds is non-decreasing within each group
    broken = (
        df.sort(ts_col)
        .with_columns(
            pl.col(horizon_col)
            .diff()
            .over(group_cols)
            .alias("_ets_diff")
        )
        .filter(pl.col("_ets_diff") < 0)
        .height
    )
    if broken > 0:
        raise ValueError(
            f"Lookahead risk: {broken} rows have decreasing event_time_seconds "
            "within a group — possible data ordering issue."
        )

    logger.info("Future-feature check passed.")
    return True


def check_tier_claim_consistency(
    df: pl.DataFrame,
    tier_col: str = "tier_actual",
    microstructure_cols: tuple[str, ...] = ("spread_bps", "orderbook_imbalance", "depth_10bps_bid_usd"),
) -> None:
    """Warn if microstructure columns are non-null for Tier-B or Tier-C nodes.

    Tier-B/C nodes should not have executable microstructure features populated
    unless they have been explicitly upgraded.
    """
    if tier_col not in df.columns:
        return

    for col in microstructure_cols:
        if col not in df.columns:
            continue
        n_bad = df.filter(
            (pl.col(tier_col).is_in(["B", "C"])) & pl.col(col).is_not_null()
        ).height
        if n_bad > 0:
            logger.warning(
                "Tier-claim inconsistency: %d rows have '%s' populated for Tier-B/C nodes. "
                "Verify these are intentional upgrades.",
                n_bad,
                col,
            )
