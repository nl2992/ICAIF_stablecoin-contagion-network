"""Cross-node basis and directional spread computation."""

from __future__ import annotations

import math

import polars as pl


def directional_basis(price_j: float, price_i: float) -> float | None:
    """Compute log(p_j) - log(p_i).

    Positive = j is more expensive than i (buying from i and selling to j profitable).
    """
    if price_i is None or price_j is None or price_i <= 0 or price_j <= 0:
        return None
    return math.log(price_j) - math.log(price_i)


def compute_pairwise_basis(
    panel: pl.DataFrame,
    node_i: str,
    node_j: str,
    price_col: str = "mid_price",
    ts_col: str = "wall_clock_utc",
) -> pl.DataFrame:
    """Compute directional basis between two nodes over time.

    Args:
        panel: Feature panel with node_id and price_col columns.
        node_i: Source node ID (denominator).
        node_j: Destination node ID (numerator).
        price_col: Column name for the price to use.
        ts_col: Timestamp column for alignment.

    Returns:
        DataFrame with columns: ts_col, basis_i_to_j (log points).
    """
    pi = (
        panel.filter(pl.col("node_id") == node_i)
        .select([ts_col, pl.col(price_col).alias("price_i")])
    )
    pj = (
        panel.filter(pl.col("node_id") == node_j)
        .select([ts_col, pl.col(price_col).alias("price_j")])
    )
    return (
        pi.join(pj, on=ts_col, how="inner")
        .with_columns(
            (pl.col("price_j").log() - pl.col("price_i").log())
            .alias(f"basis_{node_i}_to_{node_j}")
        )
        .select([ts_col, f"basis_{node_i}_to_{node_j}"])
    )


def label_basis_exceedance(
    panel: pl.DataFrame,
    basis_col: str = "basis_vs_usd",
    thresholds_bps: list[float] | None = None,
) -> pl.DataFrame:
    """Add binary exceedance labels for each threshold in thresholds_bps.

    E.g. |basis| > 10 bps → label_basis_gt10bps = True.
    """
    if thresholds_bps is None:
        thresholds_bps = [10.0, 50.0]

    result = panel
    for t in thresholds_bps:
        t_logpoints = t / 10_000
        col_name = f"label_basis_gt{int(t)}bps"
        result = result.with_columns(
            (pl.col(basis_col).abs() > t_logpoints).alias(col_name)
        )
    return result
