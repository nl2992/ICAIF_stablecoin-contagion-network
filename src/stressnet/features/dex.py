"""DEX pool feature computation from reconstructed pool states."""

from __future__ import annotations

from typing import Any

import polars as pl

from stressnet.reconstruct.dex_pool import CurvePoolState, UniswapV3PoolState
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def curve_pool_features(state: CurvePoolState, notional_usd: float = 10_000.0) -> dict[str, Any]:
    """Compute all DEX node features from a Curve pool state snapshot."""
    return {
        "reserve_imbalance": state.reserve_imbalance(token_idx=0),
        "implied_pool_price": state.implied_price(i=0, j=1),
        "pool_slippage_10k": state.slippage_bps(notional_usd, token_in_idx=0),
        "virtual_price": state.virtual_price,
        "lp_supply": state.lp_supply,
    }


def uniswap_pool_features(state: UniswapV3PoolState, notional_usd: float = 10_000.0) -> dict[str, Any]:
    """Compute all DEX node features from a Uniswap v3 pool state snapshot."""
    price = state.implied_price()
    return {
        "implied_pool_price": price,
        "pool_slippage_10k": _uniswap_slippage(state, notional_usd) if price else None,
        "tick": state.tick,
        "active_liquidity": state.liquidity,
    }


def _uniswap_slippage(state: UniswapV3PoolState, notional_usd: float) -> float | None:
    """Approximate price impact for a Uniswap v3 pool swap.

    Uses a simplified concentrated-liquidity approximation: impact ≈ notional / (2 * L * √P).
    """
    if state.liquidity <= 0 or state.sqrt_price_x96 <= 0:
        return None
    sqrt_p = state.sqrt_price_x96 / (2**96)
    # Approximate depth in USD ≈ liquidity * sqrt_p (one side)
    depth_approx = state.liquidity * sqrt_p
    if depth_approx <= 0:
        return None
    return (notional_usd / (2 * depth_approx)) * 10_000


def compute_swap_imbalance(
    swaps: pl.DataFrame,
    window_seconds: int = 60,
    ts_col: str = "timestamp",
    amount0_col: str = "amount0",
    amount1_col: str = "amount1",
) -> pl.DataFrame:
    """Compute swap imbalance (buy - sell) / total for a Uniswap pool."""
    return (
        swaps
        .with_columns([
            ((pl.col(ts_col) // window_seconds) * window_seconds).alias("ts_window"),
            pl.when(pl.col(amount0_col) > 0)
            .then(pl.col(amount0_col).abs())
            .otherwise(-pl.col(amount0_col).abs())
            .alias("signed_amount"),
            pl.col(amount0_col).abs().alias("abs_amount"),
        ])
        .group_by("ts_window")
        .agg([
            pl.col("signed_amount").sum().alias("net_signed"),
            pl.col("abs_amount").sum().alias("total_abs"),
        ])
        .with_columns(
            (pl.col("net_signed") / pl.col("total_abs").clip(lower_bound=1e-9))
            .alias("swap_imbalance")
        )
        .sort("ts_window")
    )
