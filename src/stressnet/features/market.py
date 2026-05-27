"""Market node feature computation from reconstructed order books."""

from __future__ import annotations

import math
from typing import Any

import polars as pl

from stressnet.reconstruct.orderbook import OrderBook, executable_buy_vwap, executable_sell_vwap
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_NOTIONAL_USD = 10_000.0
_DEPTH_BPS = 10.0


def book_snapshot_features(book: OrderBook, notional_usd: float = _NOTIONAL_USD) -> dict[str, Any]:
    """Compute all market node features from a single book snapshot.

    Returns a dict with keys matching the features schema in configs/features.yaml.
    Returns None values for unavailable quantities.
    """
    mid = book.mid()
    return {
        "mid_price": mid,
        "spread_bps": book.spread_bps(),
        "depth_10bps_bid_usd": _depth_usd(book, "bid", mid),
        "depth_10bps_ask_usd": _depth_usd(book, "ask", mid),
        "orderbook_imbalance": book.imbalance(bps=_DEPTH_BPS),
        "executable_price_10k_buy": _exec_buy(book, notional_usd, mid),
        "executable_price_10k_sell": _exec_sell(book, notional_usd, mid),
        "basis_vs_usd": math.log(mid) if mid and mid > 0 else None,
    }


def _depth_usd(book: OrderBook, side: str, mid: float | None) -> float | None:
    """Convert depth-within-10bps from coin units to USD notional."""
    if mid is None:
        return None
    qty = book.depth_within_bps(side, _DEPTH_BPS)
    return qty * mid


def _exec_buy(book: OrderBook, notional_usd: float, mid: float | None) -> float | None:
    if mid is None or mid <= 0:
        return None
    qty = notional_usd / mid
    return executable_buy_vwap(book, qty)


def _exec_sell(book: OrderBook, notional_usd: float, mid: float | None) -> float | None:
    if mid is None or mid <= 0:
        return None
    qty = notional_usd / mid
    return executable_sell_vwap(book, qty)


def compute_signed_trade_imbalance(
    trades: pl.DataFrame,
    window_seconds: int = 60,
    ts_col: str = "trade_ts",
    size_col: str = "qty_usd",
    side_col: str = "aggressor_side",
) -> pl.DataFrame:
    """Compute signed trade imbalance (buy - sell) / (buy + sell) in rolling windows."""
    return (
        trades
        .with_columns(
            ((pl.col(ts_col) // window_seconds) * window_seconds).alias("ts_window"),
            pl.when(pl.col(side_col) == "buy")
            .then(pl.col(size_col))
            .otherwise(-pl.col(size_col))
            .alias("signed_size"),
        )
        .group_by("ts_window")
        .agg([
            pl.col("signed_size").sum().alias("net_signed_usd"),
            pl.col(size_col).sum().alias("total_usd"),
        ])
        .with_columns(
            (pl.col("net_signed_usd") / pl.col("total_usd").clip(lower_bound=1e-9))
            .alias("signed_trade_imbalance")
        )
        .sort("ts_window")
    )
