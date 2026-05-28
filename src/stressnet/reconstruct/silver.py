"""Bronze → silver feature transforms.

Each function takes a bronze DataFrame (raw API output) and returns the
standardized silver representation with the columns expected by the gold
panel builder (script 03).

Detection of bronze type is done by column presence, so the same
`_standardize()` in script 02 can dispatch without knowing where the data
originated.

Column sets by bronze type:
  klines      : wall_clock_utc, open, high, low, close, volume, n_trades
  bookTicker  : wall_clock_utc, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty
  pool_events : wall_clock_utc, reserve_imbalance | implied_pool_price (from Curve/Uniswap)
  flows       : wall_clock_utc, exchange_inflow_1h | mint_burn_net_1h (from Etherscan)
  fixture     : all silver columns already present (mid_price, spread_bps, etc.)
"""

from __future__ import annotations

import polars as pl

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

# Notional size used for executable-price proxies (USD)
_EXEC_NOTIONAL = 10_000.0


# ---------------------------------------------------------------------------
# CEX reconstructions
# ---------------------------------------------------------------------------

def books_from_klines(df: pl.DataFrame) -> pl.DataFrame:
    """Convert OHLCV klines bronze to silver book features.

    Tier B: uses (high + low) / 2 as mid, (high - low) / close as spread proxy.
    Depth, imbalance, and executable price are not computable from OHLCV alone
    and are left as NULL.
    """
    result = df.with_columns(
        mid_price=((pl.col("high") + pl.col("low")) / 2.0),
        spread_bps=pl.when(pl.col("close") > 0).then(
            (pl.col("high") - pl.col("low")) / pl.col("close") * 10_000.0
        ).otherwise(None),
    ).with_columns(
        basis_vs_usd=pl.when(pl.col("mid_price") > 0)
            .then(pl.col("mid_price").log())
            .otherwise(None),
        depth_10bps_bid_usd=pl.lit(None, dtype=pl.Float64),
        depth_10bps_ask_usd=pl.lit(None, dtype=pl.Float64),
        orderbook_imbalance=pl.lit(None, dtype=pl.Float64),
        executable_price_10k_buy=pl.lit(None, dtype=pl.Float64),
        executable_price_10k_sell=pl.lit(None, dtype=pl.Float64),
    )
    # Drop raw OHLCV columns that are not part of the silver schema
    drop = [c for c in ["open", "high", "low", "close", "vwap"] if c in result.columns]
    return result.drop(drop) if drop else result


def books_from_book_ticker(df: pl.DataFrame) -> pl.DataFrame:
    """Convert bookTicker BBO bronze to silver book features.

    Tier A: mid and spread are exact. Depth proxied from best-level quantities.
    Imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty) at best level.
    Executable price = mid ± half-spread (best-level approximation; not VWAP).
    """
    result = df.with_columns(
        mid_price=((pl.col("best_bid_price") + pl.col("best_ask_price")) / 2.0),
        spread_bps=pl.when(
            (pl.col("best_bid_price") + pl.col("best_ask_price")) > 0
        ).then(
            (pl.col("best_ask_price") - pl.col("best_bid_price"))
            / ((pl.col("best_bid_price") + pl.col("best_ask_price")) / 2.0)
            * 10_000.0
        ).otherwise(None),
    ).with_columns(
        basis_vs_usd=pl.when(pl.col("mid_price") > 0)
            .then(pl.col("mid_price").log())
            .otherwise(None),
        depth_10bps_bid_usd=pl.col("best_bid_price") * pl.col("best_bid_qty"),
        depth_10bps_ask_usd=pl.col("best_ask_price") * pl.col("best_ask_qty"),
        orderbook_imbalance=pl.when(
            (pl.col("best_bid_qty") + pl.col("best_ask_qty")) > 0
        ).then(
            (pl.col("best_bid_qty") - pl.col("best_ask_qty"))
            / (pl.col("best_bid_qty") + pl.col("best_ask_qty"))
        ).otherwise(None),
        executable_price_10k_buy=(
            (pl.col("best_bid_price") + pl.col("best_ask_price")) / 2.0
            + (pl.col("best_ask_price") - pl.col("best_bid_price")) / 2.0
        ),
        executable_price_10k_sell=(
            (pl.col("best_bid_price") + pl.col("best_ask_price")) / 2.0
            - (pl.col("best_ask_price") - pl.col("best_bid_price")) / 2.0
        ),
    ).drop(["best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty"])
    return result


# ---------------------------------------------------------------------------
# DEX pool state reconstructions
# ---------------------------------------------------------------------------

def pool_from_curve_events(df: pl.DataFrame) -> pl.DataFrame:
    """Standardise Curve pool-event bronze to silver pool features.

    Expects columns: wall_clock_utc, reserve_imbalance (proxy), implied_pool_price.
    """
    result = df.clone()
    if "reserve_imbalance" not in result.columns:
        result = result.with_columns(pl.lit(None, dtype=pl.Float64).alias("reserve_imbalance"))
    if "implied_pool_price" not in result.columns:
        result = result.with_columns(pl.lit(None, dtype=pl.Float64).alias("implied_pool_price"))
    if "basis_vs_usd" not in result.columns:
        result = result.with_columns(
            pl.when(pl.col("implied_pool_price") > 0)
            .then(pl.col("implied_pool_price").log())
            .otherwise(None)
            .alias("basis_vs_usd")
        )
    if "pool_slippage_10k" not in result.columns:
        # Proxy: |reserve_imbalance| * 100 bps (very rough)
        result = result.with_columns(
            (pl.col("reserve_imbalance").abs() * 100.0).alias("pool_slippage_10k")
        )
    return result


def pool_from_uniswap_swaps(df: pl.DataFrame) -> pl.DataFrame:
    """Standardise Uniswap swap bronze to silver pool features."""
    return pool_from_curve_events(df)  # same column names, same transform


# ---------------------------------------------------------------------------
# Flow reconstructions
# ---------------------------------------------------------------------------

def flows_from_transfers(df: pl.DataFrame) -> pl.DataFrame:
    """Standardise transfer-based flow bronze to silver flow features.

    Expects one or more of: exchange_inflow_1h, exchange_outflow_1h,
    exchange_netflow_1h, mint_burn_net_1h.  Missing columns are filled NULL.
    """
    result = df.clone()
    for col in [
        "exchange_inflow_1h", "exchange_outflow_1h", "exchange_netflow_1h",
        "mint_burn_net_1h", "gas_base_fee_gwei",
    ]:
        if col not in result.columns:
            result = result.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
    if "basis_vs_usd" not in result.columns:
        result = result.with_columns(pl.lit(None, dtype=pl.Float64).alias("basis_vs_usd"))
    return result


# ---------------------------------------------------------------------------
# Auto-dispatch: detect bronze type from columns and apply correct transform
# ---------------------------------------------------------------------------

def standardize_bronze(df: pl.DataFrame) -> pl.DataFrame:
    """Detect the bronze column schema and apply the appropriate silver transform.

    Detection order (first match wins):
    1. bookTicker: has 'best_bid_price'
    2. klines: has 'close' (and no 'reserve_imbalance')
    3. pool_events: has 'reserve_imbalance' or 'implied_pool_price'
    4. flows: has 'exchange_inflow_1h' or 'mint_burn_net_1h'
    5. fixture / already-silver: pass through
    """
    cols = set(df.columns)

    if "best_bid_price" in cols:
        logger.debug("standardize_bronze → books_from_book_ticker")
        return books_from_book_ticker(df)

    if "close" in cols and "reserve_imbalance" not in cols:
        logger.debug("standardize_bronze → books_from_klines")
        return books_from_klines(df)

    if "reserve_imbalance" in cols or "implied_pool_price" in cols:
        logger.debug("standardize_bronze → pool_from_curve_events")
        return pool_from_curve_events(df)

    if "exchange_inflow_1h" in cols or "mint_burn_net_1h" in cols:
        logger.debug("standardize_bronze → flows_from_transfers")
        return flows_from_transfers(df)

    # Already in silver format (fixture or pre-processed)
    logger.debug("standardize_bronze → pass-through (already silver)")
    return df
