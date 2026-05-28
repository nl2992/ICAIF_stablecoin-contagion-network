"""Bookwalk VWAP and depth helpers for L2 order books.

These functions operate on ``stressnet.reconstruct.orderbook.OrderBook`` instances
and are designed for Tier-A feature extraction from reconstructed L2 books.

Key functions
-------------
bookwalk_vwap(side, book, notional_usd)
    Walk the book for a given notional order size; return executed VWAP or None
    if depth is insufficient.

depth_within_bps(side, book, bps)
    Total quantity (in native asset units) available within ``bps`` of the best price.

depth_usd_within_bps(side, book, bps)
    Same as above, expressed in USD (using mid price for conversion).

executable_spread_bps(book, notional_usd)
    Effective spread for a round-trip (buy then immediately sell the same notional),
    expressed in basis points.
"""

from __future__ import annotations

from stressnet.reconstruct.orderbook import OrderBook


# ---------------------------------------------------------------------------
# Bookwalk VWAP
# ---------------------------------------------------------------------------

def bookwalk_vwap(
    side: str,
    book: OrderBook,
    notional_usd: float,
) -> float | None:
    """Compute the VWAP for executing a notional order against the book.

    Args:
        side: ``"buy"`` (lift the ask) or ``"sell"`` (hit the bid).
        book: The reconstructed L2 book.
        notional_usd: USD-denominated order size.

    Returns:
        Executed VWAP price, or ``None`` if the book does not have sufficient depth
        to fill the entire order.

    Notes:
        The quantity is approximated as ``notional_usd / mid_price`` before walking.
        This is a first-order approximation; for highly stressed markets where the
        mid moves significantly through the book, the true executed quantity differs.
        Use the returned VWAP price to compute the actual fill cost.
    """
    mid = book.mid()
    if mid is None or mid <= 0:
        return None
    qty = notional_usd / mid

    if side == "buy":
        levels = book.sorted_asks()
    elif side == "sell":
        levels = book.sorted_bids()
    else:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    remaining = qty
    cost = 0.0
    filled = 0.0
    for price, size in levels:
        take = min(remaining, size)
        cost += take * price
        filled += take
        remaining -= take
        if remaining <= 1e-12:
            break

    if filled < qty - 1e-12:
        return None  # insufficient depth

    return cost / filled


# ---------------------------------------------------------------------------
# Depth helpers
# ---------------------------------------------------------------------------

def depth_within_bps(
    side: str,
    book: OrderBook,
    bps: float = 10.0,
) -> float:
    """Total quantity available within ``bps`` of the best price.

    Returns quantity in **native asset units** (not USD).

    Args:
        side: ``"bid"`` or ``"ask"``.
        bps: Basis-point range from best price.
    """
    return book.depth_within_bps(side, bps)


def depth_usd_within_bps(
    side: str,
    book: OrderBook,
    bps: float = 10.0,
) -> float | None:
    """Depth within bps of best, expressed in USD.

    Args:
        side: ``"bid"`` or ``"ask"``.
        bps: Basis-point range from best price.

    Returns:
        USD depth (native qty × mid price), or ``None`` if mid is unavailable.
    """
    mid = book.mid()
    if mid is None or mid <= 0:
        return None
    return depth_within_bps(side, book, bps) * mid


# ---------------------------------------------------------------------------
# Executable spread
# ---------------------------------------------------------------------------

def executable_spread_bps(
    book: OrderBook,
    notional_usd: float,
) -> float | None:
    """Round-trip execution cost in basis points.

    Computes:
        (buy_vwap - sell_vwap) / mid × 10_000

    This measures the actual cost of simultaneously buying and selling
    ``notional_usd`` against the book, i.e. the implementation shortfall
    for a round trip.  Higher values indicate a wider effective spread and
    greater market stress.

    Returns:
        Basis points, or ``None`` if either side lacks sufficient depth.
    """
    buy_vwap  = bookwalk_vwap("buy",  book, notional_usd)
    sell_vwap = bookwalk_vwap("sell", book, notional_usd)
    mid       = book.mid()

    if buy_vwap is None or sell_vwap is None or mid is None or mid <= 0:
        return None

    return (buy_vwap - sell_vwap) / mid * 10_000.0


# ---------------------------------------------------------------------------
# Convenience: compute all Tier-A microstructure features from a book state
# ---------------------------------------------------------------------------

def compute_microstructure_features(
    book: OrderBook,
    notional_usd: float = 10_000.0,
    depth_bps: float = 10.0,
) -> dict[str, float | None]:
    """Return all Tier-A microstructure features for a book snapshot.

    Returns:
        Dict with keys:
          mid_price, spread_bps, depth_10bps_bid_usd, depth_10bps_ask_usd,
          orderbook_imbalance, executable_price_10k_buy, executable_price_10k_sell,
          executable_spread_bps.
    """
    mid = book.mid()
    return {
        "mid_price":                mid,
        "spread_bps":               book.spread_bps(),
        "depth_10bps_bid_usd":      depth_usd_within_bps("bid", book, depth_bps),
        "depth_10bps_ask_usd":      depth_usd_within_bps("ask", book, depth_bps),
        "orderbook_imbalance":      book.imbalance(bps=depth_bps),
        "executable_price_10k_buy": bookwalk_vwap("buy",  book, notional_usd),
        "executable_price_10k_sell":bookwalk_vwap("sell", book, notional_usd),
        "executable_spread_bps":    executable_spread_bps(book, notional_usd),
    }
