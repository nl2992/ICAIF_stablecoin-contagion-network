"""In-memory order-book reconstruction and executable VWAP computation.

Ported from stressbench.book with renamed imports.
"""

from __future__ import annotations

from typing import Iterable


class OrderBook:
    """In-memory limit order book for a single instrument.

    Bids are stored as ``{price: size}`` and asks likewise.

    Example::

        book = OrderBook()
        book.apply_snapshot(bids=[("1.0001", "50000"), ("1.0000", "100000")],
                            asks=[("1.0002", "30000"), ("1.0003", "80000")])
        book.best_bid()    # 1.0001
        book.best_ask()    # 1.0002
        book.spread_bps()  # 1.0
    """

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}

    def apply_snapshot(self, bids: Iterable[tuple], asks: Iterable[tuple]) -> None:
        """Initialise the book from a full snapshot."""
        self.bids = {float(p): float(q) for p, q in bids if float(q) > 0}
        self.asks = {float(p): float(q) for p, q in asks if float(q) > 0}

    def apply_update(self, side: str, price: float | str, size: float | str) -> None:
        """Apply an incremental update; size == 0 removes the level."""
        book = self.bids if side == "bid" else self.asks
        price_f = float(price)
        size_f = float(size)
        if size_f == 0:
            book.pop(price_f, None)
        else:
            book[price_f] = size_f

    def best_bid(self) -> float | None:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> float | None:
        return min(self.asks) if self.asks else None

    def mid(self) -> float | None:
        bb, ba = self.best_bid(), self.best_ask()
        return (bb + ba) / 2.0 if (bb is not None and ba is not None) else None

    def spread(self) -> float | None:
        bb, ba = self.best_bid(), self.best_ask()
        return (ba - bb) if (bb is not None and ba is not None) else None

    def spread_bps(self) -> float | None:
        s, m = self.spread(), self.mid()
        if s is None or m is None or m == 0:
            return None
        return (s / m) * 10_000

    def is_crossed(self) -> bool:
        bb, ba = self.best_bid(), self.best_ask()
        return (bb is not None and ba is not None and bb >= ba)

    def sorted_bids(self) -> list[tuple[float, float]]:
        return sorted(self.bids.items(), key=lambda x: -x[0])

    def sorted_asks(self) -> list[tuple[float, float]]:
        return sorted(self.asks.items(), key=lambda x: x[0])

    def depth_within_bps(self, side: str, bps: float) -> float:
        """Total quantity available within ``bps`` basis points of the best price."""
        if side == "bid":
            best = self.best_bid()
            if best is None:
                return 0.0
            threshold = best / (1 + bps / 10_000)
            return sum(q for p, q in self.bids.items() if p > threshold)
        else:
            best = self.best_ask()
            if best is None:
                return 0.0
            threshold = best / (1 - bps / 10_000)
            return sum(q for p, q in self.asks.items() if p < threshold)

    def imbalance(self, bps: float = 10.0) -> float | None:
        """Order-book imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)."""
        bid_depth = self.depth_within_bps("bid", bps)
        ask_depth = self.depth_within_bps("ask", bps)
        total = bid_depth + ask_depth
        return (bid_depth - ask_depth) / total if total > 0 else None

    def __repr__(self) -> str:
        return (
            f"OrderBook(best_bid={self.best_bid()}, best_ask={self.best_ask()}, "
            f"bid_levels={len(self.bids)}, ask_levels={len(self.asks)})"
        )


# ---------------------------------------------------------------------------
# Executable VWAP helpers
# ---------------------------------------------------------------------------

def walk_book(levels: list[tuple[float, float]], qty: float) -> float | None:
    """Walk an order-book side and return the VWAP for a given quantity.

    Returns None if the book does not have sufficient depth.
    """
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
        return None
    return cost / filled


def executable_buy_vwap(book: OrderBook, qty: float) -> float | None:
    """VWAP for buying ``qty`` units from the ask side."""
    return walk_book(book.sorted_asks(), qty)


def executable_sell_vwap(book: OrderBook, qty: float) -> float | None:
    """VWAP for selling ``qty`` units into the bid side."""
    return walk_book(book.sorted_bids(), qty)


def notional_buy_vwap(book: OrderBook, notional_usd: float) -> float | None:
    """VWAP for a buy order of ``notional_usd`` USD (approximate: uses mid for qty)."""
    mid = book.mid()
    if mid is None or mid <= 0:
        return None
    return executable_buy_vwap(book, notional_usd / mid)


def notional_sell_vwap(book: OrderBook, notional_usd: float) -> float | None:
    """VWAP for a sell order of ``notional_usd`` USD (approximate: uses mid for qty)."""
    mid = book.mid()
    if mid is None or mid <= 0:
        return None
    return executable_sell_vwap(book, notional_usd / mid)
