"""Tests for bookwalk VWAP and depth helpers."""

from __future__ import annotations

import math

import pytest

from stressnet.reconstruct.orderbook import OrderBook
from stressnet.reconstruct.bookwalk import (
    bookwalk_vwap,
    compute_microstructure_features,
    depth_usd_within_bps,
    depth_within_bps,
    executable_spread_bps,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_book(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
) -> OrderBook:
    b = OrderBook()
    b.apply_snapshot(bids, asks)
    return b


def _std_book() -> OrderBook:
    """Standard test book: 3-level bid + 3-level ask."""
    return _make_book(
        bids=[(1.0000, 100_000), (0.9999, 80_000), (0.9998, 60_000)],
        asks=[(1.0001,  90_000), (1.0002, 70_000), (1.0003, 50_000)],
    )


# ---------------------------------------------------------------------------
# bookwalk_vwap
# ---------------------------------------------------------------------------

class TestBookwalkVwap:
    def test_buy_single_level_sufficient(self):
        """Notional order fully filled at best ask."""
        b = _make_book(
            bids=[(0.999, 100_000)],
            asks=[(1.001, 100_000)],
        )
        # 10_000 USD ÷ mid(1.000) ≈ 10_000 qty; best ask has 100_000 qty
        vwap = bookwalk_vwap("buy", b, 10_000)
        assert vwap is not None
        assert abs(vwap - 1.001) < 1e-6

    def test_sell_single_level_sufficient(self):
        b = _make_book(
            bids=[(0.999, 100_000)],
            asks=[(1.001, 100_000)],
        )
        vwap = bookwalk_vwap("sell", b, 10_000)
        assert vwap is not None
        assert abs(vwap - 0.999) < 1e-6

    def test_buy_across_multiple_levels(self):
        """Large order walks across 2 ask levels; VWAP is between them."""
        b = _make_book(
            bids=[(1.000, 5_000)],
            asks=[(1.001, 5_000), (1.002, 10_000)],   # first level: 5_000 qty
        )
        # mid ≈ 1.0005; qty = 10_000 / 1.0005 ≈ 9_995
        # First 5_000 at 1.001, remaining ~4_995 at 1.002
        vwap = bookwalk_vwap("buy", b, 10_000)
        assert vwap is not None
        assert 1.001 < vwap < 1.002

    def test_insufficient_depth_returns_none(self):
        b = _make_book(
            bids=[(1.0, 1_000)],
            asks=[(1.001, 1)],   # only 1 unit on the ask
        )
        # Need ~10_000 qty, have 1 — should return None
        vwap = bookwalk_vwap("buy", b, 10_000)
        assert vwap is None

    def test_empty_book_returns_none(self):
        b = OrderBook()
        assert bookwalk_vwap("buy",  b, 10_000) is None
        assert bookwalk_vwap("sell", b, 10_000) is None

    def test_buy_vwap_greater_than_mid(self):
        b = _std_book()
        mid = b.mid()
        vwap = bookwalk_vwap("buy", b, 10_000)
        assert vwap is not None
        assert vwap > mid

    def test_sell_vwap_less_than_mid(self):
        b = _std_book()
        mid = b.mid()
        vwap = bookwalk_vwap("sell", b, 10_000)
        assert vwap is not None
        assert vwap < mid

    def test_invalid_side_raises(self):
        b = _std_book()
        with pytest.raises(ValueError, match="side must be"):
            bookwalk_vwap("up", b, 10_000)


# ---------------------------------------------------------------------------
# depth_within_bps
# ---------------------------------------------------------------------------

class TestDepthWithinBps:
    def test_bid_depth_positive(self):
        b = _std_book()
        d = depth_within_bps("bid", b, 10.0)
        assert d > 0

    def test_ask_depth_positive(self):
        b = _std_book()
        d = depth_within_bps("ask", b, 10.0)
        assert d > 0

    def test_zero_bps_returns_zero(self):
        """Zero bps range should return 0 (no levels within 0 bps)."""
        b = _std_book()
        d = depth_within_bps("bid", b, 0.0)
        assert d == 0.0

    def test_very_large_bps_includes_all_levels(self):
        b = _make_book(
            bids=[(1.000, 100), (0.500, 200)],  # 50% away = 5000 bps
            asks=[(1.001, 100), (1.500, 200)],
        )
        d_narrow = depth_within_bps("bid", b, 10.0)
        d_wide   = depth_within_bps("bid", b, 10_000.0)
        assert d_wide >= d_narrow

    def test_empty_book_returns_zero(self):
        b = OrderBook()
        assert depth_within_bps("bid", b, 10.0) == 0.0
        assert depth_within_bps("ask", b, 10.0) == 0.0


class TestDepthUsdWithinBps:
    def test_returns_float(self):
        b = _std_book()
        d = depth_usd_within_bps("bid", b, 10.0)
        assert d is not None
        assert d > 0

    def test_empty_book_returns_none(self):
        b = OrderBook()
        assert depth_usd_within_bps("bid", b, 10.0) is None


# ---------------------------------------------------------------------------
# executable_spread_bps
# ---------------------------------------------------------------------------

class TestExecutableSpreadBps:
    def test_spread_positive(self):
        b = _std_book()
        s = executable_spread_bps(b, 10_000)
        assert s is not None
        assert s > 0

    def test_spread_larger_than_quoted_spread(self):
        """Executable spread must be >= quoted spread (bps)."""
        b = _std_book()
        quoted = b.spread_bps()
        executable = executable_spread_bps(b, 10_000)
        assert executable is not None and quoted is not None
        assert executable >= quoted - 0.001   # allow tiny float error

    def test_insufficient_depth_returns_none(self):
        b = _make_book(
            bids=[(1.0, 1)],   # 1 unit only
            asks=[(1.001, 1)],
        )
        # Can't fill 10_000 USD from a 1-unit book
        s = executable_spread_bps(b, 10_000)
        assert s is None

    def test_empty_book_returns_none(self):
        b = OrderBook()
        assert executable_spread_bps(b, 10_000) is None


# ---------------------------------------------------------------------------
# compute_microstructure_features
# ---------------------------------------------------------------------------

class TestComputeMicrostructureFeatures:
    def test_all_keys_present(self):
        b = _std_book()
        feat = compute_microstructure_features(b, notional_usd=10_000, depth_bps=10.0)
        expected_keys = [
            "mid_price", "spread_bps", "depth_10bps_bid_usd", "depth_10bps_ask_usd",
            "orderbook_imbalance", "executable_price_10k_buy", "executable_price_10k_sell",
            "executable_spread_bps",
        ]
        for key in expected_keys:
            assert key in feat, f"Missing key '{key}'"

    def test_mid_price_value(self):
        b = _std_book()
        feat = compute_microstructure_features(b)
        # mid = (1.0000 + 1.0001) / 2 = 1.00005
        assert abs(feat["mid_price"] - 1.00005) < 1e-6

    def test_buy_vwap_above_mid(self):
        b = _std_book()
        feat = compute_microstructure_features(b)
        assert feat["executable_price_10k_buy"] > feat["mid_price"]

    def test_sell_vwap_below_mid(self):
        b = _std_book()
        feat = compute_microstructure_features(b)
        assert feat["executable_price_10k_sell"] < feat["mid_price"]

    def test_empty_book_all_none(self):
        b = OrderBook()
        feat = compute_microstructure_features(b)
        assert feat["mid_price"] is None
        assert feat["spread_bps"] is None
        assert feat["depth_10bps_bid_usd"] is None
