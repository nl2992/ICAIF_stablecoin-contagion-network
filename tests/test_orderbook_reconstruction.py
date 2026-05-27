"""Tests for OrderBook and VWAP computation."""

import pytest
from stressnet.reconstruct.orderbook import (
    OrderBook,
    executable_buy_vwap,
    executable_sell_vwap,
    notional_buy_vwap,
)


@pytest.fixture
def stable_book():
    book = OrderBook()
    book.apply_snapshot(
        bids=[("0.9999", "100000"), ("0.9998", "200000"), ("0.9995", "500000")],
        asks=[("1.0001", "100000"), ("1.0002", "200000"), ("1.0005", "500000")],
    )
    return book


def test_best_bid_ask(stable_book):
    assert stable_book.best_bid() == pytest.approx(0.9999)
    assert stable_book.best_ask() == pytest.approx(1.0001)


def test_mid(stable_book):
    assert stable_book.mid() == pytest.approx(1.0, abs=0.001)


def test_spread_bps(stable_book):
    assert stable_book.spread_bps() == pytest.approx(2.0, abs=0.1)


def test_imbalance_balanced(stable_book):
    imb = stable_book.imbalance(bps=10)
    assert imb is not None
    assert abs(imb) < 0.1  # roughly balanced


def test_apply_update_removes_level():
    book = OrderBook()
    book.apply_snapshot(bids=[("1.0000", "100")], asks=[("1.0001", "100")])
    book.apply_update("bid", "1.0000", "0")
    assert book.best_bid() is None


def test_apply_update_adds_level():
    book = OrderBook()
    book.apply_snapshot(bids=[], asks=[("1.0001", "100")])
    book.apply_update("bid", "0.9999", "50")
    assert book.best_bid() == pytest.approx(0.9999)


def test_is_crossed_false(stable_book):
    assert not stable_book.is_crossed()


def test_executable_buy_vwap(stable_book):
    mid = stable_book.mid()
    vwap = notional_buy_vwap(stable_book, 10_000)
    assert vwap is not None
    assert vwap >= stable_book.best_ask() - 1e-6  # must buy at ask or worse


def test_insufficient_depth():
    book = OrderBook()
    book.apply_snapshot(bids=[("1.0", "1")], asks=[("1.001", "1")])
    result = executable_buy_vwap(book, 1_000_000)
    assert result is None  # book too thin


def test_depth_within_bps(stable_book):
    depth = stable_book.depth_within_bps("ask", 10)
    assert depth > 0
