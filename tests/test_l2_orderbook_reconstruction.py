"""Tests for L2 order-book reconstruction and BookManifest diagnostics."""

from __future__ import annotations

import math

import polars as pl
import pytest

from stressnet.reconstruct.orderbook_l2 import (
    BookManifest,
    BookUpdate,
    L2BookReconstructor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_update(
    exchange_ts: float,
    side: str,
    price: float,
    size: float,
    update_type: str = "delta",
    sequence_id: int = -1,
    local_ts: float | None = None,
) -> BookUpdate:
    return BookUpdate(
        exchange_ts  = exchange_ts,
        local_ts     = local_ts if local_ts is not None else exchange_ts + 100_000,
        sequence_id  = sequence_id,
        side         = side,
        price        = price,
        size         = size,
        update_type  = update_type,
        row_position = 0,
    )


def _simple_book() -> L2BookReconstructor:
    """Return a reconstructor with a simple 3-level book."""
    r = L2BookReconstructor()
    r.apply(_make_update(1_000_000, "bid", 1.0000, 100_000, "snapshot"))
    r.apply(_make_update(1_000_001, "bid", 0.9999, 80_000,  "snapshot"))
    r.apply(_make_update(1_000_002, "bid", 0.9998, 60_000,  "snapshot"))
    r.apply(_make_update(1_000_003, "ask", 1.0001, 90_000,  "snapshot"))
    r.apply(_make_update(1_000_004, "ask", 1.0002, 70_000,  "snapshot"))
    r.apply(_make_update(1_000_005, "ask", 1.0003, 50_000,  "snapshot"))
    return r


# ---------------------------------------------------------------------------
# Basic reconstruction
# ---------------------------------------------------------------------------

def test_silver_rows_created():
    r = _simple_book()
    assert len(r._silver_rows) == 6


def test_mid_price_correct():
    r = _simple_book()
    # Last row: best_bid=1.0000, best_ask=1.0001 → mid=1.00005
    last = r._silver_rows[-1]
    assert abs(last["mid_price"] - 1.00005) < 1e-6


def test_spread_bps_positive():
    r = _simple_book()
    last = r._silver_rows[-1]
    assert last["spread_bps"] is not None
    assert last["spread_bps"] > 0


def test_depth_bid_positive():
    r = _simple_book()
    last = r._silver_rows[-1]
    assert last["depth_10bps_bid_usd"] is not None
    assert last["depth_10bps_bid_usd"] > 0


def test_level_removal():
    r = _simple_book()
    # Remove best bid
    r.apply(_make_update(1_000_006, "bid", 1.0000, 0.0, "delta"))
    last = r._silver_rows[-1]
    # After removing 1.0000 bid, best bid should be 0.9999
    assert abs(last["best_bid"] - 0.9999) < 1e-8


def test_executable_bookwalk_available():
    r = _simple_book()
    last = r._silver_rows[-1]
    assert last["is_executable_bookwalk"] is True
    assert last["executable_price_10k_buy"]  is not None
    assert last["executable_price_10k_sell"] is not None


# ---------------------------------------------------------------------------
# apply_from_df
# ---------------------------------------------------------------------------

def test_apply_from_df():
    df = pl.DataFrame({
        "exchange_ts":  [1_000_000.0, 1_001_000.0, 1_002_000.0],
        "local_ts":     [1_000_100.0, 1_001_100.0, 1_002_100.0],
        "sequence_id":  [10, 11, 12],
        "side":         ["bid", "ask", "bid"],
        "price":        [1.0000, 1.0001, 0.9999],
        "size":         [100_000.0, 90_000.0, 80_000.0],
        "update_type":  ["snapshot", "snapshot", "delta"],
        "row_position": [0, 1, 2],
    })
    r = L2BookReconstructor()
    r.apply_from_df(df)
    assert r._n_messages == 3


# ---------------------------------------------------------------------------
# Sequence gap detection
# ---------------------------------------------------------------------------

def test_sequence_gap_detected():
    r = L2BookReconstructor()
    r.apply(_make_update(1e6, "bid", 1.0, 100_000, "snapshot", sequence_id=1))
    r.apply(_make_update(2e6, "bid", 0.9999, 80_000, "delta",   sequence_id=2))
    r.apply(_make_update(3e6, "ask", 1.0001, 90_000, "delta",   sequence_id=5))  # gap 3,4
    assert r._seq_gap_count == 2


def test_no_gap_on_contiguous_sequence():
    r = L2BookReconstructor()
    for i in range(1, 10):
        r.apply(_make_update(float(i * 1e6), "bid", 1.0 - i * 0.0001, 10_000,
                             "delta", sequence_id=i))
    assert r._seq_gap_count == 0


def test_gap_not_counted_across_snapshot():
    """Snapshots legitimately reset sequence; should not count as gap."""
    r = L2BookReconstructor()
    r.apply(_make_update(1e6, "bid", 1.0, 100_000, "snapshot", sequence_id=1))
    r.apply(_make_update(2e6, "ask", 1.0001, 90_000, "snapshot", sequence_id=999))
    assert r._seq_gap_count == 0


# ---------------------------------------------------------------------------
# Resync counting
# ---------------------------------------------------------------------------

def test_resync_count():
    r = L2BookReconstructor()
    r.apply(_make_update(1e6, "bid", 1.0, 100_000, "snapshot"))   # first snapshot — expected
    r.apply(_make_update(2e6, "ask", 1.001, 90_000, "delta"))
    r.apply(_make_update(3e6, "bid", 1.0, 100_000, "snapshot"))   # resync #1
    r.apply(_make_update(4e6, "ask", 1.001, 90_000, "snapshot"))  # resync #2
    manifest = r.manifest()
    assert manifest.resync_count == 2


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_coverage_full_window():
    """Every 60s bucket populated → coverage should be ~1.0."""
    r = L2BookReconstructor()
    for i in range(60):   # one update per minute for 60 minutes
        ts_us = float(i * 60 * 1_000_000)
        r.apply(_make_update(ts_us, "bid", 1.0, 100_000.0, "delta"))

    start_us = 0.0
    end_us   = 60 * 60 * 1_000_000.0  # 1 hour
    m = r.manifest(start_ts_us=start_us, end_ts_us=end_us)
    # 60 distinct 1-minute buckets out of 60 expected = 100%
    assert m.coverage_pct >= 0.95


def test_coverage_empty():
    r = L2BookReconstructor()
    m = r.manifest(start_ts_us=1e15, end_ts_us=2e15)
    assert m.coverage_pct == 0.0


# ---------------------------------------------------------------------------
# BookManifest tier rules
# ---------------------------------------------------------------------------

def test_tier_a_clean_book():
    m = BookManifest(
        exchange="binance", symbol="USDCUSDT",
        start_ts_us=0, end_ts_us=1e9,
        n_messages=10_000,
        coverage_pct=0.95,
        gap_rate=0.0,
        resync_count=0,
        clock_skew_abs_ms=10.0,
    )
    assert m.tier_actual == "A"
    assert m.tier_downgrade_reason == ""


def test_tier_b_low_coverage():
    m = BookManifest(
        exchange="binance", symbol="USDCUSDT",
        start_ts_us=0, end_ts_us=1e9,
        coverage_pct=0.30,  # below 50%
    )
    assert m.tier_actual == "B"
    assert "incomplete_coverage" in m.tier_downgrade_reason


def test_tier_b_high_gap_rate():
    m = BookManifest(
        exchange="binance", symbol="USDCUSDT",
        start_ts_us=0, end_ts_us=1e9,
        coverage_pct=0.90,
        gap_rate=0.05,  # above 1%
    )
    assert m.tier_actual == "B"
    assert "sequence_gaps" in m.tier_downgrade_reason


def test_tier_b_resync_long():
    m = BookManifest(
        exchange="binance", symbol="USDCUSDT",
        start_ts_us=0, end_ts_us=1e9,
        coverage_pct=0.90,
        gap_rate=0.0,
        resync_count=5,
        cumulative_resync_seconds=1500.0,  # > 300s
    )
    assert m.tier_actual == "B"
    assert "resync" in m.tier_downgrade_reason


def test_tier_b_clock_skew():
    m = BookManifest(
        exchange="binance", symbol="USDCUSDT",
        start_ts_us=0, end_ts_us=1e9,
        coverage_pct=0.90,
        clock_skew_abs_ms=6_000.0,  # > 5000ms
    )
    assert m.tier_actual == "B"
    assert "clock_unreliable" in m.tier_downgrade_reason


def test_tier_b_multiple_reasons():
    m = BookManifest(
        exchange="binance", symbol="USDCUSDT",
        start_ts_us=0, end_ts_us=1e9,
        coverage_pct=0.30,
        gap_rate=0.05,
    )
    assert m.tier_actual == "B"
    reasons = m.tier_downgrade_reason.split(",")
    assert "incomplete_coverage" in reasons
    assert "sequence_gaps" in reasons


# ---------------------------------------------------------------------------
# to_silver_df
# ---------------------------------------------------------------------------

def test_to_silver_df_columns():
    r = _simple_book()
    silver = r.to_silver_df()
    expected_cols = ["mid_price", "spread_bps", "depth_10bps_bid_usd",
                     "orderbook_imbalance", "depth_source"]
    for col in expected_cols:
        assert col in silver.columns, f"Missing column '{col}' in silver"


def test_to_silver_df_grid_resampling():
    r = L2BookReconstructor()
    grid_us = 60 * 1_000_000  # 60s
    # 10 updates spread across 2 buckets
    for i in range(5):
        r.apply(_make_update(float(i * 10_000_000), "bid", 1.0, 100_000.0, "delta"))
        r.apply(_make_update(float(i * 10_000_000 + 1), "ask", 1.0001, 90_000.0, "delta"))
    silver = r.to_silver_df(grid_seconds=60)
    # Should produce ≤10 rows (one per 60s bucket with updates)
    assert len(silver) <= 10
    assert len(silver) >= 1


def test_to_silver_df_window_filter():
    r = _simple_book()
    # All updates at ts ~1e6; request window far in the future
    silver = r.to_silver_df(start_ts_us=1e12, end_ts_us=2e12)
    assert silver.is_empty()
