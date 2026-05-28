"""Tests for Tardis L2 bronze-schema normalisation."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from stressnet.data.tardis_l2 import (
    _BRONZE_SCHEMA,
    _BRONZE_COLS,
    _empty_bronze,
    _parse_incremental_book_l2,
    _parse_book_snapshot_25,
    _parse_ts,
    load_bronze,
)


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

def test_bronze_cols_match_schema():
    """Every column in _BRONZE_COLS must have a type in _BRONZE_SCHEMA."""
    for col in _BRONZE_COLS:
        assert col in _BRONZE_SCHEMA, f"Column '{col}' in _BRONZE_COLS but missing from _BRONZE_SCHEMA"


def test_empty_bronze_has_correct_schema():
    df = _empty_bronze()
    assert df.is_empty()
    for col, dtype in _BRONZE_SCHEMA.items():
        assert col in df.columns, f"Missing column '{col}' in empty bronze schema"


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected,tol", [
    (1_678_460_000_000_000, 1_678_460_000_000_000.0, 1.0),   # int microseconds
    (1_678_460_000.0, 1_678_460_000.0, 1.0),                  # float
    ("2023-03-10T12:00:00Z", 1_678_449_600_000_000.0, 1e6),   # ISO string
    ("2023-03-10T12:00:00.000000Z", 1_678_449_600_000_000.0, 1e6),
    (None, 0.0, 0.0),
    ("", 0.0, 0.0),
])
def test_parse_ts(raw, expected, tol):
    result = _parse_ts(raw)
    assert abs(result - expected) <= tol


# ---------------------------------------------------------------------------
# incremental_book_L2 parsing
# ---------------------------------------------------------------------------

_INC_L2_CSV = (
    "exchange\tsymbol\ttimestamp\tlocal_timestamp\tis_snapshot\tside\tprice\tamount\tsequence_id\n"
    "binance\tUSDCUSDT\t1678460000000000\t1678460000100000\ttrue\tbid\t0.9999\t50000.0\t100\n"
    "binance\tUSDCUSDT\t1678460001000000\t1678460001100000\tfalse\task\t1.0001\t30000.0\t101\n"
    "binance\tUSDCUSDT\t1678460002000000\t1678460002100000\tfalse\tbid\t0.9998\t0.0\t102\n"
)


def test_parse_incremental_book_l2_row_count():
    raw = _INC_L2_CSV.encode()
    df = _parse_incremental_book_l2(raw, "binance", "USDCUSDT")
    assert len(df) == 3


def test_parse_incremental_book_l2_schema():
    raw = _INC_L2_CSV.encode()
    df = _parse_incremental_book_l2(raw, "binance", "USDCUSDT")
    for col in _BRONZE_COLS:
        assert col in df.columns, f"Missing column '{col}'"


def test_parse_incremental_book_l2_update_types():
    raw = _INC_L2_CSV.encode()
    df = _parse_incremental_book_l2(raw, "binance", "USDCUSDT")
    types = df["update_type"].to_list()
    assert types[0] == "snapshot"
    assert types[1] == "delta"
    assert types[2] == "delta"


def test_parse_incremental_book_l2_sides():
    raw = _INC_L2_CSV.encode()
    df = _parse_incremental_book_l2(raw, "binance", "USDCUSDT")
    assert df["side"].to_list() == ["bid", "ask", "bid"]


def test_parse_incremental_book_l2_prices():
    raw = _INC_L2_CSV.encode()
    df = _parse_incremental_book_l2(raw, "binance", "USDCUSDT")
    assert abs(df["price"][0] - 0.9999) < 1e-8
    assert abs(df["size"][2] - 0.0) < 1e-8   # removal


def test_parse_incremental_book_l2_exchange_fields():
    raw = _INC_L2_CSV.encode()
    df = _parse_incremental_book_l2(raw, "binance", "USDCUSDT")
    assert df["exchange"][0] == "binance"
    assert df["symbol"][0] == "USDCUSDT"


def test_parse_incremental_book_l2_sequence_ids():
    raw = _INC_L2_CSV.encode()
    df = _parse_incremental_book_l2(raw, "binance", "USDCUSDT")
    assert df["sequence_id"].to_list() == [100, 101, 102]


def test_parse_incremental_book_l2_empty_input():
    df = _parse_incremental_book_l2(b"", "binance", "USDCUSDT")
    assert df.is_empty()


def test_parse_incremental_book_l2_header_only():
    raw = b"exchange\tsymbol\ttimestamp\tlocal_timestamp\tis_snapshot\tside\tprice\tamount\tsequence_id\n"
    df = _parse_incremental_book_l2(raw, "binance", "USDCUSDT")
    assert df.is_empty()


# ---------------------------------------------------------------------------
# book_snapshot_25 parsing
# ---------------------------------------------------------------------------

def _make_snapshot_csv() -> bytes:
    """Minimal 25-level snapshot CSV."""
    headers = ["exchange", "symbol", "timestamp", "local_timestamp"]
    for side in ("bids", "asks"):
        for i in range(25):
            headers += [f"{side}[{i}].price", f"{side}[{i}].amount"]
    row_vals = ["binance", "USDCUSDT", "1678460000000000", "1678460000100000"]
    for i in range(25):
        row_vals += [str(1.0 - (i+1)*0.0001), str(50000.0 - i*100)]  # bids
    for i in range(25):
        row_vals += [str(1.0 + (i+1)*0.0001), str(30000.0 - i*100)]  # asks
    header_line = "\t".join(headers)
    data_line   = "\t".join(row_vals)
    return (header_line + "\n" + data_line + "\n").encode()


def test_parse_book_snapshot_25_schema():
    raw = _make_snapshot_csv()
    df = _parse_book_snapshot_25(raw, "binance", "USDCUSDT")
    for col in _BRONZE_COLS:
        assert col in df.columns, f"Missing column '{col}'"


def test_parse_book_snapshot_25_row_count():
    raw = _make_snapshot_csv()
    df = _parse_book_snapshot_25(raw, "binance", "USDCUSDT")
    # 1 snapshot row × 25 bid levels + 25 ask levels = 50 rows
    assert len(df) == 50


def test_parse_book_snapshot_25_update_type():
    raw = _make_snapshot_csv()
    df = _parse_book_snapshot_25(raw, "binance", "USDCUSDT")
    assert (df["update_type"] == "snapshot").all()


# ---------------------------------------------------------------------------
# load_bronze — path filtering
# ---------------------------------------------------------------------------

def test_load_bronze_returns_empty_when_no_files(tmp_path):
    df = load_bronze("binance", "USDCUSDT", "incremental_book_L2", tmp_path)
    assert df.is_empty()


def test_load_bronze_loads_existing_file(tmp_path):
    # Write a minimal bronze parquet
    df_in = pl.DataFrame(
        {col: [] for col in _BRONZE_COLS},
        schema=_BRONZE_SCHEMA,
    )
    # Add one row
    from stressnet.data.tardis_l2 import _empty_bronze
    sample = pl.DataFrame(
        {
            "wall_clock_utc": [1678460000000000.0],
            "exchange_ts":    [1678460000000000.0],
            "local_ts":       [1678460000100000.0],
            "exchange":       ["binance"],
            "symbol":         ["USDCUSDT"],
            "update_type":    ["delta"],
            "side":           ["bid"],
            "price":          [0.9999],
            "size":           [50000.0],
            "sequence_id":    [100],
            "raw_msg_id":     [0],
            "row_position":   [0],
        },
        schema=_BRONZE_SCHEMA,
    )
    fname = tmp_path / "binance_USDCUSDT_incremental_book_L2_2023-03-10.parquet"
    sample.write_parquet(fname)

    df_out = load_bronze("binance", "USDCUSDT", "incremental_book_L2", tmp_path)
    assert len(df_out) == 1
    assert abs(df_out["price"][0] - 0.9999) < 1e-8
