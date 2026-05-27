"""Tests for Hawkes stress event definition and output shapes.

Does not require tick to be installed — tests only the event-extraction logic.
"""

import numpy as np
import polars as pl
import pytest
from datetime import datetime, timezone

from stressnet.models.hawkes import define_stress_events


def _make_stress_panel() -> pl.DataFrame:
    """Create a synthetic panel with some basis exceedances."""
    n = 200
    ts = np.linspace(-100, 100, n)
    basis = np.zeros(n)
    basis[50] = 0.002   # 20 bps exceedance
    basis[51] = 0.0015
    basis[100] = -0.003  # 30 bps exceedance
    basis[150] = 0.001
    basis[151] = 0.0012
    return pl.DataFrame({
        "node_id": ["usdc_coinbase"] * n + ["curve_3pool"] * n,
        "event_time_seconds": list(ts) * 2,
        "basis_vs_usd": list(basis) + list(basis * 0.5),
    })


def test_stress_events_extracted():
    panel = _make_stress_panel()
    events = define_stress_events(panel, ["usdc_coinbase", "curve_3pool"], threshold_bps=10.0)
    assert "usdc_coinbase" in events
    assert "curve_3pool" in events


def test_stress_events_are_1d_arrays():
    panel = _make_stress_panel()
    events = define_stress_events(panel, ["usdc_coinbase"], threshold_bps=10.0)
    arr = events["usdc_coinbase"]
    assert arr.ndim == 1


def test_stress_events_are_sorted():
    panel = _make_stress_panel()
    events = define_stress_events(panel, ["usdc_coinbase"], threshold_bps=10.0)
    arr = events["usdc_coinbase"]
    assert np.all(arr[:-1] <= arr[1:])


def test_no_events_below_threshold():
    panel = _make_stress_panel()
    events = define_stress_events(panel, ["usdc_coinbase"], threshold_bps=50.0)
    # 20 bps exceedance < 50 bps threshold → should have no events
    assert len(events["usdc_coinbase"]) == 0


def test_empty_node_returns_empty_array():
    panel = pl.DataFrame({
        "node_id": ["usdc_coinbase"],
        "event_time_seconds": [0.0],
        "basis_vs_usd": [0.0],
    })
    events = define_stress_events(panel, ["usdc_coinbase", "curve_3pool"], threshold_bps=10.0)
    assert "curve_3pool" in events
    assert len(events["curve_3pool"]) == 0
