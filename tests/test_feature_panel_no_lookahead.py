"""Tests for no-lookahead validation of feature panels."""

import pytest
import polars as pl
from datetime import datetime, timezone

from stressnet.utils.validation import check_no_lookahead, check_no_future_features


def _make_panel(n: int = 100) -> pl.DataFrame:
    import numpy as np
    from datetime import timedelta
    base = datetime(2023, 3, 10, 0, 0, 0, tzinfo=timezone.utc)
    ts = [base + timedelta(seconds=i) for i in range(n)]
    return pl.DataFrame({
        "wall_clock_utc": ts,
        "event_time_seconds": list(range(n)),
        "node_id": ["usdc_coinbase"] * n,
        "basis_vs_usd": list(np.random.randn(n) * 0.001),
        "spread_bps": list(np.abs(np.random.randn(n)) * 2 + 1),
        "label_basis_gt10bps": [abs(x) > 0.001 for x in np.random.randn(n) * 0.001],
    })


def test_no_lookahead_passes():
    panel = _make_panel()
    feature_cols = ["basis_vs_usd", "spread_bps"]
    label_cols = ["label_basis_gt10bps"]
    assert check_no_lookahead(panel, feature_cols, label_cols)


def test_no_lookahead_raises_on_null_ts():
    panel = _make_panel()
    feature_cols = ["basis_vs_usd", "spread_bps"]
    label_cols = ["label_basis_gt10bps"]
    # Inject null timestamp with non-null label
    bad_row = pl.DataFrame({
        "wall_clock_utc": [None],
        "event_time_seconds": [None],
        "node_id": ["usdc_coinbase"],
        "basis_vs_usd": [None],
        "spread_bps": [None],
        "label_basis_gt10bps": [True],
    })
    bad_panel = pl.concat([panel, bad_row.cast(panel.schema)], how="vertical")
    with pytest.raises(ValueError, match="Lookahead risk"):
        check_no_lookahead(bad_panel, feature_cols, label_cols, ts_col="wall_clock_utc")


def test_no_future_features_passes():
    panel = _make_panel()
    assert check_no_future_features(panel)


def test_no_future_features_fails_on_scrambled_event_time():
    import numpy as np
    panel = _make_panel()
    # Scramble event_time_seconds so it is NOT monotone when sorted by wall_clock_utc.
    # Replace values with a decreasing sequence to guarantee negative diffs.
    n = panel.height
    scrambled = panel.with_columns(
        pl.Series("event_time_seconds", list(range(n - 1, -1, -1)), dtype=pl.Float64)
    )
    with pytest.raises(ValueError, match="Lookahead risk"):
        check_no_future_features(scrambled)
