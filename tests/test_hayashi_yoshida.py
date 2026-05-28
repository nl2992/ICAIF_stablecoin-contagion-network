import numpy as np
import polars as pl

from stressnet.models.hayashi_yoshida import (
    compute_hayashi_yoshida_table,
    hayashi_yoshida_correlation,
)


def test_hayashi_yoshida_correlation_is_positive_for_similar_series() -> None:
    tx = np.array([0, 1, 3, 6, 10], dtype=float)
    ty = np.array([0, 2, 4, 8, 10], dtype=float)
    x = np.array([1.0, 1.1, 1.3, 1.6, 2.0])
    y = np.array([1.0, 1.2, 1.35, 1.7, 2.1])

    corr = hayashi_yoshida_correlation(tx, x, ty, y)

    assert corr > 0


def test_compute_hayashi_yoshida_table_returns_pair_rows() -> None:
    panel = pl.DataFrame(
        {
            "node_id": ["a", "a", "a", "b", "b", "b"],
            "event_time_seconds": [0, 60, 180, 0, 120, 240],
            "basis_vs_usd": [0.0, 0.1, 0.2, 0.0, 0.12, 0.22],
        }
    )

    table = compute_hayashi_yoshida_table(panel, [("a", "b")])

    assert table.height == 1
    assert table["node_i"].to_list() == ["a"]
    assert "hy_corr" in table.columns
