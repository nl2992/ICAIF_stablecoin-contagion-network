"""Tests for Uniswap ingestion helpers."""

from __future__ import annotations

from stressnet.data.uniswap import parse_graph_decimal


def test_parse_graph_decimal_does_not_rescale_bigdecimal_amounts():
    """The Graph BigDecimal amounts are already token-decimal adjusted."""
    assert parse_graph_decimal("1234.56789") == 1234.56789


def test_parse_graph_decimal_handles_bad_values():
    assert parse_graph_decimal(None) == 0.0
    assert parse_graph_decimal("not-a-number") == 0.0
    assert parse_graph_decimal("nan") == 0.0
