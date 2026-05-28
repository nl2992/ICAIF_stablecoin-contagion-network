import polars as pl

from stressnet.evaluation.coverage_gate import check_empirical_coverage


def _panel(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def test_coverage_gate_fails_with_too_few_real_nodes() -> None:
    panel = _panel(
        [
            {
                "event_id": "e",
                "node_id": "a",
                "tier_actual": "B",
                "layer": "CEX",
                "basis_vs_usd": 0.0,
            },
            {
                "event_id": "e",
                "node_id": "b",
                "tier_actual": "fixture_non_empirical",
                "layer": "DEX",
                "basis_vs_usd": 0.0,
            },
        ]
    )

    result = check_empirical_coverage(panel, event_id="e", min_real_nodes=3, min_var_nodes=2)

    assert result.passes is False
    assert "real nodes" in result.reason


def test_coverage_gate_counts_var_eligible_real_nodes() -> None:
    rows = []
    for node_id, layer in [("a", "CEX"), ("b", "DEX"), ("c", "mint_burn")]:
        for t in range(25):
            rows.append(
                {
                    "event_id": "e",
                    "node_id": node_id,
                    "tier_actual": "A",
                    "layer": layer,
                    "basis_vs_usd": 0.001 * t,
                }
            )
    panel = _panel(rows)

    result = check_empirical_coverage(
        panel,
        event_id="e",
        min_real_nodes=3,
        min_var_nodes=2,
        required_layers=("CEX", "DEX"),
    )

    assert result.passes is True
    assert result.n_nodes_real == 3
    assert result.n_var_eligible_nodes == 3


def test_coverage_gate_fails_when_required_layer_missing() -> None:
    rows = []
    for node_id in ["a", "b", "c"]:
        for t in range(25):
            rows.append(
                {
                    "event_id": "e",
                    "node_id": node_id,
                    "tier_actual": "B",
                    "layer": "CEX",
                    "basis_vs_usd": 0.001 * t,
                }
            )
    panel = _panel(rows)

    result = check_empirical_coverage(
        panel,
        event_id="e",
        min_real_nodes=3,
        min_var_nodes=2,
        required_layers=("CEX", "DEX"),
    )

    assert result.passes is False
    assert "missing required real layers" in result.reason
