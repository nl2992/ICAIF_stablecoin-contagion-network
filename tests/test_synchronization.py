import polars as pl

from stressnet.features.synchronization import synchronize_node_feature


def test_synchronize_node_feature_forward_fills_with_staleness_limit() -> None:
    panel = pl.DataFrame(
        {
            "node_id": ["a", "a", "b", "b"],
            "event_time_seconds": [0, 120, 0, 180],
            "basis_vs_usd": [1.0, 2.0, 10.0, 20.0],
        }
    )

    synced = synchronize_node_feature(
        panel,
        node_ids=["a", "b"],
        feature_col="basis_vs_usd",
        grid_seconds=60,
        max_staleness_seconds=60,
    )

    a = synced.filter(pl.col("node_id") == "a").sort("event_time_seconds")
    assert a["basis_vs_usd"].to_list()[:4] == [1.0, 1.0, 2.0, 2.0]

    b = synced.filter(pl.col("node_id") == "b").sort("event_time_seconds")
    assert b["basis_vs_usd"].to_list()[:4] == [10.0, 10.0, None, 20.0]
    assert b["is_stale"].to_list()[:4] == [False, False, True, False]


def test_synchronize_node_feature_rejects_missing_feature() -> None:
    panel = pl.DataFrame({"node_id": ["a"], "event_time_seconds": [0]})

    try:
        synchronize_node_feature(
            panel,
            node_ids=["a"],
            feature_col="basis_vs_usd",
            grid_seconds=60,
        )
    except ValueError as exc:
        assert "basis_vs_usd" in str(exc)
    else:
        raise AssertionError("expected missing feature ValueError")
