"""Tests for the manifest provenance write/read system."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_row(manifest_path: Path, event_id: str = "test_event",
               node_id: str = "test_node", file_path: str = "fake_file.parquet",
               tier: str = "B", rows: int = 100) -> None:
    """Call write_manifest_row via the public API with a dummy file path."""
    from stressnet.utils.manifest import append_manifest_row
    # Create a real (empty) file so sha256 doesn't error
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_bytes(b"")
    append_manifest_row(
        manifest_path,
        event_id=event_id,
        node_id=node_id,
        source_name="test_source",
        source_tier_nominal=tier,
        source_tier_actual=tier,
        layer="CEX",
        file_stage="bronze",
        file_path=p,
        start_utc="2023-03-08T00:00:00Z",
        end_utc="2023-03-20T23:59:59Z",
        row_count=rows,
        url_or_query="https://test.example/data",
        notes="unit test row",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_write_manifest_row_creates_file(tmp_path):
    """write_manifest_row should create the manifest CSV if it does not exist."""
    manifest_path = tmp_path / "manifests" / "manifest_test.csv"
    assert not manifest_path.exists()

    fake_file = tmp_path / "fake.parquet"
    fake_file.write_bytes(b"")
    _write_row(manifest_path, file_path=str(fake_file))

    assert manifest_path.exists()
    df = pl.read_csv(manifest_path)
    assert df.height == 1
    assert df["event_id"][0] == "test_event"
    assert df["node_id"][0] == "test_node"


def test_write_manifest_row_appends_multiple_rows(tmp_path):
    """Each call to write_manifest_row should add a row."""
    manifest_path = tmp_path / "manifests" / "manifest_test.csv"

    for i in range(3):
        fake_file = tmp_path / f"file_{i}.parquet"
        fake_file.write_bytes(b"")
        _write_row(manifest_path, node_id=f"node_{i}", file_path=str(fake_file))

    df = pl.read_csv(manifest_path)
    assert df.height == 3
    assert set(df["node_id"].to_list()) == {"node_0", "node_1", "node_2"}


def test_write_manifest_row_deduplicates_by_file_path(tmp_path):
    """Writing the same file_path twice should keep only the latest row."""
    manifest_path = tmp_path / "manifests" / "manifest_test.csv"
    fake_file = tmp_path / "same_file.parquet"
    fake_file.write_bytes(b"")

    _write_row(manifest_path, node_id="node_a", rows=50, file_path=str(fake_file))
    _write_row(manifest_path, node_id="node_a_updated", rows=100, file_path=str(fake_file))

    df = pl.read_csv(manifest_path)
    # Deduplication keeps last write
    assert df.height == 1
    assert df["row_count"][0] == 100


def test_manifest_tier_columns_preserved(tmp_path):
    """tier_nominal and tier_actual should survive a write/read cycle."""
    manifest_path = tmp_path / "manifests" / "manifest_test.csv"
    fake_file = tmp_path / "f.parquet"
    fake_file.write_bytes(b"")
    _write_row(manifest_path, tier="A", file_path=str(fake_file))

    df = pl.read_csv(manifest_path)
    assert df["source_tier_nominal"][0] == "A"
    assert df["source_tier_actual"][0] == "A"


def test_manifest_sha256_recorded(tmp_path):
    """sha256 column should be a non-empty hex string."""
    manifest_path = tmp_path / "manifests" / "manifest_test.csv"
    fake_file = tmp_path / "data.parquet"
    fake_file.write_bytes(b"hello world")
    _write_row(manifest_path, file_path=str(fake_file))

    df = pl.read_csv(manifest_path)
    sha = df["sha256"][0]
    assert isinstance(sha, str) and len(sha) == 64
    # Must be valid hex
    int(sha, 16)


def test_manifest_all_required_columns_present(tmp_path):
    """All MANIFEST_COLUMNS should appear in the written CSV."""
    from stressnet.utils.manifest import MANIFEST_COLUMNS
    manifest_path = tmp_path / "manifests" / "manifest_test.csv"
    fake_file = tmp_path / "f.parquet"
    fake_file.write_bytes(b"")
    _write_row(manifest_path, file_path=str(fake_file))

    df = pl.read_csv(manifest_path)
    for col in MANIFEST_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"


def test_manifest_records_optional_quality_diagnostics(tmp_path):
    from stressnet.utils.manifest import append_manifest_row

    manifest_path = tmp_path / "manifests" / "manifest_test.csv"
    fake_file = tmp_path / "f.parquet"
    fake_file.write_bytes(b"")

    append_manifest_row(
        manifest_path,
        event_id="test_event",
        node_id="test_node",
        source_name="test_source",
        source_tier_nominal="A",
        source_tier_actual="A",
        layer="CEX",
        file_stage="silver",
        file_path=fake_file,
        start_utc="2023-03-08T00:00:00Z",
        end_utc="2023-03-20T23:59:59Z",
        row_count=100,
        url_or_query="https://test.example/data",
        coverage_pct=42.5,
        sequence_gap_count=3,
        gap_rate=0.03,
        resync_count=1,
        clock_offset_ms=120.0,
    )

    df = pl.read_csv(manifest_path)
    assert df["coverage_pct"][0] == 42.5
    assert df["sequence_gap_count"][0] == 3
    assert df["gap_rate"][0] == 0.03
    assert df["resync_count"][0] == 1
    assert df["clock_offset_ms"][0] == 120.0


def test_coverage_table_downgrades_tier_a_with_low_coverage(tmp_path, monkeypatch):
    import stressnet.utils.manifest as manifest_mod

    manifests = tmp_path / "manifests"
    results = tmp_path / "results"
    monkeypatch.setattr(manifest_mod, "manifests_root", lambda: manifests)
    monkeypatch.setattr(manifest_mod, "results_root", lambda: results)

    fake_file = tmp_path / "node.parquet"
    fake_file.write_bytes(b"")
    manifest_mod.append_manifest_row(
        manifests / "manifest_test_event.csv",
        event_id="test_event",
        node_id="tier_a_node",
        source_name="test_source",
        source_tier_nominal="A",
        source_tier_actual="A",
        layer="CEX",
        file_stage="silver",
        file_path=fake_file,
        start_utc="2023-03-08T00:00:00Z",
        end_utc="2023-03-20T23:59:59Z",
        row_count=10,
        url_or_query="https://test.example/data",
        coverage_pct=49.0,
    )

    out = manifest_mod.build_node_coverage_table()
    assert out == results / "tables" / "table_node_coverage.csv"
    coverage = pl.read_csv(out)
    assert coverage["source_tier_actual"][0] == "B"
    assert coverage["coverage_pct"][0] == 49.0
