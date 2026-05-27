"""Manifest helpers for provenance-audited data artefacts."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from stressnet.config import manifests_root, results_root

MANIFEST_COLUMNS = [
    "event_id",
    "node_id",
    "source_name",
    "source_tier_nominal",
    "source_tier_actual",
    "start_utc",
    "end_utc",
    "file_path",
    "row_count",
    "sha256",
    "created_utc",
    "notes",
]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest_row(
    event_id: str,
    node_id: str,
    source_name: str,
    source_tier_nominal: str,
    source_tier_actual: str,
    start_utc: str,
    end_utc: str,
    file_path: Path,
    row_count: int,
    notes: str = "",
) -> Path:
    """Append one artefact provenance row to the per-event manifest."""
    out_dir = manifests_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"manifest_{event_id}.csv"
    row: dict[str, Any] = {
        "event_id": event_id,
        "node_id": node_id,
        "source_name": source_name,
        "source_tier_nominal": source_tier_nominal,
        "source_tier_actual": source_tier_actual,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "file_path": str(file_path),
        "row_count": row_count,
        "sha256": sha256_file(file_path) if file_path.exists() else "",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
    }

    write_header = not manifest_path.exists()
    with manifest_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return manifest_path


def build_node_coverage_table() -> Path | None:
    """Summarise all manifests into results/tables/table_node_coverage.csv."""
    manifest_paths = sorted(manifests_root().glob("manifest_*.csv"))
    if not manifest_paths:
        return None

    frames = [pl.read_csv(path) for path in manifest_paths]
    manifest = pl.concat(frames, how="diagonal")
    coverage = (
        manifest.group_by(["event_id", "node_id"])
        .agg(
            pl.col("source_tier_nominal").first(),
            pl.col("source_tier_actual").first(),
            pl.col("source_name").unique().str.join(";").alias("sources"),
            pl.col("row_count").sum().alias("rows_available"),
            pl.col("start_utc").min().alias("start_utc"),
            pl.col("end_utc").max().alias("end_utc"),
            pl.col("file_path").n_unique().alias("artefact_count"),
        )
        .sort(["event_id", "node_id"])
    )
    out_path = results_root() / "tables" / "table_node_coverage.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coverage.write_csv(out_path)
    return out_path
