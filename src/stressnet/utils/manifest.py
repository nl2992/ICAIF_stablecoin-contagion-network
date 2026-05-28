"""Manifest helpers for provenance-audited data artefacts."""

from __future__ import annotations

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
    "layer",
    "file_stage",
    "start_utc",
    "end_utc",
    "file_path",
    "row_count",
    "sha256",
    "url_or_query",
    "created_utc",
    "downloaded_at_utc",
    "notes",
    "coverage_pct",
    "sequence_gap_count",
    "gap_rate",
    "resync_count",
    "clock_offset_ms",
]

_MANIFEST_DTYPES = {
    **{
        col: pl.Utf8
        for col in MANIFEST_COLUMNS
        if col
        not in {
            "row_count",
            "coverage_pct",
            "sequence_gap_count",
            "gap_rate",
            "resync_count",
            "clock_offset_ms",
        }
    },
    "row_count": pl.Int64,
    "coverage_pct": pl.Float64,
    "sequence_gap_count": pl.Int64,
    "gap_rate": pl.Float64,
    "resync_count": pl.Int64,
    "clock_offset_ms": pl.Float64,
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def append_manifest_row(
    manifest_path: Path,
    *,
    event_id: str,
    node_id: str,
    source_name: str,
    source_tier_nominal: str,
    source_tier_actual: str,
    layer: str,
    file_stage: str,
    file_path: Path,
    start_utc: str | None,
    end_utc: str | None,
    row_count: int,
    url_or_query: str | None,
    notes: str = "",
    coverage_pct: float | None = None,
    sequence_gap_count: int | None = None,
    gap_rate: float | None = None,
    resync_count: int | None = None,
    clock_offset_ms: float | None = None,
) -> None:
    """Append one provenance row using a stable, audit-friendly schema."""
    now = datetime.now(timezone.utc).isoformat()
    row: dict[str, Any] = {
        "event_id": event_id,
        "node_id": node_id,
        "source_name": source_name,
        "source_tier_nominal": source_tier_nominal,
        "source_tier_actual": source_tier_actual,
        "layer": layer,
        "file_stage": file_stage,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "file_path": str(file_path),
        "row_count": row_count,
        "sha256": sha256_file(file_path) if file_path.exists() else "",
        "url_or_query": url_or_query,
        "created_utc": now,
        "downloaded_at_utc": now,
        "notes": notes,
        "coverage_pct": coverage_pct,
        "sequence_gap_count": sequence_gap_count,
        "gap_rate": gap_rate,
        "resync_count": resync_count,
        "clock_offset_ms": clock_offset_ms,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    new = pl.DataFrame([row]).select(MANIFEST_COLUMNS)
    new = new.with_columns([pl.col(col).cast(dtype) for col, dtype in _MANIFEST_DTYPES.items()])
    if manifest_path.exists():
        old = pl.read_csv(manifest_path)
        for col in MANIFEST_COLUMNS:
            if col not in old.columns:
                old = old.with_columns(pl.lit(None, dtype=_MANIFEST_DTYPES[col]).alias(col))
        old = old.select(MANIFEST_COLUMNS).with_columns(
            [pl.col(col).cast(dtype) for col, dtype in _MANIFEST_DTYPES.items()]
        )
        new = pl.concat([old, new], how="diagonal").unique(
            subset=["file_path"], keep="last", maintain_order=True
        )
    new.write_csv(manifest_path)


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
    layer: str = "",
    file_stage: str = "",
    url_or_query: str | None = None,
    coverage_pct: float | None = None,
    sequence_gap_count: int | None = None,
    gap_rate: float | None = None,
    resync_count: int | None = None,
    clock_offset_ms: float | None = None,
) -> Path:
    """Append one artefact provenance row to the per-event manifest."""
    manifest_path = manifests_root() / f"manifest_{event_id}.csv"
    append_manifest_row(
        manifest_path,
        event_id=event_id,
        node_id=node_id,
        source_name=source_name,
        source_tier_nominal=source_tier_nominal,
        source_tier_actual=source_tier_actual,
        layer=layer,
        file_stage=file_stage,
        file_path=file_path,
        start_utc=start_utc,
        end_utc=end_utc,
        row_count=row_count,
        url_or_query=url_or_query,
        notes=notes,
        coverage_pct=coverage_pct,
        sequence_gap_count=sequence_gap_count,
        gap_rate=gap_rate,
        resync_count=resync_count,
        clock_offset_ms=clock_offset_ms,
    )
    return manifest_path


def _normalise_manifest_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Backfill optional manifest diagnostics for old manifest files."""
    for col in MANIFEST_COLUMNS:
        if col not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=_MANIFEST_DTYPES[col]).alias(col))
    return frame.select(MANIFEST_COLUMNS).with_columns(
        [pl.col(col).cast(dtype, strict=False) for col, dtype in _MANIFEST_DTYPES.items()]
    )


def _coverage_downgraded_tier(row: dict[str, Any]) -> str:
    """Downgrade nominally strong tiers when coverage diagnostics are weak."""
    tier = str(row.get("source_tier_actual") or "")
    if tier in {"fixture_non_empirical", "missing", "mixed", "C"}:
        return tier
    coverage_pct = row.get("coverage_pct")
    gap_rate = row.get("gap_rate")
    if tier == "A":
        if coverage_pct is not None and coverage_pct < 50.0:
            return "B"
        if gap_rate is not None and gap_rate > 0.01:
            return "B"
    return tier


def build_node_coverage_table() -> Path | None:
    """Summarise all manifests into results/tables/table_node_coverage.csv."""
    manifest_paths = sorted(manifests_root().glob("manifest_*.csv"))
    if not manifest_paths:
        return None

    frames = [_normalise_manifest_frame(pl.read_csv(path)) for path in manifest_paths]
    manifest = pl.concat(frames, how="diagonal")
    coverage = (
        manifest.group_by(["event_id", "node_id"])
        .agg(
            pl.col("source_tier_nominal").first(),
            pl.col("source_tier_actual").last(),
            pl.col("source_name").unique().str.join(";").alias("sources"),
            pl.col("row_count").sum().alias("rows_available"),
            pl.col("start_utc").min().alias("start_utc"),
            pl.col("end_utc").max().alias("end_utc"),
            pl.col("file_path").n_unique().alias("artefact_count"),
            pl.col("coverage_pct").drop_nulls().last().alias("coverage_pct"),
            pl.col("sequence_gap_count").fill_null(0).sum().alias("sequence_gap_count"),
            pl.col("gap_rate").drop_nulls().max().alias("gap_rate"),
            pl.col("resync_count").fill_null(0).sum().alias("resync_count"),
            pl.col("clock_offset_ms").drop_nulls().max().alias("clock_offset_ms"),
        )
        .sort(["event_id", "node_id"])
    )
    if coverage.height:
        coverage = coverage.with_columns(
            pl.struct(["source_tier_actual", "coverage_pct", "gap_rate"])
            .map_elements(_coverage_downgraded_tier, return_dtype=pl.Utf8)
            .alias("source_tier_actual")
        )
    out_path = results_root() / "tables" / "table_node_coverage.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coverage.write_csv(out_path)
    return out_path
