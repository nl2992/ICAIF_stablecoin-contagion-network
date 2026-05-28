"""Attach provenance tiers to result edges and gate paper claims.

The model scripts intentionally stay focused on estimation. This script is the
paper-safety layer: it joins result edges back to node provenance, caps each
edge by the weaker endpoint tier, and can fail a paper build when fixture or
missing nodes appear in claim-bearing tables.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl

from stressnet.config import load_events, manifests_root, results_root
from stressnet.graph.nodes import nodes_for_event
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

FIXTURE = "fixture_non_empirical"
MISSING = "missing"
TIER_ORDER = {"A": 0, "B": 1, "C": 2, FIXTURE: 3, MISSING: 4}
EDGE_COLUMN_CANDIDATES = [
    ("node_i", "node_j"),
    ("causing_node", "caused_node"),
    ("source", "target"),
    ("source_node", "target_node"),
]
RESULT_TABLE_PREFIXES = (
    "table_leadlag_tests",
    "table_transfer_entropy",
    "table_granger",
    "table_var_spillovers",
    "table_tvp_var_edges",
    "table_hawkes_params",
    "table_edges",
)


def _tier_rank(tier: str | None) -> int:
    return TIER_ORDER.get(str(tier) if tier is not None else MISSING, TIER_ORDER[MISSING])


def _weaker_tier(tier_i: str | None, tier_j: str | None) -> str:
    left = str(tier_i) if tier_i is not None else MISSING
    right = str(tier_j) if tier_j is not None else MISSING
    return left if _tier_rank(left) >= _tier_rank(right) else right


def _claim_allowed(tier_i: str | None, tier_j: str | None) -> bool:
    return _weaker_tier(tier_i, tier_j) not in {FIXTURE, MISSING}


def _infer_event_from_name(path: Path) -> str | None:
    event_ids = sorted(load_events().keys(), key=len, reverse=True)
    for event_id in event_ids:
        if path.stem.endswith(f"_{event_id}") or f"_{event_id}_" in path.stem:
            return event_id
    return None


def _load_tiers_from_coverage(tables_dir: Path) -> dict[str, dict[str, str]]:
    tiers: dict[str, dict[str, str]] = {}
    coverage_path = tables_dir / "table_node_coverage.csv"
    if not coverage_path.exists():
        return tiers

    coverage = pl.read_csv(coverage_path)
    tier_col = "source_tier_actual" if "source_tier_actual" in coverage.columns else "tier_actual"
    if not {"event_id", "node_id", tier_col}.issubset(coverage.columns):
        return tiers

    for row in coverage.iter_rows(named=True):
        event_id = row.get("event_id")
        node_id = row.get("node_id")
        if not event_id or not node_id or node_id == "__event_panel__":
            continue
        tiers.setdefault(str(event_id), {})[str(node_id)] = str(row.get(tier_col) or MISSING)
    return tiers


def _load_tiers_from_manifest(event_id: str) -> dict[str, str]:
    manifest_path = manifests_root() / f"manifest_{event_id}.csv"
    configured = {node.id for node in nodes_for_event(event_id)}
    if not manifest_path.exists():
        return {}

    tiers = {node_id: MISSING for node_id in configured}
    manifest = pl.read_csv(manifest_path)
    if "node_id" not in manifest.columns or "source_tier_actual" not in manifest.columns:
        return tiers
    manifest = manifest.filter(pl.col("node_id") != "__event_panel__")

    if "file_stage" in manifest.columns:
        silver = manifest.filter(pl.col("file_stage") == "silver")
        if silver.height > 0:
            manifest = silver

    for row in manifest.iter_rows(named=True):
        node_id = row.get("node_id")
        if node_id in configured:
            tiers[str(node_id)] = str(row.get("source_tier_actual") or MISSING)
    return tiers


def _load_tier_map(events: list[str], tables_dir: Path) -> dict[str, dict[str, str]]:
    tier_map = _load_tiers_from_coverage(tables_dir)
    for event_id in events:
        manifest_tiers = _load_tiers_from_manifest(event_id)
        if event_id not in tier_map:
            tier_map[event_id] = manifest_tiers
        else:
            tier_map[event_id] = {**tier_map[event_id], **manifest_tiers}
    return tier_map


def _edge_columns(df: pl.DataFrame) -> tuple[str, str] | None:
    for source_col, target_col in EDGE_COLUMN_CANDIDATES:
        if source_col in df.columns and target_col in df.columns:
            return source_col, target_col
    return None


def _is_result_table(path: Path) -> bool:
    if path.suffix != ".csv":
        return False
    return path.name.startswith(RESULT_TABLE_PREFIXES)


def _event_tables(event_id: str, tables_dir: Path) -> list[Path]:
    pattern = re.compile(rf"^({'|'.join(RESULT_TABLE_PREFIXES)}).*_{re.escape(event_id)}\.csv$")
    return sorted(path for path in tables_dir.glob("*.csv") if pattern.match(path.name))


def _paper_tables(tables_dir: Path) -> list[Path]:
    return sorted(path for path in tables_dir.glob("*.csv") if _is_result_table(path))


def _row_event(row: dict[str, object], path: Path, default_event: str | None) -> str | None:
    value = row.get("event_id")
    if value:
        return str(value)
    return default_event or _infer_event_from_name(path)


def _annotate_table(
    path: Path,
    tier_map: dict[str, dict[str, str]],
    default_event: str | None,
) -> tuple[pl.DataFrame | None, dict[str, object]]:
    df = pl.read_csv(path)
    cols = _edge_columns(df)
    summary = {
        "table": path.name,
        "event_id": default_event or "multi",
        "status": "skipped_no_edge_columns",
        "rows": df.height,
        "blocked_rows": 0,
        "fixture_or_missing_rows": 0,
    }
    if cols is None or df.height == 0:
        return None, summary

    source_col, target_col = cols
    annotated = []
    blocked_rows = 0
    fixture_or_missing_rows = 0
    for row in df.iter_rows(named=True):
        event_id = _row_event(row, path, default_event)
        event_tiers = tier_map.get(event_id or "", {})
        source = str(row.get(source_col))
        target = str(row.get(target_col))
        tier_i = event_tiers.get(source, MISSING)
        tier_j = event_tiers.get(target, MISSING)
        edge_tier = _weaker_tier(tier_i, tier_j)
        allowed = _claim_allowed(tier_i, tier_j)
        if not allowed:
            blocked_rows += 1
        if edge_tier in {FIXTURE, MISSING}:
            fixture_or_missing_rows += 1

        row["tier_i_actual"] = tier_i
        row["tier_j_actual"] = tier_j
        row["edge_tier_actual"] = edge_tier
        row["claim_allowed"] = allowed
        annotated.append(row)

    out = pl.DataFrame(annotated)
    summary.update(
        {
            "status": "annotated",
            "rows": out.height,
            "blocked_rows": blocked_rows,
            "fixture_or_missing_rows": fixture_or_missing_rows,
        }
    )
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate result claims by node provenance tiers.")
    parser.add_argument("--event", default=None, help="Annotate one event's result tables.")
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Check all paper-relevant result tables instead of one event.",
    )
    parser.add_argument(
        "--require-real",
        action="store_true",
        help="Exit nonzero if any processed edge table uses fixture or missing endpoint tiers.",
    )
    args = parser.parse_args()

    if not args.event and not args.paper:
        raise SystemExit("Provide --event or --paper.")

    tables_dir = results_root() / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    events = list(load_events().keys()) if args.paper else [args.event]
    tier_map = _load_tier_map([event for event in events if event], tables_dir)

    if args.paper:
        tables = _paper_tables(tables_dir)
        default_event = None
        summary_name = "table_claim_gate_paper.csv"
    else:
        tables = _event_tables(args.event, tables_dir)
        default_event = args.event
        summary_name = f"table_claim_gate_{args.event}.csv"

    if not tables:
        message = "No result tables found to gate."
        if args.paper or args.require_real:
            raise SystemExit(message)
        logger.info(message)
        return

    summaries = []
    blocked_total = 0
    for path in tables:
        annotated, summary = _annotate_table(path, tier_map, default_event)
        summaries.append(summary)
        if annotated is not None:
            annotated.write_csv(path)
            blocked_total += int(summary["blocked_rows"])
            logger.info(
                "Annotated %s: %d blocked rows / %d rows",
                path.name,
                summary["blocked_rows"],
                summary["rows"],
            )

    summary_df = pl.DataFrame(summaries)
    summary_path = tables_dir / summary_name
    summary_df.write_csv(summary_path)
    logger.info("Wrote %s", summary_path)

    if args.require_real and blocked_total > 0:
        raise SystemExit(
            f"Claim gate failed: {blocked_total} edge rows use fixture or missing endpoint tiers."
        )

    print(summary_df)


if __name__ == "__main__":
    main()
