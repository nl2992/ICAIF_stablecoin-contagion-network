"""Summarise real-node-only preliminary results across events.

This script is deliberately provenance-aware: any node with
``tier_actual == fixture_non_empirical`` is excluded from the real-only edge
counts. The resulting tables are suitable for checkpoint reporting, not for
strong paper claims until source provenance and placebo checks are complete.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from stressnet.config import load_events, manifests_root, results_root
from stressnet.graph.nodes import nodes_for_event
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

FIXTURE = "fixture_non_empirical"
ALPHA = 0.05


def _latest_node_tiers(event_id: str) -> dict[str, str]:
    manifest_path = manifests_root() / f"manifest_{event_id}.csv"
    if not manifest_path.exists():
        return {}

    manifest = pl.read_csv(manifest_path)
    manifest = manifest.filter(pl.col("node_id") != "__event_panel__")
    if "file_stage" in manifest.columns:
        silver = manifest.filter(pl.col("file_stage") == "silver")
        if silver.height > 0:
            manifest = silver

    tiers = {}
    for row in manifest.iter_rows(named=True):
        tiers[row["node_id"]] = row["source_tier_actual"]
    return tiers


def _edge_stats(
    path: Path,
    real_nodes: set[str],
    source_col: str,
    target_col: str,
    raw_sig_col: str,
    fdr_sig_col: str = "significant_fdr",
    p_col: str = "p_value",
) -> dict[str, int]:
    empty = {"total": 0, "raw_sig": 0, "fdr_sig": 0, "bonf_sig": 0}
    if not path.exists() or not real_nodes:
        return empty

    df = pl.read_csv(path)
    if source_col not in df.columns or target_col not in df.columns:
        return empty
    df = df.filter(pl.col(source_col).is_in(real_nodes) & pl.col(target_col).is_in(real_nodes))
    total = df.height
    if total == 0:
        return empty

    if p_col in df.columns:
        raw_sig = df.filter(pl.col(p_col) <= ALPHA).height
    elif raw_sig_col in df.columns:
        raw_sig = int(df[raw_sig_col].sum())
    else:
        raw_sig = 0

    fdr_sig = int(df[fdr_sig_col].sum()) if fdr_sig_col in df.columns else 0
    bonf_sig = df.filter(pl.col(p_col) <= (ALPHA / total)).height if p_col in df.columns else 0
    return {"total": total, "raw_sig": raw_sig, "fdr_sig": fdr_sig, "bonf_sig": bonf_sig}


def _var_status(event_id: str, real_nodes: set[str], tables_dir: Path) -> str:
    if len(real_nodes) < 2:
        return "skipped: <2 real nodes"
    path = tables_dir / f"table_granger_{event_id}.csv"
    if not path.exists():
        return "missing"
    df = pl.read_csv(path)
    if df.height == 0:
        return "skipped or empty"
    df = df.filter(
        pl.col("causing_node").is_in(real_nodes) & pl.col("caused_node").is_in(real_nodes)
    )
    if df.height == 0:
        return "empty after real-node filter"
    sig = int(df["significant"].sum()) if "significant" in df.columns else 0
    return f"ok: {sig}/{df.height} significant Granger relations"


def _network_real_edges(event_id: str, real_nodes: set[str], tables_dir: Path) -> int:
    candidates = [
        (tables_dir / f"table_var_spillovers_{event_id}.csv", "causing_node", "caused_node", "fevd_share"),
        (tables_dir / f"table_transfer_entropy_{event_id}.csv", "node_i", "node_j", "te_i_to_j"),
    ]
    for path, source_col, target_col, weight_col in candidates:
        if not path.exists():
            continue
        df = pl.read_csv(path)
        if not {source_col, target_col, weight_col}.issubset(df.columns):
            continue
        df = df.filter(
            (pl.col(source_col) != pl.col(target_col))
            & pl.col(source_col).is_in(real_nodes)
            & pl.col(target_col).is_in(real_nodes)
            & (pl.col(weight_col) > 0)
        )
        if "p_value" in df.columns:
            df = df.filter(pl.col("p_value") <= ALPHA)
        return df.height
    return 0


def _placebo_summary(event_id: str, tables_dir: Path) -> dict[str, float | int | None]:
    path = tables_dir / f"table_robustness_{event_id}.csv"
    if not path.exists():
        return {
            "true_leadlag_sig_rate": None,
            "placebo_leadlag_sig_rate": None,
            "placebo_rows": 0,
        }
    df = pl.read_csv(path)
    if "check" not in df.columns or "significant_p01" not in df.columns:
        return {
            "true_leadlag_sig_rate": None,
            "placebo_leadlag_sig_rate": None,
            "placebo_rows": 0,
        }
    base = df.filter(pl.col("check") == "baseline_60s")
    placebo = df.filter(pl.col("check") == "placebo")
    return {
        "true_leadlag_sig_rate": float(base["significant_p01"].mean()) if base.height else None,
        "placebo_leadlag_sig_rate": (
            float(placebo["significant_p01"].mean()) if placebo.height else None
        ),
        "placebo_rows": placebo.height,
    }


def main() -> None:
    events = list(load_events().keys())
    tables_dir = results_root() / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    real_rows = []
    mt_rows = []
    placebo_rows = []

    for event_id in events:
        tiers = _latest_node_tiers(event_id)
        configured = {node.id for node in nodes_for_event(event_id)}
        real_nodes = {
            node_id for node_id, tier in tiers.items()
            if node_id in configured and tier not in {FIXTURE, "missing", None}
        }
        fixture_nodes = {
            node_id for node_id, tier in tiers.items()
            if node_id in configured and tier == FIXTURE
        }

        lead = _edge_stats(
            tables_dir / f"table_leadlag_tests_{event_id}.csv",
            real_nodes,
            "node_i",
            "node_j",
            raw_sig_col="significant_p01",
        )
        te = _edge_stats(
            tables_dir / f"table_transfer_entropy_{event_id}.csv",
            real_nodes,
            "node_i",
            "node_j",
            raw_sig_col="significant_p05",
        )

        real_rows.append(
            {
                "event_id": event_id,
                "real_nodes": len(real_nodes),
                "fixture_nodes": len(fixture_nodes),
                "real_node_ids": ";".join(sorted(real_nodes)),
                "fixture_node_ids": ";".join(sorted(fixture_nodes)),
                "leadlag_real_total": lead["total"],
                "leadlag_real_raw_sig": lead["raw_sig"],
                "leadlag_real_fdr_sig": lead["fdr_sig"],
                "leadlag_real_bonf_sig": lead["bonf_sig"],
                "te_real_total": te["total"],
                "te_real_raw_sig": te["raw_sig"],
                "te_real_fdr_sig": te["fdr_sig"],
                "te_real_bonf_sig": te["bonf_sig"],
                "var_status": _var_status(event_id, real_nodes, tables_dir),
                "network_real_edges": _network_real_edges(event_id, real_nodes, tables_dir),
            }
        )

        for method, stats in [("leadlag", lead), ("transfer_entropy", te)]:
            mt_rows.append(
                {
                    "event_id": event_id,
                    "method": method,
                    "real_pairs": stats["total"],
                    "raw_significant_pairs": stats["raw_sig"],
                    "fdr_significant_pairs": stats["fdr_sig"],
                    "bonferroni_significant_pairs": stats["bonf_sig"],
                }
            )

        placebo = _placebo_summary(event_id, tables_dir)
        placebo_rows.append({"event_id": event_id, **placebo})

    real_df = pl.DataFrame(real_rows)
    mt_df = pl.DataFrame(mt_rows)
    placebo_df = pl.DataFrame(placebo_rows)

    real_path = tables_dir / "table_preliminary_real_node_summary.csv"
    mt_path = tables_dir / "table_multiple_testing_summary.csv"
    placebo_path = tables_dir / "table_placebo_summary.csv"
    real_df.write_csv(real_path)
    mt_df.write_csv(mt_path)
    placebo_df.write_csv(placebo_path)

    logger.info("Wrote %s, %s, and %s", real_path, mt_path, placebo_path)
    print(real_df)


if __name__ == "__main__":
    main()
