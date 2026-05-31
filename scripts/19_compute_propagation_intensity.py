"""Compute event-level propagation-intensity scores.

This turns the claim-gate audit into a benchmark-style ranking. The score is
not causal; it is a compact, provenance-aware summary of how much supported
stress-link evidence each event produces.

Outputs
-------
results/paper/tables/table_propagation_intensity.csv
results/tables/table_propagation_intensity.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from stressnet.config import results_root
from stressnet.evaluation.propagation_score import (
    PropagationScoreInputs,
    paper_claim_tier,
    propagation_intensity_score,
)
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def _read_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(path) if path.exists() else pl.DataFrame()


def _mean_abs_effect(tables_dir: Path, event_id: str) -> float:
    """Mean absolute effect across paper-claimable rows when available."""
    candidates = [
        tables_dir / "table_aa_paper_claimable_edges.csv",
        tables_dir / f"table_leadlag_tests_{event_id}.csv",
        tables_dir / f"table_cross_protocol_leadlag_{event_id}.csv",
    ]
    values: list[float] = []
    for path in candidates:
        df = _read_csv(path)
        if df.is_empty():
            continue
        if "event_id" in df.columns:
            df = df.filter(pl.col("event_id") == event_id)
        if "paper_claim_allowed" in df.columns:
            df = df.filter(pl.col("paper_claim_allowed").fill_null(False))
        if "peak_corr" in df.columns:
            values.extend(abs(float(v)) for v in df["peak_corr"].drop_nulls().to_list())
        if "te_i_to_j" in df.columns:
            values.extend(abs(float(v)) for v in df["te_i_to_j"].drop_nulls().to_list())
    if not values:
        return 0.0
    return sum(values) / len(values)


def _cross_protocol_aa_count(tables_dir: Path, event_id: str) -> int:
    """Count statistically supported Tier-A AMM-flow cross-protocol rows."""
    df = _read_csv(tables_dir / f"table_cross_protocol_leadlag_{event_id}.csv")
    if df.is_empty():
        return 0
    if "p_bonferroni" in df.columns:
        return df.filter(pl.col("p_bonferroni") < 0.05).height
    if "p_value_fdr" in df.columns:
        return df.filter(pl.col("p_value_fdr") < 0.05).height
    if "significant_fdr" in df.columns:
        return df.filter(pl.col("significant_fdr").fill_null(False)).height
    return 0


def _node_counts(tables_dir: Path, event_id: str) -> tuple[int, int]:
    coverage = _read_csv(tables_dir / "table_node_coverage_empirical.csv")
    if coverage.is_empty():
        coverage = _read_csv(results_root() / "tables" / "table_node_coverage.csv")
    if coverage.is_empty() or "event_id" not in coverage.columns:
        return 0, 0
    event_cov = coverage.filter(
        (pl.col("event_id") == event_id) & (~pl.col("node_id").str.starts_with("__"))
    )
    if event_cov.is_empty():
        return 0, 0
    tier_col = "source_tier_actual" if "source_tier_actual" in event_cov.columns else "tier_actual"
    fixture_nodes = event_cov.filter(pl.col(tier_col).is_in(["fixture_non_empirical", "missing"])).height
    return event_cov.height, fixture_nodes


def _placebo_rates(event_id: str) -> tuple[float | None, float | None]:
    placebo = _read_csv(results_root() / "tables" / "table_placebo_summary.csv")
    if placebo.is_empty():
        return None, None
    row = placebo.filter(pl.col("event_id") == event_id)
    if row.is_empty():
        return None, None
    data = row.row(0, named=True)
    return data.get("true_leadlag_sig_rate"), data.get("placebo_leadlag_sig_rate")


def build_table(paper_tables_dir: Path) -> pl.DataFrame:
    audit = _read_csv(paper_tables_dir / "table_claim_audit_summary.csv")
    if audit.is_empty():
        raise SystemExit(
            f"Missing {paper_tables_dir / 'table_claim_audit_summary.csv'}; run make paper_gate first."
        )

    rows = []
    for row in audit.iter_rows(named=True):
        event_id = str(row["event_id"])
        total_nodes, fixture_nodes = _node_counts(paper_tables_dir, event_id)
        true_rate, placebo_rate = _placebo_rates(event_id)
        mean_effect = _mean_abs_effect(paper_tables_dir, event_id)
        cross_aa = _cross_protocol_aa_count(paper_tables_dir, event_id)
        aa_edges = max(int(row.get("n_AA_paper_claimable") or 0), cross_aa)
        inputs = PropagationScoreInputs(
            aa_paper_edges=aa_edges,
            ab_paper_edges=int(row.get("n_AB_paper_claimable") or 0),
            bb_context_edges=int(row.get("n_BB_context") or 0),
            mean_abs_effect=mean_effect,
            true_sig_rate=true_rate,
            placebo_sig_rate=placebo_rate,
            fixture_nodes=fixture_nodes,
            total_nodes=total_nodes,
        )
        score_row = {
            **row,
            "n_AA_paper_claimable": aa_edges,
            "total_nodes": total_nodes,
            "fixture_nodes": fixture_nodes,
            "mean_abs_effect": mean_effect,
            "true_leadlag_sig_rate": true_rate,
            "placebo_leadlag_sig_rate": placebo_rate,
            "propagation_intensity_score": propagation_intensity_score(inputs),
        }
        score_row["paper_claim_tier"] = paper_claim_tier(score_row)
        rows.append(score_row)
    return pl.DataFrame(rows).sort(
        ["propagation_intensity_score", "n_AA_paper_claimable", "n_AB_paper_claimable"],
        descending=[True, True, True],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute propagation-intensity event ranking.")
    parser.add_argument(
        "--paper-tables-dir",
        type=Path,
        default=results_root() / "paper" / "tables",
        help="Directory containing claim-gated paper tables.",
    )
    args = parser.parse_args()

    table = build_table(args.paper_tables_dir)
    paper_out = args.paper_tables_dir / "table_propagation_intensity.csv"
    paper_out.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(paper_out)

    results_out = results_root() / "tables" / "table_propagation_intensity.csv"
    results_out.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(results_out)
    logger.info("Wrote %s and %s", paper_out, results_out)
    print(table)


if __name__ == "__main__":
    main()
