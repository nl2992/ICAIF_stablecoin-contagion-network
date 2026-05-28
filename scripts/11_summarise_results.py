"""Build the paper-safety preliminary result summary table.

The summary intentionally separates all-node outputs from real-node-only
outputs. A node is treated as real only when ``tier_actual`` is not
``fixture_non_empirical``. This prevents fixture-generated smooth stress paths
from silently entering paper claims.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from stressnet.config import load_events, results_root
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

FIXTURE = "fixture_non_empirical"


def _read_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(path) if path.exists() else pl.DataFrame()


def _event_coverage(event_id: str, tables_dir: Path) -> pl.DataFrame:
    event_path = tables_dir / f"table_node_coverage_{event_id}.csv"
    if event_path.exists():
        return pl.read_csv(event_path)

    all_path = tables_dir / "table_node_coverage.csv"
    if all_path.exists():
        return pl.read_csv(all_path).filter(pl.col("event_id") == event_id)
    return pl.DataFrame()


def _node_counts(coverage: pl.DataFrame) -> tuple[int, int, set[str]]:
    if coverage.is_empty():
        return 0, 0, set()
    coverage = coverage.filter(~pl.col("node_id").str.starts_with("__"))
    n_total = coverage.height
    real = coverage.filter(~pl.col("tier_actual").is_in([FIXTURE, "missing"]))
    return n_total, real.height, set(real["node_id"].to_list())


def _sig_count(
    path: Path,
    source_col: str,
    target_col: str,
    sig_col: str,
    real_nodes: set[str] | None = None,
) -> int:
    df = _read_csv(path)
    if df.is_empty() or sig_col not in df.columns:
        return 0
    if real_nodes is not None:
        if source_col not in df.columns or target_col not in df.columns:
            return 0
        df = df.filter(pl.col(source_col).is_in(real_nodes) & pl.col(target_col).is_in(real_nodes))
    return int(df[sig_col].fill_null(False).sum())


def _network_counts(event_id: str, tables_dir: Path, real_nodes: set[str] | None = None) -> tuple[int, int]:
    centrality_path = tables_dir / f"table_node_centrality_{event_id}.csv"
    centrality = _read_csv(centrality_path)
    if centrality.is_empty():
        n_nodes = 0
    elif real_nodes is None:
        n_nodes = centrality["node_id"].n_unique()
    else:
        n_nodes = centrality.filter(pl.col("node_id").is_in(real_nodes))["node_id"].n_unique()

    edge_candidates = [
        (tables_dir / f"table_var_spillovers_{event_id}.csv", "causing_node", "caused_node", "fevd_share"),
        (tables_dir / f"table_transfer_entropy_{event_id}.csv", "node_i", "node_j", "te_i_to_j"),
    ]
    n_edges = 0
    for path, source_col, target_col, weight_col in edge_candidates:
        df = _read_csv(path)
        if df.is_empty() or not {source_col, target_col, weight_col}.issubset(df.columns):
            continue
        df = df.filter((pl.col(source_col) != pl.col(target_col)) & (pl.col(weight_col) > 0))
        if "p_value" in df.columns:
            df = df.filter(pl.col("p_value") <= 0.05)
        if real_nodes is not None:
            df = df.filter(pl.col(source_col).is_in(real_nodes) & pl.col(target_col).is_in(real_nodes))
        n_edges = df.height
        break
    return n_nodes, n_edges


def _var_status(event_id: str, tables_dir: Path, real_nodes: set[str]) -> str:
    if len(real_nodes) < 2:
        return "skipped: <2 real nodes"
    path = tables_dir / f"table_granger_{event_id}.csv"
    df = _read_csv(path)
    if df.is_empty():
        return "missing or empty"
    df = df.filter(
        pl.col("causing_node").is_in(real_nodes) & pl.col("caused_node").is_in(real_nodes)
    )
    if df.is_empty():
        return "empty after real-node filter"
    sig = int(df["significant"].fill_null(False).sum())
    return f"ok: {sig}/{df.height} significant Granger relations"


def _best_prediction(event_id: str, tables_dir: Path) -> tuple[float | None, float | None]:
    path = tables_dir / f"table_prediction_metrics_{event_id}.csv"
    df = _read_csv(path)
    if df.is_empty() or "AUROC" not in df.columns:
        return None, None
    best = df.sort(["AUROC", "AUPRC"], descending=[True, True]).row(0, named=True)
    return float(best["AUROC"]), float(best["AUPRC"])


def _write_prediction_all_events(tables_dir: Path) -> None:
    frames = []
    for event_id in load_events().keys():
        path = tables_dir / f"table_prediction_metrics_{event_id}.csv"
        df = _read_csv(path)
        if df.is_empty():
            continue
        if "event_id" not in df.columns:
            df = df.with_columns(pl.lit(event_id).alias("event_id"))
        frames.append(df)
    if not frames:
        return
    pl.concat(frames, how="diagonal").write_csv(
        tables_dir / "table_prediction_metrics_all_events.csv"
    )


def _robustness_pass(event_id: str, tables_dir: Path) -> bool:
    path = tables_dir / "table_placebo_summary.csv"
    if not path.exists():
        return False
    df = pl.read_csv(path).filter(pl.col("event_id") == event_id)
    if df.is_empty():
        return False
    row = df.row(0, named=True)
    true_rate = row.get("true_leadlag_sig_rate")
    placebo_rate = row.get("placebo_leadlag_sig_rate")
    if true_rate is None or placebo_rate is None:
        return False
    return float(true_rate) > float(placebo_rate)


def _paper_claim_tier(n_real: int, n_fixture: int, robustness_pass: bool) -> str:
    if n_real < 2:
        return "insufficient_real_coverage"
    if n_fixture > 0:
        return "real_node_preliminary_fixture_contaminated"
    if not robustness_pass:
        return "real_node_preliminary_needs_placebo"
    return "paper_candidate_preliminary"


def build_summary() -> pl.DataFrame:
    tables_dir = results_root() / "tables"
    rows = []
    for event_id in load_events().keys():
        coverage = _event_coverage(event_id, tables_dir)
        n_total, n_real, real_nodes = _node_counts(coverage)
        n_fixture = max(0, n_total - n_real)
        network_nodes_all, network_edges_all = _network_counts(event_id, tables_dir)
        network_nodes_real, network_edges_real = _network_counts(event_id, tables_dir, real_nodes)
        pred_auc, pred_pr_auc = _best_prediction(event_id, tables_dir)
        robust = _robustness_pass(event_id, tables_dir)
        rows.append(
            {
                "event_id": event_id,
                "n_nodes_total": n_total,
                "n_nodes_real": n_real,
                "n_nodes_fixture": n_fixture,
                "leadlag_sig_all": _sig_count(
                    tables_dir / f"table_leadlag_tests_{event_id}.csv",
                    "node_i",
                    "node_j",
                    "significant_p01",
                ),
                "leadlag_sig_real_only": _sig_count(
                    tables_dir / f"table_leadlag_tests_{event_id}.csv",
                    "node_i",
                    "node_j",
                    "significant_p01",
                    real_nodes,
                ),
                "te_sig_all": _sig_count(
                    tables_dir / f"table_transfer_entropy_{event_id}.csv",
                    "node_i",
                    "node_j",
                    "significant_p05",
                ),
                "te_sig_real_only": _sig_count(
                    tables_dir / f"table_transfer_entropy_{event_id}.csv",
                    "node_i",
                    "node_j",
                    "significant_p05",
                    real_nodes,
                ),
                "network_nodes_all": network_nodes_all,
                "network_edges_all": network_edges_all,
                "network_nodes_real_only": network_nodes_real,
                "network_edges_real_only": network_edges_real,
                "var_status": _var_status(event_id, tables_dir, real_nodes),
                "prediction_auc": pred_auc,
                "prediction_pr_auc": pred_pr_auc,
                "robustness_pass": robust,
                "paper_claim_tier": _paper_claim_tier(n_real, n_fixture, robust),
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    out_path = out_dir / "table_preliminary_results_summary.csv"
    summary.write_csv(out_path)
    _write_prediction_all_events(out_dir)
    logger.info("Wrote %s", out_path)
    print(summary)


if __name__ == "__main__":
    main()
