"""
23_run_gnn_benchmark.py
-----------------------
Run the SnapshotGCN on each event's non-fixture panel and compare to
the flat-feature baselines from script 09b / Table 3 in the paper.

Fixture nodes are excluded before training so all results are Tier-A/B
and paper-claimable (subject to the 5pp improvement gate in temporal_gnn.py).

Usage:
    python scripts/23_run_gnn_benchmark.py

Outputs:
    results/tables/table_gnn_metrics_all.csv   -- per-event GNN metrics
    results/paper/tables/table_gnn_paper.csv   -- paper-ready comparison table
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import polars as pl

from stressnet.models.temporal_gnn import train_tgn_stub

# ---------------------------------------------------------------------------
# Flat-feature baseline AUROCs from Table 3 (paper, in-event held-out)
# ---------------------------------------------------------------------------
BASELINE_AUROC = {
    "usdt_curve_2023":  0.509,
    "usdc_svb_2023":    0.859,
    "terra_luna_2022":  0.736,
    "ftx_2022":         0.964,
    "busd_2023":        0.647,
}

EVENTS = list(BASELINE_AUROC.keys())

GOLD = ROOT / "data" / "gold"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
PAPER_TABLES = RESULTS / "paper" / "tables"
PAPER_TABLES.mkdir(parents=True, exist_ok=True)


def run_event(event_id: str) -> dict:
    """Train and evaluate GNN for one event, returning metric dict."""
    panel_path = GOLD / f"dataset_contagion_features_{event_id}.parquet"
    if not panel_path.exists():
        print(f"  [SKIP] {event_id}: panel not found")
        return {"event_id": event_id, "status": "no_panel"}

    # Filter out fixture nodes before writing to a temp parquet for train_tgn_stub
    panel = pl.read_parquet(panel_path)
    panel_clean = panel.filter(pl.col("tier_actual") != "fixture_non_empirical")
    n_fixture_rows = panel.height - panel_clean.height

    # Write cleaned panel to a temporary path
    tmp_path = GOLD / f"_tmp_gnn_{event_id}.parquet"
    panel_clean.write_parquet(tmp_path)

    print(
        f"  {event_id}: {panel_clean.height} non-fixture rows "
        f"(dropped {n_fixture_rows} fixture rows)"
    )

    try:
        result = train_tgn_stub(
            dataset_path=str(tmp_path),
            event_id=event_id,
            architecture="SnapshotGCN",
            epochs=80,
            lr=1e-3,
            baseline_auc=BASELINE_AUROC[event_id],
            output_dir=str(RESULTS),
        )
        result["n_fixture_rows_dropped"] = n_fixture_rows
        result["baseline_auroc"] = BASELINE_AUROC[event_id]
        result["auroc_lift_pp"] = (
            round((result["AUROC"] - BASELINE_AUROC[event_id]) * 100, 2)
            if result["AUROC"] == result["AUROC"]  # not nan
            else float("nan")
        )
    except Exception as e:
        print(f"  [ERROR] {event_id}: {e}")
        result = {
            "event_id": event_id,
            "status": "error",
            "error": str(e),
            "baseline_auroc": BASELINE_AUROC[event_id],
        }
    finally:
        tmp_path.unlink(missing_ok=True)

    return result


def main() -> None:
    print("=== SnapshotGCN benchmark (non-fixture panels) ===\n")
    all_results = []
    for event_id in EVENTS:
        print(f"[{event_id}]")
        r = run_event(event_id)
        all_results.append(r)
        status = r.get("status", "trained")
        if status == "trained":
            print(
                f"  AUROC={r.get('AUROC', float('nan')):.4f}  "
                f"AUPRC={r.get('AUPRC', float('nan')):.4f}  "
                f"lift={r.get('auroc_lift_pp', float('nan')):.1f}pp  "
                f"gate={'PASS' if r.get('passes_gnn_gate') else 'FAIL'}\n"
            )
        else:
            print(f"  status={status}\n")

    # Save full results
    out_all = TABLES / "table_gnn_metrics_all.csv"
    pl.DataFrame(all_results).write_csv(out_all)
    print(f"Full results → {out_all}")

    # Save paper-ready table
    paper_rows = []
    for r in all_results:
        if r.get("status", "trained") == "trained":
            paper_rows.append({
                "Event": r["event_id"],
                "GNN AUROC": round(r.get("AUROC", float("nan")), 3),
                "GNN AUPRC": round(r.get("AUPRC", float("nan")), 3),
                "Baseline AUROC": r.get("baseline_auroc", float("nan")),
                "Lift (pp)": r.get("auroc_lift_pp", float("nan")),
                "Passes gate": r.get("passes_gnn_gate", False),
                "Uses fixture": r.get("uses_fixture_data", False),
                "Train snapshots": r.get("n_train_snapshots", ""),
                "Test snapshots": r.get("n_test_snapshots", ""),
            })
    if paper_rows:
        out_paper = PAPER_TABLES / "table_gnn_paper.csv"
        pl.DataFrame(paper_rows).write_csv(out_paper)
        print(f"Paper table    → {out_paper}")

    print("\n=== Summary ===")
    for r in all_results:
        if r.get("status", "trained") == "trained":
            gate = "PASS" if r.get("passes_gnn_gate") else "FAIL"
            print(
                f"  {r['event_id']:<25s}  "
                f"AUROC={r.get('AUROC', float('nan')):.3f}  "
                f"lift={r.get('auroc_lift_pp', float('nan')):+.1f}pp  "
                f"gate={gate}"
            )


if __name__ == "__main__":
    main()
