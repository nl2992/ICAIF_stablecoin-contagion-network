"""Fail fast when an empirical event panel lacks enough real coverage."""

from __future__ import annotations

import argparse

import polars as pl

from stressnet.config import gold_root, results_root
from stressnet.evaluation.coverage_gate import check_empirical_coverage
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check minimum empirical coverage.")
    parser.add_argument("--event", required=True)
    parser.add_argument("--min-real-nodes", type=int, default=3)
    parser.add_argument("--min-var-nodes", type=int, default=2)
    parser.add_argument("--min-rows-per-node", type=int, default=20)
    parser.add_argument(
        "--require-layers",
        nargs="*",
        default=[],
        help="Optional required real layers, e.g. CEX DEX mint_burn.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Strict mode: fail loudly if ANY fixture rows are found in the gold "
            "panel for this event.  Use in CI to prevent fixture contamination "
            "from silently reaching the paper pipeline."
        ),
    )
    args = parser.parse_args()

    panel_path = gold_root() / f"dataset_contagion_features_{args.event}.parquet"
    if not panel_path.exists():
        raise SystemExit(f"Panel not found: {panel_path}. Run script 03 first.")

    panel = pl.read_parquet(panel_path)

    # ── Strict fixture check (--strict / CI mode) ────────────────────────────
    if args.strict and "tier_actual" in panel.columns:
        fixture_rows = panel.filter(
            pl.col("tier_actual") == "fixture_non_empirical"
        )
        if fixture_rows.height > 0:
            fixture_nodes = fixture_rows["node_id"].unique().to_list() if "node_id" in fixture_rows.columns else ["unknown"]
            raise SystemExit(
                f"[STRICT] Fixture rows detected in gold panel for {args.event}!\n"
                f"  {fixture_rows.height} rows from fixture nodes: {fixture_nodes}\n"
                f"  Run 'make empirical EVENT={args.event}' with ETHERSCAN_API_KEY "
                f"set to replace fixtures with real data before paper submission."
            )
        logger.info(
            "[STRICT] No fixture rows in gold panel for %s — all nodes are real data.",
            args.event,
        )
    result = check_empirical_coverage(
        panel,
        event_id=args.event,
        min_real_nodes=args.min_real_nodes,
        min_var_nodes=args.min_var_nodes,
        min_rows_per_node=args.min_rows_per_node,
        required_layers=tuple(args.require_layers),
    )

    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"table_empirical_coverage_gate_{args.event}.csv"
    pl.DataFrame(
        [
            {
                "event_id": result.event_id,
                "n_nodes_total": result.n_nodes_total,
                "n_nodes_real": result.n_nodes_real,
                "n_nodes_fixture": result.n_nodes_fixture,
                "n_var_eligible_nodes": result.n_var_eligible_nodes,
                "real_layers": ";".join(result.real_layers),
                "passes": result.passes,
                "reason": result.reason,
            }
        ]
    ).write_csv(out_path)
    logger.info("Wrote %s", out_path)

    if not result.passes:
        raise SystemExit(f"Empirical coverage gate failed for {args.event}: {result.reason}")

    logger.info("Empirical coverage gate passed for %s: %s", args.event, result.reason)


if __name__ == "__main__":
    main()
