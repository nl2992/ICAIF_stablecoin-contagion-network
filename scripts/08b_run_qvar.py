"""Run Quantile VAR (QVAR) pairwise tail-spillover analysis.

Estimates β_cross at τ=0.05, 0.50, 0.95 for each directed pair.  The key
diagnostic is whether tail quantile impulse responses (τ=0.05, 0.95) are
materially larger than the median (τ=0.50), supporting non-linear contagion.

Reads:  data/gold/dataset_contagion_features_{event}.parquet
Writes: results/tables/table_qvar_{event}.csv

Usage:
    python scripts/08b_run_qvar.py --event usdt_curve_2023
    python scripts/08b_run_qvar.py --event usdt_curve_2023 --paper-mode
    python scripts/08b_run_qvar.py --all-events --paper-mode
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import polars as pl

from stressnet.config import gold_root, load_events, results_root
from stressnet.evaluation.claim_gate import FIXTURE, load_layer_map, load_tier_map
from stressnet.graph.nodes import nodes_for_event
from stressnet.models.qvar import _HAS_STATSMODELS, run_qvar
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_FEATURE  = "usdc_net_sold_1h"
_FALLBACK_FEATURES = [
    "usdc_net_sold_1h",
    "basis_vs_usd",
    "reserve_imbalance",
    "exchange_netflow_1h",
]
_QUANTILES = (0.05, 0.50, 0.95)


def run_for_event(
    event_id: str,
    feature_col: str,
    paper_mode: bool,
    tables_dir: Path,
) -> pl.DataFrame | None:
    if not _HAS_STATSMODELS:
        raise SystemExit("statsmodels is required: pip install statsmodels")

    panel_path = gold_root() / f"dataset_contagion_features_{event_id}.parquet"
    if not panel_path.exists():
        logger.warning("Panel not found for %s; skipping.", event_id)
        return None

    panel = pl.read_parquet(panel_path)

    if feature_col not in panel.columns:
        for alt in _FALLBACK_FEATURES:
            if alt in panel.columns:
                logger.info(
                    "Feature '%s' not in panel for %s; using '%s'.",
                    feature_col, event_id, alt,
                )
                feature_col = alt
                break
        else:
            logger.warning("No usable feature column for %s; skipping.", event_id)
            return None

    nodes = nodes_for_event(event_id)
    panel_node_ids = set(panel["node_id"].unique().to_list())

    if paper_mode:
        real_ids = set(
            panel.filter(pl.col("tier_actual") != FIXTURE)["node_id"].unique().to_list()
        )
        node_ids = [n.id for n in nodes if n.id in real_ids and n.id in panel_node_ids]
        logger.info("paper-mode: %d real nodes for %s", len(node_ids), event_id)
    else:
        node_ids = [n.id for n in nodes if n.id in panel_node_ids]

    if len(node_ids) < 2:
        logger.warning("Need ≥2 nodes for QVAR on %s; found %d.", event_id, len(node_ids))
        return None

    time_col = "wall_clock_utc" if "wall_clock_utc" in panel.columns else panel.columns[0]
    try:
        wide = panel.pivot(
            values=feature_col,
            index=time_col,
            on="node_id",
            aggregate_function="mean",
        ).sort(time_col)
    except Exception as exc:
        logger.error("Pivot failed for %s/%s: %s", event_id, feature_col, exc)
        return None

    tier_map   = load_tier_map([event_id], tables_dir)
    layer_map  = load_layer_map([event_id])
    event_tiers  = tier_map.get(event_id, {})
    event_layers = layer_map.get(event_id, {})

    node_cols = [c for c in wide.columns if c in node_ids]
    pairs = list(itertools.permutations(node_cols, 2))

    rows = []
    for nid_i, nid_j in pairs:
        xi = wide[nid_i].fill_null(0.0).to_numpy().astype(float)
        xj = wide[nid_j].fill_null(0.0).to_numpy().astype(float)

        qvar_results = run_qvar(
            xi, xj, node_i=nid_i, node_j=nid_j,
            feature_col=feature_col, quantiles=_QUANTILES,
        )
        for r in qvar_results:
            rows.append({
                "event_id":     event_id,
                "node_i":       nid_i,
                "node_j":       nid_j,
                "feature_col":  feature_col,
                "tau":          r.tau,
                "beta_cross":   r.beta_cross,
                "beta_own":     r.beta_own,
                "pseudo_r2":    r.pseudo_r2,
                "t_stat":       r.t_stat,
                "p_value":      r.p_value,
                "n_obs":        r.n_obs,
                "converged":    r.converged,
                "significant_p05": r.p_value < 0.05 if r.converged else False,
                "tier_i":       event_tiers.get(nid_i, "missing"),
                "tier_j":       event_tiers.get(nid_j, "missing"),
                "layer_i":      event_layers.get(nid_i, ""),
                "layer_j":      event_layers.get(nid_j, ""),
            })

    if not rows:
        return None

    df = pl.DataFrame(rows)

    # Tail amplification: ratio of mean(|β_cross|) at tails vs median
    # Summarised per-pair for the paper table
    tail_rows = df.filter(pl.col("tau").is_in([0.05, 0.95]))
    med_rows  = df.filter(pl.col("tau") == 0.50)
    if tail_rows.height > 0 and med_rows.height > 0:
        tail_mean = tail_rows["beta_cross"].abs().mean()
        med_mean  = med_rows["beta_cross"].abs().mean()
        ratio = (tail_mean / med_mean) if (med_mean and med_mean != 0) else float("nan")
        logger.info(
            "%s: tail β_cross mean=%.4f, median β_cross mean=%.4f, ratio=%.2f",
            event_id, tail_mean, med_mean, ratio,
        )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantile VAR tail-spillover analysis.")
    parser.add_argument("--event", default=None)
    parser.add_argument("--all-events", action="store_true")
    parser.add_argument("--feature-col", default=_DEFAULT_FEATURE)
    parser.add_argument("--paper-mode", action="store_true")
    args = parser.parse_args()

    tables_dir = results_root() / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if args.all_events:
        event_ids = list(load_events().keys())
    elif args.event:
        event_ids = [args.event]
    else:
        raise SystemExit("Specify --event EVENT or --all-events.")

    all_frames = []
    for event_id in event_ids:
        logger.info("Running QVAR for %s", event_id)
        df = run_for_event(
            event_id=event_id,
            feature_col=args.feature_col,
            paper_mode=args.paper_mode,
            tables_dir=tables_dir,
        )
        if df is not None:
            out_path = tables_dir / f"table_qvar_{event_id}.csv"
            df.write_csv(out_path)
            logger.info("Wrote %s (%d rows)", out_path.name, df.height)
            all_frames.append(df)
        else:
            logger.warning("No QVAR results for %s", event_id)

    if all_frames and args.all_events:
        combined = pl.concat(all_frames, how="diagonal")
        out_path = tables_dir / "table_qvar_all.csv"
        combined.write_csv(out_path)
        logger.info("Wrote combined QVAR table: %s (%d rows)", out_path.name, combined.height)


if __name__ == "__main__":
    main()
