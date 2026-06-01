"""Run Forbes-Rigobon heteroskedasticity-corrected contagion tests.

Tests whether stress-period correlations exceed tranquil-period correlations
after the Forbes-Rigobon (2002) bias correction.  A significant result
(contagion=True) distinguishes true contagion from mere interdependence.

Reads:  data/gold/dataset_contagion_features_{event}.parquet
Writes: results/tables/table_forbes_rigobon_{event}.csv

Usage:
    python scripts/07b_run_forbes_rigobon.py --event usdc_svb_2023
    python scripts/07b_run_forbes_rigobon.py --event usdc_svb_2023 --paper-mode
    python scripts/07b_run_forbes_rigobon.py --all-events --paper-mode
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import polars as pl

from stressnet.config import gold_root, load_events, results_root
from stressnet.evaluation.claim_gate import (
    FIXTURE,
    load_layer_map,
    load_tier_map,
)
from stressnet.graph.nodes import nodes_for_event
from stressnet.models.forbes_rigobon import run_forbes_rigobon
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_FEATURE = "usdc_net_sold_1h"
_FALLBACK_FEATURES = [
    "usdc_net_sold_1h",
    "basis_vs_usd",
    "reserve_imbalance",
    "exchange_netflow_1h",
]
_P_THRESHOLD = 0.05


def _stress_mask(panel: pl.DataFrame, event_id: str) -> np.ndarray:
    """Derive binary stress indicator from event_phase or a time window."""
    if "event_phase" in panel.columns:
        phases = panel["event_phase"].to_list()
        return np.array([str(p) in ("stress", "core") for p in phases], dtype=bool)
    # Fall back: top-25th-percentile absolute basis as stress proxy
    if "basis_vs_usd" in panel.columns:
        vals = panel["basis_vs_usd"].abs().fill_null(0.0).to_numpy()
        threshold = float(np.nanpercentile(vals, 75))
        return vals >= threshold
    # Last resort: all rows treated as stress (conservative; FR test will show delta≈0)
    logger.warning("No event_phase column; using full sample as stress for %s", event_id)
    return np.ones(panel.height, dtype=bool)


def run_for_event(
    event_id: str,
    feature_col: str,
    paper_mode: bool,
    p_threshold: float,
    tables_dir: Path,
) -> pl.DataFrame | None:
    panel_path = gold_root() / f"dataset_contagion_features_{event_id}.parquet"
    if not panel_path.exists():
        logger.warning("Panel not found for %s; skipping.", event_id)
        return None

    panel = pl.read_parquet(panel_path)

    # Resolve feature column
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
            logger.warning("No usable feature column in panel for %s; skipping.", event_id)
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
        logger.warning("Need ≥2 nodes for FR test on %s; found %d.", event_id, len(node_ids))
        return None

    # Pivot to wide: rows=time, cols=node_id
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

    stress_mask = _stress_mask(
        panel.sort(time_col).unique(subset=[time_col], keep="first").sort(time_col),
        event_id,
    )
    # Align stress_mask length to wide table
    if len(stress_mask) != wide.height:
        n = wide.height
        stress_mask = stress_mask[:n] if len(stress_mask) >= n else np.pad(
            stress_mask, (0, n - len(stress_mask)), constant_values=False
        )

    # Load tiers and layers for claim gate annotation
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

        result = run_forbes_rigobon(
            xi, xj, stress_mask,
            node_i=nid_i, node_j=nid_j,
            feature_col=feature_col,
            p_threshold=p_threshold,
        )

        tier_i = event_tiers.get(nid_i, "missing")
        tier_j = event_tiers.get(nid_j, "missing")

        rows.append({
            "event_id":              event_id,
            "node_i":                nid_i,
            "node_j":                nid_j,
            "feature_col":           feature_col,
            "rho_unconditional":     result.rho_unconditional,
            "rho_stress_raw":        result.rho_stress_raw,
            "rho_stress_corrected":  result.rho_stress_corrected,
            "delta":                 result.delta,
            "z_stat":                result.z_stat,
            "p_value":               result.p_value,
            "n_tranquil":            result.n_tranquil,
            "n_stress":              result.n_stress,
            "contagion":             result.contagion,
            "tier_i":                tier_i,
            "tier_j":                tier_j,
            "layer_i":               event_layers.get(nid_i, ""),
            "layer_j":               event_layers.get(nid_j, ""),
        })

    if not rows:
        return None

    df = pl.DataFrame(rows)

    # Bonferroni correction for multiple pairs
    n_tests = len(rows)
    df = df.with_columns(
        (pl.col("p_value") * n_tests).clip(upper_bound=1.0).alias("p_bonferroni"),
        (pl.col("p_value") * n_tests < p_threshold).alias("contagion_bonferroni"),
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Forbes-Rigobon contagion test.")
    parser.add_argument("--event", default=None)
    parser.add_argument("--all-events", action="store_true")
    parser.add_argument("--feature-col", default=_DEFAULT_FEATURE)
    parser.add_argument("--p-threshold", type=float, default=_P_THRESHOLD)
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
        logger.info("Running Forbes-Rigobon for %s", event_id)
        df = run_for_event(
            event_id=event_id,
            feature_col=args.feature_col,
            paper_mode=args.paper_mode,
            p_threshold=args.p_threshold,
            tables_dir=tables_dir,
        )
        if df is not None:
            out_path = tables_dir / f"table_forbes_rigobon_{event_id}.csv"
            df.write_csv(out_path)
            logger.info("Wrote %s (%d rows)", out_path.name, df.height)
            all_frames.append(df)
        else:
            logger.warning("No FR results for %s", event_id)

    if all_frames and args.all_events:
        combined = pl.concat(all_frames, how="diagonal")
        out_path = tables_dir / "table_forbes_rigobon_all.csv"
        combined.write_csv(out_path)
        logger.info("Wrote combined FR table: %s (%d rows)", out_path.name, combined.height)

        # Print summary
        sig = combined.filter(pl.col("contagion_bonferroni"))
        logger.info(
            "Contagion (Bonferroni-corrected p<%.2f): %d / %d pairs across all events",
            args.p_threshold, sig.height, combined.height,
        )
        if sig.height > 0:
            print("\nContagion pairs (Bonferroni-corrected):")
            print(sig[["event_id", "node_i", "node_j", "rho_stress_corrected",
                        "p_bonferroni", "tier_i", "tier_j"]].sort("p_bonferroni"))


if __name__ == "__main__":
    main()
