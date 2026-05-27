"""Build temporal contagion network from estimated directed edges.

Writes:
    results/tables/table_node_centrality_{event}.csv
    results/figures/figure_contagion_map_{event}.png
"""

import argparse
from pathlib import Path

import polars as pl

from stressnet.config import results_root
from stressnet.graph.centrality import compute_centrality, detect_communities
from stressnet.graph.edges import build_networkx_graph, edges_from_table
from stressnet.plotting.networks import plot_contagion_map
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def load_best_edge_table(event_id: str, tables_dir: Path) -> pl.DataFrame | None:
    """Load the best available edge table: prefer Hawkes, then VAR, then TE."""
    for name in [
        f"table_hawkes_params_{event_id}.csv",
        f"table_var_spillovers_{event_id}.csv",
        f"table_transfer_entropy_{event_id}.csv",
    ]:
        path = tables_dir / name
        if path.exists():
            logger.info("Using edge table: %s", name)
            return pl.read_csv(path)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and visualise contagion network.")
    parser.add_argument("--event", required=True)
    parser.add_argument("--weight-threshold", type=float, default=0.0)
    parser.add_argument("--p-threshold", type=float, default=0.05)
    args = parser.parse_args()

    tables_dir = results_root() / "tables"
    figures_dir = results_root() / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    edge_df = load_best_edge_table(args.event, tables_dir)
    if edge_df is None:
        raise SystemExit("No edge table found. Run scripts 05, 06, or 07 first.")

    # Determine edge columns
    if {"causing_node", "caused_node"}.issubset(edge_df.columns):
        src_col, tgt_col = "causing_node", "caused_node"
    else:
        src_col = "node_i" if "node_i" in edge_df.columns else "source"
        tgt_col = "node_j" if "node_j" in edge_df.columns else "target"
    w_col = "branching_ratio_ij" if "branching_ratio_ij" in edge_df.columns else (
        "fevd_share" if "fevd_share" in edge_df.columns else "te_i_to_j"
    )
    if src_col in edge_df.columns and tgt_col in edge_df.columns:
        edge_df = edge_df.filter(pl.col(src_col) != pl.col(tgt_col))

    edges = edges_from_table(
        edge_df,
        source_col=src_col,
        target_col=tgt_col,
        weight_col=w_col,
        method_col="method" if "method" in edge_df.columns else "method_missing",
        p_col="p_value" if "p_value" in edge_df.columns else None,
        event_id=args.event,
    )
    G = build_networkx_graph(edges, weight_threshold=args.weight_threshold,
                             p_threshold=args.p_threshold)
    logger.info("Network: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    centrality = compute_centrality(G)
    centrality = centrality.with_columns(pl.lit(args.event).alias("event_id"))
    centrality.write_csv(tables_dir / f"table_node_centrality_{args.event}.csv")
    logger.info("Wrote node centrality table")

    communities = detect_communities(G)
    node_roles = {row["node_id"]: row["role"] for row in centrality.iter_rows(named=True)}

    plot_contagion_map(
        G,
        node_roles=node_roles,
        output_path=figures_dir / f"figure_contagion_map_{args.event}.png",
        title=f"Stablecoin Contagion Network: {args.event}",
    )
    print(centrality)


if __name__ == "__main__":
    main()
