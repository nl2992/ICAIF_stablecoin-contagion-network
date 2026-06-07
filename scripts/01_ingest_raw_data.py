"""Ingest raw/bronze data for one event.

Provenance-first: tries real data sources in priority order, falls back to
deterministic fixtures only on failure (or when --no-fixture is NOT passed
and real data is genuinely unavailable).

Real data sources (no API key unless noted):
  CEX  Binance   : Binance Vision bookTicker (Tier B: BBO only) → klines/1m (Tier B)
  CEX  Coinbase  : Coinbase Exchange public REST candles (Tier B)
  CEX  Kraken    : Kraken public REST OHLC (Tier B)
  DEX  Curve     : Etherscan getLogs + event decode (Tier A flow, Tier B proxies; needs ETHERSCAN_API_KEY)
  DEX  Uniswap   : Etherscan Swap logs (Tier A, needs ETHERSCAN_API_KEY)
                   or The Graph swaps (Tier B, needs THE_GRAPH_API_KEY)
  Flow mint_burn : Etherscan tokentx to/from null address (Tier A, needs ETHERSCAN_API_KEY)
  Flow exchange  : Etherscan tokentx + known exchange addresses (Tier B, needs ETHERSCAN_API_KEY)

Fixture files are marked tier_actual = fixture_non_empirical in the manifest.
Real files are marked with the actual Tier A/B achieved.
"""

from __future__ import annotations

import argparse
import math
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from stressnet.config import bronze_root, load_events
from stressnet.graph.nodes import Node, nodes_for_event
from stressnet.utils.logging import get_logger
from stressnet.utils.manifest import build_node_coverage_table, write_manifest_row
from stressnet.utils.time import parse_iso_utc

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Event window helpers
# ---------------------------------------------------------------------------

def _event_bounds(event_id: str):
    cfg = load_events()[event_id]
    start_str = f"{cfg['analysis_window_utc'][0]}T00:00:00Z"
    end_str   = f"{cfg['analysis_window_utc'][1]}T23:59:59Z"
    start_utc = parse_iso_utc(start_str)
    end_utc   = parse_iso_utc(end_str)
    start_date = date(start_utc.year, start_utc.month, start_utc.day)
    end_date   = date(end_utc.year,   end_utc.month,   end_utc.day)
    return start_str, end_str, start_utc, end_utc, start_date, end_date


# ---------------------------------------------------------------------------
# Real-data ingestion helpers
# ---------------------------------------------------------------------------

def _try_real_cex(node: Node, start_date: date, end_date: date,
                  out_dir: Path) -> tuple[Path | None, str, str]:
    """Try real CEX ingestion. Tries each symbol in the node's symbol list in order.

    Returns (path, kind, tier) or (None, kind, 'fixture_non_empirical').
    """
    import shutil
    symbols = node.metadata.get("symbols", [])
    if not symbols:
        return None, "books", "fixture_non_empirical"

    if node.venue == "Binance":
        from stressnet.data.binance import ingest_binance_range
        for symbol in symbols:
            # Priority: bookTicker (Tier B BBO) → klines/1m (Tier B candles)
            for data_type in ("bookTicker", "klines/1m"):
                try:
                    path, tier = ingest_binance_range(
                        symbol, start_date, end_date,
                        out_dir / "_raw", data_type=data_type,
                    )
                except Exception as exc:
                    logger.warning("Binance %s ingest error (%s): %s", data_type, symbol, exc)
                    path, tier = None, "fixture_non_empirical"
                if path is not None:
                    dest = out_dir / f"{node.id}_books.parquet"
                    shutil.copy2(path, dest)
                    return dest, "books", tier
        return None, "books", "fixture_non_empirical"

    elif node.venue == "Coinbase":
        from stressnet.data.coinbase import ingest_coinbase_range
        for symbol in symbols:
            try:
                path, tier = ingest_coinbase_range(
                    symbol, start_date, end_date, out_dir / "_raw",
                )
            except Exception as exc:
                logger.warning("Coinbase ingest error (%s): %s", symbol, exc)
                path, tier = None, "fixture_non_empirical"
            if path is not None:
                dest = out_dir / f"{node.id}_books.parquet"
                shutil.copy2(path, dest)
                return dest, "books", tier
        return None, "books", "fixture_non_empirical"

    elif node.venue == "Kraken":
        from stressnet.data.kraken import ingest_kraken_ohlc
        for symbol in symbols:
            pair = symbol.replace("/", "")
            try:
                path, tier = ingest_kraken_ohlc(pair, start_date, end_date, out_dir / "_raw")
            except Exception as exc:
                logger.warning("Kraken ingest error (%s): %s", pair, exc)
                path, tier = None, "fixture_non_empirical"
            if path is not None:
                dest = out_dir / f"{node.id}_books.parquet"
                shutil.copy2(path, dest)
                return dest, "books", tier
        return None, "books", "fixture_non_empirical"

    return None, "books", "fixture_non_empirical"


def _try_real_dex(
    node: Node, start_utc, end_utc, out_dir: Path, event_id: str = ""
) -> tuple[Path | None, str, str]:
    """Try real DEX ingestion. Returns (path, kind, tier) or (None, kind, fallback)."""
    import os
    contract = node.metadata.get("contract") or node.metadata.get("pool_address")
    if not contract:
        return None, "pool_events", "fixture_non_empirical"

    if node.venue == "Curve":
        if not os.environ.get("ETHERSCAN_API_KEY"):
            logger.debug("ETHERSCAN_API_KEY not set; skipping Curve ingest for %s", node.id)
            return None, "pool_events", "fixture_non_empirical"
        from stressnet.data.curve import ingest_curve_pool_events
        from stressnet.data.etherscan import get_block_number_by_timestamp
        try:
            start_block = get_block_number_by_timestamp(int(start_utc.timestamp()))
            end_block   = get_block_number_by_timestamp(int(end_utc.timestamp()), closest="after")
            path, tier  = ingest_curve_pool_events(
                contract, start_block, end_block, out_dir, event_id, node.id,
            )
        except Exception as exc:
            logger.warning("Curve ingest error (%s): %s", node.id, exc)
            path, tier = None, "fixture_non_empirical"
        return path, "pool_events", tier

    elif node.venue == "Uniswap":
        pool_addr = node.metadata.get("pool_address")
        if not pool_addr:
            return None, "pool_events", "fixture_non_empirical"

        # Prefer direct on-chain Swap logs: same provenance class as Curve
        # TokenExchange logs and paper-claimable when paired with Tier-A nodes.
        if os.environ.get("ETHERSCAN_API_KEY"):
            from stressnet.data.etherscan import get_block_number_by_timestamp
            from stressnet.data.uniswap_etherscan import ingest_uniswap_pool_events
            try:
                start_block = get_block_number_by_timestamp(int(start_utc.timestamp()))
                end_block = get_block_number_by_timestamp(int(end_utc.timestamp()), closest="after")
                path, tier = ingest_uniswap_pool_events(
                    pool_addr,
                    start_block,
                    end_block,
                    out_dir,
                    event_id,
                    node.id,
                )
                if path is not None:
                    return path, "pool_events", tier
            except Exception as exc:
                logger.warning("Uniswap Etherscan ingest error (%s): %s", node.id, exc)

        if not os.environ.get("THE_GRAPH_API_KEY"):
            logger.debug(
                "No ETHERSCAN_API_KEY/THE_GRAPH_API_KEY; skipping Uniswap ingest for %s",
                node.id,
            )
            return None, "pool_events", "fixture_non_empirical"
        from stressnet.data.uniswap import ingest_uniswap_pool_swaps
        try:
            path, tier = ingest_uniswap_pool_swaps(
                pool_addr,
                int(start_utc.timestamp()), int(end_utc.timestamp()),
                out_dir, event_id, node.id,
            )
        except Exception as exc:
            logger.warning("Uniswap ingest error (%s): %s", node.id, exc)
            path, tier = None, "fixture_non_empirical"
        return path, "pool_events", tier

    return None, "pool_events", "fixture_non_empirical"


def _try_real_flow(
    node: Node, start_utc, end_utc, out_dir: Path, event_id: str = ""
) -> tuple[Path | None, str, str]:
    """Try real flow/mint_burn ingestion (Ethereum only; Tron gracefully skips)."""
    import os
    # Tron uses a separate API — not supported by Etherscan block lookups.
    if node.metadata.get("chain", "").lower() == "tron":
        logger.debug("Tron chain not supported via Etherscan; skipping %s (fixture fallback).", node.id)
        return None, "flows", "fixture_non_empirical"
    token_contract = node.metadata.get("token_contract")
    if not token_contract:
        logger.debug("No token_contract for %s; skipping real flow ingest.", node.id)
        return None, "flows", "fixture_non_empirical"
    if not os.environ.get("ETHERSCAN_API_KEY"):
        logger.debug("ETHERSCAN_API_KEY not set; skipping flow ingest for %s", node.id)
        return None, "flows", "fixture_non_empirical"

    from stressnet.data.etherscan import get_block_number_by_timestamp
    try:
        start_block = get_block_number_by_timestamp(int(start_utc.timestamp()))
        end_block   = get_block_number_by_timestamp(int(end_utc.timestamp()), closest="after")
    except Exception as exc:
        logger.warning("Block lookup failed for %s: %s", node.id, exc)
        return None, "flows", "fixture_non_empirical"

    if node.layer == "mint_burn":
        # Dispatch based on event_encoding metadata.
        # Tether (USDT) emits Issue/Redeem instead of standard Transfer events.
        event_encoding = node.metadata.get("event_encoding", "erc20_transfer")
        if event_encoding == "issue_redeem":
            from stressnet.data.etherscan import ingest_tether_issue_redeem
            try:
                path, tier = ingest_tether_issue_redeem(
                    token_contract, start_block, end_block, out_dir, event_id, node.id,
                )
            except Exception as exc:
                logger.warning("Tether Issue/Redeem ingest error (%s): %s", node.id, exc)
                path, tier = None, "fixture_non_empirical"
        else:
            from stressnet.data.etherscan import ingest_mint_burn
            try:
                path, tier = ingest_mint_burn(
                    token_contract, start_block, end_block, out_dir, event_id, node.id,
                )
            except Exception as exc:
                logger.warning("Mint/burn ingest error (%s): %s", node.id, exc)
                path, tier = None, "fixture_non_empirical"
        return path, "flows", tier

    else:  # onchain_flow, bridge_flow
        from stressnet.data.etherscan import ingest_exchange_flows
        try:
            path, tier = ingest_exchange_flows(
                token_contract, start_block, end_block, out_dir, event_id, node.id,
            )
        except Exception as exc:
            logger.warning("Exchange flow ingest error (%s): %s", node.id, exc)
            path, tier = None, "fixture_non_empirical"
        return path, "flows", tier


# ---------------------------------------------------------------------------
# Deterministic fixture generators (fallback only)
# ---------------------------------------------------------------------------

def _fixture_times(start, hours: int = 72, step_minutes: int = 5):
    n = int(hours * 60 / step_minutes)
    return [start + timedelta(minutes=step_minutes * i) for i in range(n)]


def _stress_shape(i: int, n: int, node_shift: int) -> float:
    center = int(n * 0.38) + node_shift
    width  = max(n / 16, 1)
    return math.exp(-((i - center) ** 2) / (2 * width ** 2))


def _market_fixture(event_id: str, node: Node, start) -> pl.DataFrame:
    ts = _fixture_times(start)
    n  = len(ts)
    shift    = sum(ord(c) for c in node.id) % 18
    severity = 0.006 if node.asset == "USDC" else 0.0015
    rows = []
    for i, t in enumerate(ts):
        pulse  = _stress_shape(i, n, shift)
        wobble = math.sin(i / 11 + shift) * 0.00008
        mid    = 1.0 - severity * pulse + wobble
        spread = 1.5 + 18 * pulse + (shift % 4)
        depth  = max(50_000, 1_500_000 * (1 - 0.72 * pulse))
        rows.append({
            "wall_clock_utc":           t,
            "mid_price":                mid,
            "spread_bps":               spread,
            "depth_10bps_bid_usd":      depth * (1 - 0.08 * pulse),
            "depth_10bps_ask_usd":      depth * (1 + 0.05 * pulse),
            "orderbook_imbalance":      -0.45 * pulse + math.sin(i / 17) * 0.03,
            "executable_price_10k_buy": mid * (1 + spread / 20_000),
            "executable_price_10k_sell":mid * (1 - spread / 20_000),
            "basis_vs_usd":             math.log(mid),
        })
    return pl.DataFrame(rows)


def _pool_fixture(event_id: str, node: Node, start) -> pl.DataFrame:
    ts    = _fixture_times(start)
    n     = len(ts)
    shift = 8 if "uniswap" in node.id else 14
    rows  = []
    for i, t in enumerate(ts):
        pulse   = _stress_shape(i, n, shift)
        implied = 1.0 - 0.0045 * pulse + math.sin(i / 19) * 0.00005
        rows.append({
            "wall_clock_utc":     t,
            "reserve_imbalance":  0.55 * pulse,
            "implied_pool_price": implied,
            "pool_slippage_10k":  0.8 + 28 * pulse,
            "virtual_price":      1.0 + i * 1e-8,
            "lp_supply":          400_000_000 * (1 - 0.05 * pulse),
            "basis_vs_usd":       math.log(implied),
        })
    return pl.DataFrame(rows)


def _flow_fixture(event_id: str, node: Node, start) -> pl.DataFrame:
    ts    = _fixture_times(start, step_minutes=60)
    n     = len(ts)
    shift = sum(ord(c) for c in node.id) % 8
    rows  = []
    for i, t in enumerate(ts):
        pulse    = _stress_shape(i, n, shift)
        inflow   = 3_000_000 + 50_000_000 * pulse
        outflow  = 2_500_000 + 25_000_000 * max(0.0, pulse - 0.15)
        is_flow  = node.layer in ("onchain_flow", "bridge_flow", "flow")
        is_mb    = node.layer == "mint_burn"
        rows.append({
            "wall_clock_utc":      t,
            "exchange_inflow_1h":  inflow  if is_flow else None,
            "exchange_outflow_1h": outflow if is_flow else None,
            "exchange_netflow_1h": (inflow - outflow) if is_flow else None,
            "mint_burn_net_1h":    -35_000_000 * pulse if is_mb else None,
            "gas_base_fee_gwei":   22 + 70 * pulse,
            "basis_vs_usd":        -0.0008 * pulse,
        })
    return pl.DataFrame(rows)


def _fixture_for_node(event_id: str, node: Node, start) -> tuple[str, pl.DataFrame]:
    if node.layer == "CEX":
        return "books", _market_fixture(event_id, node, start)
    if node.layer == "DEX":
        return "pool_events", _pool_fixture(event_id, node, start)
    return "flows", _flow_fixture(event_id, node, start)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest raw/bronze data for one event.")
    parser.add_argument("--event", required=True)
    parser.add_argument("--nodes", nargs="+", default=None)
    parser.add_argument(
        "--no-fixture", action="store_true",
        help="Fail instead of generating fixture files when real data is absent.",
    )
    parser.add_argument(
        "--fixture-only", action="store_true",
        help="Skip real data fetching; write deterministic fixtures (pipeline test mode).",
    )
    args = parser.parse_args()

    events = load_events()
    if args.event not in events:
        raise SystemExit(f"Unknown event '{args.event}'. Available: {list(events)}")

    start_str, end_str, start_utc, end_utc, start_date, end_date = _event_bounds(args.event)

    nodes = nodes_for_event(args.event)
    if args.nodes:
        requested = set(args.nodes)
        nodes = [n for n in nodes if n.id in requested]
    if not nodes:
        raise SystemExit("No configured nodes selected.")

    out_dir = bronze_root() / args.event
    out_dir.mkdir(parents=True, exist_ok=True)

    real_node_count = 0  # tracks nodes with non-fixture data

    for node in nodes:
        path: Path | None = None
        kind: str = "books"
        tier: str = "fixture_non_empirical"

        # ---- attempt real data ----
        if not args.fixture_only:
            if node.layer == "CEX":
                path, kind, tier = _try_real_cex(node, start_date, end_date, out_dir)
            elif node.layer == "DEX":
                path, kind, tier = _try_real_dex(node, start_utc, end_utc, out_dir, args.event)
            elif node.layer in ("onchain_flow", "bridge_flow", "mint_burn", "flow"):
                path, kind, tier = _try_real_flow(node, start_utc, end_utc, out_dir, args.event)

        # ---- fallback to fixture ----
        if path is None:
            if args.no_fixture:
                logger.warning(
                    "No real data for %s and --no-fixture is set; skipping node "
                    "(will appear as missing in panel).", node.id
                )
                continue  # skip this node entirely — no fixture written

            if tier != "fixture_non_empirical":
                logger.info("Real ingest returned no data for %s; using fixture.", node.id)
            else:
                logger.debug("Writing fixture for %s (fixture_only or real ingest skipped).", node.id)

            kind_fix, df_fix = _fixture_for_node(args.event, node, start_utc)
            out_path = out_dir / f"{node.id}_{kind_fix}.parquet"
            df_fix.write_parquet(out_path)
            path = out_path
            kind = kind_fix
            tier = "fixture_non_empirical"

        if tier != "fixture_non_empirical":
            real_node_count += 1

        write_manifest_row(
            event_id=args.event,
            node_id=node.id,
            source_name=(
                "deterministic_pipeline_fixture" if tier == "fixture_non_empirical"
                else f"real_{node.venue or node.layer}_tier_{tier}"
            ),
            source_tier_nominal=node.tier,
            source_tier_actual=tier,
            start_utc=start_str,
            end_utc=end_str,
            file_path=path,
            row_count=pl.read_parquet(path).height,
            notes=(
                "Generated fixture for pipeline validation only; not empirical evidence."
                if tier == "fixture_non_empirical"
                else f"Real {node.venue or node.layer} data ingested (Tier {tier})."
            ),
            layer=node.layer,
            file_stage="bronze",
            url_or_query=(
                "fixture://deterministic_pipeline_fixture"
                if tier == "fixture_non_empirical"
                else f"real://{node.venue or node.layer}"
            ),
        )
        logger.info(
            "Bronze %-35s  %-25s  tier=%-24s  rows=%d",
            node.id, f"{kind}.parquet", tier, pl.read_parquet(path).height,
        )

    if real_node_count < 2:
        logger.warning(
            "[WARNING] Only %d real nodes ingested for %s; minimum recommended is 2 for VAR "
            "and 3 for TE/lead-lag. Consider running without --no-fixture or adding data sources.",
            real_node_count, args.event,
        )

    coverage_path = build_node_coverage_table()
    if coverage_path:
        logger.info("Updated coverage table: %s", coverage_path)


if __name__ == "__main__":
    main()
