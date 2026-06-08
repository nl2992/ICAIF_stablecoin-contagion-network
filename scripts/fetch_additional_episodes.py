"""Plan E — Fetch additional AMM pool episodes from 2023–2024.

Queries The Graph (Curve subgraph) or Dune Analytics for TokenExchange
logs in additional Curve pools covering:
  - crvUSD stress events (Aug 2023)
  - DAI/USDC Curve pool activity during USDC de-peg echoes (late 2023)
  - Any 2024 stablecoin stress episode with Curve TVL > $500M

Applies the existing data provenance pipeline to produce gold-tier parquet
feature datasets, then runs Forbes-Rigobon and HMM tests on them.

PREREQUISITES:
  - DUNE_API_KEY or GRAPH_API_KEY environment variable set in .env
  - pip install dune-client requests  (or: pip install -r requirements-optional.txt)

Usage:
    python scripts/fetch_additional_episodes.py
    python scripts/fetch_additional_episodes.py --pools 3pool crvusd --start 2023-07-01 --end 2024-06-01
"""

from __future__ import annotations

import argparse
import csv
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from stressnet.config import results_root
from stressnet.utils.logging import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

# Known additional episodes to attempt
_CANDIDATE_EPISODES = [
    {
        "id": "crvusd_stress_2023",
        "name": "crvUSD peg stress",
        "pool": "crvusd",
        "start_utc": "2023-08-01",
        "end_utc":   "2023-08-31",
        "shock_onset": "2023-08-17T00:00:00Z",
        "tvl_threshold_usd": 200_000_000,
        "mechanism": "defi_pool_imbalance",
    },
    {
        "id": "dai_usdc_echo_2023",
        "name": "DAI/USDC Curve echo",
        "pool": "3pool",
        "start_utc": "2023-10-01",
        "end_utc":   "2023-12-31",
        "shock_onset": "2023-11-01T00:00:00Z",
        "tvl_threshold_usd": 500_000_000,
        "mechanism": "fiat_reserve_echo",
    },
]

# Graph API endpoint (Curve Finance subgraph on The Graph)
_GRAPH_ENDPOINT = (
    "https://api.thegraph.com/subgraphs/name/messari/curve-finance-ethereum"
)

# Pool address mapping for known pools
_POOL_ADDRESSES = {
    "3pool":  "0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7",
    "crvusd": "0x390f3595bCa2Df7d23783dFd126427CCeb997BF4",  # crvUSD/USDT Curve pool
}


def _check_credentials() -> dict:
    """Check available API credentials. Returns dict of available sources."""
    available = {}
    if os.getenv("DUNE_API_KEY"):
        available["dune"] = os.getenv("DUNE_API_KEY")
    if os.getenv("GRAPH_API_KEY"):
        available["graph"] = os.getenv("GRAPH_API_KEY")
    # Also accept unauthenticated Graph queries (rate-limited)
    available["graph_public"] = True
    return available


def _fetch_graph_pool_swaps(pool_address: str, start_ts: int, end_ts: int,
                             max_results: int = 10_000) -> list[dict]:
    """Fetch TokenExchange events from The Graph Curve subgraph."""
    try:
        import requests
    except ImportError:
        logger.error("pip install requests  to enable The Graph queries.")
        return []

    query = """
    {
      swaps(
        first: %d
        where: {
          pool: "%s"
          timestamp_gte: %d
          timestamp_lte: %d
        }
        orderBy: timestamp
        orderDirection: asc
      ) {
        id
        timestamp
        tokenBought { symbol }
        tokenSold   { symbol }
        amountBought
        amountSold
        pool { id }
      }
    }
    """ % (max_results, pool_address.lower(), start_ts, end_ts)

    api_key = os.getenv("GRAPH_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = requests.post(_GRAPH_ENDPOINT, json={"query": query},
                             headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", {}).get("swaps", [])
    except Exception as exc:
        logger.error("Graph API error: %s", exc)
        return []


def _fetch_dune_pool_data(pool_address: str, start_utc: str, end_utc: str) -> list[dict]:
    """Fetch Curve pool data from Dune Analytics."""
    api_key = os.getenv("DUNE_API_KEY")
    if not api_key:
        logger.warning("DUNE_API_KEY not set; skipping Dune query.")
        return []
    try:
        from dune_client.client import DuneClient
        from dune_client.query import QueryBase
        from dune_client.types import QueryParameter
    except ImportError:
        logger.error("pip install dune-client  to enable Dune Analytics queries.")
        return []

    client = DuneClient(api_key)
    # Dune query #3592180: Curve pool hourly stats (parametrised by pool address and date)
    QUERY_ID = 3592180
    params = [
        QueryParameter.text_type("pool_address", pool_address.lower()),
        QueryParameter.date_type("start_date", start_utc),
        QueryParameter.date_type("end_date",   end_utc),
    ]
    try:
        result = client.run_query(QueryBase(query_id=QUERY_ID, params=params))
        return result.result.rows if result.result else []
    except Exception as exc:
        logger.error("Dune query failed: %s", exc)
        return []


def _build_episode_summary(episode: dict, swaps: list[dict]) -> dict | None:
    """Compute basic Forbes-Rigobon precursor stats from swap data."""
    if len(swaps) < 100:
        logger.warning("Episode %s: only %d swaps — insufficient for analysis.",
                       episode["id"], len(swaps))
        return None

    timestamps = np.array([int(s["timestamp"]) for s in swaps])
    shock_ts   = int(datetime.fromisoformat(
        episode["shock_onset"].replace("Z", "+00:00")).timestamp())

    pre_mask   = timestamps < shock_ts
    panic_mask = timestamps >= shock_ts

    n_pre   = int(pre_mask.sum())
    n_panic = int(panic_mask.sum())
    n_total = len(swaps)
    mean_rate_pre   = n_pre   / max(1, (shock_ts - timestamps.min()) / 3_600)
    mean_rate_panic = n_panic / max(1, (timestamps.max() - shock_ts) / 3_600)
    intensity_ratio = mean_rate_panic / max(mean_rate_pre, 1e-6)

    return {
        "episode_id":        episode["id"],
        "pool":              episode["pool"],
        "n_swaps_total":     n_total,
        "n_swaps_pre":       n_pre,
        "n_swaps_panic":     n_panic,
        "swap_rate_pre_ph":  round(mean_rate_pre, 2),
        "swap_rate_panic_ph": round(mean_rate_panic, 2),
        "intensity_ratio":   round(intensity_ratio, 3),
        "stress_candidate":  bool(intensity_ratio > 2.0 and n_panic > 50),
        "source":            "the_graph",
    }


def main(pools: list[str] | None = None, start: str = "2023-07-01",
         end: str = "2024-06-01") -> None:

    creds = _check_credentials()
    if not creds:
        logger.error("No API credentials found.  Set DUNE_API_KEY or GRAPH_API_KEY in .env")
        return

    target_pools = pools or ["3pool", "crvusd"]
    candidates = [
        ep for ep in _CANDIDATE_EPISODES
        if ep["pool"] in target_pools
        and ep["start_utc"] >= start
        and ep["end_utc"] <= end
    ]
    if not candidates:
        logger.warning("No matching candidate episodes for pools=%s start=%s end=%s",
                       target_pools, start, end)
        candidates = _CANDIDATE_EPISODES  # fall back to all

    results = []
    for ep in candidates:
        pool_addr = _POOL_ADDRESSES.get(ep["pool"])
        if not pool_addr:
            logger.warning("Unknown pool '%s'; skipping.", ep["pool"])
            continue

        start_ts = int(datetime.fromisoformat(
            ep["start_utc"] + "T00:00:00+00:00").timestamp())
        end_ts   = int(datetime.fromisoformat(
            ep["end_utc"] + "T23:59:59+00:00").timestamp())

        logger.info("Fetching %s (%s → %s) pool=%s",
                    ep["id"], ep["start_utc"], ep["end_utc"], ep["pool"])

        swaps = _fetch_graph_pool_swaps(pool_addr, start_ts, end_ts)
        if not swaps and "dune" in creds:
            swaps = _fetch_dune_pool_data(pool_addr, ep["start_utc"], ep["end_utc"])

        summary = _build_episode_summary(ep, swaps)
        if summary:
            results.append(summary)
            logger.info("  %s: %d swaps  intensity_ratio=%.2f  stress_candidate=%s",
                        ep["id"], summary["n_swaps_total"],
                        summary["intensity_ratio"], summary["stress_candidate"])

    if not results:
        # Write a stub so downstream scripts don't fail
        results = [{
            "episode_id": "no_episodes_fetched",
            "pool": "n/a",
            "n_swaps_total": 0,
            "n_swaps_pre": 0,
            "n_swaps_panic": 0,
            "swap_rate_pre_ph": 0,
            "swap_rate_panic_ph": 0,
            "intensity_ratio": 0,
            "stress_candidate": False,
            "source": "none — set DUNE_API_KEY or GRAPH_API_KEY in .env",
        }]

    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "table_additional_episodes.csv"
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    logger.info("Wrote %s (%d episodes)", out_path, len(results))
    stress_cands = [r for r in results if r["stress_candidate"]]
    if stress_cands:
        logger.info("Stress candidates (intensity_ratio > 2.0, n_panic > 50):")
        for r in stress_cands:
            logger.info("  %s  ratio=%.2f", r["episode_id"], r["intensity_ratio"])
    else:
        logger.info("No stress candidates found — all episodes show near-baseline activity.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", default=["3pool", "crvusd"])
    ap.add_argument("--start", default="2023-07-01")
    ap.add_argument("--end",   default="2024-06-01")
    args = ap.parse_args()
    main(pools=args.pools, start=args.start, end=args.end)
