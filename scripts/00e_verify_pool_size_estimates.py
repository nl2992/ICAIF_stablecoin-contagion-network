"""Verify hardcoded pool-size estimates against on-chain balances.

For each event, calls ``get_balances()`` on the relevant Curve pool
contract at the block nearest the event-start timestamp, then compares
the on-chain balance to the hardcoded ``pool_size_usd`` estimate in
``src/stressnet/data/curve.py``.

Results are written to ``results/tables/table_pool_size_verification.csv``.

A ratio outside [0.8, 1.2] triggers a WARNING — the hardcoded estimate
is off by more than 20% and should be updated in curve.py's
``_POOL_CONFIGS``.  The Tier-B designation for ``reserve_imbalance`` is
preserved regardless, but reviewers may ask for the verification evidence.

Requires:
    ETHERSCAN_API_KEY environment variable

Usage:
    python scripts/00e_verify_pool_size_estimates.py
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from stressnet.config import load_events, results_root
from stressnet.data.curve import _POOL_CONFIGS, CURVE_3POOL_ADDRESS
from stressnet.data.etherscan import _get, get_block_number_by_timestamp
from stressnet.utils.logging import get_logger
from stressnet.utils.time import parse_iso_utc

logger = get_logger(__name__)

# Mapping from contract address to its events and token decimals
_POOL_EVENT_MAP: dict[str, dict[str, Any]] = {
    CURVE_3POOL_ADDRESS.lower(): {
        "node_id": "curve_3pool",
        "events": ["usdc_svb_2023", "usdt_curve_2023", "ftx_2022", "busd_2023"],
        "token_decimals": [6, 6, 18],  # USDC, USDT, DAI
    },
    "0x390f3595bca2df7d23783dfd126427cceb997bf4": {
        "node_id": "curve_crvusd_usdt",
        "events": ["usdt_curve_2023"],
        "token_decimals": [18, 6],  # crvUSD, USDT
    },
    "0xceaf7747579696a2f0bb206a14210e3c9e6fb269": {
        "node_id": "curve_ust_wormhole",
        "events": ["terra_luna_2022"],
        "token_decimals": [18, 18, 18, 18],  # UST, 3CRV components
    },
}

# get_balances() ABI call data (function selector = keccak256("get_balances()")[0:4])
_GET_BALANCES_DATA = "0x92e3cc2d"


def _eth_call(to: str, data: str, block: str | int = "latest") -> str | None:
    """Execute a raw eth_call via Etherscan's proxy endpoint."""
    block_tag = hex(block) if isinstance(block, int) else block
    params = {
        "module": "proxy",
        "action": "eth_call",
        "to": to,
        "data": data,
        "tag": block_tag,
    }
    result = _get(params).get("result", "")
    if not result or result == "0x":
        return None
    return result


def _decode_balances(hex_result: str, decimals: list[int]) -> list[float]:
    """Decode ABI-encoded uint256[] from get_balances() return value."""
    if not hex_result:
        return []
    raw = hex_result.removeprefix("0x")
    # First 64 hex chars = offset to array (0x20), next 64 = length
    # After that, each element is 64 hex chars (32 bytes)
    try:
        # Skip the dynamic array offset word (64 chars)
        raw = raw[64:]
        n_tokens = int(raw[:64], 16)
        raw = raw[64:]
        balances = []
        for i, dec in enumerate(decimals[:n_tokens]):
            word = raw[i * 64 : (i + 1) * 64]
            raw_int = int(word, 16)
            balances.append(raw_int / (10 ** dec))
        return balances
    except Exception as exc:
        logger.warning("Failed to decode get_balances() result: %s", exc)
        return []


def verify_pool_sizes() -> list[dict[str, Any]]:
    """Check each pool's on-chain balance at each event-start block."""
    if not os.environ.get("ETHERSCAN_API_KEY"):
        logger.warning(
            "ETHERSCAN_API_KEY not set; cannot call Etherscan.  "
            "Pool-size verification requires the API key."
        )
        return []

    events = load_events()
    rows: list[dict[str, Any]] = []

    for contract_addr, info in _POOL_EVENT_MAP.items():
        node_id   = info["node_id"]
        token_dec = info["token_decimals"]
        pool_cfg  = _POOL_CONFIGS.get(contract_addr.lower())
        if pool_cfg is None:
            logger.warning("No PoolConfig for %s (%s); skipping.", node_id, contract_addr[:10])
            continue

        hardcoded_usd = pool_cfg.pool_size_usd

        for event_id in info["events"]:
            if event_id not in events:
                continue
            evt = events[event_id]
            start_str = f"{evt['analysis_window_utc'][0]}T00:00:00Z"
            start_ts  = int(parse_iso_utc(start_str).timestamp())
            block_num = get_block_number_by_timestamp(start_ts)

            if not block_num:
                logger.warning(
                    "Could not resolve block for %s / %s", event_id, node_id
                )
                rows.append({
                    "event_id":          event_id,
                    "node_id":           node_id,
                    "contract":          contract_addr[:10] + "…",
                    "block_number":      None,
                    "onchain_tvl_usd":   None,
                    "hardcoded_tvl_usd": hardcoded_usd,
                    "ratio":             None,
                    "status":            "block_lookup_failed",
                })
                continue

            logger.info(
                "Calling get_balances() for %s at block %d (%s)",
                node_id, block_num, event_id,
            )
            hex_result = _eth_call(contract_addr, _GET_BALANCES_DATA, block_num)
            balances   = _decode_balances(hex_result or "", token_dec)

            if balances:
                onchain_tvl = sum(balances)
                ratio = onchain_tvl / hardcoded_usd if hardcoded_usd else None
                status = "OK" if ratio is not None and 0.8 <= ratio <= 1.2 else "RATIO_OUTSIDE_20PCT"
                if status != "OK":
                    logger.warning(
                        "Pool-size ratio %.2f for %s/%s is outside [0.8, 1.2] — "
                        "consider updating pool_size_usd in curve.py._POOL_CONFIGS",
                        ratio, node_id, event_id,
                    )
            else:
                onchain_tvl = None
                ratio       = None
                status      = "decode_failed"

            rows.append({
                "event_id":          event_id,
                "node_id":           node_id,
                "contract":          contract_addr[:10] + "…",
                "block_number":      block_num,
                "onchain_tvl_usd":   round(onchain_tvl, 2) if onchain_tvl else None,
                "hardcoded_tvl_usd": hardcoded_usd,
                "ratio":             round(ratio, 4) if ratio else None,
                "status":            status,
            })

    return rows


def main() -> None:
    rows = verify_pool_sizes()
    if not rows:
        logger.info("No rows produced; check ETHERSCAN_API_KEY.")
        return

    out_dir = results_root() / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "table_pool_size_verification.csv"

    fieldnames = [
        "event_id", "node_id", "contract", "block_number",
        "onchain_tvl_usd", "hardcoded_tvl_usd", "ratio", "status",
    ]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_count   = sum(1 for r in rows if r["status"] == "OK")
    warn_count = sum(1 for r in rows if r["status"] == "RATIO_OUTSIDE_20PCT")
    logger.info(
        "Pool-size verification complete: %d OK, %d outside [0.8, 1.2] → %s",
        ok_count, warn_count, out_path,
    )
    if warn_count:
        logger.warning(
            "%d pool(s) have ratios outside [0.8, 1.2].  "
            "Update pool_size_usd in curve.py._POOL_CONFIGS.",
            warn_count,
        )


if __name__ == "__main__":
    main()
