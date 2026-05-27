"""Curve pool event fetching via Etherscan logs or The Graph."""

from __future__ import annotations

from typing import Any

from stressnet.data.etherscan import get_logs

# Curve 3pool (DAI/USDC/USDT) on Ethereum
CURVE_3POOL_ADDRESS = "0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7"

# Event topic0 hashes (keccak256 of event signatures)
TOPIC_TOKEN_EXCHANGE = "0x8b3e96f2b889fa771c53c981b40daf005f63f637f1869f707052d15a3dd97140"
TOPIC_ADD_LIQUIDITY = "0x189f0db4e11e3d2a5dd7bfbcd5e79e4ae43d59db8b2668a7fc4f92a2f2c2a5e2"
TOPIC_REMOVE_LIQUIDITY = "0x9878ca375e106f2a43c3b599fc624568131c4c9a4ba66a14563715763be9d59d"
TOPIC_REMOVE_LIQUIDITY_IMBALANCE = "0x2b5508378d7e19e0d5fa338419034731416c4f5b219a10379956f764317fd47e"


def fetch_3pool_events(
    start_block: int,
    end_block: int,
    event_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch Curve 3pool events from Etherscan logs.

    Args:
        start_block: Start block number.
        end_block: End block number.
        event_types: List of ['TokenExchange', 'AddLiquidity', 'RemoveLiquidity'].
                     Defaults to all three.

    Returns:
        Flat list of log dicts from Etherscan.
    """
    topic_map = {
        "TokenExchange": TOPIC_TOKEN_EXCHANGE,
        "AddLiquidity": TOPIC_ADD_LIQUIDITY,
        "RemoveLiquidity": TOPIC_REMOVE_LIQUIDITY,
    }
    if event_types is None:
        event_types = list(topic_map.keys())

    all_logs: list[dict[str, Any]] = []
    for event_type in event_types:
        topic0 = topic_map.get(event_type)
        if not topic0:
            continue
        logs = get_logs(
            contract_address=CURVE_3POOL_ADDRESS,
            topic0=topic0,
            from_block=start_block,
            to_block=end_block,
        )
        for log in logs:
            log["_event_type"] = event_type
        all_logs.extend(logs)

    return sorted(all_logs, key=lambda x: (int(x.get("blockNumber", "0x0"), 16),
                                            int(x.get("logIndex", "0x0"), 16)))
