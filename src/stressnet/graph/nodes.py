"""Node definitions and metadata for the contagion network."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from stressnet.config import load_nodes


@dataclass
class Node:
    """A node in the contagion network."""

    id: str
    layer: str          # 'CEX' | 'DEX' | 'onchain_flow' | 'bridge_flow' | 'mint_burn'
    asset: str | None = None
    venue: str | None = None
    chain: str | None = None
    tier: str = "B"
    events_covered: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_all_nodes() -> list[Node]:
    """Load node definitions from configs/nodes.yaml."""
    config = load_nodes()
    nodes = []

    for node_dict in config.get("market_nodes", []):
        nodes.append(Node(
            id=node_dict["id"],
            layer=node_dict.get("layer", "CEX"),
            asset=node_dict.get("asset"),
            venue=node_dict.get("venue"),
            tier=node_dict.get("tier", "B"),
            events_covered=node_dict.get("events_covered", []),
            metadata=node_dict,
        ))

    for node_dict in config.get("pool_nodes", []):
        nodes.append(Node(
            id=node_dict["id"],
            layer=node_dict.get("layer", "DEX"),
            venue=node_dict.get("venue"),
            chain=node_dict.get("chain"),
            tier=node_dict.get("tier", "A"),
            events_covered=node_dict.get("events_covered", []),
            metadata=node_dict,
        ))

    for node_dict in config.get("flow_nodes", []):
        nodes.append(Node(
            id=node_dict["id"],
            layer=node_dict.get("layer", "onchain_flow"),
            asset=node_dict.get("asset"),
            chain=node_dict.get("chain"),
            tier=node_dict.get("tier", "B"),
            events_covered=node_dict.get("events_covered", []),
            metadata=node_dict,
        ))

    return nodes


def nodes_for_event(event_id: str) -> list[Node]:
    """Return all nodes that are configured for a given event."""
    return [n for n in load_all_nodes() if event_id in n.events_covered]
