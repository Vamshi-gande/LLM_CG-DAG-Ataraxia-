"""
Lookup resolver.

Simple single-hop lookups ("what language is this?", "what version?")
need no chain walking and no synthesis — just the top activated nodes
by score.
"""

from typing import Dict, List

from src.graph.graph import Graph
from src.graph.node import Node


def resolve_lookup(
    activated: Dict[str, float],
    graph: Graph,
    max_nodes: int = 3,
) -> List[Node]:
    """
    Return up to max_nodes activated Node objects, sorted by activation
    score descending.

    Ghost activated entries (graph.get_node() returns None) are skipped.
    Returns [] if activated is empty or contains only ghost entries.
    """
    valid_nodes: List[Node] = []
    for node_id in activated:
        node = graph.get_node(node_id)
        if node is None:
            continue
        valid_nodes.append(node)

    valid_nodes.sort(key=lambda n: activated[n.id], reverse=True)
    return valid_nodes[:max_nodes]