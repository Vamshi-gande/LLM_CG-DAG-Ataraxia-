"""
Chain resolver.

Walks DEPENDENCY and CAUSAL edges within the activated subgraph and
returns (conclusion, support) pairs. Intermediate chain nodes are
intentionally discarded — the LLM never sees them. This is the
architectural core of pre-resolution: the graph walks the causal chain
internally, the LLM only does 1-hop articulation over the result.

CHAIN_EDGE_TYPES matches the edge types that propagate at full strength
(damping multiplier 1.0) in M3's spreading_activation() — DEPENDENCY and
CAUSAL are what M3 favors for chain reasoning, so they are also what M4
follows here.
"""

from typing import Dict, List, Tuple

from src.graph.graph import Graph
from src.graph.node import Node
from src.graph.edge import EdgeType


CHAIN_EDGE_TYPES = {EdgeType.DEPENDENCY, EdgeType.CAUSAL}


def find_chain_terminals(
    activated: Dict[str, float],
    graph: Graph,
) -> List[Node]:
    """
    Find terminal nodes in the activated subgraph.

    A terminal node:
      - is present in `activated`
      - has NO outgoing DEPENDENCY/CAUSAL edge to another activated node
      - has AT LEAST ONE incoming DEPENDENCY/CAUSAL edge from another
        activated node (i.e. it is the END of a chain, not a disconnected
        or isolated node)

    Nodes with no chain edges at all (in or out) are NOT terminals —
    they are disconnected and must not appear as conclusions.

    Ghost activated entries (graph.get_node() returns None) are skipped,
    matching M3's ghost-entry guard pattern.

    Returns terminals sorted by activation score descending.
    """
    terminals: List[Node] = []

    for node_id in activated:
        node = graph.get_node(node_id)
        if node is None:
            continue

        # Check for an outgoing chain edge into another activated node.
        has_outgoing_chain_edge = False
        for neighbor_node, edge in graph.neighbors(node_id):
            if edge.type in CHAIN_EDGE_TYPES and neighbor_node.id in activated:
                has_outgoing_chain_edge = True
                break

        if has_outgoing_chain_edge:
            continue

        # Check for at least one incoming chain edge from another
        # activated node — required to qualify as a genuine terminal.
        has_incoming_chain_edge = False
        for neighbor_node, edge in graph.reverse_neighbors(node_id):
            if edge.type in CHAIN_EDGE_TYPES and neighbor_node.id in activated:
                has_incoming_chain_edge = True
                break

        if not has_incoming_chain_edge:
            continue

        terminals.append(node)

    terminals.sort(key=lambda n: activated[n.id], reverse=True)
    return terminals


def get_chain_support(
    terminal: Node,
    activated: Dict[str, float],
    graph: Graph,
    max_support: int = 3,
) -> List[Node]:
    """
    Get 1-hop support nodes for a terminal conclusion.

    Support nodes are the DIRECT predecessors of `terminal` via
    DEPENDENCY/CAUSAL edges that are also present in `activated`.
    Only 1 hop back — do NOT recurse into the rest of the chain.

    Ghost predecessors (graph.get_node() returns None) are skipped.

    Returns up to `max_support` nodes sorted by activation score
    descending.
    """
    support: List[Node] = []

    for neighbor_node, edge in graph.reverse_neighbors(terminal.id):
        if edge.type not in CHAIN_EDGE_TYPES:
            continue
        if neighbor_node.id not in activated:
            continue
        node = graph.get_node(neighbor_node.id)
        if node is None:
            continue
        support.append(node)

    support.sort(key=lambda n: activated[n.id], reverse=True)
    return support[:max_support]


def resolve_chain(
    activated: Dict[str, float],
    graph: Graph,
    max_conclusions: int = 5,
) -> List[Tuple[Node, List[Node]]]:
    """
    Walk dependency chains in the activated subgraph.

    Returns a list of (conclusion_node, [support_nodes]) pairs, sorted by
    the conclusion's activation score descending, capped at
    `max_conclusions`.

    A terminal is only included if it has non-empty support — a terminal
    with no chain-edge predecessors is a disconnected node, not a chain
    conclusion, and is silently discarded.

    Intermediate chain nodes never appear in the output. Pure read
    operation: does not modify the graph or add nodes.
    """
    if not activated:
        return []

    terminals = find_chain_terminals(activated, graph)

    resolved: List[Tuple[Node, List[Node]]] = []
    for terminal in terminals:
        support = get_chain_support(terminal, activated, graph)
        if not support:
            continue
        resolved.append((terminal, support))
        if len(resolved) >= max_conclusions:
            break

    return resolved