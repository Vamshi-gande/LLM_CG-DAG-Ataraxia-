"""
M5 — DAG Extraction

Converts a PreResolvedContext (from src.preresolve) into an ordered,
acyclic, token-budget-trimmed DAG structure ready for serialization.

Pipeline: build_subgraph -> detect_cycles -> topological_sort ->
trim_to_budget, orchestrated by extract_dag().

active_dag_ids lifetime: set at the start of extract_dag(), cleared in a
finally block so compression engines (M8) are never permanently locked
out of a node set if extraction raises partway through.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

from src.graph.graph import Graph
from src.graph.node import Node
from src.graph.edge import Edge
from src.preresolve.classify import QueryType
from src.preresolve.preresolve import PreResolvedContext


@dataclass
class DAG:
    """
    Output of DAG extraction. Consumed by the serializer (src.serialize).

    nodes_ordered:      topologically sorted nodes, foundation first.
    edges:               acyclic, budget-trimmed edges between those nodes.
    query_type:          carried from PreResolvedContext for serializer routing.
    resolved_pairs:      carried from PreResolvedContext (CHAIN queries).
    synthesis_node:      carried from PreResolvedContext (SYNTHESIS queries).
    lookup_nodes:        carried from PreResolvedContext (LOOKUP queries).
    activation_scores:   node_id -> score, for trim/tie-break decisions.
    token_estimate:      estimated token count of the trimmed node content.
    """
    nodes_ordered: List[Node]
    edges: List[Edge]
    query_type: QueryType
    resolved_pairs: List[Tuple[Node, List[Node]]]
    synthesis_node: Optional[Node]
    lookup_nodes: List[Node]
    activation_scores: Dict[str, float]
    token_estimate: int = 0


def _estimate_tokens(nodes: List[Node], chars_per_token: float) -> float:
    total_chars = sum(len(n.content) for n in nodes)
    return total_chars / chars_per_token


def build_subgraph(
    context: PreResolvedContext,
    graph: Graph,
    max_candidates: int = 50,
) -> Tuple[List[Node], List[Edge]]:
    """
    Build candidate node + edge lists from the activated set.

    context.activated is a Dict[node_id -> score]. Dict iteration order
    is insertion order, NOT value order — it is NOT guaranteed to already
    be sorted by score. It must be explicitly sorted here rather than
    sliced directly, or the "top max_candidates" selection silently picks
    whatever nodes were inserted first instead of the highest-activation
    ones.

    Does NOT modify graph. Does NOT set active_dag_ids — that happens in
    extract_dag(), which wraps this function.
    """
    ranked_ids = [
        node_id
        for node_id, _score in sorted(
            context.activated.items(), key=lambda item: item[1], reverse=True
        )
    ][:max_candidates]

    candidate_nodes: List[Node] = []
    for node_id in ranked_ids:
        node = graph.get_node(node_id)
        if node is None:
            # Ghost entry — activated dict references a node no longer in
            # the graph. Same defensive skip pattern used in M3/M4.
            continue
        candidate_nodes.append(node)

    candidate_ids: Set[str] = {n.id for n in candidate_nodes}

    included_edges: List[Edge] = []
    seen_edge_ids: Set[str] = set()
    for node in candidate_nodes:
        for edge in graph.get_edges_from(node.id):
            if edge.to_node in candidate_ids and edge.id not in seen_edge_ids:
                included_edges.append(edge)
                seen_edge_ids.add(edge.id)

    return candidate_nodes, included_edges


def detect_cycles(
    nodes: List[Node],
    edges: List[Edge],
) -> List[Edge]:
    """
    DFS-based cycle detection. Returns edges to REMOVE to make the
    subgraph acyclic — does not mutate the input edge list.

    When a back-edge closes a cycle, the lowest-weight edge in that
    cycle is marked for removal. Ties broken by removing the more
    recently created edge (keep the earlier one).
    """
    node_ids = {n.id for n in nodes}
    adjacency: Dict[str, List[Edge]] = {nid: [] for nid in node_ids}
    for e in edges:
        if e.from_node in adjacency and e.to_node in node_ids:
            adjacency[e.from_node].append(e)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {nid: WHITE for nid in node_ids}
    to_remove: List[Edge] = []
    removed_ids: Set[str] = set()

    def dfs(u: str, path_edges: List[Edge]) -> None:
        color[u] = GRAY
        for edge in adjacency.get(u, []):
            if edge.id in removed_ids:
                continue
            v = edge.to_node
            if color.get(v, WHITE) == WHITE:
                path_edges.append(edge)
                dfs(v, path_edges)
                path_edges.pop()
            elif color.get(v) == GRAY:
                # Back-edge: v is an ancestor of u on the current DFS
                # path. The cycle is path_edges[from v onward] + this
                # back-edge.
                try:
                    start_index = next(
                        i for i, pe in enumerate(path_edges) if pe.from_node == v
                    )
                    cycle_edges = path_edges[start_index:] + [edge]
                except StopIteration:
                    cycle_edges = [edge]

                weakest = min(
                    cycle_edges,
                    key=lambda ce: (ce.weight, -ce.created_at),
                )
                if weakest.id not in removed_ids:
                    to_remove.append(weakest)
                    removed_ids.add(weakest.id)
            # BLACK neighbor: cross/forward edge in this DFS variant — ignore.
        color[u] = BLACK

    for n in nodes:
        if color[n.id] == WHITE:
            dfs(n.id, [])

    return to_remove


def topological_sort(
    nodes: List[Node],
    edges: List[Edge],
    activation_scores: Dict[str, float],
) -> List[Node]:
    """
    Kahn's algorithm topological sort. Foundation nodes (no incoming
    edges within this subgraph) come first; derived nodes follow their
    dependencies.

    activation_scores is REQUIRED (not optional) — it breaks ties among
    simultaneously-available zero-in-degree nodes, and orders
    disconnected nodes appended after the connected component. Without
    it, tie ordering is arbitrary and non-deterministic across runs.

    Disconnected nodes (no edges to/from them at all in this subgraph)
    are appended after the connected component, sorted by activation
    score descending.
    """
    node_by_id: Dict[str, Node] = {n.id: n for n in nodes}
    node_ids: Set[str] = set(node_by_id.keys())

    in_degree: Dict[str, int] = {nid: 0 for nid in node_ids}
    out_edges: Dict[str, List[str]] = {nid: [] for nid in node_ids}

    for e in edges:
        if e.from_node in node_ids and e.to_node in node_ids:
            out_edges[e.from_node].append(e.to_node)
            in_degree[e.to_node] += 1

    def score_of(nid: str) -> float:
        return activation_scores.get(nid, 0.0)

    connected_ids: Set[str] = {
        nid for nid in node_ids
        if in_degree[nid] > 0 or len(out_edges[nid]) > 0
    }

    queue: List[str] = sorted(
        (nid for nid in connected_ids if in_degree[nid] == 0),
        key=score_of,
        reverse=True,
    )

    result_ids: List[str] = []
    visited: Set[str] = set()

    while queue:
        # Re-sort each pop: multiple nodes can reach zero-in-degree in the
        # same step, and ties must resolve deterministically by score.
        queue.sort(key=score_of, reverse=True)
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        result_ids.append(current)
        for neighbor in out_edges[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0 and neighbor not in visited:
                queue.append(neighbor)

    disconnected_ids = [nid for nid in node_ids if nid not in connected_ids]
    disconnected_ids.sort(key=score_of, reverse=True)

    ordered_ids = result_ids + disconnected_ids
    return [node_by_id[nid] for nid in ordered_ids]


def trim_to_budget(
    nodes: List[Node],
    edges: List[Edge],
    activation_scores: Dict[str, float],
    token_budget: int = 800,
    chars_per_token: float = 3.5,
) -> Tuple[List[Node], List[Edge]]:
    """
    Remove lowest-activation leaf nodes until estimated token count fits
    within budget.

    "Leaf" is determined from the ORIGINAL edge set (out-degree == 0),
    not recomputed after each removal. This matters: if leaf status were
    recomputed against the shrinking edge set, a root/branching node
    would eventually lose all its outgoing edges once its children are
    pruned and would itself become "removable" — silently violating the
    invariant that structurally important (originally non-leaf) nodes
    are never removed. Freezing eligibility to the original structure
    keeps that invariant intact regardless of how much trimming happens.

    Stops trimming (without hitting budget) if no eligible leaf remains —
    never removes a non-leaf node, since that would disconnect the DAG.

    Preserves the input node order.
    """
    current_nodes = list(nodes)
    current_edges = list(edges)

    original_out_degree: Dict[str, int] = {n.id: 0 for n in nodes}
    for e in edges:
        if e.from_node in original_out_degree:
            original_out_degree[e.from_node] += 1

    eligible_leaf_ids: Set[str] = {
        nid for nid, deg in original_out_degree.items() if deg == 0
    }

    while _estimate_tokens(current_nodes, chars_per_token) > token_budget:
        present_ids = {n.id for n in current_nodes}
        removable = eligible_leaf_ids & present_ids
        if not removable:
            break

        weakest_id = min(removable, key=lambda nid: activation_scores.get(nid, 0.0))

        current_nodes = [n for n in current_nodes if n.id != weakest_id]
        current_edges = [
            e for e in current_edges
            if e.from_node != weakest_id and e.to_node != weakest_id
        ]

    return current_nodes, current_edges


def extract_dag(
    context: PreResolvedContext,
    graph: Graph,
    max_candidates: int = 50,
    token_budget: int = 800,
    chars_per_token: float = 3.5,
) -> DAG:
    """
    Full DAG extraction pipeline. Main entry point consumed by M6's proxy.

    active_dag_ids is set BEFORE extraction begins and cleared in a
    finally block, so it is always cleared even if an exception occurs
    partway through — leaving it populated would silently and permanently
    block every compression engine (M8) from ever touching those nodes.

    Does NOT call the LLM. Does NOT modify node content. Pure read + compute.
    """
    graph.active_dag_ids.update(context.activated.keys())
    try:
        nodes, edges = build_subgraph(context, graph, max_candidates=max_candidates)

        cycle_edges = detect_cycles(nodes, edges)
        cycle_edge_ids = {e.id for e in cycle_edges}
        acyclic_edges = [e for e in edges if e.id not in cycle_edge_ids]

        sorted_nodes = topological_sort(nodes, acyclic_edges, context.activated)

        trimmed_nodes, trimmed_edges = trim_to_budget(
            sorted_nodes,
            acyclic_edges,
            context.activated,
            token_budget=token_budget,
            chars_per_token=chars_per_token,
        )

        token_estimate = int(_estimate_tokens(trimmed_nodes, chars_per_token))

        return DAG(
            nodes_ordered=trimmed_nodes,
            edges=trimmed_edges,
            query_type=context.query_type,
            resolved_pairs=context.resolved_pairs,
            synthesis_node=context.synthesis_node,
            lookup_nodes=context.lookup_nodes,
            activation_scores=context.activated,
            token_estimate=token_estimate,
        )
    finally:
        graph.active_dag_ids.difference_update(context.activated.keys())