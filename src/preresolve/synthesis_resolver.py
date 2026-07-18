"""
Synthesis resolver.

For synthesis-type queries ("who would benefit", "compare X and Y"), the
useful signal is combining facts from graph regions that are NOT closely
connected — i.e. genuinely distant activated nodes. This module finds the
most relevant distant pair and builds a temporary, never-persisted
synthesis node summarizing them.

Traversal is UNDIRECTED: synthesis distance measures semantic separation,
not information flow direction. If A -> B exists, (B, A) is still
considered connected. Both graph.neighbors() (outgoing) and
graph.reverse_neighbors() (incoming) are used when computing reachability.
"""

import time
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.graph.graph import Graph
from src.graph.node import Node, NodeType


def _bfs_reachable_within(
    start_id: str,
    graph: Graph,
    max_hops: int,
) -> set:
    """
    Undirected BFS from start_id. Returns the set of node ids reachable
    within 1..max_hops hops (does NOT include start_id itself).
    """
    visited = {start_id}
    reachable: set = set()
    frontier = [start_id]

    for _ in range(max_hops):
        next_frontier: List[str] = []
        for node_id in frontier:
            node = graph.get_node(node_id)
            if node is None:
                continue
            neighbor_pairs = graph.neighbors(node_id) + graph.reverse_neighbors(node_id)
            for neighbor_node, _edge in neighbor_pairs:
                nbr_id = neighbor_node.id
                if nbr_id not in visited:
                    visited.add(nbr_id)
                    reachable.add(nbr_id)
                    next_frontier.append(nbr_id)
        frontier = next_frontier
        if not frontier:
            break

    return reachable


def find_distant_node_pairs(
    activated: Dict[str, float],
    graph: Graph,
    max_hop_distance: int = 3,
) -> List[Tuple[str, str]]:
    """
    Identify pairs of activated nodes that are NOT connected within
    max_hop_distance hops, using undirected traversal over the graph.

    Pair scoring: score(a, b) = activated[a] + activated[b].
    Returns pairs sorted by score descending.

    Ghost activated entries (graph.get_node() returns None) are skipped.
    Returns [] if all activated nodes are mutually reachable within the
    hop limit (or if fewer than 2 valid activated nodes exist).
    """
    valid_ids = [
        node_id for node_id in activated if graph.get_node(node_id) is not None
    ]

    if len(valid_ids) < 2:
        return []

    reachable_within: Dict[str, set] = {
        node_id: _bfs_reachable_within(node_id, graph, max_hop_distance)
        for node_id in valid_ids
    }

    scored_pairs: List[Tuple[str, str, float]] = []
    seen_pairs: set = set()

    for i, a in enumerate(valid_ids):
        for b in valid_ids[i + 1:]:
            if b in reachable_within[a]:
                continue
            pair_key = tuple(sorted((a, b)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            score = activated[a] + activated[b]
            scored_pairs.append((a, b, score))

    scored_pairs.sort(key=lambda item: item[2], reverse=True)
    return [(a, b) for a, b, _score in scored_pairs]


def _safe_average_embedding(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """
    Average two L2-normalized embeddings and re-normalize.

    Guards against emb_a ~= -emb_b, where the average is ~zero and
    normalizing would divide by (near) zero. In that degenerate case,
    fall back to emb_a (already normalized) rather than crashing.
    """
    avg = (emb_a + emb_b) / 2.0
    norm = np.linalg.norm(avg)
    if norm < 1e-9:
        return emb_a.copy()
    return (avg / norm).astype(np.float32)


def resolve_synthesis(
    activated: Dict[str, float],
    graph: Graph,
) -> Optional[Node]:
    """
    Create a temporary synthesis node combining content from the
    highest-scoring distant activated node pair.

    Returns None if no distant pairs are found (caller should then use
    the activated nodes directly instead of a synthesis node).

    The returned Node is NEVER added to the graph, SQLite, or the HNSW
    index — it exists only for serialization (M6) and is discarded
    afterward. Its id is prefixed "temp_" as a safety/debugging marker.
    """
    pairs = find_distant_node_pairs(activated, graph)
    if not pairs:
        return None

    node_a_id, node_b_id = pairs[0]
    node_a = graph.get_node(node_a_id)
    node_b = graph.get_node(node_b_id)

    if node_a is None or node_b is None:
        return None

    combined_embedding = _safe_average_embedding(node_a.embedding, node_b.embedding)
    combined_priority = (activated[node_a.id] + activated[node_b.id]) / 2.0

    temp = Node(
        id="temp_" + str(uuid.uuid4()),
        type=NodeType.SUMMARY,
        content=f"{node_a.content} | {node_b.content}",
        embedding=combined_embedding,
        priority=combined_priority,
        created_at=time.time(),
        updated_at=time.time(),
        access_count=0,
        confidence=1.0,
        version=1,
        last_reconciled_version=0,
    )

    return temp 