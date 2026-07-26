"""
Graph updater — closes the feedback loop after an LLM response.

process_response() is the entry point, meant to be scheduled with
asyncio.create_task() from the proxy request handlers — never awaited
directly in the request path, since extraction (especially Layer 2) must
not add latency to the response the caller is waiting on.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.graph.graph import Graph
from src.graph.node import Node, NodeType
from src.graph.edge import Edge, EdgeType
from src.influence.table import add_influence, InfluenceEntry
from src.updater.extractor import extract_layer1, extract_layer2


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for L2-normalized vectors (= dot product)."""
    return float(np.dot(a, b))


async def update_graph_node(
    node: Node,
    graph: Graph,
    hnsw: Any,
    storage: Any,
) -> bool:
    """
    Eagerly add or update a single node, then recompute edge weights to
    its direct (hop-1) neighbors in memory only (no separate SQLite write
    for edge weights — they're recalculated at access time elsewhere too).

    Returns True if the node was added or updated.

    Note: this function and add_node_with_contradiction_check() are two
    separate entry points for getting a node into the graph. Use exactly
    one per node — add_node_with_contradiction_check() already calls
    graph.add_node() internally (M4 fix), so chaining both would
    double-add. Route through add_node_with_contradiction_check() only
    when a contradiction check against same-type nodes is actually
    needed; use update_graph_node() otherwise.
    """
    existing_same_type = graph.get_nodes_by_type(node.type)
    existing = next(
        (n for n in existing_same_type if n.content.strip().lower() == node.content.strip().lower()),
        None,
    )

    if existing is not None:
        existing.embedding = node.embedding
        graph.update_node(existing)
        await storage.queue_save_node(existing)
        target = existing
    else:
        graph.add_node(node)
        await storage.queue_save_node(node)
        target = node

    for neighbor, edge in graph.neighbors(target.id):
        edge.weight = round(_cosine_similarity(target.embedding, neighbor.embedding), 4)

    return True


async def add_graph_edge(
    from_content: str,
    to_content: str,
    edge_type: EdgeType,
    graph: Graph,
    storage: Any,
) -> Optional[Edge]:
    """
    Add a typed edge between two nodes identified by content
    (case-insensitive). Skips silently if either endpoint isn't in the
    graph yet — extraction may reference content the graph hasn't caught
    up to, which is expected, not an error.
    """
    from_node = None
    to_node = None
    from_key = from_content.strip().lower()
    to_key = to_content.strip().lower()
    for n in graph.get_all_nodes():
        content_key = n.content.strip().lower()
        if from_node is None and content_key == from_key:
            from_node = n
        if to_node is None and content_key == to_key:
            to_node = n
        if from_node is not None and to_node is not None:
            break

    if from_node is None or to_node is None:
        return None

    edge = Edge.new(from_node.id, to_node.id, edge_type, weight=0.8)
    graph.add_edge(edge)
    await storage.queue_save_edge(edge)
    return edge


async def process_response(
    response_text: str,
    graph: Graph,
    hnsw: Any,
    embedder: Any,
    storage: Any,
    influence_table: Dict[str, List[InfluenceEntry]],
    nlp: Any = None,
) -> None:
    """
    Main entry point. Schedule with asyncio.create_task() — do not await
    directly from a request handler.

    1. Layer 1 extraction (always).
    2. Layer 2 extraction (if nlp given and response is long enough).
    3. Merge + dedupe nodes by content (case-insensitive), Layer 1 wins.
    4. update_graph_node() for each merged node.
    5. add_graph_edge() for each relation.
    6. add_influence() for indirect (hop >= 2) neighbors of every updated node.
    """
    if not response_text or not response_text.strip():
        return

    l1_nodes, l1_relations = extract_layer1(response_text, embedder)

    l2_nodes: List[Node] = []
    l2_relations: List[Tuple[str, str, EdgeType]] = []
    if nlp is not None:
        l2_nodes, l2_relations = await extract_layer2(response_text, nlp, embedder)

    seen = {n.content.strip().lower() for n in l1_nodes}
    merged_nodes = list(l1_nodes)
    for n in l2_nodes:
        key = n.content.strip().lower()
        if key not in seen:
            seen.add(key)
            merged_nodes.append(n)

    merged_relations = list(l1_relations) + list(l2_relations)

    for node in merged_nodes:
        await update_graph_node(node, graph, hnsw, storage)

    for from_content, to_content, edge_type in merged_relations:
        await add_graph_edge(from_content, to_content, edge_type, graph, storage)

    for node in merged_nodes:
        resolved = next(
            (n for n in graph.get_all_nodes() if n.content.strip().lower() == node.content.strip().lower()),
            node,
        )
        add_influence(influence_table, resolved.id, resolved.version, graph, hop_limit=3)