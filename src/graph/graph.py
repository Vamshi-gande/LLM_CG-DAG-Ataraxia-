"""
Graph engine for Graph-DAG Middleware.
Manages nodes, edges, adjacency lists, and optional HNSW index integration.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from .node import Node, NodeType
from .edge import Edge, EdgeType

if TYPE_CHECKING:
    from src.hnsw import HNSWIndex


class Graph:
    """
    In-memory graph with adjacency list representation.

    HNSW index is optional: wired in via set_hnsw() during startup.
    Graph operates correctly without HNSW (unit test mode).

    Invariants:
    - add_edge() raises ValueError if either endpoint is not in _nodes.
    - Edges MUST be loaded after ALL nodes during startup sequence.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[str, Edge] = {}
        self._adj:   Dict[str, List[Edge]] = {}   # outgoing edges per node
        self._radj:  Dict[str, List[Edge]] = {}   # incoming edges per node
        self._hnsw: Optional["HNSWIndex"] = None
        # M3 will add: self.active_dag_ids: set = set()

    # ── HNSW integration ─────────────────────────────────────────────────────

    def set_hnsw(self, hnsw_index: "HNSWIndex") -> None:
        """Wire HNSW index after construction. Called from startup sequence."""
        self._hnsw = hnsw_index

    # ── Node operations ───────────────────────────────────────────────────────

    def add_node(self, node: Node) -> None:
        """Add a node to the graph. Initialises empty adjacency entries."""
        self._nodes[node.id] = node
        if node.id not in self._adj:
            self._adj[node.id] = []
        if node.id not in self._radj:
            self._radj[node.id] = []

        if self._hnsw is not None:
            if self._hnsw.contains(node.id):
                self._hnsw.update(node.id, node.embedding)
            else:
                self._hnsw.add(node.id, node.embedding)

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def update_node(self, node: Node) -> None:
        """Replace a node's data in-place. Bumps version and syncs HNSW."""
        node.bump_version()
        self._nodes[node.id] = node

        if self._hnsw is not None:
            self._hnsw.update(node.id, node.embedding)

    def remove_node(self, node_id: str) -> None:
        """
        Remove a node and all its edges from the graph.

        NOTE (M8): Compression scheduler must call queue_delete_edge() for
        all connected edges BEFORE archiving a node, or startup will crash.
        """
        if node_id not in self._nodes:
            return

        edges_to_remove = list(self._adj.get(node_id, [])) + list(self._radj.get(node_id, []))
        for edge in edges_to_remove:
            self._edges.pop(edge.id, None)
            other = edge.to_node if edge.from_node == node_id else edge.from_node
            self._adj[other]  = [e for e in self._adj.get(other, [])  if e.id != edge.id]
            self._radj[other] = [e for e in self._radj.get(other, []) if e.id != edge.id]

        del self._adj[node_id]
        del self._radj[node_id]
        del self._nodes[node_id]

        if self._hnsw is not None:
            self._hnsw.remove(node_id)

    def get_all_nodes(self) -> List[Node]:
        return list(self._nodes.values())

    def get_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        """O(N) — called off hot path during post-response graph update."""
        return [n for n in self._nodes.values() if n.type == node_type]

    def node_count(self) -> int:
        return len(self._nodes)

    # ── Edge operations ───────────────────────────────────────────────────────

    def add_edge(self, edge: Edge) -> None:
        """
        Add a directed edge.
        Raises ValueError if either endpoint node is not in the graph.
        """
        if edge.from_node not in self._nodes:
            raise ValueError(
                f"add_edge: from_node '{edge.from_node}' not in graph. "
                "Load all nodes before loading edges."
            )
        if edge.to_node not in self._nodes:
            raise ValueError(
                f"add_edge: to_node '{edge.to_node}' not in graph. "
                "Load all nodes before loading edges."
            )
        self._edges[edge.id] = edge
        self._adj[edge.from_node].append(edge)
        self._radj[edge.to_node].append(edge)

    def get_edges_from(self, node_id: str) -> List[Edge]:
        return list(self._adj.get(node_id, []))

    def get_edges_to(self, node_id: str) -> List[Edge]:
        return list(self._radj.get(node_id, []))

    def get_all_edges_for(self, node_id: str) -> List[Edge]:
        seen = set()
        result = []
        for e in self._adj.get(node_id, []) + self._radj.get(node_id, []):
            if e.id not in seen:
                seen.add(e.id)
                result.append(e)
        return result

    def get_all_edges(self) -> List[Edge]:
        return list(self._edges.values())

    def edge_count(self) -> int:
        return len(self._edges)

    # ── Traversal helpers ─────────────────────────────────────────────────────

    def neighbors(self, node_id: str) -> List[Tuple[Node, Edge]]:
        """Return (node, edge) pairs for outgoing edges."""
        result = []
        for edge in self._adj.get(node_id, []):
            n = self._nodes.get(edge.to_node)
            if n is not None:
                result.append((n, edge))
        return result

    def reverse_neighbors(self, node_id: str) -> List[Tuple[Node, Edge]]:
        """Return (node, edge) pairs for incoming edges."""
        result = []
        for edge in self._radj.get(node_id, []):
            n = self._nodes.get(edge.from_node)
            if n is not None:
                result.append((n, edge))
        return result

    # ── Contradiction check ───────────────────────────────────────────────────

    def add_node_with_contradiction_check(
        self,
        node: Node,
        existing_nodes_same_type: List[Node],
    ) -> List[Edge]:
        """
        Two-gate check:
          Gate 1: reversal keyword in content (cheap string check)
          Gate 2: cosine similarity >= 0.95 (L2-normalised => dot product)

        NOTE: 0.95 here is the CONTRADICTION threshold.
              config merge.similarity_threshold is also 0.95 — different operation.
        """
        REVERSAL_KEYWORDS = [
            "switched from", "no longer", "changed to",
            "instead of", "replaced", "moved from", "stopped using",
        ]
        COSINE_THRESHOLD = 0.95

        contradictions: List[Edge] = []
        content_lower = node.content.lower()
        has_reversal = any(kw in content_lower for kw in REVERSAL_KEYWORDS)

        if not has_reversal:
            return contradictions

        for existing in existing_nodes_same_type:
            similarity = float(np.dot(node.embedding, existing.embedding))
            if similarity >= COSINE_THRESHOLD:
                edge = Edge.new(
                    from_node=node.id,
                    to_node=existing.id,
                    type=EdgeType.CONTRADICTS,
                    weight=round(1.0 - similarity, 4),
                )
                self.add_edge(edge)
                existing.confidence = 0.5
                contradictions.append(edge)

        return contradictions