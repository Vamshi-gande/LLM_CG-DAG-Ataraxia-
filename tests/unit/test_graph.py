"""
Unit tests for src/graph/graph.py, node.py, edge.py.

All 12 tests specified in the Milestone 1 prompt are covered.
"""
import numpy as np
import pytest

from src.graph.node import Node, NodeType
from src.graph.edge import Edge, EdgeType
from src.graph.graph import Graph


# ── Helpers ───────────────────────────────────────────────────────────────────

def _emb(seed: int = 0, dim: int = 384) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _node(node_id: str = "n1", content: str = "test", seed: int = 0,
          node_type: NodeType = NodeType.CONCEPT) -> Node:
    return Node(
        id=node_id,
        type=node_type,
        content=content,
        embedding=_emb(seed),
    )


def _edge(edge_id: str = "e1", from_node: str = "n1", to_node: str = "n2",
          etype: EdgeType = EdgeType.CAUSAL, weight: float = 0.8) -> Edge:
    return Edge(
        id=edge_id,
        from_node=from_node,
        to_node=to_node,
        type=etype,
        weight=weight,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_add_and_get_node():
    g = Graph()
    n = _node("n1", "Systems Programming")
    g.add_node(n)

    retrieved = g.get_node("n1")
    assert retrieved is not None
    assert retrieved.id == "n1"
    assert retrieved.content == "Systems Programming"
    assert retrieved.type == NodeType.CONCEPT
    np.testing.assert_array_equal(retrieved.embedding, n.embedding)


def test_get_nonexistent_node_returns_none():
    g = Graph()
    assert g.get_node("does-not-exist") is None


def test_add_edge_appears_in_adjacency():
    g = Graph()
    g.add_node(_node("n1"))
    g.add_node(_node("n2", seed=1))

    e = _edge("e1", "n1", "n2")
    g.add_edge(e)

    edges = g.get_edges_from("n1")

    assert len(edges) == 1
    assert edges[0].id == "e1"
    assert edges[0].to_node == "n2"


def test_add_edge_appears_in_reverse_adjacency():
    g = Graph()

    g.add_node(_node("n1"))
    g.add_node(_node("n2", seed=1))

    g.add_edge(_edge("e1", "n1", "n2"))

    edges = g.get_edges_to("n2")

    assert len(edges) == 1
    assert edges[0].id == "e1"
    assert edges[0].from_node == "n1"


def test_node_count_increments():
    g = Graph()

    assert g.node_count() == 0

    g.add_node(_node("n1"))
    assert g.node_count() == 1

    g.add_node(_node("n2", seed=1))
    assert g.node_count() == 2


def test_edge_count_increments():
    g = Graph()

    g.add_node(_node("n1"))
    g.add_node(_node("n2", seed=1))
    g.add_node(_node("n3", seed=2))

    assert g.edge_count() == 0

    g.add_edge(_edge("e1", "n1", "n2"))
    assert g.edge_count() == 1

    g.add_edge(_edge("e2", "n2", "n3"))
    assert g.edge_count() == 2


def test_update_node_bumps_version():
    g = Graph()

    n = _node("n1")
    initial_version = n.version

    g.add_node(n)

    n.content = "Updated content"
    g.update_node(n)

    retrieved = g.get_node("n1")

    assert retrieved.version == initial_version + 1


def test_update_node_updates_timestamp():
    import time

    g = Graph()

    n = _node("n1")
    n.updated_at = 1000.0

    g.add_node(n)

    before = time.time()

    n.content = "Changed"
    g.update_node(n)

    after = time.time()

    retrieved = g.get_node("n1")

    assert retrieved.updated_at >= before
    assert retrieved.updated_at <= after


def test_neighbors_returns_correct_pairs():
    g = Graph()

    n1 = _node("n1", seed=0)
    n2 = _node("n2", seed=1)

    g.add_node(n1)
    g.add_node(n2)

    e = _edge("e1", "n1", "n2", EdgeType.SEMANTIC, 0.7)
    g.add_edge(e)

    pairs = g.neighbors("n1")

    assert len(pairs) == 1

    neighbor_node, connecting_edge = pairs[0]

    assert neighbor_node.id == "n2"
    assert connecting_edge.id == "e1"
    assert connecting_edge.type == EdgeType.SEMANTIC


def test_small_graph_fixture_loads(small_graph):
    nodes = small_graph["nodes"]
    edges = small_graph["edges"]

    g = Graph()

    for n in nodes:
        g.add_node(n)

    for e in edges:
        g.add_edge(e)

    assert g.node_count() == 6
    assert g.edge_count() == 5

    # Spot check
    target = nodes[3]
    retrieved = g.get_node(target.id)

    assert retrieved is not None
    assert retrieved.content == target.content


def test_causal_chain_adjacency(small_graph):
    nodes = small_graph["nodes"]
    edges = small_graph["edges"]

    g = Graph()

    for n in nodes:
        g.add_node(n)

    for e in edges:
        g.add_edge(e)

    chain = nodes

    # n1 -> n2
    pairs = g.neighbors(chain[0].id)
    neighbor_ids = {nb.id for nb, _ in pairs}

    assert chain[1].id in neighbor_ids

    # n2 -> n3
    pairs = g.neighbors(chain[1].id)
    neighbor_ids = {nb.id for nb, _ in pairs}

    assert chain[2].id in neighbor_ids

    # n3 -> n4
    pairs = g.neighbors(chain[2].id)

    assert any(nb.id == chain[3].id for nb, _ in pairs)


def test_remove_node_removes_from_adjacency():
    g = Graph()

    g.add_node(_node("n1"))
    g.add_node(_node("n2", seed=1))

    g.add_edge(_edge("e1", "n1", "n2"))

    g.remove_node("n1")

    assert g.get_node("n1") is None
    assert g.get_edges_from("n1") == []

    # n2 should no longer see incoming edge
    assert g.get_edges_to("n2") == []


def test_add_edge_to_nonexistent_source_raises():
    """
    add_edge() must reject edges whose source node is not in the graph.
    """
    g = Graph()

    g.add_node(_node("n2", seed=1))

    e = _edge("e1", from_node="ghost", to_node="n2")

    with pytest.raises(
        ValueError,
        match="not in graph"
    ):
        g.add_edge(e)

    assert g.edge_count() == 0


def test_add_edge_to_nonexistent_target_raises():
    """
    add_edge() must reject edges whose target node is not in the graph.
    """
    g = Graph()

    g.add_node(_node("n1", seed=0))

    e = _edge("e1", from_node="n1", to_node="ghost")

    with pytest.raises(
        ValueError,
        match="not in graph"
    ):
        g.add_edge(e)

    assert g.edge_count() == 0