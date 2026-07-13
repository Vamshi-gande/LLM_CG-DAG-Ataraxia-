"""
Tests that Graph correctly syncs with HNSWIndex
when add_node / update_node / remove_node are called.
"""
import pytest
import numpy as np
from src.graph import Graph, Node, NodeType, Edge, EdgeType
from src.hnsw import HNSWIndex


@pytest.fixture
def wired_graph(dummy_embedder):
    """Graph with HNSW wired in."""
    g = Graph()
    idx = HNSWIndex(dim=384, M=16, ef_construction=200,
                    ef_search=50, max_elements=1000)
    g.set_hnsw(idx)
    return g, idx


def test_add_node_syncs_to_hnsw(wired_graph, dummy_embedder):
    g, idx = wired_graph
    node = Node.new(NodeType.CONCEPT, "test content", dummy_embedder("test content"))
    g.add_node(node)
    assert idx.contains(node.id)


def test_remove_node_removes_from_hnsw(wired_graph, dummy_embedder):
    g, idx = wired_graph
    node = Node.new(NodeType.CONCEPT, "to remove", dummy_embedder("to remove"))
    g.add_node(node)
    g.remove_node(node.id)
    assert not idx.contains(node.id)


def test_update_node_updates_hnsw(wired_graph, dummy_embedder):
    g, idx = wired_graph
    node = Node.new(NodeType.CONCEPT, "original", dummy_embedder("original"))
    g.add_node(node)
    node.embedding = dummy_embedder("updated content")
    g.update_node(node)
    assert idx.contains(node.id)


def test_graph_without_hnsw_does_not_crash(dummy_embedder):
    """Graph works correctly when no HNSW wired (unit test mode)."""
    g = Graph()
    node = Node.new(NodeType.CONCEPT, "no hnsw", dummy_embedder("no hnsw"))
    g.add_node(node)
    assert g.get_node(node.id) is not None


def test_get_nodes_by_type(wired_graph, dummy_embedder):
    g, idx = wired_graph
    g.add_node(Node.new(NodeType.CONCEPT, "c1", dummy_embedder("c1")))
    g.add_node(Node.new(NodeType.CONCEPT, "c2", dummy_embedder("c2")))
    g.add_node(Node.new(NodeType.GOAL, "g1", dummy_embedder("g1")))
    concepts = g.get_nodes_by_type(NodeType.CONCEPT)
    goals    = g.get_nodes_by_type(NodeType.GOAL)
    assert len(concepts) == 2
    assert len(goals) == 1


# Fix #6 — deterministic contradiction tests
# Use manually constructed near-identical embeddings so the outcome is guaranteed

def test_contradiction_creates_edge_when_both_gates_pass(dummy_embedder):
    """
    Gate 1 (reversal keyword) + Gate 2 (cosine >= 0.95) both satisfied.
    Contradiction edge must be created and old node confidence set to 0.5.
    """
    g = Graph()

    base_emb = dummy_embedder("User prefers Python")

    # Perturb very slightly — cosine will be ~0.9999, well above 0.95 threshold
    rng = np.random.default_rng(0)
    near_identical = base_emb + rng.standard_normal(384).astype(np.float32) * 0.005
    near_identical = (near_identical / np.linalg.norm(near_identical)).astype(np.float32)

    old_node = Node.new(NodeType.PREFERENCE, "User prefers Python", base_emb)
    g.add_node(old_node)

    new_node = Node.new(
        NodeType.PREFERENCE,
        "User switched from Python to Go",
        near_identical,
    )
    g.add_node(new_node)

    contradictions = g.add_node_with_contradiction_check(new_node, [old_node])

    assert len(contradictions) == 1
    assert contradictions[0].type == EdgeType.CONTRADICTS
    assert old_node.confidence == 0.5


def test_contradiction_does_not_trigger_without_reversal_keyword(dummy_embedder):
    """
    Gate 1 fails (no reversal keyword) — contradiction must NOT be created
    even when embeddings are near-identical.
    """
    g = Graph()

    base_emb = dummy_embedder("User prefers Python")

    rng = np.random.default_rng(0)
    near_identical = base_emb + rng.standard_normal(384).astype(np.float32) * 0.005
    near_identical = (near_identical / np.linalg.norm(near_identical)).astype(np.float32)

    old_node = Node.new(NodeType.PREFERENCE, "User prefers Python", base_emb)
    g.add_node(old_node)

    # High cosine similarity but NO reversal keyword
    new_node = Node.new(
        NodeType.PREFERENCE,
        "User also enjoys Go",
        near_identical,
    )
    g.add_node(new_node)

    contradictions = g.add_node_with_contradiction_check(new_node, [old_node])

    assert len(contradictions) == 0
    assert old_node.confidence == 1.0  # unchanged


def test_contradiction_does_not_trigger_when_embeddings_dissimilar(dummy_embedder):
    """
    Gate 1 passes (reversal keyword present) but Gate 2 fails (cosine < 0.95).
    Contradiction must NOT be created.
    """
    g = Graph()

    old_node = Node.new(
        NodeType.PREFERENCE,
        "User prefers Python",
        dummy_embedder("User prefers Python"),
    )
    g.add_node(old_node)

    # Reversal keyword present, but completely different embedding
    new_node = Node.new(
        NodeType.PREFERENCE,
        "User switched from Python to Go",
        dummy_embedder("completely unrelated topic about weather"),
    )
    g.add_node(new_node)

    contradictions = g.add_node_with_contradiction_check(new_node, [old_node])

    assert len(contradictions) == 0
    assert old_node.confidence == 1.0  # unchanged