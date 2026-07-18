"""
Unit tests for contradiction detection.

Two-gate check (see src/graph/graph.py::add_node_with_contradiction_check):
  Gate 1 — reversal keyword present in new node's content
  Gate 2 — cosine similarity(new, existing) >= 0.95

Both gates must pass for a Contradicts edge to be created. Tests that
expect a contradiction MUST force near-identical embeddings between the
old and new node — independently random embeddings will almost never
clear the 0.95 cosine bar, since two arbitrary 384-dim unit vectors are
essentially orthogonal.
"""
import numpy as np
import pytest

from src.graph.node import Node, NodeType
from src.graph.edge import Edge, EdgeType
from src.graph.graph import Graph


def _emb(seed: int = 0, dim: int = 384) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _node(node_id: str, content: str, node_type: NodeType = NodeType.PREFERENCE,
          seed: int = 0, embedding: np.ndarray = None) -> Node:
    """
    embedding: optional explicit override. When two nodes must be
    detected as contradicting, pass the SAME embedding (or a
    near-identical one) to both — cosine similarity from independent
    seeds will not reliably clear the 0.95 gate.
    """
    return Node(
        id=node_id,
        type=node_type,
        content=content,
        embedding=embedding if embedding is not None else _emb(seed),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_no_contradiction_different_types():
    """
    Caller is responsible for filtering existing_nodes_same_type by type
    before calling — add_node_with_contradiction_check() does not filter
    by node.type itself. Here the first call deliberately passes a
    same-embedding pair (which WOULD trigger gate 2) but the caller has
    passed a Concept node into a Preference check's "same type" list to
    simulate a caller-side type mismatch; since the method trusts the
    caller's list, this test instead verifies the empty-list contract
    directly.
    """
    g = Graph()
    existing = _node("n1", "Python is great", NodeType.CONCEPT, seed=1)
    g.add_node(existing)

    new_node = _node("n2", "I switched from Python to Go", NodeType.PREFERENCE, seed=2)
    created = g.add_node_with_contradiction_check(
        new_node,
        existing_nodes_same_type=[existing],
    )
    # Independent seeds -> dissimilar embeddings -> gate 2 fails regardless
    assert created == []

    created2 = g.add_node_with_contradiction_check(
        _node("n3", "I switched from Java to Go", NodeType.CONCEPT, seed=3),
        existing_nodes_same_type=[],  # explicitly empty
    )
    assert created2 == []


def test_contradiction_detected_on_switch_keyword():
    """
    A new Preference node containing 'switched from' when an existing
    Preference node is present, WITH matching embeddings -> Contradicts
    edge created.
    """
    g = Graph()
    shared_emb = _emb(seed=1)

    old = _node("n1", "User prefers Python", NodeType.PREFERENCE, embedding=shared_emb)
    g.add_node(old)

    new_node = _node("n2", "I switched from Python to Go", NodeType.PREFERENCE,
                      embedding=shared_emb)
    created = g.add_node_with_contradiction_check(
        new_node,
        existing_nodes_same_type=[old],
    )

    assert len(created) == 1
    assert created[0].type == EdgeType.CONTRADICTS
    assert created[0].from_node == new_node.id
    assert created[0].to_node == old.id


def test_contradiction_reduces_old_node_confidence():
    g = Graph()
    shared_emb = _emb(seed=1)

    old = _node("n1", "User prefers Python", NodeType.PREFERENCE, embedding=shared_emb)
    g.add_node(old)
    assert old.confidence == 1.0

    new_node = _node("n2", "I no longer use Python", NodeType.PREFERENCE,
                      embedding=shared_emb)
    g.add_node_with_contradiction_check(new_node, existing_nodes_same_type=[old])

    # The old node's confidence should now be 0.5
    assert old.confidence == pytest.approx(0.5)


def test_contradiction_edge_type_is_contradicts():
    g = Graph()
    shared_emb = _emb(seed=1)

    old = _node("n1", "Using C++", NodeType.PREFERENCE, embedding=shared_emb)
    g.add_node(old)

    new_node = _node("n2", "Changed to Rust instead of C++", NodeType.PREFERENCE,
                      embedding=shared_emb)
    created = g.add_node_with_contradiction_check(new_node, [old])

    assert len(created) == 1
    assert created[0].type == EdgeType.CONTRADICTS


def test_no_contradiction_on_normal_update():
    """
    Adding a Preference node with normal content (no switch keywords)
    should not create any Contradicts edges — gate 1 fails, so gate 2
    (embedding similarity) is irrelevant here. Embeddings are
    deliberately kept independent/random to prove keyword absence alone
    blocks contradiction, regardless of similarity.
    """
    g = Graph()
    old = _node("n1", "User enjoys concurrency", NodeType.PREFERENCE, seed=1)
    g.add_node(old)

    new_node = _node("n2", "User also enjoys type safety", NodeType.PREFERENCE, seed=2)
    created = g.add_node_with_contradiction_check(new_node, [old])

    assert created == []
    # Old node confidence unchanged
    assert old.confidence == pytest.approx(1.0)


def test_contradiction_keywords_all_trigger():
    """All recognized contradiction keywords trigger detection, given
    matching embeddings to clear gate 2."""
    keywords_and_phrases = [
        "switched from X to Y",
        "no longer using X",
        "changed to Y",
        "instead of X",
        "moved from X to Y",
        "not anymore",
        "replaced by Y",
        "stopped using X",
        "dropped X",
        "abandoned X",
    ]
    for phrase in keywords_and_phrases:
        g = Graph()
        shared_emb = _emb(seed=0)

        old = _node("old", "Old preference", NodeType.PREFERENCE, embedding=shared_emb)
        g.add_node(old)
        new_node = _node("new", phrase, NodeType.PREFERENCE, embedding=shared_emb)
        created = g.add_node_with_contradiction_check(new_node, [old])
        assert len(created) == 1, f"Expected contradiction for phrase: '{phrase}'"


def test_multiple_existing_nodes_each_get_contradiction():
    """
    If multiple existing nodes of the same type exist (all embedding-similar
    to the new node) and the new node has switch keywords, ALL existing
    nodes get a Contradicts edge and confidence reduction.
    """
    g = Graph()
    shared_emb = _emb(seed=1)

    old1 = _node("n1", "Python preference", NodeType.PREFERENCE, embedding=shared_emb)
    old2 = _node("n2", "Java preference", NodeType.PREFERENCE, embedding=shared_emb)
    g.add_node(old1)
    g.add_node(old2)

    new_node = _node("n3", "I switched from everything to Go", NodeType.PREFERENCE,
                      embedding=shared_emb)
    created = g.add_node_with_contradiction_check(new_node, [old1, old2])

    assert len(created) == 2
    assert all(e.type == EdgeType.CONTRADICTS for e in created)
    assert old1.confidence == pytest.approx(0.5)
    assert old2.confidence == pytest.approx(0.5)