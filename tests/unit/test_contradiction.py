"""
Unit tests for keyword-based contradiction detection.

At this milestone, detection uses only content keyword heuristics.
Embedding-based detection (cosine similarity) is added in M2.
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
          seed: int = 0) -> Node:
    return Node(
        id=node_id,
        type=node_type,
        content=content,
        embedding=_emb(seed),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_no_contradiction_different_types():
    """
    A Concept node and a Preference node with switch keywords should NOT
    create a Contradicts edge — type must match.
    """
    g = Graph()
    existing = _node("n1", "Python is great", NodeType.CONCEPT, seed=1)
    g.add_node(existing)

    new_node = _node("n2", "I switched from Python to Go", NodeType.PREFERENCE, seed=2)
    created = g.add_node_with_contradiction_check(
        new_node,
        existing_nodes_same_type=[existing],  # same type list is empty effectively
    )
    # The existing node is Concept; new is Preference — types differ
    # But the caller controls existing_nodes_same_type — pass empty list
    created2 = g.add_node_with_contradiction_check(
        _node("n3", "I switched from Java to Go", NodeType.CONCEPT, seed=3),
        existing_nodes_same_type=[],  # explicitly empty
    )
    assert created == []
    assert created2 == []


def test_contradiction_detected_on_switch_keyword():
    """
    A new Preference node containing 'switched from' when an existing
    Preference node is present → Contradicts edge created.
    """
    g = Graph()
    old = _node("n1", "User prefers Python", NodeType.PREFERENCE, seed=1)
    g.add_node(old)

    new_node = _node("n2", "I switched from Python to Go", NodeType.PREFERENCE, seed=2)
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
    old = _node("n1", "User prefers Python", NodeType.PREFERENCE, seed=1)
    g.add_node(old)
    assert old.confidence == 1.0

    new_node = _node("n2", "I no longer use Python", NodeType.PREFERENCE, seed=2)
    g.add_node_with_contradiction_check(new_node, existing_nodes_same_type=[old])

    # The old node's confidence should now be 0.5
    assert old.confidence == pytest.approx(0.5)


def test_contradiction_edge_type_is_contradicts():
    g = Graph()
    old = _node("n1", "Using C++", NodeType.PREFERENCE, seed=1)
    g.add_node(old)

    new_node = _node("n2", "Changed to Rust instead of C++", NodeType.PREFERENCE, seed=2)
    created = g.add_node_with_contradiction_check(new_node, [old])

    assert len(created) == 1
    assert created[0].type == EdgeType.CONTRADICTS


def test_no_contradiction_on_normal_update():
    """
    Adding a Preference node with normal content (no switch keywords)
    should not create any Contradicts edges.
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
    """All recognized contradiction keywords trigger detection."""
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
        old = _node("old", "Old preference", NodeType.PREFERENCE, seed=0)
        g.add_node(old)
        new_node = _node("new", phrase, NodeType.PREFERENCE, seed=1)
        created = g.add_node_with_contradiction_check(new_node, [old])
        assert len(created) == 1, f"Expected contradiction for phrase: '{phrase}'"


def test_multiple_existing_nodes_each_get_contradiction():
    """
    If multiple existing nodes of the same type exist and the new node has
    switch keywords, ALL existing nodes get a Contradicts edge and confidence
    reduction.
    """
    g = Graph()
    old1 = _node("n1", "Python preference", NodeType.PREFERENCE, seed=1)
    old2 = _node("n2", "Java preference", NodeType.PREFERENCE, seed=2)
    g.add_node(old1)
    g.add_node(old2)

    new_node = _node("n3", "I switched from everything to Go", NodeType.PREFERENCE, seed=3)
    created = g.add_node_with_contradiction_check(new_node, [old1, old2])

    assert len(created) == 2
    assert all(e.type == EdgeType.CONTRADICTS for e in created)
    assert old1.confidence == pytest.approx(0.5)
    assert old2.confidence == pytest.approx(0.5)