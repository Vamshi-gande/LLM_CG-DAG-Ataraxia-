"""
Unit tests for HNSWIndex.
Uses dummy_embedder fixture — no ONNX required.
"""
import pytest
import numpy as np
from src.hnsw import HNSWIndex


@pytest.fixture
def index():
    return HNSWIndex(dim=384, M=16, ef_construction=200,
                     ef_search=50, space="cosine", max_elements=1000)


def test_add_and_contains(index, dummy_embedder):
    index.add("node-1", dummy_embedder("test node"))
    assert index.contains("node-1")


def test_size_increments(index, dummy_embedder):
    index.add("a", dummy_embedder("a"))
    index.add("b", dummy_embedder("b"))
    assert index.size() == 2


def test_add_duplicate_raises(index, dummy_embedder):
    index.add("node-1", dummy_embedder("content"))
    with pytest.raises(ValueError, match="already in index"):
        index.add("node-1", dummy_embedder("content"))


def test_remove_makes_not_contained(index, dummy_embedder):
    index.add("node-1", dummy_embedder("content"))
    index.remove("node-1")
    assert not index.contains("node-1")


def test_remove_nonexistent_does_not_raise(index):
    index.remove("does-not-exist")


def test_update_changes_embedding(index, dummy_embedder):
    index.add("node-1", dummy_embedder("original"))
    index.update("node-1", dummy_embedder("updated content"))
    assert index.contains("node-1")


def test_update_nonexistent_raises(index, dummy_embedder):
    with pytest.raises(ValueError, match="not in index"):
        index.update("ghost", dummy_embedder("x"))


def test_search_returns_list(index, dummy_embedder):
    for i in range(10):
        index.add(f"node-{i}", dummy_embedder(f"content {i}"))
    results = index.search(dummy_embedder("content 3"), k=5)
    assert isinstance(results, list)
    assert len(results) <= 5


def test_search_result_format(index, dummy_embedder):
    index.add("node-x", dummy_embedder("test"))
    results = index.search(dummy_embedder("test"), k=1)
    assert len(results) == 1
    node_id, dist = results[0]
    assert isinstance(node_id, str)
    assert isinstance(dist, float)


def test_search_top_result_is_most_similar(index, dummy_embedder):
    target_emb = dummy_embedder("systems programming in Go")
    index.add("target",    target_emb)
    index.add("unrelated", dummy_embedder("weather forecast tomorrow"))
    results = index.search(target_emb, k=2)
    assert results[0][0] == "target"


def test_search_empty_index_returns_empty(index, dummy_embedder):
    results = index.search(dummy_embedder("anything"), k=5)
    assert results == []


def test_deleted_count_tracks(index, dummy_embedder):
    index.add("node-1", dummy_embedder("a"))
    index.remove("node-1")
    assert index.deleted_count() == 1


def test_rebuild_from_nodes(dummy_embedder):
    from src.graph.node import Node, NodeType
    nodes = [
        Node.new(NodeType.CONCEPT, f"node content {i}", dummy_embedder(f"node content {i}"))
        for i in range(5)
    ]
    index = HNSWIndex(dim=384, M=16, ef_construction=200,
                      ef_search=50, max_elements=1000)
    index.rebuild(nodes)
    assert index.size() == 5
    for n in nodes:
        assert index.contains(n.id)


# Fix #3 — rebuild must preserve constructor parameters
def test_rebuild_preserves_constructor_params(dummy_embedder):
    from src.graph.node import Node, NodeType
    index = HNSWIndex(dim=384, M=32, ef_construction=400,
                      ef_search=150, max_elements=1000)
    nodes = [
        Node.new(NodeType.CONCEPT, f"content {i}", dummy_embedder(f"content {i}"))
        for i in range(3)
    ]
    index.rebuild(nodes)
    # Internal params must match constructor, not hardcoded defaults
    assert index._M == 32
    assert index._ef_construction == 400
    assert index._ef_search == 150


# Fix #11 — dimension validation
def test_add_wrong_dimension_raises(index):
    bad_emb = np.ones(128, dtype=np.float32)
    with pytest.raises(ValueError, match="Expected embedding of shape"):
        index.add("node-bad", bad_emb)


def test_update_wrong_dimension_raises(index, dummy_embedder):
    index.add("node-1", dummy_embedder("original"))
    bad_emb = np.ones(128, dtype=np.float32)
    with pytest.raises(ValueError, match="Expected embedding of shape"):
        index.update("node-1", bad_emb)