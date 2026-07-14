"""
Unit tests for spreading activation.
Uses dummy_embedder and small_graph fixtures — no ONNX, no live Ollama.
"""
import pytest
import uuid
import time
import numpy as np

from src.graph.graph import Graph
from src.graph.node import Node, NodeType
from src.graph.edge import Edge, EdgeType
from src.hnsw.index import HNSWIndex
from src.propagation import spreading_activation, seed_activation, spread
from src.propagation.activation import EDGE_TYPE_DAMPING_MULTIPLIERS


# ── Helpers ────────────────────────────────────────────────────────────────

@pytest.fixture
def loaded_graph(small_graph):
    """Graph + HNSW loaded with small_graph fixture nodes and edges."""
    g = Graph()
    idx = HNSWIndex(dim=384, M=16, ef_construction=200,
                     ef_search=50, max_elements=1000)
    g.set_hnsw(idx)
    for node in small_graph["nodes"]:
        g.add_node(node)
    for edge in small_graph["edges"]:
        g.add_edge(edge)
    return g, idx, small_graph["nodes"]


# ── seed_activation tests ────────────────────────────────────────────────

def test_seed_returns_dict(loaded_graph, dummy_embedder):
    g, idx, nodes = loaded_graph
    result = seed_activation(dummy_embedder("middleware"), g, idx, k=3)
    assert isinstance(result, dict)


def test_seed_returns_node_ids_as_keys(loaded_graph, dummy_embedder):
    g, idx, nodes = loaded_graph
    result = seed_activation(dummy_embedder("query"), g, idx, k=3)
    known_ids = {n.id for n in nodes}
    for key in result:
        assert key in known_ids, f"Unknown node_id in seed result: {key}"


def test_seed_scores_are_positive(loaded_graph, dummy_embedder):
    g, idx, nodes = loaded_graph
    result = seed_activation(dummy_embedder("query"), g, idx, k=3)
    for score in result.values():
        assert score >= 0.0


def test_seed_returns_at_most_k_results(loaded_graph, dummy_embedder):
    g, idx, nodes = loaded_graph
    result = seed_activation(dummy_embedder("query"), g, idx, k=3)
    assert len(result) <= 3


def test_seed_empty_graph_returns_empty(dummy_embedder):
    g = Graph()
    idx = HNSWIndex(dim=384, M=16, ef_construction=200,
                     ef_search=50, max_elements=100)
    g.set_hnsw(idx)
    result = seed_activation(dummy_embedder("query"), g, idx, k=5)
    assert result == {}


def test_seed_high_priority_node_scores_higher(dummy_embedder):
    """Node with higher priority should score higher given same embedding."""
    g = Graph()
    idx = HNSWIndex(dim=384, M=16, ef_construction=200,
                     ef_search=50, max_elements=100)
    g.set_hnsw(idx)

    emb = dummy_embedder("shared content")

    now = time.time()
    lo = Node(id=str(uuid.uuid4()), type=NodeType.CONCEPT,
              content="low priority", embedding=emb.copy(),
              priority=0.1, updated_at=now)
    hi = Node(id=str(uuid.uuid4()), type=NodeType.CONCEPT,
              content="high priority", embedding=emb.copy(),
              priority=0.9, updated_at=now)
    g.add_node(lo)
    g.add_node(hi)

    result = seed_activation(emb, g, idx, k=2)
    assert result[hi.id] > result[lo.id]


def test_seed_activation_excludes_active_dag_ids(loaded_graph, dummy_embedder):
    g, idx, nodes = loaded_graph
    g.active_dag_ids.add(nodes[0].id)
    result = seed_activation(dummy_embedder("middleware"), g, idx, k=10)
    assert nodes[0].id not in result
    g.active_dag_ids.clear()


def test_zero_updated_at_does_not_zero_activation(loaded_graph, dummy_embedder):
    g, idx, nodes = loaded_graph
    for n in nodes:
        n.updated_at = 0.0
    result = seed_activation(dummy_embedder("middleware"), g, idx, k=5)
    assert any(score > 0.0 for score in result.values())


def test_seed_activation_calls_reconcile(loaded_graph, dummy_embedder):
    g, idx, nodes = loaded_graph
    target = nodes[0]
    target.last_reconciled_version = 1
    influence = {target.id: {"source_version": 7}}
    seed_activation(dummy_embedder("middleware"), g, idx, k=5,
                     influence_table=influence)
    assert target.last_reconciled_version == 7


def test_seed_activation_empty_search_results(loaded_graph, dummy_embedder, monkeypatch):
    g, idx, nodes = loaded_graph
    monkeypatch.setattr(idx, "search", lambda *a, **kw: [])
    result = seed_activation(dummy_embedder("middleware"), g, idx, k=5)
    assert result == {}


def test_ghost_hnsw_entry_is_skipped(dummy_embedder, make_node):
    g = Graph()
    idx = HNSWIndex(dim=384, M=16, ef_construction=200, ef_search=50, max_elements=100)
    g.set_hnsw(idx)
    real_node = make_node("real", NodeType.CONCEPT)
    g.add_node(real_node)
    # Insert directly into HNSW without registering in the graph — simulates drift
    idx.add("ghost-id-not-in-graph", dummy_embedder("ghost"))
    result = seed_activation(dummy_embedder("real"), g, idx, k=5)
    assert "ghost-id-not-in-graph" not in result


# ── spread() tests ───────────────────────────────────────────────────────

def test_spread_activates_neighbors(loaded_graph, dummy_embedder):
    g, idx, nodes = loaded_graph
    seeds = {nodes[0].id: 1.0}
    result = spread(seeds, g, damping=0.6, hop_limit=3)
    assert nodes[1].id in result


def test_spread_decays_with_hops(loaded_graph, dummy_embedder):
    g, idx, nodes = loaded_graph
    seeds = {nodes[0].id: 1.0}
    result = spread(seeds, g, damping=0.6, hop_limit=3)
    if nodes[1].id in result and nodes[2].id in result:
        assert result[nodes[1].id] >= result[nodes[2].id]


def test_spread_respects_hop_limit(loaded_graph, dummy_embedder):
    g, idx, nodes = loaded_graph
    seeds = {nodes[0].id: 1.0}
    result_1 = spread(seeds, g, damping=0.6, hop_limit=1)
    result_3 = spread(seeds, g, damping=0.6, hop_limit=3)
    assert len(result_3) >= len(result_1)


def test_spread_skips_active_dag_ids(loaded_graph, dummy_embedder):
    g, idx, nodes = loaded_graph
    g.active_dag_ids.add(nodes[1].id)
    seeds = {nodes[0].id: 1.0}
    result = spread(seeds, g, damping=0.6, hop_limit=3)
    assert nodes[1].id not in result
    g.active_dag_ids.clear()


def test_spread_accumulates_from_multiple_seeds(loaded_graph, dummy_embedder):
    g, idx, nodes = loaded_graph
    seeds = {nodes[0].id: 0.5, nodes[2].id: 0.5}
    result = spread(seeds, g, damping=0.6, hop_limit=2)
    assert nodes[1].id in result or nodes[3].id in result


def test_spread_empty_seeds_returns_empty(loaded_graph):
    g, idx, nodes = loaded_graph
    result = spread({}, g, damping=0.6, hop_limit=3)
    assert result == {}


def test_edge_type_damping_differentiates_propagation(dummy_embedder, make_node, make_edge):
    g = Graph()
    idx = HNSWIndex(dim=384, M=16, ef_construction=200, ef_search=50, max_elements=100)
    g.set_hnsw(idx)

    root = make_node("root", NodeType.CONCEPT)
    dep_target = make_node("dependency target", NodeType.CONCEPT)
    sem_target = make_node("semantic target", NodeType.CONCEPT)
    g.add_node(root)
    g.add_node(dep_target)
    g.add_node(sem_target)
    g.add_edge(make_edge(root.id, dep_target.id, EdgeType.DEPENDENCY, 0.9))
    g.add_edge(make_edge(root.id, sem_target.id, EdgeType.SEMANTIC, 0.9))

    result = spread({root.id: 1.0}, g, damping=0.6, hop_limit=1)
    assert result[dep_target.id] > result[sem_target.id]
    expected_dep = 1.0 * 0.9 * 0.6 * EDGE_TYPE_DAMPING_MULTIPLIERS[EdgeType.DEPENDENCY]
    expected_sem = 1.0 * 0.9 * 0.6 * EDGE_TYPE_DAMPING_MULTIPLIERS[EdgeType.SEMANTIC]
    assert result[dep_target.id] == pytest.approx(expected_dep)
    assert result[sem_target.id] == pytest.approx(expected_sem)


def test_contradicts_edge_heavily_suppressed(dummy_embedder, make_node, make_edge):
    g = Graph()
    idx = HNSWIndex(dim=384, M=16, ef_construction=200, ef_search=50, max_elements=100)
    g.set_hnsw(idx)
    a = make_node("a", NodeType.CONCEPT)
    b = make_node("b", NodeType.CONCEPT)
    g.add_node(a)
    g.add_node(b)
    g.add_edge(make_edge(a.id, b.id, EdgeType.CONTRADICTS, 0.9))
    result = spread({a.id: 1.0}, g, damping=0.6, hop_limit=1)
    assert result[b.id] < 1.0 * 0.9 * 0.6 * 0.5


def test_cycle_does_not_blow_up_or_hang(dummy_embedder, make_node, make_edge):
    g = Graph()
    idx = HNSWIndex(dim=384, M=16, ef_construction=200, ef_search=50, max_elements=100)
    g.set_hnsw(idx)
    a = make_node("a", NodeType.CONCEPT)
    b = make_node("b", NodeType.CONCEPT)
    c = make_node("c", NodeType.CONCEPT)
    g.add_node(a)
    g.add_node(b)
    g.add_node(c)
    g.add_edge(make_edge(a.id, b.id, EdgeType.CAUSAL, 1.0))
    g.add_edge(make_edge(b.id, c.id, EdgeType.CAUSAL, 1.0))
    g.add_edge(make_edge(c.id, a.id, EdgeType.CAUSAL, 1.0))  # cycle back to a

    result = spread({a.id: 1.0}, g, damping=0.6, hop_limit=10)
    assert result[a.id] < 5.0
    assert all(score < 5.0 for score in result.values())


def test_incremental_frontier_only_propagates_new_delta(dummy_embedder, make_node, make_edge):
    """A node reached twice (once directly, once via a longer path) should
    accumulate both contributions in its own score, but each hop must only
    forward the delta gained that hop, not the running total."""
    g = Graph()
    idx = HNSWIndex(dim=384, M=16, ef_construction=200, ef_search=50, max_elements=100)
    g.set_hnsw(idx)
    a = make_node("a", NodeType.CONCEPT)
    b = make_node("b", NodeType.CONCEPT)
    c = make_node("c", NodeType.CONCEPT)  # reached from both a and b
    g.add_node(a)
    g.add_node(b)
    g.add_node(c)
    g.add_edge(make_edge(a.id, b.id, EdgeType.CAUSAL, 1.0))
    g.add_edge(make_edge(a.id, c.id, EdgeType.CAUSAL, 1.0))
    g.add_edge(make_edge(b.id, c.id, EdgeType.CAUSAL, 1.0))

    result = spread({a.id: 1.0}, g, damping=0.6, hop_limit=2)
    expected_c = (1.0 * 0.6) + (1.0 * 0.6 * 0.6)
    assert result[c.id] == pytest.approx(expected_c)


# ── spreading_activation() full pipeline tests ───────────────────────────

def test_full_activation_returns_dict(loaded_graph, dummy_embedder):
    g, idx, nodes = loaded_graph
    result = spreading_activation(dummy_embedder("query"), g, idx)
    assert isinstance(result, dict)


def test_full_activation_filters_below_threshold(loaded_graph, dummy_embedder):
    g, idx, nodes = loaded_graph
    result = spreading_activation(
        dummy_embedder("query"), g, idx,
        activation_threshold=0.05
    )
    for score in result.values():
        assert score >= 0.05


def test_full_activation_calls_touch_on_activated_nodes(loaded_graph, dummy_embedder):
    g, idx, nodes = loaded_graph
    for n in nodes:
        n.access_count = 0
    result = spreading_activation(dummy_embedder("middleware"), g, idx)
    activated_ids = set(result.keys())
    for node in nodes:
        if node.id in activated_ids:
            assert node.access_count == 1, \
                f"touch() should increment exactly once, got {node.access_count}"


def test_full_activation_does_not_touch_inactive_nodes(loaded_graph, dummy_embedder):
    g, idx, nodes = loaded_graph
    for n in nodes:
        n.access_count = 0
    result = spreading_activation(dummy_embedder("middleware"), g, idx,
                                   activation_threshold=0.05)
    activated_ids = set(result.keys())
    for node in nodes:
        if node.id not in activated_ids:
            assert node.access_count == 0, \
                f"touch() incorrectly called on inactive node {node.id}"


def test_priority_decay_applied_to_inactive_nodes(loaded_graph, dummy_embedder):
    g, idx, nodes = loaded_graph
    for n in nodes:
        n.priority = 0.5
    result = spreading_activation(
        dummy_embedder("middleware"), g, idx,
        activation_threshold=0.05,
        priority_decay=0.999,
        activation_boost=0.05,
    )
    activated_ids = set(result.keys())
    for node in nodes:
        if node.id not in activated_ids:
            assert node.priority < 0.5, \
                f"Priority decay not applied to inactive node {node.id}"


def test_priority_boost_applied_to_activated_nodes(loaded_graph, dummy_embedder):
    g, idx, nodes = loaded_graph
    for n in nodes:
        n.priority = 0.5
    result = spreading_activation(
        dummy_embedder("middleware"), g, idx,
        activation_threshold=0.05,
        priority_decay=0.999,
        activation_boost=0.05,
    )
    activated_ids = set(result.keys())
    for node in nodes:
        if node.id in activated_ids:
            assert node.priority > 0.5, \
                f"Priority boost not applied to activated node {node.id}"


def test_priority_does_not_exceed_1(loaded_graph, dummy_embedder):
    g, idx, nodes = loaded_graph
    for n in nodes:
        n.priority = 0.99
    spreading_activation(dummy_embedder("middleware"), g, idx)
    for node in nodes:
        assert node.priority <= 1.0


def test_active_dag_ids_skipped_in_full_pipeline(loaded_graph, dummy_embedder):
    g, idx, nodes = loaded_graph
    g.active_dag_ids.add(nodes[0].id)
    result = spreading_activation(dummy_embedder("middleware"), g, idx)
    assert nodes[0].id not in result
    g.active_dag_ids.clear()


def test_full_activation_on_empty_graph(dummy_embedder):
    g = Graph()
    idx = HNSWIndex(dim=384, M=16, ef_construction=200,
                     ef_search=50, max_elements=100)
    g.set_hnsw(idx)
    result = spreading_activation(dummy_embedder("anything"), g, idx)
    assert result == {}


def test_touch_increments_exactly_once(loaded_graph, dummy_embedder):
    g, idx, nodes = loaded_graph
    for n in nodes:
        n.access_count = 0
    result = spreading_activation(dummy_embedder("middleware"), g, idx)
    for node_id in result:
        node = g.get_node(node_id)
        assert node.access_count == 1


def test_priority_decay_on_guaranteed_unreachable_node(dummy_embedder, make_node):
    g = Graph()
    idx = HNSWIndex(dim=384, M=16, ef_construction=200, ef_search=50, max_elements=100)
    g.set_hnsw(idx)
    connected = make_node("connected topic", NodeType.CONCEPT)
    isolated = make_node("completely unrelated isolated node", NodeType.CONCEPT)
    connected.priority = 0.5
    isolated.priority = 0.5
    g.add_node(connected)
    g.add_node(isolated)  # no edges — cannot be reached by spread from any seed

    result = spreading_activation(connected.embedding, g, idx, seed_k=1,
                                   activation_threshold=0.9)
    assert isolated.id not in result
    assert isolated.priority < 0.5


def test_final_filter_strips_active_dag_ids_added_after_spread(loaded_graph, dummy_embedder, monkeypatch):
    """Simulate a node that only becomes active_dag_ids AFTER spread() has
    already returned it, to prove spreading_activation()'s own final filter
    (not just spread()'s internal filter) is what enforces the invariant."""
    g, idx, nodes = loaded_graph
    import src.propagation.activation as activation_module
    original_spread = activation_module.spread

    def spread_then_mark_active(seeds, graph, damping=0.6, hop_limit=3):
        result = original_spread(seeds, graph, damping=damping, hop_limit=hop_limit)
        if nodes[1].id in result:
            graph.active_dag_ids.add(nodes[1].id)
        return result

    monkeypatch.setattr(activation_module, "spread", spread_then_mark_active)

    result = spreading_activation(dummy_embedder("middleware"), g, idx)
    assert nodes[1].id not in result
    g.active_dag_ids.clear()