"""
Unit tests for DAG extraction.
Uses conftest fixtures — no ONNX, no live Ollama.
Constructs PreResolvedContext manually rather than running full pipeline.
"""
import pytest
from src.graph.graph import Graph
from src.graph.node import Node, NodeType
from src.graph.edge import Edge, EdgeType
from src.preresolve.classify import QueryType
from src.preresolve.preresolve import PreResolvedContext
from src.dag import extract_dag, build_subgraph, detect_cycles, \
                    topological_sort, trim_to_budget, DAG


# ── Helpers ──────────────────────────────────────────────────────────────────

@pytest.fixture
def loaded_graph(small_graph):
    """Graph loaded with small_graph 6-node DEPENDENCY chain."""
    g = Graph()
    for n in small_graph["nodes"]:
        g.add_node(n)
    for e in small_graph["edges"]:
        g.add_edge(e)
    return g, small_graph["nodes"], small_graph["edges"]


def make_chain_context(nodes, edges, scores=None):
    """Build a PreResolvedContext for a chain query from a node list."""
    if scores is None:
        scores = {n.id: 0.9 - i * 0.1 for i, n in enumerate(nodes)}
    # Simulate chain pre-resolution: last node is the conclusion
    conclusion = nodes[-1]
    support = [nodes[-2]] if len(nodes) >= 2 else []
    return PreResolvedContext(
        query_type=QueryType.CHAIN,
        activated=scores,
        resolved_pairs=[(conclusion, support)],
        synthesis_node=None,
        lookup_nodes=[],
    )


# ── build_subgraph() tests ────────────────────────────────────────────────────

def test_build_subgraph_returns_nodes_and_edges(loaded_graph):
    g, nodes, edges = loaded_graph
    activated = {n.id: 0.9 - i * 0.1 for i, n in enumerate(nodes)}
    ctx = make_chain_context(nodes, edges, activated)
    result_nodes, result_edges = build_subgraph(ctx, g, max_candidates=10)
    assert len(result_nodes) > 0
    assert isinstance(result_nodes, list)
    assert isinstance(result_edges, list)

def test_build_subgraph_respects_max_candidates(loaded_graph):
    g, nodes, edges = loaded_graph
    activated = {n.id: 0.9 - i * 0.1 for i, n in enumerate(nodes)}
    ctx = make_chain_context(nodes, edges, activated)
    result_nodes, _ = build_subgraph(ctx, g, max_candidates=3)
    assert len(result_nodes) <= 3

def test_build_subgraph_selects_highest_activation_candidates(loaded_graph):
    """
    Regression test for the M5 'already sorted, just slice' bug: the
    activated dict is NOT guaranteed to be insertion-ordered by score.
    build_subgraph must explicitly rank by score, not rely on dict order.
    """
    g, nodes, edges = loaded_graph
    # Deliberately insert into the dict in an order that does NOT match
    # descending score, to prove build_subgraph doesn't just slice as-is.
    activated = {
        nodes[0].id: 0.1,   # low score, inserted first
        nodes[5].id: 0.95,  # highest score, inserted second
        nodes[1].id: 0.2,
        nodes[4].id: 0.9,
        nodes[2].id: 0.3,
        nodes[3].id: 0.5,
    }
    ctx = PreResolvedContext(
        query_type=QueryType.LOOKUP,
        activated=activated,
        resolved_pairs=[],
        synthesis_node=None,
        lookup_nodes=[],
    )
    result_nodes, _ = build_subgraph(ctx, g, max_candidates=2)
    result_ids = {n.id for n in result_nodes}
    # Top 2 by score are nodes[5] (0.95) and nodes[4] (0.9), regardless
    # of insertion order in the activated dict.
    assert result_ids == {nodes[5].id, nodes[4].id}

def test_build_subgraph_only_includes_edges_between_candidates(loaded_graph):
    g, nodes, edges = loaded_graph
    # Only include first 2 nodes
    activated = {nodes[0].id: 0.9, nodes[1].id: 0.8}
    ctx = PreResolvedContext(
        query_type=QueryType.LOOKUP,
        activated=activated,
        resolved_pairs=[],
        synthesis_node=None,
        lookup_nodes=[nodes[0], nodes[1]],
    )
    result_nodes, result_edges = build_subgraph(ctx, g, max_candidates=2)
    # All returned edges must connect nodes within the candidate set
    candidate_ids = {n.id for n in result_nodes}
    for edge in result_edges:
        assert edge.from_node in candidate_ids
        assert edge.to_node in candidate_ids

def test_build_subgraph_skips_ghost_node_ids(loaded_graph):
    """Activated entries with no corresponding graph node are skipped."""
    g, nodes, edges = loaded_graph
    activated = {
        nodes[0].id: 0.9,
        "ghost-uuid-does-not-exist": 0.8,
        nodes[1].id: 0.7,
    }
    ctx = PreResolvedContext(
        query_type=QueryType.LOOKUP,
        activated=activated,
        resolved_pairs=[],
        synthesis_node=None,
        lookup_nodes=[],
    )
    result_nodes, _ = build_subgraph(ctx, g, max_candidates=10)
    result_ids = {n.id for n in result_nodes}
    assert "ghost-uuid-does-not-exist" not in result_ids


# ── detect_cycles() tests ─────────────────────────────────────────────────────

def test_detect_cycles_returns_empty_for_dag(loaded_graph):
    """A linear chain has no cycles."""
    g, nodes, edges = loaded_graph
    to_remove = detect_cycles(nodes, edges)
    assert to_remove == []

def test_detect_cycles_finds_cycle(make_node, make_edge):
    """A → B → C → A should produce one edge to remove."""
    g = Graph()
    n1 = make_node("A")
    n2 = make_node("B")
    n3 = make_node("C")
    for n in [n1, n2, n3]:
        g.add_node(n)
    e1 = make_edge(n1.id, n2.id, EdgeType.DEPENDENCY, 0.9)
    e2 = make_edge(n2.id, n3.id, EdgeType.DEPENDENCY, 0.8)
    e3 = make_edge(n3.id, n1.id, EdgeType.DEPENDENCY, 0.5)  # back-edge, lowest weight
    edges = [e1, e2, e3]
    to_remove = detect_cycles([n1, n2, n3], edges)
    assert len(to_remove) >= 1
    # Lowest-weight back-edge (e3, weight=0.5) should be removed
    remove_ids = {e.id for e in to_remove}
    assert e3.id in remove_ids

def test_detect_cycles_keeps_highest_weight_edge(make_node, make_edge):
    n1 = make_node("A")
    n2 = make_node("B")
    e_high = make_edge(n1.id, n2.id, EdgeType.DEPENDENCY, 0.9)
    e_low  = make_edge(n2.id, n1.id, EdgeType.DEPENDENCY, 0.3)
    to_remove = detect_cycles([n1, n2], [e_high, e_low])
    remove_ids = {e.id for e in to_remove}
    assert e_high.id not in remove_ids
    assert e_low.id in remove_ids


# ── topological_sort() tests ──────────────────────────────────────────────────

def test_topological_sort_root_before_leaf(loaded_graph):
    g, nodes, edges = loaded_graph
    activation_scores = {n.id: 0.9 - i * 0.1 for i, n in enumerate(nodes)}
    sorted_nodes = topological_sort(nodes, edges, activation_scores)
    # nodes[0] must appear before nodes[-1] in sorted order
    positions = {n.id: i for i, n in enumerate(sorted_nodes)}
    assert positions[nodes[0].id] < positions[nodes[-1].id]

def test_topological_sort_respects_dependency_order(loaded_graph):
    g, nodes, edges = loaded_graph
    activation_scores = {n.id: 0.5 for n in nodes}
    sorted_nodes = topological_sort(nodes, edges, activation_scores)
    positions = {n.id: i for i, n in enumerate(sorted_nodes)}
    # Every node must come after all its predecessors
    for edge in edges:
        assert positions[edge.from_node] < positions[edge.to_node], \
            f"Edge {edge.from_node}→{edge.to_node} violated topological order"

def test_topological_sort_handles_disconnected_nodes(make_node, make_edge):
    """Disconnected nodes appended at end sorted by score."""
    n1 = make_node("connected A")
    n2 = make_node("connected B")
    n3 = make_node("isolated high score")
    n4 = make_node("isolated low score")
    e1 = make_edge(n1.id, n2.id, EdgeType.DEPENDENCY, 0.9)
    activation_scores = {n1.id: 0.8, n2.id: 0.7, n3.id: 0.9, n4.id: 0.1}
    sorted_nodes = topological_sort([n1, n2, n3, n4], [e1], activation_scores)
    assert len(sorted_nodes) == 4
    # n1 must come before n2 (dependency)
    positions = {n.id: i for i, n in enumerate(sorted_nodes)}
    assert positions[n1.id] < positions[n2.id]
    # n3 (score 0.9) must come before n4 (score 0.1) among disconnected
    assert positions[n3.id] < positions[n4.id]


# ── trim_to_budget() tests ────────────────────────────────────────────────────

def test_trim_removes_lowest_activation_leaf(make_node, make_edge):
    n1 = make_node("root with long content padding " * 5)
    n2 = make_node("leaf high score " * 5)
    n3 = make_node("leaf low score " * 5)
    e1 = make_edge(n1.id, n2.id, EdgeType.DEPENDENCY, 0.9)
    e2 = make_edge(n1.id, n3.id, EdgeType.DEPENDENCY, 0.8)
    activation_scores = {n1.id: 0.9, n2.id: 0.7, n3.id: 0.2}
    # Set very small budget to force trimming
    trimmed_nodes, trimmed_edges = trim_to_budget(
        [n1, n2, n3], [e1, e2], activation_scores,
        token_budget=10, chars_per_token=3.5
    )
    trimmed_ids = {n.id for n in trimmed_nodes}
    # n3 (lowest activation leaf) should be removed first
    assert n3.id not in trimmed_ids

def test_trim_preserves_non_leaf_nodes(make_node, make_edge):
    """Root node is not a leaf — must never be removed."""
    n1 = make_node("root " * 20)
    n2 = make_node("leaf " * 20)
    e1 = make_edge(n1.id, n2.id, EdgeType.DEPENDENCY, 0.9)
    activation_scores = {n1.id: 0.9, n2.id: 0.1}
    trimmed_nodes, _ = trim_to_budget(
        [n1, n2], [e1], activation_scores,
        token_budget=1, chars_per_token=3.5
    )
    trimmed_ids = {n.id for n in trimmed_nodes}
    # n1 is not a leaf (has outgoing edge) — must remain even at budget=1
    assert n1.id in trimmed_ids

def test_trim_within_budget_unchanged(loaded_graph):
    g, nodes, edges = loaded_graph
    activation_scores = {n.id: 0.5 for n in nodes}
    trimmed_nodes, trimmed_edges = trim_to_budget(
        nodes, edges, activation_scores,
        token_budget=10000, chars_per_token=3.5
    )
    assert len(trimmed_nodes) == len(nodes)


# ── extract_dag() — full pipeline tests ──────────────────────────────────────

def test_extract_dag_returns_dag(loaded_graph):
    g, nodes, edges = loaded_graph
    ctx = make_chain_context(nodes, edges)
    dag = extract_dag(ctx, g)
    assert isinstance(dag, DAG)

def test_extract_dag_clears_active_dag_ids(loaded_graph):
    g, nodes, edges = loaded_graph
    ctx = make_chain_context(nodes, edges)
    extract_dag(ctx, g)
    # After extraction, active_dag_ids must be empty
    assert len(g.active_dag_ids) == 0

def test_extract_dag_clears_active_dag_ids_on_exception(loaded_graph, monkeypatch):
    """active_dag_ids must be cleared even if extraction raises."""
    g, nodes, edges = loaded_graph
    ctx = make_chain_context(nodes, edges)

    import src.dag.extractor as extractor_module

    def boom(*args, **kwargs):
        raise RuntimeError("simulated failure mid-extraction")

    monkeypatch.setattr(extractor_module, "detect_cycles", boom)

    with pytest.raises(RuntimeError):
        extract_dag(ctx, g)

    assert len(g.active_dag_ids) == 0

def test_extract_dag_topological_order(loaded_graph):
    g, nodes, edges = loaded_graph
    ctx = make_chain_context(nodes, edges)
    dag = extract_dag(ctx, g)
    positions = {n.id: i for i, n in enumerate(dag.nodes_ordered)}
    # Root (nodes[0]) must appear before terminal (nodes[-1])
    if nodes[0].id in positions and nodes[-1].id in positions:
        assert positions[nodes[0].id] < positions[nodes[-1].id]

def test_extract_dag_carries_context_fields(loaded_graph):
    g, nodes, edges = loaded_graph
    ctx = make_chain_context(nodes, edges)
    dag = extract_dag(ctx, g)
    assert dag.query_type == QueryType.CHAIN
    assert dag.resolved_pairs == ctx.resolved_pairs
    assert dag.synthesis_node is None