"""
Unit tests for pre-resolution engine.
Tests chain resolution, synthesis, and lookup independently.
Uses manually constructed activated dicts — does NOT depend on HNSW
or spreading_activation() so logic is tested in isolation.
"""
import pytest
import uuid
import numpy as np
from src.graph.graph import Graph
from src.graph.node import Node, NodeType
from src.graph.edge import Edge, EdgeType
from src.preresolve import (
    QueryType, classify_and_preresolve, PreResolvedContext,
    resolve_chain, resolve_synthesis, resolve_lookup,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def chain_graph(small_graph, dummy_embedder):
    """
    Load small_graph (6-node DEPENDENCY chain) into a Graph.
    Also returns a manually constructed activated dict with all 6 nodes.
    This tests chain resolution in isolation from HNSW.
    """
    g = Graph()
    nodes = small_graph["nodes"]
    edges = small_graph["edges"]
    for node in nodes:
        g.add_node(node)
    for edge in edges:
        g.add_edge(edge)

    # Manually set activation scores — decreasing along the chain
    # This simulates spreading activation having found all 6 nodes
    activated = {
        nodes[0].id: 0.8,
        nodes[1].id: 0.7,
        nodes[2].id: 0.6,
        nodes[3].id: 0.5,
        nodes[4].id: 0.4,
        nodes[5].id: 0.3,
    }
    return g, activated, nodes


@pytest.fixture
def disconnected_graph(make_node, make_edge, dummy_embedder):
    """
    Two disconnected groups of nodes — no edges between groups.
    Used for synthesis tests (distant node detection).
    """
    g = Graph()
    # Group A: 2 nodes about Go/middleware
    a1 = make_node("Go middleware layer", NodeType.CONCEPT)
    a2 = make_node("consumer GPU constraint", NodeType.CONCEPT)

    # Group B: 2 nodes about privacy/organizations (no edges to Group A)
    b1 = make_node("privacy-conscious organizations", NodeType.ENTITY)
    b2 = make_node("on-premise deployment required", NodeType.CONCEPT)

    for n in [a1, a2, b1, b2]:
        g.add_node(n)

    # Only intra-group edges — no cross-group edges
    g.add_edge(make_edge(a1.id, a2.id, EdgeType.DEPENDENCY, 0.8))
    g.add_edge(make_edge(b1.id, b2.id, EdgeType.SEMANTIC, 0.7))

    activated = {a1.id: 0.8, a2.id: 0.6, b1.id: 0.7, b2.id: 0.5}
    return g, activated, [a1, a2, b1, b2]


@pytest.fixture
def close_graph(make_node, make_edge, dummy_embedder):
    """
    3 nodes, all mutually reachable within max_hop_distance=3 (undirected).
    Used to test the genuine 'no distant pairs' path in resolve_synthesis.
    Edges: n1-n2 (DEPENDENCY), n2-n3 (DEPENDENCY)
    Undirected hop distances: n1-n2=1, n2-n3=1, n1-n3=2 — all <= 3.
    """
    g = Graph()
    n1 = make_node("Go middleware layer", NodeType.CONCEPT)
    n2 = make_node("targets Ollama ecosystem", NodeType.ENTITY)
    n3 = make_node("runs on consumer GPU", NodeType.CONCEPT)

    for n in [n1, n2, n3]:
        g.add_node(n)

    g.add_edge(make_edge(n1.id, n2.id, EdgeType.DEPENDENCY, 0.9))
    g.add_edge(make_edge(n2.id, n3.id, EdgeType.DEPENDENCY, 0.9))

    activated = {n1.id: 0.8, n2.id: 0.7, n3.id: 0.6}
    return g, activated, [n1, n2, n3]


# ── Chain resolution tests ─────────────────────────────────────────────────

def test_chain_finds_terminal_node(chain_graph):
    g, activated, nodes = chain_graph
    pairs = resolve_chain(activated, g)
    assert len(pairs) >= 1
    # Terminal is the last node in the chain (no outgoing dep edges)
    conclusion_ids = [p[0].id for p in pairs]
    assert nodes[-1].id in conclusion_ids

def test_chain_excludes_intermediate_nodes(chain_graph):
    g, activated, nodes = chain_graph
    pairs = resolve_chain(activated, g)
    # Intermediate nodes (nodes[1] through nodes[4]) should NOT be
    # conclusions — they have both incoming and outgoing dep edges
    conclusion_ids = {p[0].id for p in pairs}
    for intermediate in nodes[1:-1]:
        assert intermediate.id not in conclusion_ids, \
            f"Intermediate node {intermediate.content!r} incorrectly " \
            f"included as conclusion"

def test_chain_support_is_1_hop_only(chain_graph):
    g, activated, nodes = chain_graph
    pairs = resolve_chain(activated, g)
    if not pairs:
        pytest.skip("No chain pairs found — check edge types")
    conclusion, support = pairs[0]
    # Support should only be direct predecessors (1 hop), not full chain
    support_ids = {n.id for n in support}
    # The root node (nodes[0]) should NOT be in support of nodes[-1]
    # since it is more than 1 hop away
    assert nodes[0].id not in support_ids

def test_chain_returns_list_of_tuples(chain_graph):
    g, activated, nodes = chain_graph
    pairs = resolve_chain(activated, g)
    assert isinstance(pairs, list)
    for item in pairs:
        assert isinstance(item, tuple)
        assert len(item) == 2
        conclusion, support = item
        assert isinstance(conclusion, Node)
        assert isinstance(support, list)

def test_chain_empty_activated_returns_empty(small_graph):
    g = Graph()
    for n in small_graph["nodes"]:
        g.add_node(n)
    for e in small_graph["edges"]:
        g.add_edge(e)
    result = resolve_chain({}, g)
    assert result == []

def test_chain_no_dep_edges_returns_empty(make_node, make_edge):
    """
    All activated nodes connected by SEMANTIC edges only — no chain edges.
    With Issue 3 fix: terminals require at least one chain-edge predecessor.
    Since neither node has chain-edge predecessors, neither qualifies as a
    terminal. resolve_chain must return empty list.
    """
    g = Graph()
    n1 = make_node("node one")
    n2 = make_node("node two")
    g.add_node(n1)
    g.add_node(n2)
    g.add_edge(make_edge(n1.id, n2.id, EdgeType.SEMANTIC, 0.8))
    activated = {n1.id: 0.8, n2.id: 0.6}
    result = resolve_chain(activated, g)
    assert result == [], \
        "Expected empty list when no DEPENDENCY/CAUSAL edges exist"


# ── Synthesis resolution tests ────────────────────────────────────────────

def test_synthesis_creates_temporary_node(disconnected_graph):
    g, activated, nodes = disconnected_graph
    result = resolve_synthesis(activated, g)
    if result is not None:
        assert isinstance(result, Node)
        assert result.type == NodeType.SUMMARY

def test_synthesis_node_not_in_graph(disconnected_graph):
    g, activated, nodes = disconnected_graph
    node_count_before = g.node_count()
    result = resolve_synthesis(activated, g)
    # Graph must be unchanged — synthesis node is temporary
    assert g.node_count() == node_count_before

def test_synthesis_temp_node_has_combined_content(disconnected_graph):
    g, activated, nodes = disconnected_graph
    result = resolve_synthesis(activated, g)
    if result is None:
        pytest.skip("No distant pairs found")
    # Temporary node content must reference content from both source nodes
    # (exact format depends on implementation — just check it's non-empty)
    assert len(result.content) > 0

def test_synthesis_returns_none_when_all_connected(close_graph):
    """
    All activated nodes are within max_hop_distance (default 3) of each
    other via undirected traversal — find_distant_node_pairs() must find
    no distant pairs, so resolve_synthesis() must return None strictly.

    NOTE: chain_graph (6-node chain) is NOT suitable for this test — at
    max_hop_distance=3, node 0 and node 4/5 exceed the hop limit and DO
    count as distant, so resolve_synthesis() would correctly return a
    Node for that fixture, not None. Use close_graph instead, where every
    pair is provably within the hop limit.
    """
    g, activated, nodes = close_graph
    result = resolve_synthesis(activated, g)
    assert result is None, \
        "Expected None when all activated nodes are mutually reachable " \
        "within max_hop_distance"

def test_synthesis_finds_distant_pair_beyond_hop_limit(chain_graph):
    """
    chain_graph is a 6-node chain. Undirected hop distance from node 0
    to node 4 is 4, and to node 5 is 5 — both exceed max_hop_distance=3.
    resolve_synthesis() should find this pair and return a synthesis Node.
    """
    g, activated, nodes = chain_graph
    result = resolve_synthesis(activated, g)
    assert result is not None, \
        "Expected a synthesis node — root and terminal exceed hop limit"
    assert isinstance(result, Node)
    assert result.type == NodeType.SUMMARY


# ── Lookup resolution tests ───────────────────────────────────────────────

def test_lookup_returns_list_of_nodes(chain_graph):
    g, activated, nodes = chain_graph
    result = resolve_lookup(activated, g, max_nodes=3)
    assert isinstance(result, list)
    assert all(isinstance(n, Node) for n in result)

def test_lookup_respects_max_nodes(chain_graph):
    g, activated, nodes = chain_graph
    result = resolve_lookup(activated, g, max_nodes=2)
    assert len(result) <= 2

def test_lookup_sorted_by_activation_score(chain_graph):
    g, activated, nodes = chain_graph
    result = resolve_lookup(activated, g, max_nodes=6)
    scores = [activated[n.id] for n in result]
    assert scores == sorted(scores, reverse=True), \
        "Lookup results not sorted by activation score descending"

def test_lookup_empty_activated_returns_empty(small_graph):
    g = Graph()
    for n in small_graph["nodes"]:
        g.add_node(n)
    result = resolve_lookup({}, g)
    assert result == []


# ── classify_and_preresolve() integration tests ───────────────────────────

def test_chain_query_produces_chain_context(chain_graph):
    g, activated, nodes = chain_graph
    ctx = classify_and_preresolve(
        "why does this project need graph compression?",
        activated, g
    )
    assert ctx.query_type == QueryType.CHAIN
    assert isinstance(ctx.resolved_pairs, list)
    assert ctx.synthesis_node is None
    assert ctx.activated is activated

def test_lookup_query_produces_lookup_context(chain_graph):
    g, activated, nodes = chain_graph
    ctx = classify_and_preresolve(
        "what language is this project in?",
        activated, g
    )
    assert ctx.query_type == QueryType.LOOKUP
    assert isinstance(ctx.lookup_nodes, list)
    assert ctx.synthesis_node is None
    assert ctx.resolved_pairs == []

def test_context_always_carries_activated_dict(chain_graph):
    g, activated, nodes = chain_graph
    ctx = classify_and_preresolve("what is this?", activated, g)
    assert ctx.activated is activated  # same object, not a copy

def test_preresolve_does_not_modify_graph(chain_graph):
    g, activated, nodes = chain_graph
    count_before = g.node_count()
    edge_count_before = g.edge_count()
    classify_and_preresolve("why does X require Y?", activated, g)
    assert g.node_count() == count_before
    assert g.edge_count() == edge_count_before

def test_5hop_chain_preresolve_validates_serialization_experiment(chain_graph):
    """
    Key validation: the same 5-hop chain that scored 3/5 in the
    serialization experiment must be pre-resolved correctly here,
    without any LLM call.

    Chain: Go→Ollama→GPU→VRAM→4K→compression
    Expected: terminal node (compression) identified as conclusion.
    Intermediate nodes NOT in the resolved output.
    """
    g, activated, nodes = chain_graph
    ctx = classify_and_preresolve(
        "why does this project need graph compression?",
        activated, g
    )
    assert ctx.query_type == QueryType.CHAIN
    conclusion_ids = {pair[0].id for pair in ctx.resolved_pairs}

    # Terminal node (requires graph memory compression) should be conclusion
    assert nodes[-1].id in conclusion_ids, \
        "Terminal chain node not found in pre-resolved conclusions"

    # Root node (Go middleware layer) should NOT be a conclusion
    assert nodes[0].id not in conclusion_ids, \
        "Root node incorrectly identified as conclusion"