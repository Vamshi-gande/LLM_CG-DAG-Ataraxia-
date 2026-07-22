"""
Unit tests for serialization strategies.
No graph, no ONNX. Operates purely on Node objects and DAG dataclass.
"""
import pytest
from src.graph.node import Node, NodeType
from src.preresolve.classify import QueryType
from src.serialize import (
    serialize_chain, serialize_synthesis, serialize_lookup,
    serialize_dag, REASONING_PRIMER,
)
from src.dag import DAG


# ── Helper ────────────────────────────────────────────────────────────────────

def make_dag(query_type, nodes, resolved_pairs=None, synthesis_node=None,
             lookup_nodes=None, activation_scores=None):
    return DAG(
        nodes_ordered=nodes,
        edges=[],
        query_type=query_type,
        resolved_pairs=resolved_pairs or [],
        synthesis_node=synthesis_node,
        lookup_nodes=lookup_nodes or [],
        activation_scores=activation_scores or {n.id: 0.5 for n in nodes},
        token_estimate=sum(len(n.content) for n in nodes) // 4,
    )


# ── serialize_chain() tests ────────────────────────────────────────────────────

def test_chain_output_contains_resolved_context_header(make_node):
    conclusion = make_node("graph compression is needed")
    support    = make_node("4K context limit on consumer GPUs")
    result = serialize_chain([(conclusion, [support])])
    assert "[RESOLVED CONTEXT]" in result

def test_chain_output_contains_conclusion_content(make_node):
    conclusion = make_node("graph compression is needed")
    result = serialize_chain([(conclusion, [])])
    assert "graph compression is needed" in result

def test_chain_output_contains_support_content(make_node):
    conclusion = make_node("conclusion node")
    support    = make_node("the reason for this conclusion")
    result = serialize_chain([(conclusion, [support])])
    assert "the reason for this conclusion" in result

def test_chain_empty_pairs_returns_empty_string():
    result = serialize_chain([])
    assert result == ""

def test_chain_multiple_conclusions(make_node):
    pairs = [
        (make_node(f"conclusion {i}"), [make_node(f"support {i}")])
        for i in range(3)
    ]
    result = serialize_chain(pairs)
    for i in range(3):
        assert f"conclusion {i}" in result


# ── serialize_synthesis() tests ────────────────────────────────────────────────

def test_synthesis_output_contains_relevant_context_header(make_node):
    nodes = [make_node("node content")]
    result = serialize_synthesis(None, nodes)
    assert "[RELEVANT CONTEXT]" in result

def test_synthesis_uses_synthesis_node_content_when_present(make_node, dummy_embedder):
    import numpy as np, uuid, time
    emb = dummy_embedder("synthesis topic")
    synthesis = Node(
        id="temp_" + str(uuid.uuid4()),
        type=NodeType.SUMMARY,
        content="synthesized topic about privacy and performance",
        embedding=emb,
        priority=0.6,
        created_at=time.time(),
        updated_at=time.time(),
    )
    nodes = [make_node("supporting detail")]
    result = serialize_synthesis(synthesis, nodes)
    assert "synthesized topic about privacy and performance" in result

def test_synthesis_falls_back_to_nodes_when_synthesis_none(make_node):
    nodes = [make_node("fallback content")]
    result = serialize_synthesis(None, nodes)
    assert "fallback content" in result

def test_synthesis_empty_nodes_returns_header_only(make_node):
    result = serialize_synthesis(None, [])
    assert "[RELEVANT CONTEXT]" in result


# ── serialize_lookup() tests ───────────────────────────────────────────────────

def test_lookup_output_contains_current_state_header(make_node):
    nodes = [make_node("language: Go")]
    result = serialize_lookup(nodes)
    assert "[CURRENT STATE]" in result

def test_lookup_contains_node_content(make_node):
    node = make_node("implementation language: Go")
    result = serialize_lookup([node])
    assert "implementation language: Go" in result

def test_lookup_empty_returns_empty_string():
    result = serialize_lookup([])
    assert result == ""


# ── serialize_dag() — full pipeline tests ─────────────────────────────────────

def test_serialize_dag_prepends_reasoning_primer(make_node):
    nodes = [make_node("some content")]
    dag = make_dag(QueryType.LOOKUP, nodes, lookup_nodes=nodes)
    result = serialize_dag(dag)
    assert result.startswith(REASONING_PRIMER) or REASONING_PRIMER in result

def test_serialize_dag_routes_chain(make_node):
    conclusion = make_node("conclusion")
    support    = make_node("support")
    dag = make_dag(
        QueryType.CHAIN, [conclusion, support],
        resolved_pairs=[(conclusion, [support])]
    )
    result = serialize_dag(dag)
    assert "[RESOLVED CONTEXT]" in result

def test_serialize_dag_routes_synthesis(make_node):
    nodes = [make_node("content")]
    dag = make_dag(QueryType.SYNTHESIS, nodes)
    result = serialize_dag(dag)
    assert "[RELEVANT CONTEXT]" in result

def test_serialize_dag_routes_lookup(make_node):
    nodes = [make_node("language: Go")]
    dag = make_dag(QueryType.LOOKUP, nodes, lookup_nodes=nodes)
    result = serialize_dag(dag)
    assert "[CURRENT STATE]" in result

def test_serialize_dag_enforces_budget(make_node):
    # Create many nodes with long content to exceed budget
    nodes = [make_node("x " * 200) for _ in range(20)]
    dag = make_dag(QueryType.LOOKUP, nodes, lookup_nodes=nodes)
    result = serialize_dag(dag, chars_per_token=3.5, tier2_budget=50)
    estimated_tokens = len(result) / 3.5
    # Allow 20% overage for header text
    assert estimated_tokens <= 50 * 1.2, \
        f"Budget not enforced: estimated {estimated_tokens:.0f} tokens for budget=50"

def test_serialize_dag_truncation_preserves_primer_prefix(make_node):
    """Budget truncation must cut from the END, never the START — the
    primer must survive even under an extremely tight budget."""
    nodes = [make_node("x " * 500) for _ in range(20)]
    dag = make_dag(QueryType.LOOKUP, nodes, lookup_nodes=nodes)
    result = serialize_dag(dag, chars_per_token=3.5, tier2_budget=5)
    assert result == REASONING_PRIMER[:len(result)]

def test_serialize_dag_chain_with_empty_resolved_pairs(make_node):
    """Chain with no resolved pairs should not crash."""
    nodes = [make_node("content")]
    dag = make_dag(QueryType.CHAIN, nodes, resolved_pairs=[])
    result = serialize_dag(dag)
    assert isinstance(result, str)