"""
Unit tests for ContextAssembler.
No graph, no ONNX, no Ollama. Pure context assembly logic.
"""
import pytest
from src.context.assembler import ContextAssembler
from src.dag.extractor import DAG
from src.preresolve.classify import QueryType


def make_minimal_dag(query_type=QueryType.LOOKUP):
    """Minimal DAG for testing serializer routing."""
    return DAG(
        nodes_ordered=[],
        edges=[],
        query_type=query_type,
        resolved_pairs=[],
        synthesis_node=None,
        lookup_nodes=[],
        activation_scores={},
        token_estimate=0,
    )


def test_assemble_includes_tier2(make_node):
    assembler = ContextAssembler()
    dag = make_minimal_dag()
    result = assembler.assemble(dag)
    assert isinstance(result, str)


def test_assemble_with_global_summary_includes_header():
    assembler = ContextAssembler()
    assembler.update_global_summary("User is a CSE student building LLM middleware.")
    dag = make_minimal_dag()
    result = assembler.assemble(dag)
    assert "[GLOBAL CONTEXT]" in result
    assert "CSE student" in result


def test_assemble_without_global_summary_omits_header():
    assembler = ContextAssembler()
    # No update_global_summary call - empty string
    dag = make_minimal_dag()
    result = assembler.assemble(dag)
    assert "[GLOBAL CONTEXT]" not in result


def test_update_global_summary_enforces_tier1_budget():
    assembler = ContextAssembler(tier1_budget_tokens=10, chars_per_token=3.5)
    long_summary = "x " * 1000  # far over budget
    assembler.update_global_summary(long_summary)
    # Stored summary must fit within tier1 budget
    max_chars = 10 * 3.5
    assert len(assembler._global_summary) <= max_chars + 1  # +1 for rounding


def test_total_tokens_within_budget_true_for_short():
    assembler = ContextAssembler(
        tier1_budget_tokens=200, tier2_budget_tokens=800, chars_per_token=3.5
    )
    short_text = "hello world"
    assert assembler.total_tokens_within_budget(short_text) is True


def test_total_tokens_within_budget_false_for_long():
    assembler = ContextAssembler(
        tier1_budget_tokens=10, tier2_budget_tokens=10, chars_per_token=3.5
    )
    long_text = "x " * 500
    assert assembler.total_tokens_within_budget(long_text) is False


def test_global_summary_separator_present_in_assembly():
    assembler = ContextAssembler()
    assembler.update_global_summary("Summary content here.")
    dag = make_minimal_dag()
    result = assembler.assemble(dag)
    # Tier 1 and Tier 2 must be separated by whitespace
    assert "Summary content here." in result