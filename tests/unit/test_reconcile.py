"""
Unit tests for lazy reconciliation hook.
"""
import pytest
import uuid
from src.graph.node import Node, NodeType
from src.graph.graph import Graph
from src.propagation.reconcile import check_and_reconcile


@pytest.fixture
def sample_node(dummy_embedder):
    return Node(
        id=str(uuid.uuid4()),
        type=NodeType.CONCEPT,
        content="test node",
        embedding=dummy_embedder("test node"),
        last_reconciled_version=0,
    )


def test_none_influence_table_returns_false(sample_node):
    g = Graph()
    result = check_and_reconcile(sample_node, g, influence_table=None)
    assert result is False


def test_none_influence_table_does_not_modify_node(sample_node):
    g = Graph()
    original_version = sample_node.last_reconciled_version
    check_and_reconcile(sample_node, g, influence_table=None)
    assert sample_node.last_reconciled_version == original_version


def test_missing_entry_returns_false(sample_node):
    g = Graph()
    result = check_and_reconcile(sample_node, g,
                                  influence_table={"other_id": {"source_version": 5}})
    assert result is False


def test_stale_node_reconciles(sample_node):
    g = Graph()
    sample_node.last_reconciled_version = 2
    influence = {sample_node.id: {"source_version": 5}}
    result = check_and_reconcile(sample_node, g, influence_table=influence)
    assert result is True
    assert sample_node.last_reconciled_version == 5


def test_already_current_returns_false(sample_node):
    g = Graph()
    sample_node.last_reconciled_version = 5
    influence = {sample_node.id: {"source_version": 5}}
    result = check_and_reconcile(sample_node, g, influence_table=influence)
    assert result is False