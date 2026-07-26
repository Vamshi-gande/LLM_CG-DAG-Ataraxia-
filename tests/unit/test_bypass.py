"""
Unit tests for bypass mode detection.
"""
import pytest
from src.graph.graph import Graph
from src.proxy.bypass import should_bypass


@pytest.fixture
def empty_graph():
    return Graph()


@pytest.fixture
def populated_graph(small_graph):
    g = Graph()
    for n in small_graph["nodes"]:
        g.add_node(n)
    return g


def test_bypass_when_graph_small_and_turns_low(empty_graph):
    assert should_bypass(empty_graph, turn_count=5) is True


def test_no_bypass_when_graph_large_enough(populated_graph):
    # small_graph has 6 nodes - below min_nodes=20 threshold
    # but we can override min_nodes for testing
    assert should_bypass(populated_graph, turn_count=5, min_nodes=5) is False


def test_no_bypass_when_turns_high_enough(empty_graph):
    # Even with small graph, if turns >= min_turns AND is AND logic
    # should_bypass returns True only when BOTH conditions met
    # Empty graph (0 nodes < 20) + 15 turns (>= 10) -> False (AND logic)
    assert should_bypass(empty_graph, turn_count=15) is False


def test_bypass_requires_both_conditions(populated_graph):
    # 6 nodes < 20 AND 5 turns < 10 -> True
    assert should_bypass(populated_graph, turn_count=5) is True


def test_bypass_false_when_nodes_enough():
    g = Graph()
    # Can't add 20 real nodes easily - use min_nodes override
    assert should_bypass(g, turn_count=5, min_nodes=0) is False


def test_bypass_configurable_thresholds(empty_graph):
    assert should_bypass(
        empty_graph, turn_count=5, min_nodes=100, min_turns=100
    ) is True
    assert should_bypass(
        empty_graph, turn_count=5, min_nodes=0, min_turns=0
    ) is False