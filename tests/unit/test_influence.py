"""
Unit tests for the influence table.
"""
from src.influence.table import (
    InfluenceEntry, add_influence, get_pending_influences,
    clear_reconciled, INFLUENCE_MEDIUM, INFLUENCE_WEAK,
)
from src.graph.graph import Graph


def _loaded_graph(small_graph):
    g = Graph()
    for n in small_graph["nodes"]:
        g.add_node(n)
    for e in small_graph["edges"]:
        g.add_edge(e)
    return g, small_graph["nodes"]


def test_add_influence_populates_hop2_and_hop3_only(small_graph):
    """
    Chain: n0 -> n1 -> n2 -> n3 -> n4 -> n5.
    Updating n0: hop1 = n1 (eager, NEVER in the table), hop2 = n2 (medium),
    hop3 = n3 (weak). n4 is hop4 — beyond hop_limit=3, not tracked.
    """
    g, nodes = _loaded_graph(small_graph)
    influence_table = {}
    add_influence(influence_table, nodes[0].id, version=2, graph=g, hop_limit=3)

    assert nodes[1].id not in influence_table, "hop-1 neighbor must never be written to the influence table"
    assert nodes[2].id in influence_table
    assert nodes[3].id in influence_table
    assert nodes[4].id not in influence_table  # hop 4, beyond hop_limit


def test_add_influence_uses_correct_strength(small_graph):
    g, nodes = _loaded_graph(small_graph)
    influence_table = {}
    add_influence(influence_table, nodes[0].id, version=1, graph=g, hop_limit=3)

    hop2_entries = [e for e in influence_table[nodes[2].id] if e.source_node_id == nodes[0].id]
    assert any(e.strength == INFLUENCE_MEDIUM for e in hop2_entries)

    hop3_entries = [e for e in influence_table[nodes[3].id] if e.source_node_id == nodes[0].id]
    assert any(e.strength == INFLUENCE_WEAK for e in hop3_entries)


def test_add_influence_updates_version(small_graph):
    g, nodes = _loaded_graph(small_graph)
    influence_table = {}
    add_influence(influence_table, nodes[0].id, version=1, graph=g, hop_limit=3)
    add_influence(influence_table, nodes[0].id, version=3, graph=g, hop_limit=3)

    # nodes[2] is hop 2 — a valid, always-tracked target for this chain/hop_limit
    entries = [e for e in influence_table[nodes[2].id] if e.source_node_id == nodes[0].id]
    assert len(entries) == 1  # upsert, not append — never duplicates per source
    assert entries[0].source_version == 3


def test_add_influence_hop_limit_of_one_records_nothing(small_graph):
    """hop_limit < 2 means there is no hop-2+ to reach — table stays empty."""
    g, nodes = _loaded_graph(small_graph)
    influence_table = {}
    add_influence(influence_table, nodes[0].id, version=1, graph=g, hop_limit=1)
    assert influence_table == {}


def test_get_pending_returns_stale_entries(small_graph):
    g, nodes = _loaded_graph(small_graph)
    influence_table = {
        nodes[1].id: [
            InfluenceEntry(source_node_id=nodes[0].id, strength=INFLUENCE_MEDIUM, source_version=5)
        ]
    }
    pending = get_pending_influences(nodes[1].id, 0, influence_table)
    assert len(pending) == 1
    assert pending[0].source_version == 5


def test_get_pending_skips_already_reconciled(small_graph):
    g, nodes = _loaded_graph(small_graph)
    influence_table = {
        nodes[1].id: [
            InfluenceEntry(source_node_id=nodes[0].id, strength=INFLUENCE_MEDIUM, source_version=3)
        ]
    }
    pending = get_pending_influences(nodes[1].id, 3, influence_table)
    assert len(pending) == 0


def test_clear_reconciled_removes_old_entries(small_graph):
    g, nodes = _loaded_graph(small_graph)
    influence_table = {
        nodes[1].id: [
            InfluenceEntry(nodes[0].id, INFLUENCE_MEDIUM, source_version=2),
            InfluenceEntry(nodes[0].id, INFLUENCE_WEAK, source_version=5),
        ]
    }
    clear_reconciled(nodes[1].id, reconciled_version=3, influence_table=influence_table)
    remaining = influence_table.get(nodes[1].id, [])
    versions = [e.source_version for e in remaining]
    assert 2 not in versions
    assert 5 in versions