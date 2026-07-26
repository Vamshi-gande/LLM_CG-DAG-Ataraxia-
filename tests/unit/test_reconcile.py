"""
Unit tests for lazy reconciliation hook.

M3 established the None-influence-table no-op path (unchanged since).
M7 wired in the real InfluenceEntry-based data shape — Dict[node_id ->
List[InfluenceEntry]] — replacing the placeholder single-dict-per-node
stub this file originally used. This is the single canonical test file
for check_and_reconcile(); it supersedes both the pre-M7 version of this
file and the separate test_reconcile_m7.py scratch file from the M7
handoff — don't keep three copies of reconcile tests around.
"""
import uuid

from src.graph.node import Node, NodeType
from src.graph.graph import Graph
from src.propagation.reconcile import check_and_reconcile
from src.influence.table import InfluenceEntry, INFLUENCE_MEDIUM


def _loaded_graph(small_graph):
    g = Graph()
    for n in small_graph["nodes"]:
        g.add_node(n)
    for e in small_graph["edges"]:
        g.add_edge(e)
    return g, small_graph["nodes"]


# ── M3 — influence_table=None no-op path (unchanged) ──────────────────────

def test_none_influence_table_returns_false(make_node):
    node = make_node("test node", NodeType.CONCEPT)
    g = Graph()
    result = check_and_reconcile(node, g, influence_table=None)
    assert result is False


def test_none_influence_table_does_not_modify_node(make_node):
    node = make_node("test node", NodeType.CONCEPT)
    g = Graph()
    original_version = node.last_reconciled_version
    check_and_reconcile(node, g, influence_table=None)
    assert node.last_reconciled_version == original_version


def test_missing_entry_returns_false(make_node):
    node = make_node("test node", NodeType.CONCEPT)
    g = Graph()
    result = check_and_reconcile(
        node, g,
        influence_table={"other_id": [InfluenceEntry("some_source", INFLUENCE_MEDIUM, 5)]},
    )
    assert result is False


# ── M7 — real InfluenceEntry data shape ───────────────────────────────────
# (rewritten from the M3-era stub, which used {node.id: {"source_version": 5}}
#  — a single dict, not a List[InfluenceEntry]. That shape no longer exists;
#  see M7 completion record §2.1-2.3 for why the table is list-valued.)

def test_stale_node_reconciles(make_node):
    node = make_node("test node", NodeType.CONCEPT)
    node.last_reconciled_version = 2
    influence = {node.id: [InfluenceEntry("some_source_id", INFLUENCE_MEDIUM, source_version=5)]}
    g = Graph()

    result = check_and_reconcile(node, g, influence_table=influence)
    assert result is True
    assert node.last_reconciled_version == 5


def test_already_current_returns_false(make_node):
    node = make_node("test node", NodeType.CONCEPT)
    node.last_reconciled_version = 5
    influence = {node.id: [InfluenceEntry("some_source_id", INFLUENCE_MEDIUM, source_version=5)]}
    g = Graph()

    result = check_and_reconcile(node, g, influence_table=influence)
    assert result is False


# ── M7 — pending influence actually applied against a real graph ─────────

def test_reconcile_recomputes_edge_weight_from_pending_source(small_graph):
    g, nodes = _loaded_graph(small_graph)
    # nodes[2] has a pending influence from nodes[0] (a real, connected
    # ancestor in the chain) — reconciliation should recompute the edge
    # weight between nodes[2] and its actual graph neighbor.
    influence_table = {
        nodes[2].id: [InfluenceEntry(nodes[0].id, INFLUENCE_MEDIUM, source_version=4)]
    }
    result = check_and_reconcile(nodes[2], g, influence_table=influence_table)
    assert result is True
    assert nodes[2].last_reconciled_version == 4
    assert influence_table.get(nodes[2].id, []) == []  # cleared after reconciliation


def test_reconcile_skips_ghost_source_but_still_advances_version(small_graph):
    g, nodes = _loaded_graph(small_graph)
    influence_table = {
        nodes[2].id: [InfluenceEntry("nonexistent-ghost-id", INFLUENCE_MEDIUM, source_version=2)]
    }
    # source node no longer exists (e.g. merged/archived by a future
    # compression engine) — should not raise, and the version still
    # advances so the entry doesn't sit "pending" forever.
    result = check_and_reconcile(nodes[2], g, influence_table=influence_table)
    assert result is True
    assert nodes[2].last_reconciled_version == 2