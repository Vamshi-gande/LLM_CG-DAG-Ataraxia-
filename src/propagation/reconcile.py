"""
Lazy reconciliation check (M3 scaffold, wired to real data in M7).

Before M7: influence_table was always None, and this function was a
documented no-op path — check_and_reconcile() always returned False when
influence_table is None, and that path is unchanged here.

After M7: when a real influence_table dict is supplied, pending
InfluenceEntry records are recomputed into edge weights and the node's
last_reconciled_version is advanced.
"""
from typing import Dict, Optional

import numpy as np

from src.graph.node import Node
from src.graph.graph import Graph
from src.influence.table import get_pending_influences, clear_reconciled, InfluenceEntry


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for L2-normalized vectors (= dot product)."""
    return float(np.dot(a, b))


def check_and_reconcile(
    node: Node,
    graph: Graph,
    influence_table: Optional[Dict[str, "list[InfluenceEntry]"]] = None,
) -> bool:
    """
    Check for and apply pending influences on `node`.

    Returns True if a reconciliation actually happened (edge weights were
    recomputed and last_reconciled_version advanced), False otherwise
    (including the M3 no-op path when influence_table is None, and the
    case where there is simply nothing pending).
    """
    if influence_table is None:
        return False

    pending = get_pending_influences(node.id, node.last_reconciled_version, influence_table)
    if not pending:
        return False

    max_version_seen = node.last_reconciled_version
    direct_neighbors = graph.neighbors(node.id)

    for entry in pending:
        if entry.source_version > max_version_seen:
            max_version_seen = entry.source_version

        source = graph.get_node(entry.source_node_id)
        if source is None:
            # Ghost entry: source node no longer exists (e.g. merged/archived
            # by a compression engine). The version still advances — this
            # entry is considered "handled" — but there's no embedding left
            # to recompute an edge weight against. Same defensive pattern
            # M3 used for ghost HNSW entries.
            continue

        for neighbor, edge in direct_neighbors:
            if neighbor.id == source.id:
                edge.weight = round(_cosine_similarity(node.embedding, source.embedding), 4)

    node.last_reconciled_version = max_version_seen
    clear_reconciled(node.id, node.last_reconciled_version, influence_table)
    return True