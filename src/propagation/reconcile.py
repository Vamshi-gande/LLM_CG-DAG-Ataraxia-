from typing import Optional, Dict, Any
from src.graph.node import Node
from src.graph.graph import Graph


def check_and_reconcile(
    node: Node,
    graph: Graph,
    influence_table: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Check if node has pending influences and reconcile if stale.

    In M3: influence_table is always None (not yet implemented). Returns
    False immediately — nothing to reconcile.

    In M7: influence_table will be a Dict[node_id -> InfluenceEntry]. If an
    entry exists for node.id with source_version > node.last_reconciled_version,
    edge weights are recomputed from influential nodes and
    last_reconciled_version is bumped.

    Returns True if reconciliation occurred, False otherwise.
    """
    if influence_table is None:
        return False
    entry = influence_table.get(node.id)
    if entry is None:
        return False
    if entry.get("source_version", 0) > node.last_reconciled_version:
        node.last_reconciled_version = entry["source_version"]
        return True
    return False