import time
import math
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from src.graph.graph import Graph
from src.graph.node import Node
from src.graph.edge import EdgeType
from src.hnsw.index import HNSWIndex
from src.propagation.reconcile import check_and_reconcile


# Per-edge-type damping multipliers, applied on top of the base damping
# factor and the edge's own weight:
#   delta = gained * edge.weight * damping * EDGE_TYPE_DAMPING_MULTIPLIERS[edge.type]
#
# Causal/Dependency edges carry chain-reasoning relevance at full strength
# — these are exactly what the M4 pre-resolution engine walks, so they
# should not be artificially weakened here. Hierarchical and Temporal
# edges propagate slightly below full strength. Semantic edges are loose
# association and propagate more weakly. Contradicts edges are heavily
# suppressed — a conflicting node should not freely spread positive
# activation to its neighbors. Reinforces edges propagate at 0.9 since
# M7's eager local update already boosts the target's priority directly
# on write; propagation is a secondary signal for them.
EDGE_TYPE_DAMPING_MULTIPLIERS: Dict[EdgeType, float] = {
    EdgeType.CAUSAL:       1.0,
    EdgeType.DEPENDENCY:   1.0,
    EdgeType.HIERARCHICAL: 0.85,
    EdgeType.TEMPORAL:     0.8,
    EdgeType.REINFORCES:   0.9,
    EdgeType.SEMANTIC:     0.6,
    EdgeType.CONTRADICTS:  0.2,
}


def _recency_decay(updated_at: float, freshness_lambda: float,
                    now: Optional[float] = None) -> float:
    """
    exp(-lambda * days_since_update).

    Treats updated_at <= 0 (unset / fixture default of 0.0) as "just
    updated" -> recency_decay = 1.0, rather than letting it decay to ~0
    from a 1970 epoch timestamp (M3 review finding #2). This only affects
    nodes that have literally never been written; any real node has
    updated_at set via Node.bump_version() / Node.new().
    """
    if updated_at <= 0:
        return 1.0
    now = now if now is not None else time.time()
    days_since_update = max(0.0, (now - updated_at) / 86400.0)
    return math.exp(-freshness_lambda * days_since_update)


def seed_activation(
    query_embedding: np.ndarray,
    graph: Graph,
    hnsw: HNSWIndex,
    k: int = 10,
    freshness_lambda: float = 0.1,
    influence_table: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    ANN search -> initial activation scores for seed nodes.

        cosine_similarity = 1.0 - hnsw_distance   (clipped to >= 0.0)
        recency_decay     = exp(-lambda * days_since_update), 1.0 if updated_at <= 0
        activation[node]  = cosine_similarity * node.priority * recency_decay

    - Guards graph.node_count() == 0 and empty hnsw.search() results before
      doing anything else (finding #4).
    - Nodes in graph.active_dag_ids are excluded from seeding entirely —
      they must never enter the activation map, not even at hop 0
      (finding #1).
    - Ghost HNSW entries (node_id present in the index but not in
      graph._nodes) are silently skipped (finding #9).
    - influence_table is threaded through and check_and_reconcile() is
      called for every seed candidate before scoring (finding #3). In M3
      this is always None (no-op passthrough). M7 wires in the real table
      with zero changes to this function.
    """
    if graph.node_count() == 0:
        return {}

    results = hnsw.search(query_embedding, k=k)
    if not results:
        return {}

    activation: Dict[str, float] = {}
    for node_id, distance in results:
        if node_id in graph.active_dag_ids:
            continue

        node = graph.get_node(node_id)
        if node is None:
            # ghost HNSW entry — index and graph have drifted, skip safely
            continue

        check_and_reconcile(node, graph, influence_table)

        cosine_similarity = max(0.0, 1.0 - distance)
        recency = _recency_decay(node.updated_at, freshness_lambda)
        score = cosine_similarity * node.priority * recency
        activation[node_id] = score

    return activation


def spread(
    seeds: Dict[str, float],
    graph: Graph,
    damping: float = 0.6,
    hop_limit: int = 3,
) -> Dict[str, float]:
    """
    Incremental BFS spreading activation from seed nodes.

    Frontier semantics (finding #6, resolved): each hop propagates only
    the activation *newly gained* in the previous hop, not the node's full
    accumulated total. A node reached via two paths, or revisited through
    a cycle, only re-propagates the incremental delta it just received —
    never its whole running score. Combined with damping < 1.0 per hop,
    this guarantees the propagated delta strictly shrinks hop over hop
    even around cycles, so the pass always terminates within hop_limit
    iterations regardless of graph topology (finding #7, resolved).

    Per-edge-type damping multipliers (EDGE_TYPE_DAMPING_MULTIPLIERS) are
    applied on top of the base damping factor and edge.weight.

    - Nodes in graph.active_dag_ids are skipped entirely: they neither
      receive activation nor propagate it onward (finding #1).
    - Does NOT call node.touch() — that happens in spreading_activation()
      after threshold filtering.
    """
    if not seeds:
        return {}

    activation: Dict[str, float] = dict(seeds)
    frontier: Dict[str, float] = {
        node_id: score for node_id, score in seeds.items()
        if node_id not in graph.active_dag_ids
    }

    for _hop in range(hop_limit):
        if not frontier:
            break
        next_frontier: Dict[str, float] = {}

        for node_id, gained in frontier.items():
            if node_id in graph.active_dag_ids:
                continue

            for neighbor, edge in graph.neighbors(node_id):
                if neighbor.id in graph.active_dag_ids:
                    continue

                multiplier = EDGE_TYPE_DAMPING_MULTIPLIERS.get(edge.type, 1.0)
                delta = gained * edge.weight * damping * multiplier
                if delta <= 0.0:
                    continue

                activation[neighbor.id] = activation.get(neighbor.id, 0.0) + delta
                next_frontier[neighbor.id] = next_frontier.get(neighbor.id, 0.0) + delta

        frontier = next_frontier

    # Seeds excluded from the frontier (active_dag_ids) were still copied
    # into `activation` via dict(seeds) above — strip them back out.
    for node_id in list(activation.keys()):
        if node_id in graph.active_dag_ids:
            del activation[node_id]

    return activation


def spreading_activation(
    query_embedding: np.ndarray,
    graph: Graph,
    hnsw: HNSWIndex,
    seed_k: int = 10,
    damping: float = 0.6,
    hop_limit: int = 3,
    activation_threshold: float = 0.05,
    freshness_lambda: float = 0.1,
    priority_decay: float = 0.999,
    activation_boost: float = 0.05,
    influence_table: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Full spreading activation pipeline. Main entry point for M4+.

    1. seed_activation()  - active_dag_ids excluded at seed time
    2. spread()           - incremental propagation, active_dag_ids excluded throughout
    3. threshold filter
    4. final active_dag_ids safety filter (finding #1, defense in depth):
       even though seed_activation() and spread() already exclude these
       nodes, this final pass guarantees the invariant holds regardless of
       how the two are combined or extended by future milestones.
    5. node.touch() on every surviving node — increments access_count by
       exactly 1 per call (finding #10).
    6. Priority decay / boost (Engine 1), inline, O(N) over all graph nodes.
    7. Return final activation map.
    """
    seeds = seed_activation(
        query_embedding, graph, hnsw,
        k=seed_k, freshness_lambda=freshness_lambda,
        influence_table=influence_table,
    )

    spread_result = spread(seeds, graph, damping=damping, hop_limit=hop_limit)

    activated = {
        node_id: score for node_id, score in spread_result.items()
        if score >= activation_threshold
    }

    # Defense-in-depth: strip any active_dag_ids node that slipped through.
    activated = {
        node_id: score for node_id, score in activated.items()
        if node_id not in graph.active_dag_ids
    }

    activated_ids = set(activated.keys())

    for node_id in activated_ids:
        node = graph.get_node(node_id)
        if node is not None:
            node.touch()

    for node in graph.get_all_nodes():
        if node.id in activated_ids:
            node.priority = min(1.0, node.priority + activation_boost)
        else:
            node.priority *= priority_decay

    return activated