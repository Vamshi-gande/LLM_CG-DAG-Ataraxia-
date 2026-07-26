"""
Semantic influence table — lazy indirect consistency mechanism.

When a node is updated, its DIRECT (hop-1) neighbors are updated eagerly
by the caller (src.updater.updater.update_graph_node). This module tracks
INDIRECT neighbors (hop 2 and hop 3): instead of eagerly recomputing their
edge weights too, we record that an update happened, and defer the actual
recomputation until the indirect node is next accessed during spreading
activation (src.propagation.reconcile.check_and_reconcile).

Hop-1 nodes are deliberately never written into this table. They are
already up to date the moment update_graph_node() eagerly recomputes
their edge weight — inserting a hop-1 entry here would just cause
check_and_reconcile() to redundantly redo work that's already correct.
"""
from dataclasses import dataclass
from typing import Dict, List

INFLUENCE_STRONG = "strong"    # hop 1 (label reserved; never written here — see module docstring)
INFLUENCE_MEDIUM = "medium"    # hop 2
INFLUENCE_WEAK = "weak"        # hop 3

STRENGTH_ORDER = {INFLUENCE_STRONG: 3, INFLUENCE_MEDIUM: 2, INFLUENCE_WEAK: 1}

_HOP_STRENGTH = {
    2: INFLUENCE_MEDIUM,
    3: INFLUENCE_WEAK,
}


@dataclass
class InfluenceEntry:
    """
    Tracks that node `source_node_id` at `source_version` has an indirect
    influence on the node this entry is stored under.

    strength: "medium" (hop 2) or "weak" (hop 3). Hop 1 entries are never
        created — direct neighbors are updated eagerly, not lazily.
    source_version: the version of source_node at the time influence was
        recorded. If source_node has since been updated again
        (version > source_version), the influenced node needs
        reconciliation the next time it's accessed.
    """
    source_node_id: str
    strength: str
    source_version: int


def add_influence(
    influence_table: Dict[str, List[InfluenceEntry]],
    updated_node_id: str,
    version: int,
    graph: "object",
    hop_limit: int = 3,
) -> None:
    """
    Populate influence entries for all INDIRECT (hop >= 2) neighbors of
    updated_node_id, up to hop_limit hops away.

    Walks outward from updated_node_id via BFS using graph.neighbors().
    Hop 1 is traversed (to reach hop 2/3) but never written to the table —
    direct neighbors are the caller's responsibility (eager update).

    Version tracking: if influence_table[M.id] already has an entry for
    updated_node_id at a lower source_version, it is bumped to the new
    version. If the existing entry's source_version is already >= the new
    version, it is left alone (M has already reconciled against this
    version or a newer one — no redundant downgrade).

    Pure write to influence_table. Does not modify the graph. Not
    thread-safe by design — callers run this from a single asyncio task
    (process_response), so no concurrent-modification guard is needed.
    """
    if hop_limit < 2:
        return  # nothing to record — hop 1 is never recorded, and there's
                # no hop 2+ to reach

    visited = {updated_node_id}
    frontier = [updated_node_id]

    # Nodes discovered while expanding the frontier during loop iteration
    # `hop` are at graph-distance `hop` from updated_node_id (frontier
    # starts at distance 0, so its first expansion reaches distance 1).
    for hop in range(1, hop_limit + 1):
        next_frontier: List[str] = []
        for node_id in frontier:
            for neighbor, _edge in graph.neighbors(node_id):
                if neighbor.id in visited:
                    continue
                visited.add(neighbor.id)
                next_frontier.append(neighbor.id)

                if hop < 2:
                    continue  # distance 1 (direct neighbor) — eager, never lazy-tracked
                strength = _HOP_STRENGTH.get(hop)
                if strength is None:
                    continue  # beyond weak (hop 3) — not tracked
                _upsert_entry(
                    influence_table, neighbor.id,
                    InfluenceEntry(
                        source_node_id=updated_node_id,
                        strength=strength,
                        source_version=version,
                    ),
                )
        frontier = next_frontier
        if not frontier:
            break


def _upsert_entry(
    influence_table: Dict[str, List[InfluenceEntry]],
    target_node_id: str,
    new_entry: InfluenceEntry,
) -> None:
    entries = influence_table.setdefault(target_node_id, [])
    for i, existing in enumerate(entries):
        if existing.source_node_id == new_entry.source_node_id:
            if new_entry.source_version > existing.source_version:
                entries[i] = new_entry
            return  # found existing entry for this source — update or skip, never duplicate
    entries.append(new_entry)


def get_pending_influences(
    node_id: str,
    node_last_reconciled_version: int,
    influence_table: Dict[str, List[InfluenceEntry]],
) -> List[InfluenceEntry]:
    """
    Return influence entries for node_id where source_version is strictly
    greater than node_last_reconciled_version (i.e. not yet reconciled).
    Returns [] if there is nothing pending or no entries at all.
    """
    entries = influence_table.get(node_id, [])
    return [e for e in entries if e.source_version > node_last_reconciled_version]


def clear_reconciled(
    node_id: str,
    reconciled_version: int,
    influence_table: Dict[str, List[InfluenceEntry]],
) -> None:
    """
    Remove influence entries for node_id where source_version <=
    reconciled_version. Called after reconciliation completes so a
    future access doesn't redo work already accounted for.
    """
    if node_id not in influence_table:
        return
    remaining = [e for e in influence_table[node_id] if e.source_version > reconciled_version]
    if remaining:
        influence_table[node_id] = remaining
    else:
        del influence_table[node_id]