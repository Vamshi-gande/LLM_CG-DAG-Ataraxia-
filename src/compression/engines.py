"""
Compression Scheduler — Engines 2-5.

Engine 1 (Priority Decay) is NOT here — it runs inline inside
spreading_activation() on every query (M3). This module implements the
four background engines that run as long-lived asyncio Tasks and never
block the inference path.

Design invariants carried in from the M8 milestone spec (do not relax
these when editing):

  * Cold storage = Option A (cold_node_ids set on Graph, node stays
    resident in the main HNSW index rather than a second cold index).
    See docstring on `graph.cold_node_ids` and the thaw check added to
    `src/propagation/activation.py`.

  * Asyncio mutation safety: every engine completes ALL in-memory graph
    mutations (no `await`) before issuing any `await storage.queue_*()`
    calls. Between two `await`s, no other coroutine can run, so the
    in-memory graph is always internally consistent at every await
    boundary. Storage (SQLite) may lag briefly behind memory; that is
    fine because SQLite is reloaded as source of truth on restart.

  * M8-B invariant: for any node that gets archived, every edge attached
    to it MUST be deleted from SQLite (queue_delete_edge) BEFORE the
    node itself is archived (queue_archive_node). Violating this order
    causes `add_edge()` to raise ValueError on the NEXT proxy restart,
    because load_all_edges() runs after load_all_nodes() and will try to
    attach an edge to a node that was never loaded (archived nodes are
    filtered out of load_all_nodes()).

  * active_dag_ids vs cold_node_ids are different sets with different
    semantics. active_dag_ids = "in use by a live query right now, never
    touch." cold_node_ids = "archived to SQLite, still present in the
    main HNSW index, thaw candidate on next ANN hit." Never confuse or
    merge these two sets.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np

from src.graph.graph import Graph
from src.graph.node import Node, NodeType
from src.graph.edge import Edge, EdgeType
from src.hnsw.index import HNSWIndex
from src.storage.sqlite import SQLiteStorage
from src.embedding.onnx_embedder import ONNXEmbedder
from src.context.assembler import ContextAssembler

if TYPE_CHECKING:
    # Import deferred to type-checking only. src.proxy.ollama_client lives
    # inside the src.proxy PACKAGE, and src/proxy/__init__.py does
    # `from .server import app` — server.py imports CompScheduler from
    # this very package (src.compression.scheduler), which imports
    # engines.py (this file). A module-level `from src.proxy.ollama_client
    # import OllamaClient` here forces Python to fully initialize
    # src.proxy's __init__.py first, which re-enters src.compression.scheduler
    # before it has finished initializing -> ImportError: cannot import
    # name 'CompScheduler' from partially initialized module. Since this
    # file has `from __future__ import annotations` at the top, every type
    # hint below is evaluated as a lazy string, not a real reference, so
    # OllamaClient is never actually needed at import time — only for
    # static type checkers / IDEs, which do see it via this guarded import.
    from src.proxy.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector; return it unchanged if the norm is ~0."""
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return v
    return (v / norm).astype(np.float32)


def compute_urgency(node: Node, now: float) -> float:
    """
    urgency = (1 - priority) x age_factor x (1 / (access_count + 1)) x (1 - confidence)

    age_factor = min(1.0, days_since_update / 30.0)
    where days_since_update = (now - node.updated_at) / 86400

    High urgency = low priority + old + rarely accessed + low confidence.
    Nodes with high urgency are compression candidates.

    NOTE: this function has no active_dag_ids guard of its own — the
    calling engine is responsible for excluding active_dag_ids (and,
    where relevant, cold_node_ids) nodes from the candidate list before
    calling compute_urgency() on them at all.
    """
    days_since_update = max(0.0, (now - node.updated_at)) / 86400.0
    age_factor = min(1.0, days_since_update / 30.0)
    access_factor = 1.0 / (node.access_count + 1)
    return (1.0 - node.priority) * age_factor * access_factor * (1.0 - node.confidence)


def _rank_by_urgency(nodes: List[Node], now: float, top_n: int) -> List[Node]:
    scored = sorted(nodes, key=lambda n: compute_urgency(n, now), reverse=True)
    return scored[:top_n]


def _build_redirected_edge(
    graph: Graph,
    edge: Edge,
    old_id: str,
    new_id: str,
) -> Optional[Edge]:
    """
    Redirect `edge` so that any endpoint equal to old_id becomes new_id.
    Returns None (and adds nothing) if the redirect would create a
    self-loop, or if an equivalent edge (same endpoints + type) already
    exists — both guards match Engine 2's Issue 3.3/3.4 fixes and are
    reused verbatim by Engine 4.
    """
    new_from = new_id if edge.from_node == old_id else edge.from_node
    new_to = new_id if edge.to_node == old_id else edge.to_node

    if new_from == new_to:
        return None  # self-loop guard

    existing = graph.get_edges_from(new_from)
    already_exists = any(e.to_node == new_to and e.type == edge.type for e in existing)
    if already_exists:
        return None  # duplicate guard

    return Edge.new(new_from, new_to, edge.type, weight=edge.weight)


async def _run_forever(cycle_fn, interval_seconds: float, engine_name: str) -> None:
    """
    Shared driver loop: run `cycle_fn()` (a zero-arg async callable) in a
    try/except per cycle so a single bad cycle never kills the whole
    background task. CancelledError always re-raises for clean shutdown.
    """
    while True:
        try:
            await cycle_fn()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("%s: cycle failed, continuing", engine_name)
        await asyncio.sleep(interval_seconds)


# ─────────────────────────────────────────────────────────────────────────────
# Engine 2 — Semantic Merge
# ─────────────────────────────────────────────────────────────────────────────

async def run_engine2_semantic_merge(
    graph: Graph,
    hnsw: HNSWIndex,
    storage: SQLiteStorage,
    interval_seconds: float = 300.0,
    top_n_candidates: int = 50,
    similarity_threshold: float = 0.95,
    min_age_hours: float = 24.0,
    min_access_count: int = 1,
) -> None:
    """Background task: find and merge near-duplicate nodes. Runs forever."""

    async def _cycle() -> None:
        now = time.time()
        live_nodes = [
            n for n in graph.get_all_nodes()
            if n.id not in graph.active_dag_ids and n.id not in graph.cold_node_ids
        ]
        candidates = _rank_by_urgency(live_nodes, now, top_n_candidates)

        for a in candidates:
            if graph.get_node(a.id) is None:
                continue  # a was merged away earlier in this same cycle
            if a.id in graph.active_dag_ids:
                continue

            neighbor_hits = hnsw.search(a.embedding, top_k=10)
            merged_this_candidate = False

            for hit in neighbor_hits:
                b_id, distance = hit[0], hit[1]
                if b_id == a.id:
                    continue
                if b_id in graph.active_dag_ids or b_id in graph.cold_node_ids:
                    continue

                b = graph.get_node(b_id)
                if b is None:
                    continue  # ghost HNSW entry

                similarity = 1.0 - distance
                if similarity < similarity_threshold:
                    continue

                age_a_hours = (now - a.updated_at) / 3600.0
                age_b_hours = (now - b.updated_at) / 3600.0
                if age_a_hours < min_age_hours or age_b_hours < min_age_hours:
                    continue

                if a.access_count < min_access_count or b.access_count < min_access_count:
                    continue

                await _merge_nodes(graph, storage, a, b)
                merged_this_candidate = True
                break  # re-score next cycle rather than chain-merge further

            if merged_this_candidate:
                continue

    await _run_forever(_cycle, interval_seconds, "engine2_semantic_merge")


async def _merge_nodes(graph: Graph, storage: SQLiteStorage, node_a: Node, node_b: Node) -> None:
    """
    winner = higher priority (ties: more recently updated).
    loser absorbed into winner; loser -> cold storage (Option A).
    """
    if node_a.priority > node_b.priority:
        winner, loser = node_a, node_b
    elif node_b.priority > node_a.priority:
        winner, loser = node_b, node_a
    else:
        winner, loser = (node_a, node_b) if node_a.updated_at >= node_b.updated_at else (node_b, node_a)

    # Content cap: prevent unbounded growth across repeated merges.
    winner_content = winner.content[:250]
    loser_content = loser.content[:250]
    merged_content = (winner_content + " | " + loser_content)[:500]

    merged_embedding = _safe_normalize((winner.embedding + loser.embedding) / 2.0)
    merged_priority = max(winner.priority, loser.priority)
    merged_access_count = winner.access_count + loser.access_count

    loser_edges = graph.get_all_edges_for(loser.id)

    # ---- PHASE 1: in-memory mutations only, no await ----
    winner.content = merged_content
    winner.embedding = merged_embedding
    winner.priority = merged_priority
    winner.access_count = merged_access_count

    new_edges: List[Edge] = []
    for edge in loser_edges:
        redirected = _build_redirected_edge(graph, edge, loser.id, winner.id)
        if redirected is None:
            continue
        new_edges.append(redirected)
        graph.add_edge(redirected)

    graph.remove_node(loser.id)
    graph.update_node(winner)
    graph.cold_node_ids.add(loser.id)
    # ---- end PHASE 1 ----

    # ---- PHASE 2: async storage writes, may yield ----
    for edge in loser_edges:
        await storage.queue_delete_edge(edge.id)
    await storage.queue_archive_node(loser.id)
    for new_edge in new_edges:
        await storage.queue_save_edge(new_edge)
    await storage.queue_save_node(winner)


# ─────────────────────────────────────────────────────────────────────────────
# Engine 3 — Hierarchical Abstraction
# ─────────────────────────────────────────────────────────────────────────────

async def run_engine3_hierarchical_abstraction(
    graph: Graph,
    hnsw: HNSWIndex,
    storage: SQLiteStorage,
    embedder: ONNXEmbedder,
    interval_seconds: float = 900.0,
    min_cluster_size: int = 5,
    priority_threshold: float = 0.4,
    hierarchical_edge_weight: float = 0.8,
    child_priority_multiplier: float = 0.7,
) -> None:
    """Background task: HDBSCAN-cluster low-priority nodes into Summary parents."""

    async def _cycle() -> None:
        import hdbscan  # imported lazily so unit tests without the dep can mock this module

        live_nodes = [
            n for n in graph.get_all_nodes()
            if n.id not in graph.active_dag_ids and n.id not in graph.cold_node_ids
        ]
        if len(live_nodes) < min_cluster_size * 2:
            return

        embeddings = np.stack([n.embedding for n in live_nodes]).astype(np.float64)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = await asyncio.to_thread(clusterer.fit_predict, embeddings)

        clusters: Dict[int, List[Node]] = {}
        for node, label in zip(live_nodes, labels):
            if label == -1:
                continue  # noise — never gets a synthetic parent
            clusters.setdefault(int(label), []).append(node)

        for cluster_nodes in clusters.values():
            if len(cluster_nodes) < min_cluster_size:
                continue

            avg_priority = sum(n.priority for n in cluster_nodes) / len(cluster_nodes)
            if avg_priority >= priority_threshold:
                continue  # cluster still "hot", leave alone

            if any(n.id in graph.active_dag_ids for n in cluster_nodes):
                continue  # a member is in use by a live query right now

            child_id_set = {n.id for n in cluster_nodes}
            duplicate_parent_exists = False
            for existing_node in graph.get_all_nodes():
                if existing_node.type != NodeType.SUMMARY:
                    continue
                existing_children = {
                    e.to_node for e in graph.get_edges_from(existing_node.id)
                    if e.type == EdgeType.HIERARCHICAL
                }
                if existing_children & child_id_set:
                    duplicate_parent_exists = True
                    break
            if duplicate_parent_exists:
                continue

            cluster_embeddings = np.stack([n.embedding for n in cluster_nodes])
            parent_content = "; ".join(n.content[:50] for n in cluster_nodes[:5])
            parent = Node(
                id=str(uuid.uuid4()),
                type=NodeType.SUMMARY,
                content=parent_content,
                embedding=_safe_normalize(np.mean(cluster_embeddings, axis=0)),
                priority=min(1.0, avg_priority * 1.2),
                created_at=time.time(),
                updated_at=time.time(),
                access_count=0,
                confidence=1.0,
                version=1,
                last_reconciled_version=0,
            )

            # ---- PHASE 1: in-memory mutations only, no await ----
            graph.add_node(parent)
            child_edges: List[Edge] = []
            for child in cluster_nodes:
                edge = Edge.new(parent.id, child.id, EdgeType.HIERARCHICAL,
                                 weight=hierarchical_edge_weight)
                graph.add_edge(edge)
                child_edges.append(edge)
                child.priority = child.priority * child_priority_multiplier
            # ---- end PHASE 1 ----

            # ---- PHASE 2: async storage writes ----
            await storage.queue_save_node(parent)
            for edge in child_edges:
                await storage.queue_save_edge(edge)
            for child in cluster_nodes:
                await storage.queue_save_node(child)

    await _run_forever(_cycle, interval_seconds, "engine3_hierarchical_abstraction")


# ─────────────────────────────────────────────────────────────────────────────
# Engine 4 — Temporal Compression
# ─────────────────────────────────────────────────────────────────────────────

async def run_engine4_temporal_compression(
    graph: Graph,
    hnsw: HNSWIndex,
    storage: SQLiteStorage,
    embedder: ONNXEmbedder,
    interval_seconds: float = 1800.0,
    min_age_days: float = 7.0,
    max_access_count: int = 3,
    max_priority: float = 0.3,
    rebuild_threshold_ratio: float = 0.10,
) -> None:
    """Background task: archive old, rarely-accessed, low-priority nodes to cold storage."""

    async def _cycle() -> None:
        now = time.time()
        candidates = [
            n for n in graph.get_all_nodes()
            if n.id not in graph.active_dag_ids
            and n.id not in graph.cold_node_ids
            and n.type != NodeType.SUMMARY
            and (now - n.updated_at) / 86400.0 > min_age_days
            and n.access_count < max_access_count
            and n.priority < max_priority
        ]

        for candidate in candidates:
            # candidate may have already been merged/archived earlier
            # in this same cycle by a redirected edge touching it
            if graph.get_node(candidate.id) is None:
                continue

            summary = Node(
                id=str(uuid.uuid4()),
                type=NodeType.SUMMARY,
                content=f"[archived {time.strftime('%Y-%m')}] {candidate.content[:400]}",
                embedding=candidate.embedding.copy(),
                priority=candidate.priority,
                created_at=time.time(),
                updated_at=time.time(),
                access_count=0,
                confidence=candidate.confidence,
                version=1,
                last_reconciled_version=0,
            )

            connected = graph.get_all_edges_for(candidate.id)

            transfer_edges: List[Edge] = []
            for edge in connected:
                redirected = _build_redirected_edge(graph, edge, candidate.id, summary.id)
                if redirected is not None:
                    transfer_edges.append(redirected)

            # ---- PHASE 1: in-memory mutations only, no await ----
            graph.add_node(summary)
            for te in transfer_edges:
                graph.add_edge(te)
            graph.remove_node(candidate.id)
            graph.cold_node_ids.add(candidate.id)
            # ---- end PHASE 1 ----

            # ---- PHASE 2: async storage writes ----
            await storage.queue_save_node(summary)
            for te in transfer_edges:
                await storage.queue_save_edge(te)
            for edge in connected:
                await storage.queue_delete_edge(edge.id)  # BEFORE archive — M8-B invariant
            await storage.queue_archive_node(candidate.id)  # AFTER edges deleted

        # HNSW rebuild trigger (M8-D)
        deleted = hnsw.deleted_count() if hasattr(hnsw, "deleted_count") else 0
        max_elements = getattr(hnsw, "_max_elements", None)
        if max_elements and deleted > max_elements * rebuild_threshold_ratio:
            hnsw.rebuild(graph.get_all_nodes())

    await _run_forever(_cycle, interval_seconds, "engine4_temporal_compression")


# ─────────────────────────────────────────────────────────────────────────────
# Engine 5 — Global Summary Regenerator (the ONLY engine that calls the LLM)
# ─────────────────────────────────────────────────────────────────────────────

async def run_engine5_global_summary(
    graph: Graph,
    storage: SQLiteStorage,
    ollama: OllamaClient,
    assembler: ContextAssembler,
    interval_seconds: float = 3600.0,
    update_threshold_ratio: float = 0.20,
    top_n_nodes: int = 30,
    min_nodes_before_generation: int = 15,
    model: str = "llama3.2:3b",
) -> None:
    """
    Background task: regenerate the Tier 1 global summary from the
    top-priority nodes. This is the only engine permitted to call the
    local LLM; all others are pure computation. LLM failures must never
    crash this background task — skip the cycle and try again later.
    """

    async def _cycle() -> None:
        if graph.node_count() < min_nodes_before_generation:
            return

        top_nodes = sorted(graph.get_all_nodes(), key=lambda n: n.priority, reverse=True)[:top_n_nodes]
        content = "\n".join(f"- {n.content}" for n in top_nodes)
        prompt = (
            "Summarize the following knowledge into a 200-token context "
            f"header for a coding assistant:\n{content}"
        )

        try:
            response = await ollama.generate(model=model, prompt=prompt)
        except Exception:
            logger.exception("engine5_global_summary: Ollama call failed, skipping this cycle")
            return

        summary = (response.get("response", "") if isinstance(response, dict) else "").strip()
        if not summary:
            return

        assembler.update_global_summary(summary)
        await storage.queue_save_meta("global_summary", summary)

    await _run_forever(_cycle, interval_seconds, "engine5_global_summary")