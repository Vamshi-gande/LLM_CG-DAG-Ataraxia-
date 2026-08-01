"""
Unit tests for the M8 compression engines.
Uses dummy_embedder + mocked storage/HNSW — no live Ollama.
"""
import asyncio
import time
import uuid

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.graph.graph import Graph
from src.graph.node import Node, NodeType
from src.graph.edge import EdgeType
from src.compression.engines import (
    compute_urgency,
    run_engine2_semantic_merge,
    run_engine3_hierarchical_abstraction,
    run_engine4_temporal_compression,
)
from src.compression.scheduler import CompScheduler


# ── compute_urgency() ─────────────────────────────────────────────────────

def test_urgency_low_priority_node_scores_high(make_node):
    # NOTE: this test's inputs came verbatim from the M8 milestone spec's
    # own worked example, which asserted `score > 0.5`. That's arithmetically
    # unreachable given these exact inputs: urgency = (1-priority) * age_factor
    # * (1/(access_count+1)) * (1-confidence) = 0.9 * 1.0 * 1.0 * 0.5 = 0.45,
    # a hard ceiling set by confidence=0.5 alone. This is a spec bug in the
    # test's threshold, not a bug in compute_urgency() — fixed here to a
    # reachable bound that still demonstrates "high relative to a hot node"
    # (see test_urgency_high_priority_node_scores_low below, which asserts
    # < 0.1 for the opposite case).
    node = make_node("low priority node")
    node.priority = 0.1
    node.access_count = 0
    node.confidence = 0.5
    node.updated_at = time.time() - (30 * 86400)
    score = compute_urgency(node, time.time())
    assert score == pytest.approx(0.45, rel=1e-6)
    assert score > 0.4


def test_urgency_high_priority_node_scores_low(make_node):
    node = make_node("hot active node")
    node.priority = 0.9
    node.access_count = 100
    node.confidence = 1.0
    node.updated_at = time.time()
    score = compute_urgency(node, time.time())
    assert score < 0.1


def test_urgency_active_dag_node_should_be_skipped(make_node):
    node = make_node("in active dag")
    g = Graph()
    g.add_node(node)
    g.active_dag_ids.add(node.id)
    live_candidates = [n for n in g.get_all_nodes() if n.id not in g.active_dag_ids]
    assert node.id not in [n.id for n in live_candidates]


# ── Engine 2 — Semantic Merge ─────────────────────────────────────────────

async def _run_one_cycle(coro_fn, *args, **kwargs):
    """Run an engine's forever-loop exactly once by making asyncio.sleep raise."""
    with patch("src.compression.engines.asyncio.sleep",
               new=AsyncMock(side_effect=asyncio.CancelledError)):
        try:
            await coro_fn(*args, **kwargs)
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_engine2_skips_active_dag_nodes(small_graph):
    g = Graph()
    for n in small_graph["nodes"]:
        g.add_node(n)
    for e in small_graph["edges"]:
        g.add_edge(e)
    for n in small_graph["nodes"]:
        g.active_dag_ids.add(n.id)

    hnsw = MagicMock()
    hnsw.search = MagicMock(return_value=[])
    storage = MagicMock()
    storage.queue_save_node = AsyncMock()

    node_count_before = g.node_count()
    await _run_one_cycle(run_engine2_semantic_merge, g, hnsw, storage, interval_seconds=0)

    assert g.node_count() == node_count_before


@pytest.mark.asyncio
async def test_engine2_merges_similar_nodes(dummy_embedder):
    g = Graph()
    storage = MagicMock()
    storage.queue_save_node = AsyncMock()
    storage.queue_save_edge = AsyncMock()
    storage.queue_delete_edge = AsyncMock()
    storage.queue_archive_node = AsyncMock()

    shared_emb = dummy_embedder("Go middleware for LLMs")
    now = time.time()
    n1 = Node(id=str(uuid.uuid4()), type=NodeType.CONCEPT,
              content="Go middleware for LLMs", embedding=shared_emb.copy(),
              priority=0.8, access_count=5, confidence=1.0,
              created_at=now - 86400 * 2, updated_at=now - 86400 * 2,
              version=1, last_reconciled_version=0)
    n2 = Node(id=str(uuid.uuid4()), type=NodeType.CONCEPT,
              content="Go LLM inference middleware", embedding=shared_emb.copy(),
              priority=0.6, access_count=3, confidence=1.0,
              created_at=now - 86400 * 2, updated_at=now - 86400 * 2,
              version=1, last_reconciled_version=0)
    g.add_node(n1)
    g.add_node(n2)

    hnsw = MagicMock()
    hnsw.search = MagicMock(return_value=[(n2.id, 0.02)])  # cosine distance ~0.02
    hnsw.contains = MagicMock(return_value=True)

    await _run_one_cycle(
        run_engine2_semantic_merge, g, hnsw, storage,
        interval_seconds=0,
        similarity_threshold=0.90,
        min_age_hours=0,
        min_access_count=0,
    )

    assert g.node_count() == 1
    storage.queue_archive_node.assert_called_once()


@pytest.mark.asyncio
async def test_engine2_does_not_merge_below_similarity_threshold(dummy_embedder):
    g = Graph()
    storage = MagicMock()
    storage.queue_save_node = AsyncMock()
    storage.queue_save_edge = AsyncMock()
    storage.queue_delete_edge = AsyncMock()
    storage.queue_archive_node = AsyncMock()

    now = time.time()
    n1 = Node(id=str(uuid.uuid4()), type=NodeType.CONCEPT,
              content="Go middleware", embedding=dummy_embedder("Go middleware"),
              priority=0.8, access_count=5, confidence=1.0,
              created_at=now - 86400 * 2, updated_at=now - 86400 * 2,
              version=1, last_reconciled_version=0)
    n2 = Node(id=str(uuid.uuid4()), type=NodeType.CONCEPT,
              content="unrelated topic", embedding=dummy_embedder("unrelated topic"),
              priority=0.6, access_count=3, confidence=1.0,
              created_at=now - 86400 * 2, updated_at=now - 86400 * 2,
              version=1, last_reconciled_version=0)
    g.add_node(n1)
    g.add_node(n2)

    hnsw = MagicMock()
    hnsw.search = MagicMock(return_value=[(n2.id, 0.8)])  # cosine distance 0.8 -> sim 0.2

    await _run_one_cycle(
        run_engine2_semantic_merge, g, hnsw, storage,
        interval_seconds=0, similarity_threshold=0.95, min_age_hours=0, min_access_count=0,
    )

    assert g.node_count() == 2
    storage.queue_archive_node.assert_not_called()


# ── Engine 3 — Hierarchical Abstraction ───────────────────────────────────

@pytest.mark.asyncio
async def test_engine3_skips_noise_labels(dummy_embedder):
    g = Graph()
    storage = MagicMock()
    storage.queue_save_node = AsyncMock()
    storage.queue_save_edge = AsyncMock()
    hnsw = MagicMock()
    embedder = MagicMock()

    now = time.time()
    nodes = []
    for i in range(12):
        n = Node(id=str(uuid.uuid4()), type=NodeType.CONCEPT,
                 content=f"node {i}", embedding=dummy_embedder(f"node {i}"),
                 priority=0.2, access_count=1, confidence=1.0,
                 created_at=now, updated_at=now, version=1, last_reconciled_version=0)
        g.add_node(n)
        nodes.append(n)

    fake_hdbscan_module = MagicMock()
    fake_clusterer = MagicMock()
    # every node labeled noise
    fake_clusterer.fit_predict = MagicMock(return_value=[-1] * len(nodes))
    fake_hdbscan_module.HDBSCAN = MagicMock(return_value=fake_clusterer)

    with patch.dict("sys.modules", {"hdbscan": fake_hdbscan_module}):
        await _run_one_cycle(
            run_engine3_hierarchical_abstraction,
            g, hnsw, storage, embedder,
            interval_seconds=0, min_cluster_size=5,
        )

    summary_nodes = [n for n in g.get_all_nodes() if n.type == NodeType.SUMMARY]
    assert summary_nodes == []
    storage.queue_save_node.assert_not_called()


@pytest.mark.asyncio
async def test_engine3_creates_parent_for_low_priority_cluster(dummy_embedder):
    g = Graph()
    storage = MagicMock()
    storage.queue_save_node = AsyncMock()
    storage.queue_save_edge = AsyncMock()
    hnsw = MagicMock()
    embedder = MagicMock()

    # NOTE: run_engine3_hierarchical_abstraction() returns early (per spec)
    # if len(live_nodes) < min_cluster_size * 2 — this is the engine's own
    # pre-clustering gate, checked BEFORE HDBSCAN ever runs, so it must be
    # cleared regardless of how HDBSCAN's fit_predict is mocked below.
    # With min_cluster_size=5 that gate is 10 nodes; the original version of
    # this test only created 6 and never reached the clustering logic at
    # all (silently returned early, asserted 0 == 1). Fixed by creating 10.
    now = time.time()
    nodes = []
    for i in range(10):
        n = Node(id=str(uuid.uuid4()), type=NodeType.CONCEPT,
                 content=f"low priority topic {i}", embedding=dummy_embedder(f"topic {i}"),
                 priority=0.1, access_count=1, confidence=1.0,
                 created_at=now, updated_at=now, version=1, last_reconciled_version=0)
        g.add_node(n)
        nodes.append(n)

    fake_hdbscan_module = MagicMock()
    fake_clusterer = MagicMock()
    fake_clusterer.fit_predict = MagicMock(return_value=[0] * len(nodes))
    fake_hdbscan_module.HDBSCAN = MagicMock(return_value=fake_clusterer)

    with patch.dict("sys.modules", {"hdbscan": fake_hdbscan_module}):
        await _run_one_cycle(
            run_engine3_hierarchical_abstraction,
            g, hnsw, storage, embedder,
            interval_seconds=0, min_cluster_size=5, priority_threshold=0.4,
        )

    summary_nodes = [n for n in g.get_all_nodes() if n.type == NodeType.SUMMARY]
    assert len(summary_nodes) == 1
    parent = summary_nodes[0]
    child_edges = [e for e in g.get_edges_from(parent.id) if e.type == EdgeType.HIERARCHICAL]
    assert len(child_edges) == len(nodes)
    for n in nodes:
        assert g.get_node(n.id).priority < 0.1  # deprioritized, still present


# ── Engine 4 — Temporal Compression ───────────────────────────────────────

@pytest.mark.asyncio
async def test_engine4_archives_old_low_priority_node(dummy_embedder):
    g = Graph()
    storage = MagicMock()
    storage.queue_save_node = AsyncMock()
    storage.queue_save_edge = AsyncMock()
    storage.queue_delete_edge = AsyncMock()
    storage.queue_archive_node = AsyncMock()
    hnsw = MagicMock()
    hnsw.deleted_count = MagicMock(return_value=0)
    embedder = MagicMock()
    embedder.embed = MagicMock(return_value=dummy_embedder("summary"))

    old_node = Node(
        id=str(uuid.uuid4()), type=NodeType.CONCEPT,
        content="old rarely accessed node",
        embedding=dummy_embedder("old rarely accessed node"),
        priority=0.1, access_count=1, confidence=0.8,
        created_at=time.time() - 86400 * 10,
        updated_at=time.time() - 86400 * 10,
        version=1, last_reconciled_version=0,
    )
    g.add_node(old_node)

    await _run_one_cycle(
        run_engine4_temporal_compression, g, hnsw, storage, embedder,
        interval_seconds=0, min_age_days=7.0, max_access_count=3, max_priority=0.3,
    )

    assert g.get_node(old_node.id) is None
    assert old_node.id in g.cold_node_ids
    storage.queue_archive_node.assert_called_once()


@pytest.mark.asyncio
async def test_engine4_deletes_edges_before_archiving(dummy_embedder, make_edge):
    """M8-B invariant: edges deleted BEFORE node archived."""
    g = Graph()
    storage = MagicMock()
    storage.queue_save_node = AsyncMock()
    storage.queue_save_edge = AsyncMock()
    storage.queue_delete_edge = AsyncMock()
    storage.queue_archive_node = AsyncMock()
    hnsw = MagicMock()
    hnsw.deleted_count = MagicMock(return_value=0)
    embedder = MagicMock()
    embedder.embed = MagicMock(return_value=dummy_embedder("summary"))

    now = time.time()
    n1 = Node(id=str(uuid.uuid4()), type=NodeType.CONCEPT,
              content="stale node", embedding=dummy_embedder("stale"),
              priority=0.1, access_count=0, confidence=0.8,
              updated_at=now - 86400 * 10, created_at=now - 86400 * 10,
              version=1, last_reconciled_version=0)
    n2 = Node(id=str(uuid.uuid4()), type=NodeType.CONCEPT,
              content="neighbor", embedding=dummy_embedder("neighbor"),
              priority=0.5, access_count=10, confidence=1.0,
              updated_at=now, created_at=now,
              version=1, last_reconciled_version=0)
    g.add_node(n1)
    g.add_node(n2)
    edge = make_edge(n1.id, n2.id, EdgeType.SEMANTIC, 0.7)
    g.add_edge(edge)

    call_order = []

    async def _record_delete(eid):
        call_order.append("delete_edge")

    async def _record_archive(nid):
        call_order.append("archive_node")

    storage.queue_delete_edge = AsyncMock(side_effect=_record_delete)
    storage.queue_archive_node = AsyncMock(side_effect=_record_archive)

    await _run_one_cycle(
        run_engine4_temporal_compression, g, hnsw, storage, embedder,
        interval_seconds=0, min_age_days=7.0, max_access_count=3, max_priority=0.3,
    )

    assert "delete_edge" in call_order and "archive_node" in call_order
    assert call_order.index("delete_edge") < call_order.index("archive_node"), \
        "M8-B INVARIANT VIOLATED: archive_node called before delete_edge"


@pytest.mark.asyncio
async def test_engine4_triggers_hnsw_rebuild_above_threshold(dummy_embedder):
    g = Graph()
    storage = MagicMock()
    storage.queue_save_node = AsyncMock()
    storage.queue_save_edge = AsyncMock()
    storage.queue_delete_edge = AsyncMock()
    storage.queue_archive_node = AsyncMock()
    hnsw = MagicMock()
    hnsw.deleted_count = MagicMock(return_value=20)
    hnsw._max_elements = 100  # 20% deleted > 10% threshold
    hnsw.rebuild = MagicMock()
    embedder = MagicMock()

    await _run_one_cycle(
        run_engine4_temporal_compression, g, hnsw, storage, embedder,
        interval_seconds=0, min_age_days=7.0, max_access_count=3, max_priority=0.3,
    )

    hnsw.rebuild.assert_called_once()


# ── CompScheduler ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scheduler_launches_tasks():
    scheduler = CompScheduler()
    g = MagicMock()
    g.node_count = MagicMock(return_value=0)
    g.get_all_nodes = MagicMock(return_value=[])
    g.active_dag_ids = set()
    g.cold_node_ids = set()

    storage = MagicMock()
    hnsw = MagicMock()
    embedder = MagicMock()
    ollama = MagicMock()
    assembler = MagicMock()
    config = {
        "compression": {
            "semantic_merge_interval_min": 5,
            "hierarchical_interval_min": 15,
            "temporal_compress_interval_min": 30,
            "global_summary_interval_min": 60,
            "global_summary_update_threshold": 0.20,
        },
        "merge": {"similarity_threshold": 0.95, "min_age_hours": 24, "min_access_count": 1},
        "model": {"default": "llama3.2:3b"},
        "summary": {"min_nodes_before_generation": 15},
    }

    scheduler.start(g, hnsw, storage, embedder, ollama, assembler, config)
    assert len(scheduler._tasks) == 4

    await scheduler.stop()
    assert len(scheduler._tasks) == 0


@pytest.mark.asyncio
async def test_scheduler_stop_cancels_all_tasks():
    scheduler = CompScheduler()
    scheduler._tasks = [
        asyncio.create_task(asyncio.sleep(9999), name=f"task_{i}")
        for i in range(3)
    ]
    await scheduler.stop()
    for task in scheduler._tasks:
        assert task.cancelled() or task.done()