"""
Unit tests for src/storage/sqlite.py.

Uses on-disk temp SQLite databases (SQLiteStorage manages its own schema
via CREATE TABLE IF NOT EXISTS — tests should not pre-create schema,
since a stale pre-existing table silently blocks the canonical schema
from being applied).
"""
import sqlite3
import time

import numpy as np
import pytest

from src.graph.node import Node, NodeType
from src.graph.edge import Edge, EdgeType
from src.storage.sqlite import SQLiteStorage


def _make_db_path(tmp_path, name: str = "test.db") -> str:
    """Return a path to a fresh, non-existent DB file. SQLiteStorage's
    own __init__ creates the canonical schema — we must NOT pre-create
    it here, or CREATE TABLE IF NOT EXISTS will silently keep whatever
    schema already exists (e.g. missing the 'archived' column)."""
    return str(tmp_path / name)


def _rand_emb(seed: int = 0, dim: int = 384) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _sample_node(node_id: str = "n1", seed: int = 0) -> Node:
    return Node(
        id=node_id,
        type=NodeType.CONCEPT,
        content="Test concept",
        embedding=_rand_emb(seed),
        priority=0.6,
        created_at=1000.0,
        updated_at=1001.0,
        access_count=3,
        confidence=0.9,
        version=2,
        last_reconciled_version=1,
    )


def _sample_edge(edge_id: str = "e1") -> Edge:
    return Edge(
        id=edge_id,
        from_node="n1",
        to_node="n2",
        type=EdgeType.CAUSAL,
        weight=0.75,
        created_at=1000.0,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_embedding_to_blob_and_back():
    emb = _rand_emb(42)
    blob = SQLiteStorage.embedding_to_blob(emb)
    recovered = SQLiteStorage.blob_to_embedding(blob)
    np.testing.assert_array_almost_equal(emb, recovered, decimal=6)
    assert recovered.dtype == np.float32


@pytest.mark.asyncio
async def test_save_and_load_node(tmp_path):
    db_path = _make_db_path(tmp_path)
    storage = SQLiteStorage(db_path)
    node = _sample_node("n1")

    await storage.start_write_queue()
    await storage.queue_save_node(node)
    await storage.stop_write_queue()  # flushes + commits, clears drain task

    loaded = storage.load_all_nodes()
    assert len(loaded) == 1
    n = loaded[0]
    assert n.id == "n1"
    assert n.type == NodeType.CONCEPT
    assert n.content == "Test concept"
    assert n.priority == pytest.approx(0.6)
    assert n.access_count == 3
    assert n.confidence == pytest.approx(0.9)
    assert n.version == 2
    assert n.last_reconciled_version == 1
    np.testing.assert_array_almost_equal(n.embedding, node.embedding, decimal=6)


@pytest.mark.asyncio
async def test_save_and_load_edge(tmp_path):
    db_path = _make_db_path(tmp_path)
    storage = SQLiteStorage(db_path)
    edge = _sample_edge("e1")

    await storage.start_write_queue()
    await storage.queue_save_edge(edge)
    await storage.stop_write_queue()

    loaded = storage.load_all_edges()
    assert len(loaded) == 1
    e = loaded[0]
    assert e.id == "e1"
    assert e.from_node == "n1"
    assert e.to_node == "n2"
    assert e.type == EdgeType.CAUSAL
    assert e.weight == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_save_meta_and_load(tmp_path):
    db_path = _make_db_path(tmp_path)
    storage = SQLiteStorage(db_path)

    await storage.start_write_queue()
    await storage.queue_save_meta("global_summary", "User is building middleware")
    await storage.stop_write_queue()

    value = storage.load_meta("global_summary")
    assert value == "User is building middleware"


def test_load_all_nodes_empty_db(tmp_path):
    db_path = _make_db_path(tmp_path)
    storage = SQLiteStorage(db_path)
    assert storage.load_all_nodes() == []


def test_schema_validation_passes(tmp_path):
    db_path = _make_db_path(tmp_path)
    # Should not raise
    storage = SQLiteStorage(db_path)
    assert storage is not None


def test_schema_validation_fails_on_missing_column(tmp_path):
    db_path = _make_db_path(tmp_path)
    # Pre-create a corrupt 'nodes' table on an otherwise-empty DB. Because
    # SQLiteStorage's own schema init uses CREATE TABLE IF NOT EXISTS, this
    # corrupt table survives construction and validation must catch it.
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE nodes (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL
                -- missing: priority, created_at, updated_at, etc.
            );
        """)

    with pytest.raises(RuntimeError, match="missing columns"):
        SQLiteStorage(db_path)


@pytest.mark.asyncio
async def test_write_queue_does_not_block(tmp_path):
    db_path = _make_db_path(tmp_path)
    storage = SQLiteStorage(db_path)
    # Deliberately do NOT start the drain task — enqueueing must still be
    # fast, since _enqueue() only does asyncio.Queue.put_nowait() and
    # never touches the DB directly.

    # Pre-generate nodes OUTSIDE the timed block — embedding generation
    # (numpy RNG + normalization) is real CPU work and must not be
    # counted against the enqueue-latency budget.
    nodes = [_sample_node(f"n{i}", seed=i) for i in range(100)]

    start = time.perf_counter()
    for n in nodes:
        await storage.queue_save_node(n)
    elapsed = time.perf_counter() - start

    # 100 enqueue-only calls should complete well under 100ms even with
    # asyncio/Windows scheduling overhead. This is a "doesn't block on
    # I/O" smoke test, not a strict performance benchmark — the DB write
    # itself only happens later in _drain_loop(), never here.
    assert elapsed < 0.100, f"Enqueue took {elapsed*1000:.1f}ms — expected < 100ms"

@pytest.mark.asyncio
async def test_full_round_trip(tmp_path):
    from src.graph.graph import Graph

    db_path = _make_db_path(tmp_path)
    storage = SQLiteStorage(db_path)

    g1 = Graph()
    nodes = [_sample_node(f"n{i}", seed=i) for i in range(4)]
    edges = [
        _sample_edge("e0"),
        Edge(id="e1", from_node="n1", to_node="n2",
             type=EdgeType.SEMANTIC, weight=0.5),
        Edge(id="e2", from_node="n2", to_node="n3",
             type=EdgeType.TEMPORAL, weight=0.4),
    ]
    for n in nodes:
        g1.add_node(n)
    for e in edges:
        g1.add_edge(e)

    await storage.start_write_queue()
    for n in g1.get_all_nodes():
        await storage.queue_save_node(n)
    for e in g1.get_all_edges():
        await storage.queue_save_edge(e)
    await storage.stop_write_queue()

    g2 = Graph()
    for n in storage.load_all_nodes():
        g2.add_node(n)
    for e in storage.load_all_edges():
        g2.add_edge(e)

    assert g2.node_count() == g1.node_count()
    assert g2.edge_count() == g1.edge_count()