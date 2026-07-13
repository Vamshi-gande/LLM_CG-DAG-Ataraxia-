"""
Unit tests for src/storage/sqlite.py.

Uses ':memory:' SQLite databases for isolation.
The in-memory DB must be pre-initialized with the correct schema before each
test (mirrors what the setup script does on disk).
"""
import asyncio
import sqlite3
import time

import numpy as np
import pytest

from src.graph.node import Node, NodeType
from src.graph.edge import Edge, EdgeType
from src.storage.sqlite import SQLiteStorage


# ── Schema initialization helper ──────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,
    priority REAL DEFAULT 0.5,
    created_at REAL DEFAULT 0.0,
    updated_at REAL DEFAULT 0.0,
    access_count INTEGER DEFAULT 0,
    confidence REAL DEFAULT 1.0,
    version INTEGER DEFAULT 1,
    last_reconciled_version INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    from_node TEXT NOT NULL,
    to_node TEXT NOT NULL,
    type TEXT NOT NULL,
    weight REAL NOT NULL,
    created_at REAL DEFAULT 0.0
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at REAL DEFAULT 0.0
);
"""


def _make_db(tmp_path, name: str = "test.db") -> str:
    """Create a temp SQLite DB with correct schema; return path."""
    db_path = str(tmp_path / name)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SCHEMA_SQL)
    return db_path


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
    db_path = _make_db(tmp_path)
    storage = SQLiteStorage(db_path, batch_interval=0.01)
    node = _sample_node("n1")

    await storage.queue_save_node(node)
    await storage.flush_for_test()

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
    db_path = _make_db(tmp_path)
    storage = SQLiteStorage(db_path, batch_interval=0.01)
    edge = _sample_edge("e1")

    await storage.queue_save_edge(edge)
    await storage.flush_for_test()

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
    db_path = _make_db(tmp_path)
    storage = SQLiteStorage(db_path, batch_interval=0.01)

    await storage.queue_save_meta("global_summary", "User is building middleware")
    await storage.flush_for_test()

    value = storage.load_meta("global_summary")
    assert value == "User is building middleware"


def test_load_all_nodes_empty_db(tmp_path):
    db_path = _make_db(tmp_path)
    storage = SQLiteStorage(db_path)
    assert storage.load_all_nodes() == []


def test_schema_validation_passes(tmp_path):
    db_path = _make_db(tmp_path)
    # Should not raise
    storage = SQLiteStorage(db_path)
    assert storage is not None


def test_schema_validation_fails_on_missing_column(tmp_path):
    db_path = _make_db(tmp_path)
    # Corrupt the schema by dropping a column (recreate table without it)
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            DROP TABLE nodes;
            CREATE TABLE nodes (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL
                -- missing: priority, created_at, updated_at, etc.
            );
        """)

    with pytest.raises(RuntimeError, match="nodes table missing columns"):
        SQLiteStorage(db_path)


@pytest.mark.asyncio
async def test_write_queue_does_not_block(tmp_path):
    db_path = _make_db(tmp_path)
    storage = SQLiteStorage(db_path, batch_interval=10.0)  # drain won't auto-fire

    start = time.perf_counter()
    for i in range(100):
        await storage.queue_save_node(_sample_node(f"n{i}", seed=i))
    elapsed = time.perf_counter() - start

    # 100 async enqueue calls should take well under 10ms
    assert elapsed < 0.010, f"Enqueue took {elapsed*1000:.1f}ms — expected < 10ms"


@pytest.mark.asyncio
async def test_full_round_trip(tmp_path):
    from src.graph.graph import Graph

    db_path = _make_db(tmp_path)
    storage = SQLiteStorage(db_path, batch_interval=0.01)

    # Build and save a small graph
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
        nodes[0].id  # silence unused warning

    for n in g1.get_all_nodes():
        await storage.queue_save_node(n)
    for e in g1.get_all_edges():
        await storage.queue_save_edge(e)
    await storage.flush_for_test()

    # Rebuild a new graph from storage
    g2 = Graph()
    for n in storage.load_all_nodes():
        g2.add_node(n)
    for e in storage.load_all_edges():
        g2.add_edge(e)

    assert g2.node_count() == g1.node_count()
    assert g2.edge_count() == g1.edge_count()