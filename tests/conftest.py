# tests/conftest.py
# Shared pytest fixtures for all test suites.
# Uses in-memory SQLite — never touches the real graph.db.

import pytest
import asyncio
import sqlite3
import numpy as np
import time
import uuid
from unittest.mock import AsyncMock, MagicMock

# ── Event loop (required for pytest-asyncio) ──────────────────────────────────
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── In-memory SQLite DB ───────────────────────────────────────────────────────
@pytest.fixture
def in_memory_db():
    """Fresh in-memory SQLite DB for each test. Isolated, no disk I/O."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.executescript("""
    PRAGMA journal_mode=WAL;

    CREATE TABLE nodes (
        id                      TEXT PRIMARY KEY,
        type                    TEXT NOT NULL,
        content                 TEXT NOT NULL,
        embedding               BLOB NOT NULL,
        priority                REAL NOT NULL DEFAULT 0.5,
        created_at              REAL NOT NULL,
        updated_at              REAL NOT NULL,
        access_count            INTEGER NOT NULL DEFAULT 0,
        confidence              REAL NOT NULL DEFAULT 0.8,
        version                 INTEGER NOT NULL DEFAULT 1,
        last_reconciled_version INTEGER NOT NULL DEFAULT 0
    );

    CREATE TABLE edges (
        id         TEXT PRIMARY KEY,
        from_node  TEXT NOT NULL REFERENCES nodes(id),
        to_node    TEXT NOT NULL REFERENCES nodes(id),
        type       TEXT NOT NULL,
        weight     REAL NOT NULL DEFAULT 0.5,
        created_at REAL NOT NULL
    );

    CREATE TABLE meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """)
    conn.commit()
    yield conn
    conn.close()


# ── Dummy embedding function ───────────────────────────────────────────────────
@pytest.fixture
def dummy_embed():
    """
    Returns a deterministic 384-dim unit vector for any string.
    Avoids ONNX model requirement during unit tests.
    Seeds numpy RNG from hash(text) so same text → same vector.
    """
    def _embed(text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vec = rng.standard_normal(384).astype(np.float32)
        return vec / np.linalg.norm(vec)
    return _embed


# ── Sample node factory ───────────────────────────────────────────────────────
@pytest.fixture
def make_node(dummy_embed):
    """Factory: make_node(content, type='Concept', priority=0.5) → dict"""
    def _make(content: str, node_type: str = "Concept", priority: float = 0.5,
              confidence: float = 0.8) -> dict:
        now = time.time()
        return {
            "id": str(uuid.uuid4()),
            "type": node_type,
            "content": content,
            "embedding": dummy_embed(content),
            "priority": priority,
            "created_at": now,
            "updated_at": now,
            "access_count": 0,
            "confidence": confidence,
            "version": 1,
            "last_reconciled_version": 0,
        }
    return _make


# ── Sample edge factory ───────────────────────────────────────────────────────
@pytest.fixture
def make_edge():
    """Factory: make_edge(from_id, to_id, edge_type, weight) → dict"""
    def _make(from_id: str, to_id: str, edge_type: str = "Semantic",
              weight: float = 0.7) -> dict:
        return {
            "id": str(uuid.uuid4()),
            "from_node": from_id,
            "to_node": to_id,
            "type": edge_type,
            "weight": weight,
            "created_at": time.time(),
        }
    return _make


# ── Small pre-built graph ─────────────────────────────────────────────────────
@pytest.fixture
def small_graph(make_node, make_edge):
    """
    A small 6-node graph with typed edges.
    Topology:
      GPU_constraint --[Causal]--> context_limit
      context_limit  --[Causal]--> compression_needed
      compression_needed --[Dependency]--> graph_dag
      graph_dag      --[Semantic]--> hnsw_index
      user_pref_go   --[Semantic]--> graph_dag
    """
    n_gpu   = make_node("Consumer GPU limits VRAM to 4GB",        "Entity",     priority=0.8)
    n_ctx   = make_node("4K context limit on consumer GPUs",       "Concept",    priority=0.75)
    n_comp  = make_node("Graph compression is required",           "Concept",    priority=0.7)
    n_dag   = make_node("Graph-DAG middleware architecture",        "Concept",    priority=0.9)
    n_hnsw  = make_node("HNSW index for approximate NN search",    "Concept",    priority=0.6)
    n_gopref = make_node("User prefers Go for the final release",  "Preference", priority=0.65)

    nodes = [n_gpu, n_ctx, n_comp, n_dag, n_hnsw, n_gopref]
    edges = [
        make_edge(n_gpu["id"],  n_ctx["id"],   "Causal",      0.9),
        make_edge(n_ctx["id"],  n_comp["id"],  "Causal",      0.85),
        make_edge(n_comp["id"], n_dag["id"],   "Dependency",  0.8),
        make_edge(n_dag["id"],  n_hnsw["id"],  "Semantic",    0.7),
        make_edge(n_gopref["id"], n_dag["id"], "Semantic",    0.6),
    ]
    return {"nodes": nodes, "edges": edges}


# ── Mock Ollama client ────────────────────────────────────────────────────────
@pytest.fixture
def mock_ollama():
    """Mock for the Ollama HTTP client — avoids needing a live Ollama instance."""
    mock = AsyncMock()
    mock.post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "model": "llama3.2:3b",
            "message": {
                "role": "assistant",
                "content": "The Graph-DAG middleware uses HNSW for ANN search."
            },
            "done": True,
        }
    )
    return mock
