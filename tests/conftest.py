import pytest
import sqlite3
import os
import uuid
from typing import List
import numpy as np
import yaml

from src.graph.node import Node, NodeType
from src.graph.edge import Edge, EdgeType


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)

@pytest.fixture
def in_memory_db():
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY, type TEXT, content TEXT, embedding BLOB,
            priority REAL, created_at REAL, updated_at REAL,
            access_count INTEGER, confidence REAL,
            version INTEGER, last_reconciled_version INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE edges (
            id TEXT PRIMARY KEY, from_node TEXT, to_node TEXT,
            type TEXT, weight REAL, created_at REAL
        )
    """)
    conn.execute("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT, updated_at REAL)
    """)
    conn.commit()
    yield conn
    conn.close()

@pytest.fixture
def dummy_embedder():
    """
    Deterministic fake embedder. No ONNX required.
    Returns normalized 384-dim float32 vectors seeded on content hash.
    """
    def embed(text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vec = rng.standard_normal(384).astype(np.float32)
        return vec / np.linalg.norm(vec)
    return embed

@pytest.fixture
def make_node(dummy_embedder):
    def factory(content: str, node_type: NodeType = NodeType.CONCEPT) -> Node:
        return Node(
            id=str(uuid.uuid4()),
            type=node_type,
            content=content,
            embedding=dummy_embedder(content),
            priority=0.5,
            confidence=1.0,
            version=1,
        )
    return factory

@pytest.fixture
def make_edge():
    def factory(from_id: str, to_id: str,
                edge_type: EdgeType = EdgeType.SEMANTIC,
                weight: float = 0.8) -> Edge:
        return Edge(
            id=str(uuid.uuid4()),
            from_node=from_id,
            to_node=to_id,
            type=edge_type,
            weight=weight,
        )
    return factory

@pytest.fixture
def small_graph(make_node, make_edge):
    """
    6-node causal chain:
    Go middleware -> targets Ollama -> runs on consumer GPU
         -> 4GB VRAM -> 4K context limit -> needs graph compression

    Returns: {"nodes": List[Node], "edges": List[Edge]}
    Node IDs are UUIDs — never hardcode "n1", "n2" etc.
    Access via: nodes = small_graph["nodes"], edges = small_graph["edges"]
    Chain traversal: chain[i].id, chain[i+1].id
    """
    n1 = make_node("Go middleware layer", NodeType.CONCEPT)
    n2 = make_node("targets Ollama ecosystem", NodeType.ENTITY)
    n3 = make_node("runs on consumer GPU", NodeType.CONCEPT)
    n4 = make_node("4GB VRAM constraint", NodeType.CONCEPT)
    n5 = make_node("4K token context limit", NodeType.CONCEPT)
    n6 = make_node("requires graph memory compression", NodeType.CONCEPT)
    chain = [n1, n2, n3, n4, n5, n6]
    edges = [
        make_edge(chain[i].id, chain[i+1].id, EdgeType.DEPENDENCY, 0.9)
        for i in range(len(chain) - 1)
    ]
    return {"nodes": chain, "edges": edges}

@pytest.fixture
def mock_ollama_client():
    class MockOllamaClient:
        async def chat(self, model: str, messages: list) -> dict:
            return {
                "model": model,
                "message": {"role": "assistant", "content": "Mock response."},
                "done": True
            }
    return MockOllamaClient()