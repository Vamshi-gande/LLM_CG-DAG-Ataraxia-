import pytest
import sqlite3
import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
import numpy as np
import yaml


# ── Enums (duplicated here intentionally — conftest must be self-contained) ──

class NodeType(Enum):
    CONCEPT    = "Concept"
    ENTITY     = "Entity"
    EVENT      = "Event"
    PREFERENCE = "Preference"
    GOAL       = "Goal"
    SUMMARY    = "Summary"

class EdgeType(Enum):
    CAUSAL       = "Causal"
    TEMPORAL     = "Temporal"
    SEMANTIC     = "Semantic"
    DEPENDENCY   = "Dependency"
    CONTRADICTS  = "Contradicts"
    HIERARCHICAL = "Hierarchical"
    REINFORCES   = "Reinforces"


# ── Dataclasses (duplicated here — conftest must be self-contained) ──

@dataclass
class Node:
    id: str
    type: NodeType
    content: str
    embedding: np.ndarray
    priority: float = 0.5
    created_at: float = 0.0
    updated_at: float = 0.0
    access_count: int = 0
    confidence: float = 1.0
    version: int = 1
    last_reconciled_version: int = 0

@dataclass
class Edge:
    id: str
    from_node: str
    to_node: str
    type: EdgeType
    weight: float
    created_at: float = 0.0


# ── Fixtures ──

@pytest.fixture
def config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

@pytest.fixture
def in_memory_db():
    """SQLite in-memory DB with full schema. No file I/O."""
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
        CREATE TABLE meta (
            key TEXT PRIMARY KEY, value TEXT, updated_at REAL
        )
    """)
    conn.commit()
    yield conn
    conn.close()

@pytest.fixture
def dummy_embedder():
    """
    Deterministic fake embedder. Returns normalized 384-dim vectors.
    No ONNX required. Seeded on content hash for reproducibility.
    """
    def embed(text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vec = rng.standard_normal(384).astype(np.float32)
        return vec / np.linalg.norm(vec)
    return embed

@pytest.fixture
def make_node(dummy_embedder):
    """Factory: make_node(content, type) → Node with deterministic embedding."""
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
    """Factory: make_edge(from_id, to_id, type, weight) → Edge."""
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
    6-node causal chain used throughout unit tests:

    Go middleware → targets Ollama → runs on consumer GPU
         → 4GB VRAM → 4K context limit → needs graph compression

    All connected by DEPENDENCY edges in that order.
    """
    n1 = make_node("Go middleware layer", NodeType.CONCEPT)
    n2 = make_node("targets Ollama ecosystem", NodeType.ENTITY)
    n3 = make_node("runs on consumer GPU", NodeType.CONCEPT)
    n4 = make_node("4GB VRAM constraint", NodeType.CONCEPT)
    n5 = make_node("4K token context limit", NodeType.CONCEPT)
    n6 = make_node("requires graph memory compression", NodeType.CONCEPT)

    nodes = [n1, n2, n3, n4, n5, n6]
    chain = [n1, n2, n3, n4, n5, n6]
    edges = [
        make_edge(chain[i].id, chain[i+1].id, EdgeType.DEPENDENCY, 0.9)
        for i in range(len(chain) - 1)
    ]

    return {"nodes": nodes, "edges": edges}

@pytest.fixture
def mock_ollama_client():
    """Fake Ollama client. Returns a canned response without network calls."""
    class MockOllamaClient:
        async def chat(self, model: str, messages: list) -> dict:
            return {
                "model": model,
                "message": {
                    "role": "assistant",
                    "content": "Mock response from Ollama."
                },
                "done": True
            }
    return MockOllamaClient()
