"""
Milestone 0 smoke tests — 25 tests.
Verify environment, dependencies, schema, fixtures, and config.
No middleware logic tested here.
"""

import os
import sys
import sqlite3
import importlib
import pytest
import numpy as np
import yaml


# ── Group 1: Python version (1 test) ──────────────────────────────────────────

def test_python_version():
    assert sys.version_info >= (3, 11), (
        f"Python 3.11+ required, got {sys.version_info.major}.{sys.version_info.minor}"
    )


# ── Group 2: Core dependencies importable (7 tests) ──────────────────────────

def test_import_fastapi():
    import fastapi
    assert fastapi.__version__ is not None

def test_import_hnswlib():
    import hnswlib
    assert hasattr(hnswlib, "Index")

def test_import_onnxruntime():
    import onnxruntime as ort
    assert hasattr(ort, "InferenceSession")

def test_import_spacy():
    import spacy
    assert spacy.__version__ is not None

def test_import_aiosqlite():
    import aiosqlite
    assert aiosqlite is not None

def test_import_httpx():
    import httpx
    assert httpx.__version__ is not None

def test_import_yaml():
    import yaml
    assert yaml.__version__ is not None


# ── Group 3: Downloaded models exist (3 tests) ────────────────────────────────

def test_onnx_model_file_exists():
    path = os.path.join("models", "all-MiniLM-L6-v2", "model.onnx")
    assert os.path.isfile(path), f"ONNX model not found at {path}"

def test_tokenizer_file_exists():
    path = os.path.join("models", "all-MiniLM-L6-v2", "tokenizer.json")
    assert os.path.isfile(path), f"Tokenizer not found at {path}"

def test_spacy_model_loadable():
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("test sentence")
    assert len(doc) > 0


# ── Group 4: SQLite schema correct (5 tests) ──────────────────────────────────

def test_sqlite_db_file_exists():
    assert os.path.isfile(os.path.join("data", "graph.db"))

def test_sqlite_nodes_table_exists():
    conn = sqlite3.connect(os.path.join("data", "graph.db"))
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    conn.close()
    table_names = [t[0] for t in tables]
    assert "nodes" in table_names

def test_sqlite_edges_table_exists():
    conn = sqlite3.connect(os.path.join("data", "graph.db"))
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    conn.close()
    assert "edges" in [t[0] for t in tables]

def test_sqlite_meta_table_exists():
    conn = sqlite3.connect(os.path.join("data", "graph.db"))
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    conn.close()
    assert "meta" in [t[0] for t in tables]

def test_sqlite_nodes_schema():
    conn = sqlite3.connect(os.path.join("data", "graph.db"))
    cols = [row[1] for row in conn.execute("PRAGMA table_info(nodes)").fetchall()]
    conn.close()
    required = {"id", "type", "content", "embedding", "priority",
                "created_at", "updated_at", "access_count",
                "confidence", "version", "last_reconciled_version"}
    assert required.issubset(set(cols))


# ── Group 5: Config file correct (4 tests) ────────────────────────────────────

def test_config_file_exists():
    assert os.path.isfile(os.path.join("config", "config.yaml"))

def test_config_loads_without_error():
    with open(os.path.join("config", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict)

def test_config_required_top_level_keys():
    with open(os.path.join("config", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    required = {"model", "embedding", "hnsw", "graph", "propagation",
                "priority", "context", "merge", "bypass", "compression"}
    assert required.issubset(set(cfg.keys()))

def test_config_hardware_calibrated():
    with open(os.path.join("config", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    assert cfg["model"]["default"] == "llama3.2:3b"
    assert cfg["embedding"]["device"] == "cpu"
    assert cfg["embedding"]["dimension"] == 384


# ── Group 6: Fixtures work correctly (5 tests) ────────────────────────────────

def test_dummy_embedder_returns_384_dim(dummy_embedder):
    vec = dummy_embedder("hello world")
    assert vec.shape == (384,)

def test_dummy_embedder_returns_normalized(dummy_embedder):
    vec = dummy_embedder("hello world")
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-5

def test_dummy_embedder_is_deterministic(dummy_embedder):
    v1 = dummy_embedder("same text")
    v2 = dummy_embedder("same text")
    assert np.allclose(v1, v2)

def test_small_graph_has_six_nodes(small_graph):
    assert len(small_graph["nodes"]) == 6

def test_small_graph_has_five_edges(small_graph):
    assert len(small_graph["edges"]) == 5
