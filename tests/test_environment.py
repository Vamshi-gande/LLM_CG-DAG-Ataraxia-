# tests/test_environment.py
# ── Environment Smoke Tests ───────────────────────────────────────────────────
# Run first: pytest tests/test_environment.py -v
# All tests here are @unit — no Ollama, no disk, no network.
# If any fail the setup_env.sh likely didn't complete cleanly.

import pytest
import numpy as np
import sqlite3
import time
import uuid

pytestmark = pytest.mark.unit


class TestDependencies:
    """Verify all required packages import and basic functionality works."""

    def test_fastapi_import(self):
        import fastapi
        assert fastapi.__version__

    def test_hnswlib_import(self):
        import hnswlib
        # Basic index creation
        index = hnswlib.Index(space="cosine", dim=4)
        index.init_index(max_elements=10, ef_construction=50, M=8)
        vec = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        index.add_items(vec, [0])
        labels, distances = index.knn_query(vec, k=1)
        assert labels[0][0] == 0

    def test_numpy_import(self):
        arr = np.zeros(384, dtype=np.float32)
        assert arr.shape == (384,)

    def test_spacy_import(self):
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("Graph-DAG middleware uses HNSW for ANN search.")
        assert len(list(doc.sents)) >= 1

    def test_onnxruntime_import(self):
        import onnxruntime as ort
        providers = ort.get_available_providers()
        assert "CPUExecutionProvider" in providers

    def test_apscheduler_import(self):
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        sched = AsyncIOScheduler()
        assert sched is not None

    def test_pyyaml_import(self):
        import yaml
        data = yaml.safe_load("key: value\nnested:\n  a: 1")
        assert data["nested"]["a"] == 1

    def test_httpx_import(self):
        import httpx
        assert httpx.__version__


class TestSQLiteSchema:
    """Verify the DB schema is correct and queryable."""

    def test_nodes_table_exists(self, in_memory_db):
        c = in_memory_db.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='nodes'")
        assert c.fetchone() is not None

    def test_edges_table_exists(self, in_memory_db):
        c = in_memory_db.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='edges'")
        assert c.fetchone() is not None

    def test_meta_table_exists(self, in_memory_db):
        c = in_memory_db.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='meta'")
        assert c.fetchone() is not None

    def test_node_insert_and_retrieve(self, in_memory_db, make_node):
        node = make_node("Test concept for GPU memory constraint")
        emb_bytes = node["embedding"].astype(np.float32).tobytes()

        c = in_memory_db.cursor()
        c.execute("""
            INSERT INTO nodes (id, type, content, embedding, priority,
                               created_at, updated_at, access_count,
                               confidence, version, last_reconciled_version)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            node["id"], node["type"], node["content"], emb_bytes,
            node["priority"], node["created_at"], node["updated_at"],
            node["access_count"], node["confidence"],
            node["version"], node["last_reconciled_version"],
        ))
        in_memory_db.commit()

        c.execute("SELECT id, content, priority FROM nodes WHERE id=?", (node["id"],))
        row = c.fetchone()
        assert row is not None
        assert row["content"] == node["content"]
        assert abs(row["priority"] - node["priority"]) < 1e-6

    def test_edge_insert_and_retrieve(self, in_memory_db, make_node, make_edge):
        n1 = make_node("GPU VRAM constraint")
        n2 = make_node("Context window limit")
        emb1 = n1["embedding"].astype(np.float32).tobytes()
        emb2 = n2["embedding"].astype(np.float32).tobytes()

        c = in_memory_db.cursor()
        for n, emb in [(n1, emb1), (n2, emb2)]:
            c.execute("""
                INSERT INTO nodes (id, type, content, embedding, priority,
                                   created_at, updated_at, access_count,
                                   confidence, version, last_reconciled_version)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (n["id"], n["type"], n["content"], emb, n["priority"],
                  n["created_at"], n["updated_at"], n["access_count"],
                  n["confidence"], n["version"], n["last_reconciled_version"]))

        edge = make_edge(n1["id"], n2["id"], "Causal", 0.9)
        c.execute("""
            INSERT INTO edges (id, from_node, to_node, type, weight, created_at)
            VALUES (?,?,?,?,?,?)
        """, (edge["id"], edge["from_node"], edge["to_node"],
              edge["type"], edge["weight"], edge["created_at"]))
        in_memory_db.commit()

        c.execute("SELECT type, weight FROM edges WHERE from_node=?", (n1["id"],))
        row = c.fetchone()
        assert row is not None
        assert row["type"] == "Causal"
        assert abs(row["weight"] - 0.9) < 1e-6


class TestEmbeddingFixture:
    """Verify the dummy embedding fixture works correctly."""

    def test_embedding_shape(self, dummy_embed):
        vec = dummy_embed("test input text")
        assert vec.shape == (384,)
        assert vec.dtype == np.float32

    def test_embedding_is_unit_vector(self, dummy_embed):
        vec = dummy_embed("some random content about graph databases")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_embedding_deterministic(self, dummy_embed):
        vec1 = dummy_embed("identical text produces identical embedding")
        vec2 = dummy_embed("identical text produces identical embedding")
        assert np.allclose(vec1, vec2)

    def test_embedding_different_texts_different_vectors(self, dummy_embed):
        vec1 = dummy_embed("Go programming language")
        vec2 = dummy_embed("Python programming language")
        cosine_sim = float(np.dot(vec1, vec2))
        # Should not be identical
        assert cosine_sim < 0.9999


class TestNodeAndEdgeFixtures:
    """Verify the test fixtures produce valid structures."""

    def test_make_node_has_all_fields(self, make_node):
        node = make_node("User prefers Go for concurrency")
        required = ["id", "type", "content", "embedding", "priority",
                    "created_at", "updated_at", "access_count",
                    "confidence", "version", "last_reconciled_version"]
        for field in required:
            assert field in node, f"Missing field: {field}"

    def test_make_node_default_type(self, make_node):
        node = make_node("some concept")
        assert node["type"] == "Concept"

    def test_make_node_custom_type(self, make_node):
        node = make_node("User prefers ONNX", node_type="Preference")
        assert node["type"] == "Preference"

    def test_make_edge_has_all_fields(self, make_node, make_edge):
        n1 = make_node("source")
        n2 = make_node("target")
        edge = make_edge(n1["id"], n2["id"], "Causal", 0.8)
        required = ["id", "from_node", "to_node", "type", "weight", "created_at"]
        for field in required:
            assert field in edge, f"Missing field: {field}"

    def test_small_graph_structure(self, small_graph):
        assert len(small_graph["nodes"]) == 6
        assert len(small_graph["edges"]) == 5
        # All node types should be valid
        valid_types = {"Concept", "Entity", "Event", "Preference", "Goal", "Summary"}
        for node in small_graph["nodes"]:
            assert node["type"] in valid_types

    def test_small_graph_edge_types(self, small_graph):
        valid_edge_types = {
            "Causal", "Temporal", "Semantic", "Dependency",
            "Contradicts", "Hierarchical", "Reinforces"
        }
        for edge in small_graph["edges"]:
            assert edge["type"] in valid_edge_types

    def test_small_graph_topology(self, small_graph):
        """Verify the causal chain: GPU → ctx_limit → compression → dag exists."""
        edges_by_type = {}
        for e in small_graph["edges"]:
            edges_by_type.setdefault(e["type"], []).append(e)
        assert len(edges_by_type.get("Causal", [])) == 2
        assert len(edges_by_type.get("Dependency", [])) == 1


class TestConfigLoad:
    """Verify config.yaml loads correctly."""

    def test_config_loads(self):
        import yaml, os
        config_path = "config/config.yaml"
        if not os.path.exists(config_path):
            pytest.skip("config.yaml not found — run from project root")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        assert "ollama" in cfg
        assert "graph" in cfg
        assert "embedding" in cfg
        assert "context" in cfg
        assert "compression" in cfg

    def test_token_budget_adds_up(self):
        import yaml, os
        config_path = "config/config.yaml"
        if not os.path.exists(config_path):
            pytest.skip("config.yaml not found")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        ctx = cfg["context"]
        assert ctx["tier1_tokens"] + ctx["tier2_tokens"] == ctx["total_budget_tokens"]

    def test_hardware_profile_present(self):
        import yaml, os
        config_path = "config/config.yaml"
        if not os.path.exists(config_path):
            pytest.skip("config.yaml not found")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        hw = cfg["hardware"]
        assert hw["vram_gb"] == 4       # GTX 1650
        assert hw["ram_gb"] == 8
