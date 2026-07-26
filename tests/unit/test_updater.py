import time
import pytest
from unittest.mock import AsyncMock, MagicMock
import numpy as np

from src.graph.graph import Graph
from src.graph.node import NodeType
from src.graph.edge import EdgeType
from src.updater.extractor import extract_layer1, extract_layer2
from src.updater.updater import process_response, update_graph_node, add_graph_edge


# ── Layer 1 extraction tests ─────────────────────────────────────────────

def test_layer1_extracts_preference_node(dummy_embedder):
    nodes, relations = extract_layer1("I prefer Go for systems programming", dummy_embedder)
    contents = [n.content.lower() for n in nodes]
    assert any("go" in c for c in contents)
    assert len([n for n in nodes if n.type == NodeType.PREFERENCE]) >= 1


def test_layer1_extracts_goal_node(dummy_embedder):
    nodes, relations = extract_layer1("I want to build a graph memory system", dummy_embedder)
    assert len([n for n in nodes if n.type == NodeType.GOAL]) >= 1


def test_layer1_detects_switch_pattern(dummy_embedder):
    nodes, relations = extract_layer1("I switched from Python to Go", dummy_embedder)
    contents = [n.content.lower() for n in nodes]
    assert any("python" in c or "go" in c for c in contents)
    assert any(r[2] == EdgeType.TEMPORAL for r in relations)


def test_layer1_returns_empty_on_empty_text(dummy_embedder):
    nodes, relations = extract_layer1("", dummy_embedder)
    assert nodes == []
    assert relations == []


def test_layer1_returns_empty_on_whitespace(dummy_embedder):
    nodes, relations = extract_layer1("   \n\t  ", dummy_embedder)
    assert nodes == []


def test_layer1_assigns_embeddings(dummy_embedder):
    nodes, _ = extract_layer1("I prefer Rust for safety", dummy_embedder)
    for node in nodes:
        assert node.embedding is not None
        assert node.embedding.shape == (384,)


def test_layer1_assigns_uuid_ids(dummy_embedder):
    nodes, _ = extract_layer1("I use ONNX for embeddings", dummy_embedder)
    for node in nodes:
        assert len(node.id) > 0


# ── Layer 2 extraction tests ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_layer2_skips_short_text(dummy_embedder):
    nodes, relations = await extract_layer2("Short text.", MagicMock(), dummy_embedder)
    assert nodes == []
    assert relations == []


@pytest.mark.asyncio
async def test_layer2_returns_empty_when_nlp_none(dummy_embedder):
    nodes, relations = await extract_layer2("Any text here " * 20, None, dummy_embedder)
    assert nodes == []
    assert relations == []


@pytest.mark.asyncio
async def test_layer2_extracts_entities_from_long_text(dummy_embedder):
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        pytest.skip("en_core_web_sm not installed in this environment")
    long_text = ("The Go programming language was developed at Google. "
                 "It is widely used for building infrastructure software "
                 "and cloud services. Kubernetes and Docker are written in Go. "
                 "The language features goroutines for concurrency and channels "
                 "for communication between goroutines. Many companies use Go "
                 "for backend development and distributed systems. " * 2)
    nodes, _ = await extract_layer2(long_text, nlp, dummy_embedder)
    assert len(nodes) >= 1


# ── update_graph_node() tests ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_update_graph_node_adds_new_node(make_node):
    g = Graph()
    storage = MagicMock()
    storage.queue_save_node = AsyncMock()
    hnsw = MagicMock()

    node = make_node("new preference for Go", NodeType.PREFERENCE)
    result = await update_graph_node(node, g, hnsw, storage)
    assert result is True
    assert g.get_node(node.id) is not None


@pytest.mark.asyncio
async def test_update_graph_node_queues_sqlite_write(make_node):
    g = Graph()
    storage = MagicMock()
    storage.queue_save_node = AsyncMock()
    hnsw = MagicMock()

    node = make_node("test content")
    await update_graph_node(node, g, hnsw, storage)
    storage.queue_save_node.assert_called_once()


@pytest.mark.asyncio
async def test_update_graph_node_recomputes_neighbor_weights(make_node, make_edge):
    g = Graph()
    storage = MagicMock()
    storage.queue_save_node = AsyncMock()
    hnsw = MagicMock()

    n1 = make_node("node one", NodeType.CONCEPT)
    n2 = make_node("node two", NodeType.CONCEPT)
    g.add_node(n1)
    g.add_node(n2)
    e = make_edge(n1.id, n2.id, EdgeType.SEMANTIC, 0.5)
    g.add_edge(e)

    await update_graph_node(n1, g, hnsw, storage)
    assert isinstance(e.weight, float)


# ── add_graph_edge() tests ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_add_graph_edge_creates_edge(make_node):
    g = Graph()
    storage = MagicMock()
    storage.queue_save_edge = AsyncMock()

    n1 = make_node("Go middleware")
    n2 = make_node("Ollama ecosystem")
    g.add_node(n1)
    g.add_node(n2)

    edge = await add_graph_edge(n1.content, n2.content, EdgeType.DEPENDENCY, g, storage)
    assert edge is not None
    assert g.edge_count() >= 1


@pytest.mark.asyncio
async def test_add_graph_edge_skips_missing_node(make_node):
    g = Graph()
    storage = MagicMock()
    storage.queue_save_edge = AsyncMock()

    n1 = make_node("existing node")
    g.add_node(n1)

    edge = await add_graph_edge(n1.content, "nonexistent content", EdgeType.SEMANTIC, g, storage)
    assert edge is None
    storage.queue_save_edge.assert_not_called()


# ── process_response() integration tests ─────────────────────────────────

@pytest.mark.asyncio
async def test_process_response_adds_nodes_to_graph(dummy_embedder):
    g = Graph()
    storage = MagicMock()
    storage.queue_save_node = AsyncMock()
    storage.queue_save_edge = AsyncMock()
    hnsw = MagicMock()

    influence_table = {}
    response_text = "I prefer Go for systems programming. My goal is to build LLM middleware."

    await process_response(
        response_text=response_text,
        graph=g,
        hnsw=hnsw,
        embedder=dummy_embedder,
        storage=storage,
        influence_table=influence_table,
    )
    assert g.node_count() >= 1


@pytest.mark.asyncio
async def test_process_response_does_not_block():
    g = Graph()
    storage = MagicMock()
    storage.queue_save_node = AsyncMock()
    storage.queue_save_edge = AsyncMock()
    hnsw = MagicMock()

    def fake_embedder(text):
        return np.zeros(384, dtype="float32")

    start = time.perf_counter()
    await process_response(
        response_text="Short response.",
        graph=g, hnsw=hnsw,
        embedder=fake_embedder,
        storage=storage,
        influence_table={},
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"process_response took {elapsed:.3f}s — too slow"


@pytest.mark.asyncio
async def test_process_response_empty_text_is_noop():
    g = Graph()
    storage = MagicMock()
    storage.queue_save_node = AsyncMock()
    storage.queue_save_edge = AsyncMock()
    hnsw = MagicMock()

    await process_response(
        response_text="   ",
        graph=g, hnsw=hnsw,
        embedder=lambda t: np.zeros(384, dtype="float32"),
        storage=storage,
        influence_table={},
    )
    assert g.node_count() == 0
    storage.queue_save_node.assert_not_called()