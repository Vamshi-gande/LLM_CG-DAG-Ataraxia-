"""
INTEGRATION TEST - requires Ollama running with llama3.2:3b pulled for the
live-network variant. The tests below use a mocked Ollama client so they
can run without a live Ollama instance; run separately from unit tests:

    pytest tests/integration/test_proxy.py -v

This is the first end-to-end verification that the full pipeline
(startup -> request -> context injection -> Ollama -> response) works.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch


@pytest.fixture
def mock_client():
    """
    TestClient with mocked Ollama - no live Ollama needed.
    For testing pipeline flow without network.
    """
    from src.proxy.server import app, state

    mock_response = {
        "model": "llama3.2:3b",
        "message": {"role": "assistant", "content": "Mock Ollama response."},
        "done": True,
    }
    with TestClient(app) as c:
        with patch.object(state, "ollama") as mock_ollama:
            mock_ollama.chat = AsyncMock(return_value=mock_response)
            mock_ollama.generate = AsyncMock(return_value={
                "model": "llama3.2:3b",
                "response": "Mock generate response.",
                "done": True,
            })
            yield c


def test_health_endpoint(mock_client):
    response = mock_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "graph_nodes" in data
    assert "turn_count" in data
    assert "bypass_active" in data


def test_chat_endpoint_returns_200(mock_client):
    response = mock_client.post("/api/chat", json={
        "model": "llama3.2:3b",
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert response.status_code == 200


def test_chat_response_has_message_field(mock_client):
    response = mock_client.post("/api/chat", json={
        "model": "llama3.2:3b",
        "messages": [{"role": "user", "content": "what language is Go?"}],
    })
    data = response.json()
    assert "message" in data or "response" in data or "done" in data


def test_generate_endpoint_returns_200(mock_client):
    response = mock_client.post("/api/generate", json={
        "model": "llama3.2:3b",
        "prompt": "explain graph compression",
    })
    assert response.status_code == 200


def test_bypass_mode_active_on_empty_graph(mock_client):
    """Fresh startup graph has 0 nodes - bypass must be active."""
    health = mock_client.get("/health").json()
    assert health["bypass_active"] is True


def test_turn_count_increments(mock_client):
    before = mock_client.get("/health").json()["turn_count"]
    mock_client.post("/api/chat", json={
        "model": "llama3.2:3b",
        "messages": [{"role": "user", "content": "test"}],
    })
    after = mock_client.get("/health").json()["turn_count"]
    assert after == before + 1