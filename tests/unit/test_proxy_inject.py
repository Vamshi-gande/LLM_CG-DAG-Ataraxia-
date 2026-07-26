"""
Unit tests for context injection into messages.
No live Ollama needed.
"""
import pytest
from src.proxy.server import _inject_context


def test_inject_adds_system_message_when_none():
    messages = [{"role": "user", "content": "hello"}]
    result = _inject_context(messages, "context text")
    assert result[0]["role"] == "system"
    assert "context text" in result[0]["content"]
    assert result[1]["role"] == "user"


def test_inject_prepends_to_existing_system_message():
    messages = [
        {"role": "system", "content": "existing system"},
        {"role": "user", "content": "hello"},
    ]
    result = _inject_context(messages, "new context")
    assert result[0]["role"] == "system"
    assert "new context" in result[0]["content"]
    assert "existing system" in result[0]["content"]


def test_inject_does_not_mutate_original():
    messages = [{"role": "user", "content": "hello"}]
    original_len = len(messages)
    _inject_context(messages, "context")
    assert len(messages) == original_len  # original unchanged


def test_inject_empty_messages():
    result = _inject_context([], "context text")
    assert len(result) == 1
    assert result[0]["role"] == "system"
    assert "context text" in result[0]["content"]


def test_inject_preserves_message_order():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    result = _inject_context(messages, "ctx")
    # system prepended, rest preserved in order
    assert result[1]["content"] == "first"
    assert result[2]["content"] == "reply"
    assert result[3]["content"] == "second"