"""
Unit tests for query classifier.
No graph, no embeddings, no ONNX — pure string classification.
"""
import pytest
from src.preresolve.classify import QueryType, classify_query


# ── Chain detection ──────────────────────────────────────────────────────

def test_why_question_is_chain():
    assert classify_query("why does this project need graph compression?") \
           == QueryType.CHAIN

def test_how_does_is_chain():
    assert classify_query("how does the HNSW index work?") == QueryType.CHAIN

def test_what_causes_is_chain():
    assert classify_query("what causes the 4K token limit?") == QueryType.CHAIN

def test_explain_why_is_chain():
    assert classify_query("explain why VRAM is a constraint") == QueryType.CHAIN

def test_require_keyword_is_chain():
    assert classify_query("does middleware require Ollama to be running?") \
           == QueryType.CHAIN

def test_depend_keyword_is_chain():
    assert classify_query("what does DAG extraction depend on?") == QueryType.CHAIN


# ── Synthesis detection ──────────────────────────────────────────────────

def test_who_would_is_synthesis():
    assert classify_query("who would benefit from this project?") \
           == QueryType.SYNTHESIS

def test_compare_is_synthesis():
    assert classify_query("compare this approach to GraphRAG") \
           == QueryType.SYNTHESIS

def test_what_kind_of_is_synthesis():
    assert classify_query("what kind of organizations should use this?") \
           == QueryType.SYNTHESIS


# ── Lookup detection ──────────────────────────────────────────────────────

def test_what_language_is_lookup():
    assert classify_query("what language is this project written in?") \
           == QueryType.LOOKUP

def test_what_is_is_lookup():
    assert classify_query("what is the embedding dimension?") == QueryType.LOOKUP

def test_which_model_is_lookup():
    assert classify_query("which model is used for testing?") == QueryType.LOOKUP


# ── Ambiguous defaults to CHAIN ─────────────────────────────────────────

def test_chain_wins_over_synthesis_when_both_match():
    # "why" (chain) + "benefit" (synthesis) — chain must win (evaluated first)
    assert classify_query(
        "why would organizations benefit from this?"
    ) == QueryType.CHAIN

def test_explicit_lookup_pattern_is_lookup():
    assert classify_query("what is the embedding dimension?") == QueryType.LOOKUP

def test_ambiguous_defaults_to_chain():
    # No clear chain or synthesis signals — falls through to CHAIN fallback
    assert classify_query("tell me about this project") == QueryType.CHAIN

def test_empty_query_defaults_to_chain():
    assert classify_query("") == QueryType.CHAIN

def test_classify_returns_querytype_instance():
    result = classify_query("why does X require Y?")
    assert isinstance(result, QueryType)