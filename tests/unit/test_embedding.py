"""
Unit tests for ONNXEmbedder.
Requires: ONNX model and tokenizer at models/all-MiniLM-L6-v2/
"""
import pytest
import numpy as np
from src.embedding import ONNXEmbedder

MODEL_PATH = "models/all-MiniLM-L6-v2/model.onnx"
TOK_PATH   = "models/all-MiniLM-L6-v2/tokenizer.json"


@pytest.fixture(scope="module")
def embedder():
    return ONNXEmbedder(MODEL_PATH, TOK_PATH)


def test_embed_returns_384_dim(embedder):
    vec = embedder.embed("hello world")
    assert vec.shape == (384,)


def test_embed_returns_float32(embedder):
    vec = embedder.embed("hello world")
    assert vec.dtype == np.float32


def test_embed_is_normalized(embedder):
    vec = embedder.embed("test sentence for normalization")
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_embed_is_deterministic(embedder):
    v1 = embedder.embed("same text")
    v2 = embedder.embed("same text")
    assert np.allclose(v1, v2, atol=1e-6)


def test_similar_texts_have_high_cosine(embedder):
    v1 = embedder.embed("I prefer Go for systems programming")
    v2 = embedder.embed("Go is my preferred language for systems work")
    cosine = float(np.dot(v1, v2))
    assert cosine > 0.70, f"Expected > 0.70, got {cosine:.4f}"


def test_dissimilar_texts_have_low_cosine(embedder):
    v1 = embedder.embed("I prefer Go for systems programming")
    v2 = embedder.embed("The weather is nice today")
    cosine = float(np.dot(v1, v2))
    assert cosine < 0.60, f"Expected < 0.60, got {cosine:.4f}"


# Fix #7 + #8 — ranking-based test: more robust than absolute thresholds
# Passes regardless of model version, quantization, or export variant
def test_similar_pair_scores_higher_than_dissimilar_pair(embedder):
    similar_cosine = float(np.dot(
        embedder.embed("I prefer Go for systems programming"),
        embedder.embed("Go is my preferred language for systems work"),
    ))
    dissimilar_cosine = float(np.dot(
        embedder.embed("I prefer Go for systems programming"),
        embedder.embed("The weather is nice today"),
    ))
    assert similar_cosine > dissimilar_cosine, (
        f"Similar pair ({similar_cosine:.4f}) should score higher than "
        f"dissimilar pair ({dissimilar_cosine:.4f})"
    )


def test_batch_embed_shape(embedder):
    texts = ["first sentence", "second sentence", "third sentence"]
    vecs = embedder.batch_embed(texts)
    assert vecs.shape == (3, 384)


def test_batch_embed_each_row_normalized(embedder):
    texts = ["one", "two", "three"]
    vecs = embedder.batch_embed(texts)
    for i, row in enumerate(vecs):
        assert abs(np.linalg.norm(row) - 1.0) < 1e-5, \
            f"Row {i} not normalized: norm={np.linalg.norm(row)}"


def test_batch_matches_single(embedder):
    text = "batch should match single"
    single = embedder.embed(text)
    batch  = embedder.batch_embed([text])[0]
    assert np.allclose(single, batch, atol=1e-5)


def test_long_text_does_not_crash(embedder):
    long_text = "word " * 500
    vec = embedder.embed(long_text)
    assert vec.shape == (384,)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5