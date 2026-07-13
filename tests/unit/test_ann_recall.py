"""
ANN recall quality test.
Verifies that HNSW + real embeddings achieves >= 0.90 recall@10
on a synthetic dataset of 200 nodes.
Requires: ONNX model and tokenizer at models/all-MiniLM-L6-v2/
"""
import pytest
import numpy as np
from src.embedding import ONNXEmbedder
from src.hnsw import HNSWIndex

MODEL_PATH = "models/all-MiniLM-L6-v2/model.onnx"
TOK_PATH   = "models/all-MiniLM-L6-v2/tokenizer.json"

TOPICS = [
    "systems programming", "machine learning", "database design",
    "network protocols", "compiler optimization", "memory management",
    "distributed systems", "graph algorithms", "cryptography",
    "operating systems",
]

@pytest.fixture(scope="module")
def recall_setup():
    embedder = ONNXEmbedder(MODEL_PATH, TOK_PATH)
    index = HNSWIndex(dim=384, M=16, ef_construction=200,
                      ef_search=50, max_elements=500)

    node_ids = []
    embeddings = {}

    for i, topic in enumerate(TOPICS):
        for j in range(20):
            text = f"{topic} concept number {j} in depth"
            nid  = f"{topic.replace(' ', '_')}_{j}"
            emb  = embedder.embed(text)
            index.add(nid, emb)
            node_ids.append(nid)
            embeddings[nid] = emb

    return index, embeddings, node_ids

def test_recall_at_10(recall_setup):
    index, embeddings, node_ids = recall_setup
    hits = 0
    total = len(node_ids)

    for nid in node_ids:
        query_emb = embeddings[nid]
        results = index.search(query_emb, k=10)
        result_ids = [r[0] for r in results]
        if nid in result_ids:
            hits += 1

    recall = hits / total
    assert recall >= 0.90, \
        f"Recall@10 = {recall:.3f}, expected >= 0.90."