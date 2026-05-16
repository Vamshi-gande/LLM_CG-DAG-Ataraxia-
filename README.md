# Graph-DAG Inference Middleware
### Model-agnostic long-horizon memory for locally hosted LLMs

---

## What This Is

A transparent proxy layer that sits between any Ollama-compatible application and a locally hosted LLM. It intercepts every query, injects semantically structured context from a persistent graph memory, and updates the graph from every response — giving local models effective long-horizon memory across sessions.

- **Fully on-premise** — no data leaves the machine
- **Consumer GPU friendly** — keeps effective context to ~1000 tokens regardless of session length
- **Model agnostic** — works with any LLM behind an Ollama-compatible API
- **Zero client changes** — just point your app at `localhost:8080` instead of `localhost:11434`

---

## Hardware

### Test Machine
| Component | Spec |
|---|---|
| CPU | Ryzen 5 4600H |
| RAM | 8 GB |
| GPU | GTX 1650 (4 GB VRAM) |

### Supported LLMs for 4GB VRAM
| Model | VRAM Usage | Status |
|---|---|---|
| `llama3.2:3b` Q4_K_M | ~2.0 GB | ✅ Recommended |
| `phi3:mini` Q4_K_M | ~2.2 GB | ✅ Good alternative |
| `mistral:7b` Q4_K_M | ~4.1 GB | ❌ OOM on 4GB |
| `llama3.1:8b` Q4_K_M | ~4.9 GB | ❌ OOM on 4GB |

The middleware itself (embeddings, graph, HNSW index) runs entirely on CPU and uses ~210 MB of system RAM at 100K nodes.

---

## Project Phases

| Phase | Description | Status |
|---|---|---|
| 1 | Architecture Design | ✅ Complete |
| 2 | Python PoC + Paper | 🔄 In Progress |
| 3 | Benchmarks | ⬜ Pending |
| 4 | Paper (NeurIPS/ICLR/ACL workshop or arXiv) | ⬜ Pending |
| 5 | Go open-source release | ⬜ Pending |

---

## Quickstart

```bash
# 1. Clone and enter project
cd graph-dag-middleware

# 2. One-shot environment setup
bash scripts/setup_env.sh

# 3. Activate environment
source .venv/bin/activate

# 4. Pull recommended LLM (first time)
ollama pull llama3.2:3b

# 5. Start Ollama
ollama serve

# 6. Start middleware proxy (in a new terminal)
python3 -m src.proxy.server

# 7. Point your app at the middleware instead of Ollama directly
# Before: http://localhost:11434
# After:  http://localhost:8080
```

---

## Project Structure

```
graph-dag-middleware/
├── src/
│   ├── proxy/          # FastAPI server, Ollama-compatible /api/chat + /api/generate
│   ├── graph/          # Graph engine: nodes, edges, adjacency lists
│   ├── storage/        # SQLite persistence + async write queue
│   ├── embedding/      # ONNX wrapper for all-MiniLM-L6-v2 (384-dim, CPU)
│   ├── hnsw/           # hnswlib wrapper + index management
│   ├── propagation/    # Spreading activation engine (priority propagation)
│   ├── preresolve/     # Pre-resolution engine (chain walk, synthesis)
│   ├── dag/            # DAG extractor, cycle breaking, topological sort
│   ├── context/        # Two-tier context assembler (Tier1 global + Tier2 dynamic)
│   ├── serialize/      # Serialization strategies: briefing, flat, lookup
│   ├── updater/        # Post-response knowledge extraction + graph update
│   ├── influence/      # Influence table: lazy indirect propagation
│   └── compression/    # 5-engine background compression scheduler
├── tests/
│   ├── unit/           # Fast tests, no IO, no Ollama
│   ├── integration/    # Requires Ollama at localhost:11434
│   ├── benchmarks/     # Multi-session recall, associative retrieval, belief update
│   └── fixtures/       # Seed graphs for reproducible tests
├── config/
│   └── config.yaml     # All tunable parameters, hardware profile
├── data/
│   ├── db/graph.db     # SQLite persistent graph (source of truth)
│   └── models/         # ONNX embedding model + tokenizer
└── scripts/
    └── setup_env.sh    # One-shot environment setup
```

---

## Running Tests

```bash
# All unit tests (no Ollama needed)
pytest tests/ -m unit -v

# Just the environment smoke tests (run this first)
pytest tests/test_environment.py -v

# Integration tests (Ollama must be running)
pytest tests/ -m integration -v

# Benchmarks (slow, run explicitly)
pytest tests/ -m benchmark -v

# Coverage report
pytest tests/ -m unit --cov=src --cov-report=html
# Open data/coverage/index.html
```

---

## Architecture Reference

See `docs/graph_dag_middleware_architecture.md` for the full Phase 1 specification including:
- Graph data model (Node types, Edge types)
- Query-time pipeline with complexity analysis
- Pre-resolution engine design (empirically derived)
- 5-engine compression scheduler
- Empirical serialization experiment results
- Differentiation from GraphRAG, RAG, MemGPT
