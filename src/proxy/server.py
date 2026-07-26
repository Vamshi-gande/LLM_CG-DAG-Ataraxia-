import asyncio

import spacy
import yaml
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.graph.graph import Graph
from src.storage.sqlite import SQLiteStorage
from src.embedding.onnx_embedder import ONNXEmbedder
from src.hnsw.index import HNSWIndex
from src.propagation.activation import spreading_activation
from src.preresolve.preresolve import classify_and_preresolve
from src.dag.extractor import extract_dag
from src.context.assembler import ContextAssembler
from src.proxy.ollama_client import OllamaClient
from src.proxy.bypass import should_bypass
from src.updater.updater import process_response


# ── Application state (populated during lifespan startup) ────────────────────

class AppState:
    graph: Graph
    storage: SQLiteStorage
    embedder: ONNXEmbedder
    hnsw: HNSWIndex
    assembler: ContextAssembler
    ollama: OllamaClient
    config: dict
    turn_count: int = 0
    influence_table: dict
    spacy_nlp: object


state = AppState()


# ── Lifespan context manager ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown sequence.

    STARTUP ORDER IS A HARD CONSTRAINT (add_edge raises ValueError if
    endpoint node is not in _nodes - edges MUST load after ALL nodes):

        Step 1:  Load config
        Step 2:  Init SQLiteStorage (validates schema on init)
        Step 3:  Init Graph
        Step 4:  load_all_nodes() -> graph.add_node() for each   NODES FIRST
        Step 5:  load_all_edges() -> graph.add_edge() for each   EDGES AFTER
        Step 6:  load_meta("global_summary") -> assembler.update_global_summary()
        Step 7:  Init ONNXEmbedder
        Step 7b: Init spaCy model (M7 - Layer 2 extraction) + influence
                 table (M7 - lazy reconciliation, populated by the updater,
                 consumed by check_and_reconcile() via spreading_activation())
        Step 8:  Init HNSWIndex, hnsw.rebuild(graph.get_all_nodes())
        Step 9:  graph.set_hnsw(hnsw)
        Step 10: await storage.start_write_queue()
                 NOTE: start_write_queue() MUST have a double-call guard:
                     if self._drain_task is not None: return
                 FastAPI's TestClient can trigger lifespan multiple times.
                 Without this guard a second drain task is created and
                 double-writes occur on every flush cycle.
        Step 11: Init OllamaClient
        Step 12: launch background compression engine tasks (M8 - placeholder)
        Step 13: yield - serve requests

    SHUTDOWN:
        await storage.stop_write_queue()  - final flush guarantee
        await ollama.close()
    """
    with open("config/config.yaml") as f:
        state.config = yaml.safe_load(f)

    cfg = state.config
    embedding_cfg = cfg["embedding"]
    hnsw_cfg = cfg["hnsw"]

    # NOTE: onnx_model_path / tokenizer_path / ollama_base_url / hnsw.space
    # are not documented in config.yaml's section-20 key reference as of
    # M5. Using .get() with sane defaults so startup doesn't crash if
    # they haven't been added to config.yaml yet - add them there
    # (see config_additions.yaml in this delivery) so these fall through
    # to the real configured values instead of the fallback defaults.
    model_path = embedding_cfg.get(
        "onnx_model_path", "models/all-MiniLM-L6-v2/model.onnx"
    )
    tok_path = embedding_cfg.get(
        "tokenizer_path", "models/all-MiniLM-L6-v2/tokenizer.json"
    )
    ollama_base_url = cfg.get("model", {}).get(
        "ollama_base_url", "http://localhost:11434"
    )
    hnsw_space = hnsw_cfg.get("space", "cosine")

    # Storage + Graph
    state.storage = SQLiteStorage("data/graph.db")
    state.graph = Graph()

    for node in state.storage.load_all_nodes():     # NODES FIRST
        state.graph.add_node(node)
    for edge in state.storage.load_all_edges():     # EDGES AFTER ALL NODES
        state.graph.add_edge(edge)

    global_summary = state.storage.load_meta("global_summary") or ""

    # Embedder
    state.embedder = ONNXEmbedder(model_path, tok_path)

    # M7: spaCy model for Layer 2 extraction, influence table for lazy
    # reconciliation (populated by the updater, consumed by
    # check_and_reconcile() via spreading_activation()).
    state.spacy_nlp = spacy.load("en_core_web_sm")
    state.influence_table = {}

    # HNSW
    state.hnsw = HNSWIndex(
        dim=embedding_cfg["dimension"],
        M=hnsw_cfg["M"],
        ef_construction=hnsw_cfg["ef_construction"],
        ef_search=hnsw_cfg["ef_search"],
        space=hnsw_space,
    )
    state.hnsw.rebuild(state.graph.get_all_nodes())
    state.graph.set_hnsw(state.hnsw)

    # Write queue (double-call guard must be in start_write_queue() itself)
    await state.storage.start_write_queue()

    # Context assembler
    state.assembler = ContextAssembler(
        tier1_budget_tokens=cfg["context"]["tier1_budget_tokens"],
        tier2_budget_tokens=cfg["context"]["tier2_budget_tokens"],
        chars_per_token=cfg["context"]["chars_per_token"],
    )
    state.assembler.update_global_summary(global_summary)

    # Ollama client
    state.ollama = OllamaClient(base_url=ollama_base_url)

    # M8: launch compression engine background tasks here
    # (placeholder - not implemented until M8)

    yield  # serve requests

    # Shutdown
    await state.storage.stop_write_queue()
    await state.ollama.close()


# ── FastAPI application ───────────────────────────────────────────────────────

app = FastAPI(title="graph-dag-middleware", lifespan=lifespan)


# ── Request pipeline helpers ──────────────────────────────────────────────────

def _inject_context(messages: list, context_str: str) -> list:
    """
    Inject assembled context into the messages list.
    Prepend as a system message if no system message exists.
    Merge ahead of an existing system message's content if one already
    exists. Returns a new list - does NOT mutate the original.
    """
    if not messages:
        return [{"role": "system", "content": context_str}]

    modified = list(messages)
    if modified[0].get("role") == "system":
        modified[0] = {
            "role": "system",
            "content": context_str + "\n\n" + modified[0]["content"],
        }
    else:
        modified.insert(0, {"role": "system", "content": context_str})
    return modified


async def _run_full_pipeline(query_text: str) -> str:
    """
    Run the complete graph pipeline for a single query.
    Returns assembled context string (Tier 1 + Tier 2).
    """
    cfg = state.config
    query_emb = state.embedder.embed(query_text)

    activated = spreading_activation(
        query_emb, state.graph, state.hnsw,
        seed_k=cfg["propagation"]["seed_k"],
        damping=cfg["propagation"]["damping_factor"],
        hop_limit=cfg["propagation"]["hop_limit"],
        activation_threshold=cfg["propagation"]["activation_threshold"],
        influence_table=state.influence_table,
    )

    if not activated:
        return ""

    context = classify_and_preresolve(query_text, activated, state.graph)

    dag = extract_dag(
        context, state.graph,
        max_candidates=cfg["context"]["max_candidates"],
        token_budget=cfg["context"]["tier2_budget_tokens"],
        chars_per_token=cfg["context"]["chars_per_token"],
    )

    # assemble() calls serialize_dag() internally - do not call it again
    # here (the milestone spec's draft computed a separate `tier2` value
    # via serialize_dag() and then discarded it, doubling the work).
    return state.assembler.assemble(dag)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(request: Request):
    """
    Ollama-compatible /api/chat endpoint.

    In full pipeline mode:
        1. Extract query text from last user message
        2. Run _run_full_pipeline() to get assembled context
        3. Inject context into messages
        4. Forward to Ollama
        5. Return Ollama's response unchanged
        6. Schedule graph update (M7)

    In bypass mode:
        1. Forward to Ollama unchanged
        2. Still schedule graph update (updater runs during bypass)
        3. Increment turn_count
    """
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", state.config["model"]["default"])

    state.turn_count += 1

    # Extract query text from last user message
    query_text = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            query_text = msg.get("content", "")
            break

    if should_bypass(
        state.graph, state.turn_count,
        min_nodes=state.config["bypass"]["min_nodes"],
        min_turns=state.config["bypass"]["min_turns"],
    ):
        # Bypass: forward unchanged, graph update still runs
        response = await state.ollama.chat(model=model, messages=messages)
    else:
        # Full pipeline
        context_str = await _run_full_pipeline(query_text)
        modified_messages = (
            _inject_context(messages, context_str) if context_str else messages
        )
        response = await state.ollama.chat(model=model, messages=modified_messages)

    # M7: graph updater — scheduled as a background task so it never
    # blocks the response path. embedder=state.embedder.embed (bound
    # method, not the ONNXEmbedder instance) since process_response()
    # and its extractors call embedder(text) as a plain callable,
    # matching the dummy_embedder test-fixture convention.
    response_text = response.get("message", {}).get("content", "")
    asyncio.create_task(
        process_response(
            response_text=response_text,
            graph=state.graph,
            hnsw=state.hnsw,
            embedder=state.embedder.embed,
            storage=state.storage,
            influence_table=state.influence_table,
            nlp=state.spacy_nlp,
        )
    )

    return JSONResponse(content=response)


@app.post("/api/generate")
async def generate(request: Request):
    """
    Ollama-compatible /api/generate endpoint.
    Same pipeline as /api/chat but for the generate API style.
    """
    body = await request.json()
    prompt = body.get("prompt", "")
    model = body.get("model", state.config["model"]["default"])

    state.turn_count += 1

    if should_bypass(
        state.graph, state.turn_count,
        min_nodes=state.config["bypass"]["min_nodes"],
        min_turns=state.config["bypass"]["min_turns"],
    ):
        response = await state.ollama.generate(model=model, prompt=prompt)
    else:
        context_str = await _run_full_pipeline(prompt)
        if context_str:
            augmented_prompt = context_str + "\n\n" + prompt
        else:
            augmented_prompt = prompt
        response = await state.ollama.generate(model=model, prompt=augmented_prompt)

    # M7: graph updater — same background-task pattern as /api/chat.
    # /api/generate's response field is "response", not "message.content".
    response_text = response.get("response", "")
    asyncio.create_task(
        process_response(
            response_text=response_text,
            graph=state.graph,
            hnsw=state.hnsw,
            embedder=state.embedder.embed,
            storage=state.storage,
            influence_table=state.influence_table,
            nlp=state.spacy_nlp,
        )
    )

    return JSONResponse(content=response)


@app.get("/health")
async def health():
    """Health check endpoint. Returns graph stats."""
    return {
        "status": "ok",
        "graph_nodes": state.graph.node_count(),
        "turn_count": state.turn_count,
        "bypass_active": should_bypass(
            state.graph, state.turn_count,
            min_nodes=state.config["bypass"]["min_nodes"],
            min_turns=state.config["bypass"]["min_turns"],
        ),
    }