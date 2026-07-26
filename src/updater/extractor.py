"""
Two-layer knowledge extraction.

Layer 1 (rule-based regex) always runs, target < 5ms, no model inference.
Layer 2 (spaCy) runs only when the response is long enough to be worth
the ~50-100ms cost, and supplements — never duplicates — Layer 1 output.

Embedder convention: embedder is called as a plain callable, embedder(text)
-> np.ndarray[384], matching the dummy_embedder test fixture. See the note
in src/embedding/onnx_embedder.py if wiring against the real ONNXEmbedder.
"""
import re
import time
import uuid
from typing import Any, List, Tuple

from src.graph.node import Node, NodeType
from src.graph.edge import EdgeType

# ── Layer 1 — Rule-based regex extraction ───────────────────────────────

PREFERENCE_PATTERNS = [
    r"[Ii] prefer ([A-Za-z0-9_\-\.]+)",
    r"[Ii] use ([A-Za-z0-9_\-\.]+)",
    r"[Ii] like ([A-Za-z0-9_\-\.]+)",
]

GOAL_PATTERNS = [
    r"[Ii] want to ([^.!?]{5,60})",
    r"[Ii]'m trying to ([^.!?]{5,60})",
    r"[Mm]y goal is ([^.!?]{5,60})",
]

SWITCH_PATTERNS = [
    r"switched from ([A-Za-z0-9_\-\.]+) to ([A-Za-z0-9_\-\.]+)",
    r"moved from ([A-Za-z0-9_\-\.]+) to ([A-Za-z0-9_\-\.]+)",
    r"no longer using ([A-Za-z0-9_\-\.]+)",
]

ENTITY_PATTERN = r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b"


def _make_node(content: str, node_type: NodeType, embedder: Any) -> Node:
    now = time.time()
    return Node(
        id=str(uuid.uuid4()),
        type=node_type,
        content=content.strip(),
        embedding=embedder(content),
        priority=0.5,
        created_at=now,
        updated_at=now,
        access_count=0,
        confidence=1.0,
        version=1,
        last_reconciled_version=0,
    )


def extract_layer1(
    text: str,
    embedder: Any,
) -> Tuple[List[Node], List[Tuple[str, str, EdgeType]]]:
    """
    Rule-based extraction. Returns (nodes, relations) where relations are
    raw (from_content, to_content, EdgeType) tuples resolved to real graph
    nodes later by src.updater.updater.add_graph_edge().

    Never raises on empty/whitespace-only text — returns ([], []).
    """
    if not text or not text.strip():
        return [], []

    nodes: List[Node] = []
    relations: List[Tuple[str, str, EdgeType]] = []
    seen_content_lower: set = set()

    def _add(content: str, node_type: NodeType) -> None:
        key = content.strip().lower()
        if not key or key in seen_content_lower:
            return
        seen_content_lower.add(key)
        nodes.append(_make_node(content, node_type, embedder))

    # 1. Switch/reversal patterns — EVENT node + Temporal edge (old -> new)
    for pattern in SWITCH_PATTERNS:
        for match in re.finditer(pattern, text):
            groups = match.groups()
            if len(groups) == 2:
                old_val, new_val = groups
                event_content = f"switched from {old_val} to {new_val}"
                _add(event_content, NodeType.EVENT)
                _add(old_val, NodeType.CONCEPT)
                _add(new_val, NodeType.CONCEPT)
                relations.append((old_val, new_val, EdgeType.TEMPORAL))
            elif len(groups) == 1:
                _add(f"no longer using {groups[0]}", NodeType.EVENT)
                _add(groups[0], NodeType.CONCEPT)

    # 2. Preference patterns
    for pattern in PREFERENCE_PATTERNS:
        for match in re.finditer(pattern, text):
            _add(match.group(1), NodeType.PREFERENCE)

    # 3. Goal patterns
    for pattern in GOAL_PATTERNS:
        for match in re.finditer(pattern, text):
            _add(match.group(1), NodeType.GOAL)

    # 4. Named entity patterns (only content not already extracted)
    for match in re.finditer(ENTITY_PATTERN, text):
        _add(match.group(1), NodeType.ENTITY)

    return nodes, relations


# ── Layer 2 — spaCy NLP extraction ──────────────────────────────────────

_WORD_COUNT_THRESHOLD = 50


async def extract_layer2(
    text: str,
    nlp: Any,
    embedder: Any,
) -> Tuple[List[Node], List[Tuple[str, str, EdgeType]]]:
    """
    spaCy-based extraction, supplementing Layer 1 with NER, noun-chunk
    concepts, and SVO-triple semantic edges. Only fires when the response
    exceeds _WORD_COUNT_THRESHOLD words and nlp is available. Deduplicates
    against nothing on its own here — callers (process_response) dedupe
    against Layer 1 output by content, case-insensitive.
    """
    if nlp is None:
        return [], []
    if len(text.split()) <= _WORD_COUNT_THRESHOLD:
        return [], []

    nodes: List[Node] = []
    relations: List[Tuple[str, str, EdgeType]] = []
    seen_content_lower: set = set()

    def _add(content: str, node_type: NodeType) -> None:
        key = content.strip().lower()
        if not key or key in seen_content_lower:
            return
        seen_content_lower.add(key)
        nodes.append(_make_node(content, node_type, embedder))

    doc = nlp(text)

    for ent in doc.ents:
        _add(ent.text, NodeType.ENTITY)

    for chunk in doc.noun_chunks:
        _add(chunk.text, NodeType.CONCEPT)

    for token in doc:
        if token.dep_ == "ROOT":
            subject = next((c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")), None)
            obj = next((c for c in token.children if c.dep_ in ("dobj", "attr", "pobj")), None)
            if subject is not None and obj is not None:
                relations.append((subject.text, obj.text, EdgeType.SEMANTIC))

    return nodes, relations