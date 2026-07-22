"""
M5 — Serialization Strategies

Formats a DAG's pre-resolved content into Tier 2 prompt text. Format
selection is empirically derived from the Phase 1 serialization
experiment (architecture doc Section 10, PROJECT_CONTEXT.md Section 16):

    Chain      — format-irrelevant (all formats scored 3/5); pre-resolution
                 already solved the reasoning failure. Flat statements.
    Synthesis  — Briefing/Socratic scored 4/5 vs Flat's 3/5. Briefing
                 document prose is the default.
    Lookup     — format-irrelevant (all formats scored 5/5). Flat
                 current-state statements.

Produces Tier 2 only (~800 tokens). The Tier 1 global summary
(~200 tokens) is prepended separately by M6's context assembler — not
here.
"""
from typing import List, Tuple, Optional, TYPE_CHECKING

from src.graph.node import Node
from src.preresolve.classify import QueryType

if TYPE_CHECKING:
    from src.dag import DAG


REASONING_PRIMER = (
    "[Use the following context to inform your response, "
    "prioritizing earlier information as foundational.]"
)


def serialize_chain(
    resolved_pairs: List[Tuple[Node, List[Node]]],
) -> str:
    """
    Format pre-resolved (conclusion, support) pairs as flat statements.

    Each pair becomes its own [RESOLVED CONTEXT] block:
        [RESOLVED CONTEXT]
        {conclusion.content}
        (support: {support[0].content}; {support[1].content}; ...)

    Returns "" if resolved_pairs is empty — caller (serialize_dag)
    handles the empty case by falling back to the primer alone.
    """
    if not resolved_pairs:
        return ""

    blocks: List[str] = []
    for conclusion, support in resolved_pairs:
        lines = ["[RESOLVED CONTEXT]", conclusion.content]
        if support:
            support_text = "; ".join(s.content for s in support)
            lines.append(f"(support: {support_text})")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def serialize_synthesis(
    synthesis_node: Optional[Node],
    nodes_ordered: List[Node],
) -> str:
    """
    Format synthesis context as a briefing document.

    If synthesis_node is present, it leads as the topic sentence,
    followed by nodes_ordered (excluding the synthesis node itself, in
    case it was somehow included) as supporting detail in topological
    order.

    If synthesis_node is None (no distant pair found by the resolver),
    falls back to nodes_ordered directly with no topic sentence.

    Never calls anything on the graph for synthesis_node — it is a
    temporary node (id prefixed "temp_") that is not stored in the graph.
    """
    lines = ["[RELEVANT CONTEXT]"]

    if synthesis_node is not None:
        lines.append(synthesis_node.content)
        supporting = [n for n in nodes_ordered if n.id != synthesis_node.id]
    else:
        supporting = nodes_ordered

    for node in supporting:
        lines.append(node.content)

    return "\n".join(lines)


def serialize_lookup(
    lookup_nodes: List[Node],
) -> str:
    """
    Format top-N lookup nodes as flat current-state statements.

    Returns "" if lookup_nodes is empty.
    """
    if not lookup_nodes:
        return ""

    lines = ["[CURRENT STATE]"]
    for node in lookup_nodes:
        lines.append(node.content)

    return "\n".join(lines)


def _estimate_tokens(text: str, chars_per_token: float) -> float:
    return len(text) / chars_per_token


def serialize_dag(
    dag: "DAG",
    chars_per_token: float = 3.5,
    tier2_budget: int = 800,
) -> str:
    """
    Main serialization entry point. Routes to the correct strategy by
    dag.query_type, prepends the reasoning primer, and enforces the
    final token budget.

    This is the SECOND trim pass. Pass 1 (trim_to_budget, in DAG
    extraction) removes whole nodes from the DAG. Pass 2 (here) truncates
    the already-serialized text to the final budget.

    Truncation removes characters from the END of the string, never the
    beginning — the primer and highest-priority content sit at the top
    and must survive truncation. If this were reversed, the LLM would see
    dangling detail text with no header or primer at all.
    """
    if dag.query_type == QueryType.CHAIN:
        body = serialize_chain(dag.resolved_pairs)
    elif dag.query_type == QueryType.SYNTHESIS:
        body = serialize_synthesis(dag.synthesis_node, dag.nodes_ordered)
    else:  # QueryType.LOOKUP
        body = serialize_lookup(dag.lookup_nodes)

    if body:
        full_text = f"{REASONING_PRIMER}\n\n{body}"
    else:
        full_text = REASONING_PRIMER

    if _estimate_tokens(full_text, chars_per_token) > tier2_budget:
        max_chars = int(tier2_budget * chars_per_token)
        full_text = full_text[:max_chars]

    return full_text