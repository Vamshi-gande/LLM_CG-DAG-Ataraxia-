"""
Query classifier for the pre-resolution engine.

Classifies a raw query string into QueryType.CHAIN, QueryType.SYNTHESIS,
or QueryType.LOOKUP using pure keyword heuristics. No LLM call.

Rules are evaluated in STRICT ORDER — first match wins:
    1. CHAIN indicators
    2. SYNTHESIS indicators
    3. Explicit LOOKUP patterns
    4. CHAIN fallback (default)

The CHAIN fallback is deliberate: chain-reasoning failure is the critical
failure mode for 7B models (see empirical serialization experiment).
Misclassifying a chain query as LOOKUP means it skips pre-resolution
entirely, which is the worse error. A misclassified LOOKUP still gets
*something* useful out of the chain resolver in the worst case.
"""

from enum import Enum


class QueryType(Enum):
    CHAIN = "chain"
    SYNTHESIS = "synthesis"
    LOOKUP = "lookup"


# Order matters only within a step's own list purposes of readability;
# what matters is that CHAIN indicators are checked before SYNTHESIS,
# and SYNTHESIS before explicit LOOKUP.
_CHAIN_INDICATORS = [
    "why",
    "how does",
    "what causes",
    "explain why",
    "because",
    "depend",
    "require",
    "lead to",
    "result in",
    "what happens when",
    "why does",
    "how come",
]

_SYNTHESIS_INDICATORS = [
    "who would",
    "what kind of",
    "compare",
    "benefit",
    "suitable for",
    "use cases",
    "what type of",
    "who should",
]

_LOOKUP_INDICATORS = [
    "what is",
    "which",
    "when did",
    "what language",
    "what version",
]


def classify_query(query: str) -> QueryType:
    """
    Heuristic query classifier. No LLM call.

    Step 1 — CHAIN indicators (any match -> CHAIN immediately)
    Step 2 — SYNTHESIS indicators (any match -> SYNTHESIS)
    Step 3 — Explicit LOOKUP patterns (any match -> LOOKUP)
    Step 4 — CHAIN fallback (no match anywhere above)

    Matching is case-insensitive substring matching against the raw query.
    """
    q = (query or "").lower()

    for indicator in _CHAIN_INDICATORS:
        if indicator in q:
            return QueryType.CHAIN

    for indicator in _SYNTHESIS_INDICATORS:
        if indicator in q:
            return QueryType.SYNTHESIS

    for indicator in _LOOKUP_INDICATORS:
        if indicator in q:
            return QueryType.LOOKUP

    # Step 4 — fallback
    return QueryType.CHAIN