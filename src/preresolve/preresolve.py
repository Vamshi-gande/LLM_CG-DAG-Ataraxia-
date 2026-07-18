"""
Pre-resolution engine — main entry point.

classify_and_preresolve() is the M4 function consumed by M5's DAG
extractor. It classifies the query, dispatches to the appropriate
resolver, and packages the result into a PreResolvedContext.

No LLM call is made anywhere in this module. No graph modifications.
Pure read + compute over the activated subgraph produced by M3's
spreading_activation().
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.graph.graph import Graph
from src.graph.node import Node
from src.preresolve.classify import QueryType, classify_query
from src.preresolve.chain_resolver import resolve_chain
from src.preresolve.synthesis_resolver import resolve_synthesis
from src.preresolve.lookup_resolver import resolve_lookup


@dataclass
class PreResolvedContext:
    """
    Output of the pre-resolution engine. Consumed by M5 (DAG extractor)
    and M6 (serializer).

    query_type:      what kind of query this was
    activated:       the full activated dict from spreading_activation()
                      (same object reference — M5 needs this to build the
                      subgraph)
    resolved_pairs:  for CHAIN — (conclusion, [support]) pairs.
                      Empty list for SYNTHESIS and LOOKUP.
    synthesis_node:  for SYNTHESIS — temporary node (NOT stored in graph).
                      None for CHAIN and LOOKUP, and also None for
                      SYNTHESIS queries where no distant pair was found
                      (M5/M6 must then fall back to using activated nodes
                      directly).
    lookup_nodes:    for LOOKUP — top-N activated nodes by score.
                      Empty list for CHAIN and SYNTHESIS.
    resolved_at:     unix timestamp of resolution.
    """

    query_type: QueryType
    activated: Dict[str, float]
    resolved_pairs: List[Tuple[Node, List[Node]]] = field(default_factory=list)
    synthesis_node: Optional[Node] = None
    lookup_nodes: List[Node] = field(default_factory=list)
    resolved_at: float = field(default_factory=time.time)


def classify_and_preresolve(
    query: str,
    activated: Dict[str, float],
    graph: Graph,
) -> PreResolvedContext:
    """
    Main entry point for M4. Called by M5's DAG extractor.

    Pipeline:
        1. classify_query(query) -> QueryType
        2. Branch:
             CHAIN     -> resolve_chain(activated, graph)
             SYNTHESIS -> resolve_synthesis(activated, graph)
             LOOKUP    -> resolve_lookup(activated, graph)
        3. Package into PreResolvedContext and return

    No LLM call. No graph modifications. Pure read + compute.
    """
    query_type = classify_query(query)

    if query_type == QueryType.CHAIN:
        resolved_pairs = resolve_chain(activated, graph)
        return PreResolvedContext(
            query_type=query_type,
            activated=activated,
            resolved_pairs=resolved_pairs,
        )
    elif query_type == QueryType.SYNTHESIS:
        synthesis_node = resolve_synthesis(activated, graph)
        return PreResolvedContext(
            query_type=query_type,
            activated=activated,
            synthesis_node=synthesis_node,
        )
    else:  # LOOKUP
        lookup_nodes = resolve_lookup(activated, graph)
        return PreResolvedContext(
            query_type=query_type,
            activated=activated,
            lookup_nodes=lookup_nodes,
        )