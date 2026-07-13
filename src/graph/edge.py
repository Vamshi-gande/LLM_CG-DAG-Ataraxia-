"""
Edge dataclass and EdgeType enum.
Structurally identical to conftest.py — field names and types must not diverge.
"""
import uuid
import time
from dataclasses import dataclass, field
from enum import Enum


class EdgeType(Enum):
    CAUSAL       = "Causal"
    TEMPORAL     = "Temporal"
    SEMANTIC     = "Semantic"
    DEPENDENCY   = "Dependency"
    CONTRADICTS  = "Contradicts"
    HIERARCHICAL = "Hierarchical"
    REINFORCES   = "Reinforces"


@dataclass
class Edge:
    id: str
    from_node: str
    to_node: str
    type: EdgeType
    weight: float
    created_at: float = field(default_factory=time.time)

    @staticmethod
    def new(
        from_node: str,
        to_node: str,
        type: EdgeType,
        weight: float,
    ) -> "Edge":
        return Edge(
            id=str(uuid.uuid4()),
            from_node=from_node,
            to_node=to_node,
            type=type,
            weight=weight,
            created_at=time.time(),
        )