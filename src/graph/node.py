"""
Node dataclass and NodeType enum.
Structurally identical to conftest.py — field names and types must not diverge.
"""
import uuid
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class NodeType(Enum):
    CONCEPT    = "Concept"
    ENTITY     = "Entity"
    EVENT      = "Event"
    PREFERENCE = "Preference"
    GOAL       = "Goal"
    SUMMARY    = "Summary"


@dataclass
class Node:
    id: str
    type: NodeType
    content: str
    embedding: np.ndarray
    priority: float = 0.5
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    confidence: float = 1.0
    version: int = 1
    last_reconciled_version: int = 0

    def bump_version(self) -> None:
        """Call whenever content or embedding changes."""
        self.version += 1
        self.updated_at = time.time()

    def touch(self) -> None:
        """Call on every activation during spreading activation."""
        self.access_count += 1

    @staticmethod
    def new(
        type: NodeType,
        content: str,
        embedding: np.ndarray,
        priority: float = 0.5,
    ) -> "Node":
        now = time.time()
        return Node(
            id=str(uuid.uuid4()),
            type=type,
            content=content,
            embedding=embedding,
            priority=priority,
            created_at=now,
            updated_at=now,
        )