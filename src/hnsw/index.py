"""
hnswlib wrapper with string NodeID <-> int label mapping.

hnswlib requires integer labels internally, not strings.
We maintain:
    _str_to_int: Dict[str, int]   NodeID -> hnswlib label
    _int_to_str: Dict[int, str]   hnswlib label -> NodeID

Labels are assigned sequentially and never reused.
mark_deleted() does NOT free memory; deleted_count is tracked for rebuild trigger.

Known behaviour (documented, not a bug):
    Each update() consumes one new label and increments _deleted_count.
    Over many updates to a small live set, _next_label grows much faster
    than live node count. This triggers _resize() more often and increases
    memory use. The M8 compression scheduler's rebuild trigger
    (deleted_count > 10% of max_elements) bounds this in production.
    No action required before M8.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import hnswlib
import numpy as np


class HNSWIndex:
    def __init__(
        self,
        dim: int,
        M: int,
        ef_construction: int,
        ef_search: int,
        space: str = "cosine",
        max_elements: int = 100_000,
    ) -> None:
        self._dim = dim
        self._space = space
        self._max_elements = max_elements

        # Fix #3: store constructor params so rebuild() uses the same values
        self._M = M
        self._ef_construction = ef_construction
        self._ef_search = ef_search

        self._index = hnswlib.Index(space=space, dim=dim)
        self._index.init_index(
            max_elements=max_elements,
            M=M,
            ef_construction=ef_construction,
            random_seed=42,
        )
        self._index.set_ef(ef_search)

        self._str_to_int: Dict[str, int] = {}
        self._int_to_str: Dict[int, str] = {}
        self._next_label: int = 0
        self._deleted_count: int = 0

    def add(self, node_id: str, embedding: np.ndarray) -> None:
        """Add a node. Raises ValueError if node_id already present or embedding shape wrong."""
        # Fix #11: validate embedding dimension before passing to hnswlib
        if embedding.shape != (self._dim,):
            raise ValueError(
                f"Expected embedding of shape ({self._dim},), got {embedding.shape}."
            )
        if node_id in self._str_to_int:
            raise ValueError(
                f"node_id {node_id!r} already in index. Use update() to replace."
            )
        if self._next_label >= self._max_elements:
            self._resize()

        label = self._next_label
        self._next_label += 1
        self._str_to_int[node_id] = label
        self._int_to_str[label] = node_id
        self._index.add_items(
            embedding.reshape(1, -1).astype(np.float32), [label]
        )

    def update(self, node_id: str, new_embedding: np.ndarray) -> None:
        """
        Update embedding for existing node (mark_deleted + re-add).

        Note: each call increments _next_label and _deleted_count by 1.
        Over many updates, label space grows faster than live node count.
        Bounded in production by the M8 rebuild trigger.
        """
        # Fix #11: validate embedding dimension
        if new_embedding.shape != (self._dim,):
            raise ValueError(
                f"Expected embedding of shape ({self._dim},), got {new_embedding.shape}."
            )
        if node_id not in self._str_to_int:
            raise ValueError(f"node_id {node_id!r} not in index. Use add().")

        old_label = self._str_to_int[node_id]
        self._index.mark_deleted(old_label)
        del self._int_to_str[old_label]
        del self._str_to_int[node_id]
        self._deleted_count += 1

        if self._next_label >= self._max_elements:
            self._resize()

        new_label = self._next_label
        self._next_label += 1
        self._str_to_int[node_id] = new_label
        self._int_to_str[new_label] = node_id
        self._index.add_items(
            new_embedding.reshape(1, -1).astype(np.float32), [new_label]
        )

    def remove(self, node_id: str) -> None:
        """Mark node as deleted. Does NOT free memory."""
        if node_id not in self._str_to_int:
            return
        label = self._str_to_int.pop(node_id)
        self._int_to_str.pop(label, None)
        self._index.mark_deleted(label)
        self._deleted_count += 1

    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """Return top-k (node_id, distance) pairs."""
        live_count = len(self._str_to_int)
        if live_count == 0:
            return []
        k_actual = min(k, live_count)
        labels, distances = self._index.knn_query(
            query.reshape(1, -1).astype(np.float32), k=k_actual
        )
        results = []
        for label, dist in zip(labels[0], distances[0]):
            node_id = self._int_to_str.get(label)
            if node_id is not None:
                results.append((node_id, float(dist)))
        return results

    def contains(self, node_id: str) -> bool:
        return node_id in self._str_to_int

    def size(self) -> int:
        return len(self._str_to_int)

    def deleted_count(self) -> int:
        return self._deleted_count

    def rebuild(self, nodes: list) -> None:
        """
        Full rebuild from a list of Node objects.
        Uses the same M/ef_construction/ef_search as the original constructor.
        Used at startup and after bulk deletions (M8 trigger).
        """
        # Fix #3: use stored constructor params, not hardcoded defaults
        self._index = hnswlib.Index(space=self._space, dim=self._dim)
        self._index.init_index(
            max_elements=max(self._max_elements, len(nodes) + 1000),
            M=self._M,
            ef_construction=self._ef_construction,
            random_seed=42,
        )
        self._index.set_ef(self._ef_search)
        self._str_to_int = {}
        self._int_to_str = {}
        self._next_label = 0
        self._deleted_count = 0

        for node in nodes:
            label = self._next_label
            self._next_label += 1
            self._str_to_int[node.id] = label
            self._int_to_str[label] = node.id
            self._index.add_items(
                node.embedding.reshape(1, -1).astype(np.float32), [label]
            )

    def _resize(self) -> None:
        """Double max_elements when capacity is reached."""
        new_max = self._max_elements * 2
        self._index.resize_index(new_max)
        self._max_elements = new_max