"""
SQLite persistence layer for Graph-DAG Middleware.

Write strategy: async write queue using (sql, params) tuples.
NEVER use callable lambdas — aiosqlite requires await conn.execute(),
and lambdas silently drop all writes.

Schema validation on every startup via PRAGMA table_info(). Missing
columns on an existing DB file are auto-migrated (ALTER TABLE ADD COLUMN)
before validation runs, so an older data/graph.db (e.g. one created
before the `archived` column existed) self-heals instead of refusing
to start.
"""
from __future__ import annotations

import asyncio
import io
import logging
import sqlite3
import time
from typing import Any, List, Optional, Tuple

import aiosqlite
import numpy as np

from src.graph.node import Node, NodeType
from src.graph.edge import Edge, EdgeType

logger = logging.getLogger(__name__)

_REQUIRED_NODE_COLS = {
    "id", "type", "content", "embedding", "priority",
    "created_at", "updated_at", "access_count", "confidence",
    "version", "last_reconciled_version", "archived",
}
_REQUIRED_EDGE_COLS = {"id", "from_node", "to_node", "type", "weight", "created_at"}
_REQUIRED_META_COLS = {"key", "value", "updated_at"}

# Column -> (SQL type, default clause) used by _migrate_missing_columns_sync()
# when ALTER TABLE ADD COLUMN is needed for a column that's in the required
# set above but absent from an existing table on disk. Only additive,
# nullable-or-defaulted columns belong here — anything requiring a type
# change, rename, or NOT NULL without a default needs a real migration.
_NODE_COL_DEFAULTS = {
    "archived": ("INTEGER", "NOT NULL DEFAULT 0"),
}
_EDGE_COL_DEFAULTS = {}
_META_COL_DEFAULTS = {}


def _ndarray_to_blob(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _blob_to_ndarray(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob))


class SQLiteStorage:
    SCHEMA = """
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            priority REAL NOT NULL DEFAULT 0.5,
            created_at REAL NOT NULL DEFAULT 0.0,
            updated_at REAL NOT NULL DEFAULT 0.0,
            access_count INTEGER NOT NULL DEFAULT 0,
            confidence REAL NOT NULL DEFAULT 1.0,
            version INTEGER NOT NULL DEFAULT 1,
            last_reconciled_version INTEGER NOT NULL DEFAULT 0,
            archived INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS edges (
            id TEXT PRIMARY KEY,
            from_node TEXT NOT NULL,
            to_node TEXT NOT NULL,
            type TEXT NOT NULL,
            weight REAL NOT NULL,
            created_at REAL NOT NULL DEFAULT 0.0
        );
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at REAL NOT NULL DEFAULT 0.0
        );
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._write_queue: asyncio.Queue[Optional[Tuple[str, tuple]]] = asyncio.Queue()
        self._drain_task: Optional[asyncio.Task] = None
        self._init_schema_sync()

    def _init_schema_sync(self) -> None:
        con = sqlite3.connect(self._db_path)
        try:
            con.executescript(self.SCHEMA)          # CREATE TABLE IF NOT EXISTS
            self._migrate_missing_columns_sync(con)  # self-heal existing DB files
            con.commit()
            self._validate_schema_sync(con)
        finally:
            con.close()

    def _migrate_missing_columns_sync(self, con) -> None:
        """
        ALTER TABLE ADD COLUMN, but ONLY for columns explicitly listed in
        the *_COL_DEFAULTS maps (currently just `archived` on `nodes`) —
        known-safe additive migrations with a vetted type and default.

        Deliberately does NOT attempt to auto-heal any other missing
        required column. Guessing a generic TEXT/no-default column for
        something like `priority` or `created_at` would satisfy the
        column-name check in _validate_schema_sync() while silently
        corrupting the type contract (REAL NOT NULL DEFAULT 0.5 -> nullable
        TEXT) that every read/write in this file assumes. Anything missing
        outside this curated set is real schema corruption and must still
        surface as a RuntimeError from _validate_schema_sync(), not be
        papered over here.
        """
        checks = [
            ("nodes", _NODE_COL_DEFAULTS),
            ("edges", _EDGE_COL_DEFAULTS),
            ("meta", _META_COL_DEFAULTS),
        ]
        for table, defaults in checks:
            existing = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
            for col, (col_type, default_clause) in defaults.items():
                if col not in existing:
                    con.execute(
                        f"ALTER TABLE {table} ADD COLUMN {col} {col_type} {default_clause}"
                    )

    def _validate_schema_sync(self, con) -> None:
        checks = [
            ("nodes", _REQUIRED_NODE_COLS),
            ("edges", _REQUIRED_EDGE_COLS),
            ("meta", _REQUIRED_META_COLS),
        ]
        for table, required in checks:
            cols = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
            missing = required - cols
            if missing:
                raise RuntimeError(
                    f"SQLiteStorage: table '{table}' is missing columns: {missing}."
                )

    # ── Read operations ───────────────────────────────────────────────────────

    def load_all_nodes(self) -> List[Node]:
        con = sqlite3.connect(self._db_path)
        nodes = []
        try:
            rows = con.execute(
                "SELECT id, type, content, embedding, priority, created_at, "
                "updated_at, access_count, confidence, version, last_reconciled_version "
                "FROM nodes WHERE archived = 0 OR archived IS NULL"
            ).fetchall()
            for row in rows:
                (nid, ntype, content, emb_blob, priority, created_at,
                 updated_at, access_count, confidence, version, lrv) = row
                nodes.append(Node(
                    id=nid,
                    type=NodeType(ntype),
                    content=content,
                    embedding=_blob_to_ndarray(emb_blob),
                    priority=priority,
                    created_at=created_at,
                    updated_at=updated_at,
                    access_count=access_count,
                    confidence=confidence,
                    version=version,
                    last_reconciled_version=lrv,
                ))
        finally:
            con.close()
        return nodes

    def load_all_edges(self) -> List[Edge]:
        con = sqlite3.connect(self._db_path)
        edges = []
        try:
            rows = con.execute(
                "SELECT id, from_node, to_node, type, weight, created_at FROM edges"
            ).fetchall()
            for row in rows:
                eid, from_node, to_node, etype, weight, created_at = row
                edges.append(Edge(
                    id=eid,
                    from_node=from_node,
                    to_node=to_node,
                    type=EdgeType(etype),
                    weight=weight,
                    created_at=created_at,
                ))
        finally:
            con.close()
        return edges

    def load_meta(self, key: str) -> Optional[str]:
        con = sqlite3.connect(self._db_path)
        try:
            row = con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            return row[0] if row else None
        finally:
            con.close()

    # ── Write queue ───────────────────────────────────────────────────────────

    async def start_write_queue(self) -> None:
        """Start the async drain task. Double-call guard: no-op if already running."""
        if self._drain_task is not None:
            return
        self._drain_task = asyncio.create_task(self._drain_loop())

    async def stop_write_queue(self) -> None:
        if self._drain_task is None:
            return
        await self._write_queue.put(None)
        await self._drain_task
        self._drain_task = None

    async def _drain_loop(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            while True:
                item = await self._write_queue.get()
                if item is None:
                    await db.commit()
                    break
                sql, params = item
                await db.execute(sql, params)
                if self._write_queue.empty():
                    await db.commit()

    def _enqueue(self, sql: str, params: tuple) -> None:
        self._write_queue.put_nowait((sql, params))

    # ── Async write operations ────────────────────────────────────────────────

    async def queue_save_node(self, node: Node) -> None:
        # archived is always written as 0 here: queue_save_node() is only
        # ever called on nodes currently live in memory (M7's eager local
        # update, M1-era creation). An archived node is removed from
        # memory (M8 Engine 4) and would never reach this call — so
        # forcing archived=0 on every INSERT OR REPLACE is the correct
        # invariant, not an oversight. M8's thaw path uses a distinct
        # code path, not this method.
        sql = """
            INSERT OR REPLACE INTO nodes
                (id, type, content, embedding, priority, created_at, updated_at,
                 access_count, confidence, version, last_reconciled_version, archived)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """
        params = (
            node.id, node.type.value, node.content,
            _ndarray_to_blob(node.embedding),
            node.priority, node.created_at, node.updated_at,
            node.access_count, node.confidence,
            node.version, node.last_reconciled_version,
        )
        self._enqueue(sql, params)

    async def queue_save_edge(self, edge: Edge) -> None:
        sql = """
            INSERT OR REPLACE INTO edges (id, from_node, to_node, type, weight, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (edge.id, edge.from_node, edge.to_node, edge.type.value,
                  edge.weight, edge.created_at)
        self._enqueue(sql, params)

    async def queue_save_meta(self, key: str, value: str) -> None:
        self._enqueue(
            "INSERT OR REPLACE INTO meta (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, time.time())
        )

    async def queue_archive_node(self, node_id: str) -> None:
        """
        Mark node as archived (cold storage).
        CRITICAL (M8): All edges must be deleted BEFORE calling this.
        """
        self._enqueue("UPDATE nodes SET archived = 1 WHERE id = ?", (node_id,))

    async def queue_delete_edge(self, edge_id: str) -> None:
        """Queue a hard edge deletion. Used by compression scheduler in M8."""
        self._enqueue("DELETE FROM edges WHERE id = ?", (edge_id,))

    # ── Static blob operations ────────────────────────────────────────────────
    @staticmethod
    def embedding_to_blob(arr: np.ndarray) -> bytes:
        return _ndarray_to_blob(arr)

    @staticmethod
    def blob_to_embedding(blob: bytes) -> np.ndarray:
        return _blob_to_ndarray(blob)