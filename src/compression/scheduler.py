"""
CompScheduler — manages the four background compression engine Tasks.

Engine 1 (Priority Decay) is NOT launched here — it runs inline inside
spreading_activation() on every query (M3), not as a background task.
"""

from __future__ import annotations

import asyncio
from typing import Any, List

from src.compression.engines import (
    run_engine2_semantic_merge,
    run_engine3_hierarchical_abstraction,
    run_engine4_temporal_compression,
    run_engine5_global_summary,
)


class CompScheduler:
    """
    Launches Engines 2-5 as asyncio background tasks and cancels them
    cleanly on shutdown. Never blocks the inference path.
    """

    def __init__(self) -> None:
        self._tasks: List[asyncio.Task] = []

    def start(
        self,
        graph: Any,
        hnsw: Any,
        storage: Any,
        embedder: Any,
        ollama: Any,
        assembler: Any,
        config: dict,
    ) -> None:
        """
        Launch all four compression engines as asyncio Tasks. Call this
        from the FastAPI lifespan, after storage.start_write_queue().
        """
        cfg = config.get("compression", {})

        self._tasks.append(asyncio.create_task(
            run_engine2_semantic_merge(
                graph, hnsw, storage,
                interval_seconds=cfg.get("semantic_merge_interval_min", 5) * 60,
                similarity_threshold=config["merge"]["similarity_threshold"],
                min_age_hours=config["merge"]["min_age_hours"],
                min_access_count=config["merge"]["min_access_count"],
            ),
            name="engine2_semantic_merge",
        ))

        self._tasks.append(asyncio.create_task(
            run_engine3_hierarchical_abstraction(
                graph, hnsw, storage, embedder,
                interval_seconds=cfg.get("hierarchical_interval_min", 15) * 60,
            ),
            name="engine3_hierarchical",
        ))

        self._tasks.append(asyncio.create_task(
            run_engine4_temporal_compression(
                graph, hnsw, storage, embedder,
                interval_seconds=cfg.get("temporal_compress_interval_min", 30) * 60,
            ),
            name="engine4_temporal",
        ))

        self._tasks.append(asyncio.create_task(
            run_engine5_global_summary(
                graph, storage, ollama, assembler,
                interval_seconds=cfg.get("global_summary_interval_min", 60) * 60,
                update_threshold_ratio=cfg.get("global_summary_update_threshold", 0.20),
                model=config["model"]["default"],
                min_nodes_before_generation=config["summary"]["min_nodes_before_generation"],
            ),
            name="engine5_global_summary",
        ))

    async def stop(self) -> None:
        """
        Cancel all engine tasks cleanly and await their cancellation so no
        coroutine is left dangling at shutdown. Called from the FastAPI
        lifespan's shutdown path, after storage.stop_write_queue().
        """
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()