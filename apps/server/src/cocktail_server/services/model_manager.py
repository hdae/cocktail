from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from time import perf_counter_ns
from typing import Literal

from cocktail_server.schemas.health import ModelStatus

logger = logging.getLogger(__name__)

Role = Literal["llm", "image"]
Policy = Literal["swap", "coresident"]
Evictor = Callable[[], Awaitable[None]]


class ModelManager:
    """GPU 上の「今どれが載っているか」を追跡するシリアライザ。

    `policy="swap"` は 16GB VRAM 向け: 役割遷移のたびに前役の evictor を呼んで
    GPU を空ける。`policy="coresident"` は 24GB+ 向けで、両モデルの常駐を許す
    ため evictor は呼ばない。`asyncio.Lock` による直列化はどちらのモードでも維持。
    """

    def __init__(self, policy: Policy = "swap") -> None:
        self._gpu_lock = asyncio.Lock()
        self._active: Role | None = None
        self._status: dict[Role, ModelStatus] = {"llm": "idle", "image": "idle"}
        self._evictors: dict[Role, Evictor] = {}
        self._queue_depth = 0
        self._policy: Policy = policy

    def register_evictor(self, role: Role, fn: Evictor) -> None:
        self._evictors[role] = fn

    def set_status(self, role: Role, status: ModelStatus) -> None:
        self._status[role] = status

    def snapshot_status(self) -> dict[Role, ModelStatus]:
        return dict(self._status)

    @property
    def active(self) -> Role | None:
        return self._active

    @property
    def queue_depth(self) -> int:
        return self._queue_depth

    @property
    def policy(self) -> Policy:
        return self._policy

    @asynccontextmanager
    async def acquire(self, role: Role) -> AsyncIterator[None]:
        self._queue_depth += 1
        wait_start = perf_counter_ns()
        try:
            await self._gpu_lock.acquire()
        finally:
            self._queue_depth -= 1
        wait_ms = (perf_counter_ns() - wait_start) / 1_000_000
        try:
            evict_ms = 0.0
            prev_active = self._active
            if self._policy == "swap" and prev_active is not None and prev_active != role:
                evictor = self._evictors.get(prev_active)
                if evictor is not None:
                    evict_start = perf_counter_ns()
                    await evictor()
                    evict_ms = (perf_counter_ns() - evict_start) / 1_000_000
                    logger.info(
                        "swap %s->%s: evict=%.0f ms, wait=%.0f ms",
                        prev_active,
                        role,
                        evict_ms,
                        wait_ms,
                    )
            self._active = role
            yield
        finally:
            self._gpu_lock.release()
