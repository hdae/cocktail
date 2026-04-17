from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Literal

from cocktail_server.schemas.health import ModelStatus

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
        try:
            await self._gpu_lock.acquire()
        finally:
            self._queue_depth -= 1
        try:
            if self._policy == "swap" and self._active is not None and self._active != role:
                evictor = self._evictors.get(self._active)
                if evictor is not None:
                    await evictor()
            self._active = role
            yield
        finally:
            self._gpu_lock.release()
