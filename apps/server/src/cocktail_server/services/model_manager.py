from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Literal

from cocktail_server.schemas.health import ModelStatus

Role = Literal["llm", "image"]
Evictor = Callable[[], Awaitable[None]]


class ModelManager:
    """GPU 上の「今どれが載っているか」を追跡し、役割が変わるときに前のモデルを退避させる。

    16GB VRAM を LLM と Image で取り合うため、同時に GPU 上に載せない。
    `acquire(role)` は GPU ロックを取り、必要に応じて前役の evictor を呼び出す。
    """

    def __init__(self) -> None:
        self._gpu_lock = asyncio.Lock()
        self._active: Role | None = None
        self._status: dict[Role, ModelStatus] = {"llm": "idle", "image": "idle"}
        self._evictors: dict[Role, Evictor] = {}
        self._queue_depth = 0

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

    @asynccontextmanager
    async def acquire(self, role: Role) -> AsyncIterator[None]:
        self._queue_depth += 1
        try:
            await self._gpu_lock.acquire()
        finally:
            self._queue_depth -= 1
        try:
            if self._active is not None and self._active != role:
                evictor = self._evictors.get(self._active)
                if evictor is not None:
                    await evictor()
            self._active = role
            yield
        finally:
            self._gpu_lock.release()
