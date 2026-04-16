import asyncio

import pytest
from cocktail_server.services.model_manager import ModelManager


@pytest.mark.asyncio
async def test_acquire_without_conflict() -> None:
    mgr = ModelManager()
    async with mgr.acquire("llm"):
        assert mgr.active == "llm"
    assert mgr.active == "llm"  # 解放後も active は残る（次の acquire で判定用）


@pytest.mark.asyncio
async def test_role_change_triggers_evictor_of_previous_role() -> None:
    mgr = ModelManager()
    evicted: list[str] = []

    async def evict_llm() -> None:
        evicted.append("llm")

    async def evict_image() -> None:
        evicted.append("image")

    mgr.register_evictor("llm", evict_llm)
    mgr.register_evictor("image", evict_image)

    async with mgr.acquire("llm"):
        pass
    assert evicted == []  # 初回は前役なし

    async with mgr.acquire("image"):
        pass
    assert evicted == ["llm"]  # llm → image の遷移で llm 退避

    async with mgr.acquire("llm"):
        pass
    assert evicted == ["llm", "image"]  # image → llm の遷移で image 退避


@pytest.mark.asyncio
async def test_same_role_does_not_call_evictor() -> None:
    mgr = ModelManager()
    evicted: list[str] = []

    async def evict_llm() -> None:
        evicted.append("llm")

    mgr.register_evictor("llm", evict_llm)

    async with mgr.acquire("llm"):
        pass
    async with mgr.acquire("llm"):
        pass
    assert evicted == []


@pytest.mark.asyncio
async def test_acquire_serializes_concurrent_callers() -> None:
    mgr = ModelManager()
    order: list[str] = []

    async def worker(name: str, delay: float) -> None:
        async with mgr.acquire("llm"):
            order.append(f"{name}:start")
            await asyncio.sleep(delay)
            order.append(f"{name}:end")

    await asyncio.gather(worker("a", 0.02), worker("b", 0.01))

    # 2 つのワーカーが GPU ロックで直列化されていること
    a_start = order.index("a:start")
    a_end = order.index("a:end")
    b_start = order.index("b:start")
    b_end = order.index("b:end")
    assert a_end < b_start or b_end < a_start


def test_status_snapshot_is_a_copy() -> None:
    mgr = ModelManager()
    mgr.set_status("llm", "loaded")
    snap = mgr.snapshot_status()
    snap["llm"] = "error"
    assert mgr.snapshot_status()["llm"] == "loaded"
