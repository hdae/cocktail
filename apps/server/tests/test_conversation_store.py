from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from cocktail_server.schemas.images import GeneratedImageRef
from cocktail_server.schemas.messages import Message, TextPart
from cocktail_server.services.conversation_store import ConversationStore


def _msg(conversation_id: str, text: str, mid: str = "m") -> Message:
    return Message(
        id=mid,
        conversation_id=conversation_id,
        role="user",
        parts=[TextPart(text=text)],
        created_at=datetime.now(UTC),
    )


def _ref(
    conversation_id: str,
    *,
    image_id: str,
    created_at: datetime,
    seed: int = 42,
) -> GeneratedImageRef:
    return GeneratedImageRef(
        image_id=image_id,
        image_url=f"/images/{image_id}.webp",
        conversation_id=conversation_id,
        created_at=created_at,
        prompt_excerpt="tag1, tag2, tag3",
        seed=seed,
        aspect_ratio="portrait",
        cfg_preset="standard",
        width=896,
        height=1152,
    )


@pytest.mark.asyncio
async def test_create_returns_unique_ids() -> None:
    s = ConversationStore()
    a = await s.create()
    b = await s.create()
    assert a != b
    assert await s.exists(a)
    assert await s.exists(b)


@pytest.mark.asyncio
async def test_exists_is_false_for_unknown_id() -> None:
    s = ConversationStore()
    assert not await s.exists("00000000-0000-0000-0000-000000000000")


@pytest.mark.asyncio
async def test_append_and_list_preserves_order() -> None:
    s = ConversationStore()
    cid = await s.create()
    await s.append(cid, _msg(cid, "first", mid="m1"))
    await s.append(cid, _msg(cid, "second", mid="m2"))
    msgs = await s.list_messages(cid)
    assert [m.id for m in msgs] == ["m1", "m2"]


@pytest.mark.asyncio
async def test_append_to_unknown_conversation_raises() -> None:
    s = ConversationStore()
    with pytest.raises(KeyError):
        await s.append("no-such", _msg("no-such", "hi"))


@pytest.mark.asyncio
async def test_append_mismatched_conversation_id_raises() -> None:
    s = ConversationStore()
    cid = await s.create()
    with pytest.raises(ValueError, match="conversation_id"):
        await s.append(cid, _msg("different", "hi"))


@pytest.mark.asyncio
async def test_list_returns_snapshot_not_reference() -> None:
    s = ConversationStore()
    cid = await s.create()
    await s.append(cid, _msg(cid, "a", mid="m1"))
    snap = await s.list_messages(cid)
    snap.clear()
    again = await s.list_messages(cid)
    assert len(again) == 1


@pytest.mark.asyncio
async def test_get_last_image_seed_none_before_any_generation() -> None:
    s = ConversationStore()
    cid = await s.create()
    assert await s.get_last_image_seed(cid) is None


@pytest.mark.asyncio
async def test_record_generated_image_updates_last_seed() -> None:
    s = ConversationStore()
    cid = await s.create()
    now = datetime.now(UTC)
    await s.record_generated_image(
        cid,
        _ref(cid, image_id="11111111-1111-1111-1111-111111111111", created_at=now, seed=123),
    )
    assert await s.get_last_image_seed(cid) == 123
    await s.record_generated_image(
        cid,
        _ref(
            cid,
            image_id="22222222-2222-2222-2222-222222222222",
            created_at=now + timedelta(seconds=1),
            seed=456,
        ),
    )
    assert await s.get_last_image_seed(cid) == 456


@pytest.mark.asyncio
async def test_record_generated_image_rejects_mismatched_conversation() -> None:
    s = ConversationStore()
    cid = await s.create()
    now = datetime.now(UTC)
    with pytest.raises(ValueError, match="conversation_id"):
        await s.record_generated_image(
            cid,
            _ref("other", image_id="11111111-1111-1111-1111-111111111111", created_at=now),
        )


@pytest.mark.asyncio
async def test_list_all_generated_images_sorts_descending_across_sessions() -> None:
    s = ConversationStore()
    a = await s.create()
    b = await s.create()
    t0 = datetime(2026, 4, 17, 10, 0, 0, tzinfo=UTC)
    await s.record_generated_image(
        a, _ref(a, image_id="11111111-1111-1111-1111-111111111111", created_at=t0)
    )
    await s.record_generated_image(
        b,
        _ref(
            b,
            image_id="22222222-2222-2222-2222-222222222222",
            created_at=t0 + timedelta(seconds=10),
        ),
    )
    await s.record_generated_image(
        a,
        _ref(
            a,
            image_id="33333333-3333-3333-3333-333333333333",
            created_at=t0 + timedelta(seconds=20),
        ),
    )
    items, next_before = await s.list_all_generated_images(limit=10)
    assert [r.image_id for r in items] == [
        "33333333-3333-3333-3333-333333333333",
        "22222222-2222-2222-2222-222222222222",
        "11111111-1111-1111-1111-111111111111",
    ]
    assert next_before is None


@pytest.mark.asyncio
async def test_list_all_generated_images_paginates_with_before_cursor() -> None:
    s = ConversationStore()
    cid = await s.create()
    t0 = datetime(2026, 4, 17, 12, 0, 0, tzinfo=UTC)
    for i in range(5):
        await s.record_generated_image(
            cid,
            _ref(
                cid,
                image_id=f"{i:08d}-0000-0000-0000-000000000000",
                created_at=t0 + timedelta(seconds=i),
            ),
        )

    first, cursor = await s.list_all_generated_images(limit=2)
    assert len(first) == 2
    assert cursor is not None
    # 降順: 4, 3, (次ページは 2, 1, 0)
    assert first[0].created_at == t0 + timedelta(seconds=4)
    assert first[1].created_at == t0 + timedelta(seconds=3)

    second, cursor2 = await s.list_all_generated_images(limit=2, before=cursor)
    assert len(second) == 2
    assert second[0].created_at == t0 + timedelta(seconds=2)
    assert second[1].created_at == t0 + timedelta(seconds=1)

    third, cursor3 = await s.list_all_generated_images(limit=2, before=cursor2)
    assert len(third) == 1
    assert cursor3 is None


@pytest.mark.asyncio
async def test_list_all_generated_images_empty_store() -> None:
    s = ConversationStore()
    items, cursor = await s.list_all_generated_images(limit=10)
    assert items == []
    assert cursor is None


@pytest.mark.asyncio
async def test_get_session_returns_snapshot_with_messages_and_images() -> None:
    s = ConversationStore()
    cid = await s.create()
    await s.append(cid, _msg(cid, "hello", mid="m1"))
    now = datetime.now(UTC)
    await s.record_generated_image(
        cid, _ref(cid, image_id="11111111-1111-1111-1111-111111111111", created_at=now)
    )
    snap = await s.get_session(cid)
    assert snap.id == cid
    assert [m.id for m in snap.messages] == ["m1"]
    assert [r.image_id for r in snap.generated_images] == ["11111111-1111-1111-1111-111111111111"]
    # 返ってきたリストを変更してもストアに波及しない
    snap.messages.clear()
    again = await s.get_session(cid)
    assert len(again.messages) == 1


@pytest.mark.asyncio
async def test_get_session_raises_for_unknown_id() -> None:
    s = ConversationStore()
    with pytest.raises(KeyError):
        await s.get_session("no-such")


@pytest.mark.asyncio
async def test_subscribe_images_receives_broadcast() -> None:
    s = ConversationStore()
    cid = await s.create()
    async with s.subscribe_images() as queue:
        ref = _ref(
            cid,
            image_id="11111111-1111-1111-1111-111111111111",
            created_at=datetime.now(UTC),
        )
        await s.record_generated_image(cid, ref)
        received = await queue.get()
        assert received.image_id == ref.image_id


@pytest.mark.asyncio
async def test_subscribe_images_supports_multiple_subscribers() -> None:
    s = ConversationStore()
    cid = await s.create()
    async with s.subscribe_images() as q1, s.subscribe_images() as q2:
        ref = _ref(
            cid,
            image_id="22222222-2222-2222-2222-222222222222",
            created_at=datetime.now(UTC),
        )
        await s.record_generated_image(cid, ref)
        r1 = await q1.get()
        r2 = await q2.get()
        assert r1.image_id == r2.image_id == ref.image_id


@pytest.mark.asyncio
async def test_subscribe_images_cleans_up_on_exit() -> None:
    s = ConversationStore()
    cid = await s.create()
    async with s.subscribe_images():
        pass
    # 購読解除後に record してもエラーなく完了することを確認（残留参照が無い）
    await s.record_generated_image(
        cid,
        _ref(
            cid,
            image_id="33333333-3333-3333-3333-333333333333",
            created_at=datetime.now(UTC),
        ),
    )
    assert s._image_subscribers == set()


@pytest.mark.asyncio
async def test_subscribe_images_drops_oldest_on_overflow() -> None:
    """queue は maxsize=32。溢れた時点で古い方を捨て最新を残す。"""
    s = ConversationStore()
    cid = await s.create()
    async with s.subscribe_images() as queue:
        # 33 件連続 broadcast。drain せずに送り続ける。
        t0 = datetime(2026, 4, 18, 0, 0, 0, tzinfo=UTC)
        for i in range(33):
            ref = _ref(
                cid,
                image_id=f"{i:08x}-0000-0000-0000-000000000000",
                created_at=t0 + timedelta(seconds=i),
                seed=i,
            )
            await s.record_generated_image(cid, ref)
        # maxsize 32 を超えず、最新が先頭から 32 件分残る
        assert queue.qsize() == 32
        first = await queue.get()
        # 33 入れて 1 つ捨てたので最古は seed=1（index=0 が落ちた）
        assert first.seed == 1
