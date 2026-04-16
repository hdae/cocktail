from __future__ import annotations

from datetime import UTC, datetime

import pytest
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
