from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from cocktail_server.config import get_settings
from cocktail_server.main import create_app
from cocktail_server.schemas.images import GeneratedImageRef
from cocktail_server.schemas.messages import Message, TextPart
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("IMAGES_DIR", str(tmp_path / "images"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "models"))
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("STARTUP_PRELOAD", "false")
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_get_conversation_returns_detail_with_messages_and_images(
    client: TestClient,
) -> None:
    store = client.app.state.conversations  # type: ignore[attr-defined]
    cid = await store.create()
    msg = Message(
        id="m1",
        conversation_id=cid,
        role="user",
        parts=[TextPart(text="こんにちは")],
        created_at=datetime.now(UTC),
    )
    await store.append(cid, msg)
    await store.record_generated_image(
        cid,
        GeneratedImageRef(
            image_id="11111111-1111-1111-1111-111111111111",
            image_url="/images/11111111-1111-1111-1111-111111111111.webp",
            conversation_id=cid,
            created_at=datetime.now(UTC),
            prompt_excerpt="tag1, tag2",
            seed=42,
            aspect_ratio="portrait",
            cfg_preset="standard",
            width=896,
            height=1152,
        ),
    )

    r = client.get(f"/conversations/{cid}")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["id"] == cid
    assert len(data["messages"]) == 1
    assert data["messages"][0]["id"] == "m1"
    assert len(data["generated_images"]) == 1


def test_get_conversation_returns_404_for_unknown_id(client: TestClient) -> None:
    r = client.get("/conversations/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404


def test_list_conversations_returns_empty_when_none_exist(client: TestClient) -> None:
    r = client.get("/conversations")
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_list_conversations_returns_summaries_sorted_by_updated_at_desc(
    client: TestClient,
) -> None:
    store = client.app.state.conversations  # type: ignore[attr-defined]
    older = await store.create()
    await store.append(
        older,
        Message(
            id="m-old",
            conversation_id=older,
            role="user",
            parts=[TextPart(text="古い会話")],
            created_at=datetime.now(UTC),
        ),
    )
    newer = await store.create()
    await store.append(
        newer,
        Message(
            id="m-new",
            conversation_id=newer,
            role="user",
            parts=[TextPart(text="新しい会話のタイトルになるテキスト")],
            created_at=datetime.now(UTC),
        ),
    )

    r = client.get("/conversations")
    assert r.status_code == 200
    items = r.json()
    ids = [item["id"] for item in items]
    assert ids == [newer, older]
    assert items[0]["title"] == "新しい会話のタイトルになるテキスト"
    assert items[0]["message_count"] == 1
    assert items[1]["title"] == "古い会話"


@pytest.mark.asyncio
async def test_list_conversations_title_truncates_and_falls_back(
    client: TestClient,
) -> None:
    store = client.app.state.conversations  # type: ignore[attr-defined]
    empty = await store.create()
    long_cid = await store.create()
    long_text = "あ" * 80
    await store.append(
        long_cid,
        Message(
            id="m-long",
            conversation_id=long_cid,
            role="user",
            parts=[TextPart(text=long_text)],
            created_at=datetime.now(UTC),
        ),
    )

    r = client.get("/conversations")
    assert r.status_code == 200
    items = {item["id"]: item for item in r.json()}
    assert items[empty]["title"] == "(新規会話)"
    assert items[long_cid]["title"].endswith("…")
    assert len(items[long_cid]["title"]) == 41  # 40 文字 + "…"
