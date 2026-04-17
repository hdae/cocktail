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
