from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cocktail_server.config import get_settings
from cocktail_server.main import create_app
from cocktail_server.schemas.images import GeneratedImageRef
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
        prompt_excerpt="score_7, masterpiece, safe, 1girl",
        seed=seed,
        aspect_ratio="portrait",
        cfg_preset="standard",
        width=896,
        height=1152,
    )


def test_list_generated_images_empty_returns_empty_list(client: TestClient) -> None:
    r = client.get("/api/images")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data == {"images": [], "next_before": None}


@pytest.mark.asyncio
async def test_list_generated_images_returns_descending_across_sessions(
    client: TestClient,
) -> None:
    store = client.app.state.conversations  # type: ignore[attr-defined]
    a = await store.create()
    b = await store.create()
    t0 = datetime(2026, 4, 17, 9, 0, 0, tzinfo=UTC)
    await store.record_generated_image(
        a, _ref(a, image_id="11111111-1111-1111-1111-111111111111", created_at=t0)
    )
    await store.record_generated_image(
        b,
        _ref(
            b,
            image_id="22222222-2222-2222-2222-222222222222",
            created_at=t0 + timedelta(seconds=5),
        ),
    )
    r = client.get("/api/images?limit=10")
    assert r.status_code == 200
    data = r.json()
    ids = [img["image_id"] for img in data["images"]]
    assert ids == [
        "22222222-2222-2222-2222-222222222222",
        "11111111-1111-1111-1111-111111111111",
    ]
    assert data["next_before"] is None


@pytest.mark.asyncio
async def test_list_generated_images_pagination_with_before(client: TestClient) -> None:
    store = client.app.state.conversations  # type: ignore[attr-defined]
    cid = await store.create()
    t0 = datetime(2026, 4, 17, 9, 0, 0, tzinfo=UTC)
    for i in range(4):
        await store.record_generated_image(
            cid,
            _ref(
                cid,
                image_id=f"{i:08d}-0000-0000-0000-000000000000",
                created_at=t0 + timedelta(seconds=i),
            ),
        )

    r = client.get("/api/images?limit=2")
    data = r.json()
    assert [img["image_id"] for img in data["images"]] == [
        "00000003-0000-0000-0000-000000000000",
        "00000002-0000-0000-0000-000000000000",
    ]
    assert data["next_before"] is not None

    r2 = client.get(f"/api/images?limit=2&before={data['next_before']}")
    assert r2.status_code == 200
    data2 = r2.json()
    assert [img["image_id"] for img in data2["images"]] == [
        "00000001-0000-0000-0000-000000000000",
        "00000000-0000-0000-0000-000000000000",
    ]
    assert data2["next_before"] is None


def test_list_generated_images_rejects_invalid_limit(client: TestClient) -> None:
    r = client.get("/api/images?limit=0")
    assert r.status_code == 422
    r = client.get("/api/images?limit=500")
    assert r.status_code == 422
