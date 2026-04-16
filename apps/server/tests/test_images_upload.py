from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from cocktail_server.config import get_settings
from cocktail_server.main import create_app
from fastapi.testclient import TestClient
from PIL import Image as PILImage


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("IMAGES_DIR", str(tmp_path / "images"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "models"))
    get_settings.cache_clear()

    app = create_app()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def _png_bytes(w: int = 16, h: int = 16) -> bytes:
    img = PILImage.new("RGB", (w, h), (0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_upload_png_returns_normalized_webp_response(client: TestClient) -> None:
    r = client.post(
        "/images",
        files={"file": ("sample.png", _png_bytes(24, 32), "image/png")},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["mime"] == "image/webp"
    assert data["width"] == 24
    assert data["height"] == 32
    assert data["image_url"].endswith(".webp")
    # 書かれた webp が /images/{id}.webp で配信される
    r2 = client.get(data["image_url"])
    assert r2.status_code == 200
    assert r2.headers["content-type"] == "image/webp"


def test_upload_rejects_unsupported_mime(client: TestClient) -> None:
    r = client.post(
        "/images",
        files={"file": ("x.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 415


def test_upload_rejects_empty_file(client: TestClient) -> None:
    r = client.post(
        "/images",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert r.status_code == 400


def test_upload_rejects_corrupted_bytes(client: TestClient) -> None:
    r = client.post(
        "/images",
        files={"file": ("garbage.png", b"not-really-a-png", "image/png")},
    )
    assert r.status_code == 400
