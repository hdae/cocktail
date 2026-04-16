from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from cocktail_server.config import get_settings
from cocktail_server.main import create_app
from cocktail_server.schemas.generate import PromptSpec
from fastapi.testclient import TestClient
from PIL import Image as PILImage


class FakeLlm:
    async def build_anima_prompt(self, instruction_ja: str) -> PromptSpec:
        return PromptSpec(
            positive=(
                "score_7, masterpiece, best quality, safe, newest, 1girl, "
                "cat ears, pink hair, starry sky, smile, "
                "a smiling cat-eared girl under a starry sky."
            ),
            negative="worst quality, low quality, score_1, score_2, score_3, artist name",
            rationale="fake rationale",
        )

    def unload(self) -> None:
        return None


class FakeImageGen:
    async def generate(self, **kwargs: Any) -> PILImage.Image:
        return PILImage.new("RGB", (kwargs["width"], kwargs["height"]), (255, 128, 64))

    def unload(self) -> None:
        return None


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("IMAGES_DIR", str(tmp_path / "images"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "models"))
    get_settings.cache_clear()

    app = create_app()
    with TestClient(app) as c:
        # lifespan 初期化後に、実モデルを呼ばない fake に差し替える
        app.state.llm = FakeLlm()
        app.state.image_gen = FakeImageGen()
        yield c
    get_settings.cache_clear()


def test_health_returns_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["models"]["llm"] in {"idle", "loading", "loaded", "error"}
    assert data["models"]["image"] in {"idle", "loading", "loaded", "error"}
    assert data["queue_depth"] == 0


def test_generate_returns_expected_shape(client: TestClient, tmp_path: Path) -> None:
    r = client.post(
        "/generate",
        json={"instruction_ja": "ピンクの髪の猫耳少女が星空の下で微笑んでいる絵"},
    )
    assert r.status_code == 200, r.text

    data = r.json()
    assert data["prompt"].startswith("score_7, masterpiece, best quality, safe,")
    assert data["negative_prompt"] == (
        "worst quality, low quality, score_1, score_2, score_3, artist name"
    )
    assert data["image_url"].startswith("/images/")
    assert data["image_url"].endswith(".webp")
    assert data["params"]["width"] == 896
    assert data["params"]["height"] == 1152
    assert data["params"]["steps"] == 32
    assert data["params"]["cfg"] == 4.0

    # webp が実際に保存されていて、GET で取れること
    image_id = data["image_id"]
    r2 = client.get(f"/images/{image_id}.webp")
    assert r2.status_code == 200
    assert r2.headers["content-type"] == "image/webp"


def test_generate_rejects_bad_image_id() -> None:
    app = create_app()
    with TestClient(app) as c:
        r = c.get("/images/not-a-uuid.webp")
        assert r.status_code == 400


def test_generate_rejects_missing_image_id() -> None:
    app = create_app()
    with TestClient(app) as c:
        r = c.get("/images/00000000-0000-0000-0000-000000000000.webp")
        assert r.status_code == 404


def test_generate_rejects_empty_instruction(client: TestClient) -> None:
    r = client.post("/generate", json={"instruction_ja": ""})
    assert r.status_code == 422
