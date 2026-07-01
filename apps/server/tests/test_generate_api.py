from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from cocktail_server.config import get_settings
from cocktail_server.main import create_app
from cocktail_server.schemas.generate import GenerateImageCall
from cocktail_server.schemas.messages import Message
from cocktail_server.services.llm import LlmStreamChunk, LlmTurnComplete, LlmTurnResult
from cocktail_server.services.prompt_builder import NEGATIVE_DEFAULT
from fastapi.testclient import TestClient
from PIL import Image as PILImage

_TURBO_URN = "urn:air:anima:lora:civitai:2560840@2979642"


class FakeLlm:
    async def run_turn(self, history: list[Message]) -> AsyncIterator[LlmStreamChunk]:
        result = LlmTurnResult(
            text="テスト応答",
            thought="",
            tool_calls=[
                GenerateImageCall(
                    positive=(
                        "score_7, masterpiece, best quality, safe, newest, 1girl, "
                        "cat ears, pink hair, starry sky, smile, "
                        "a smiling cat-eared girl under a starry sky."
                    ),
                    aspect_ratio="portrait",
                    seed_action="new",
                )
            ],
        )
        yield LlmTurnComplete(result=result)

    def unload(self) -> None:
        return None


class FakeImageGen:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> PILImage.Image:
        self.calls.append(kwargs)
        return PILImage.new("RGB", (kwargs["width"], kwargs["height"]), (255, 128, 64))

    def unload(self) -> None:
        return None


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("IMAGES_DIR", str(tmp_path / "images"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "models"))
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("STARTUP_PRELOAD", "false")
    get_settings.cache_clear()

    app = create_app()
    with TestClient(app) as c:
        app.state.llm = FakeLlm()
        app.state.image_gen = FakeImageGen()
        yield c
    get_settings.cache_clear()


def test_health_returns_ok(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["startup"]["state"] == "ready"
    assert data["startup"]["error"] is None
    assert data["models"]["llm"] in {"idle", "loading", "loaded", "error"}
    assert data["models"]["image"] in {"idle", "loading", "loaded", "error"}
    assert data["queue_depth"] == 0


def test_generate_returns_expected_shape(client: TestClient, tmp_path: Path) -> None:
    r = client.post(
        "/api/generate",
        json={"instruction_ja": "ピンクの髪の猫耳少女が星空の下で微笑んでいる絵"},
    )
    assert r.status_code == 200, r.text

    data = r.json()
    assert data["prompt"].startswith("score_7, masterpiece, best quality, safe,")
    assert data["negative_prompt"] == NEGATIVE_DEFAULT
    assert data["image_url"].startswith("/api/images/")
    assert data["image_url"].endswith(".webp")
    # aspect_ratio=portrait → 896x1152 が初期値
    assert data["params"]["width"] == 896
    assert data["params"]["height"] == 1152
    assert data["params"]["steps"] == 32
    assert data["params"]["cfg"] == 4.0
    # seed は未指定時サーバがランダム割当するので int で入っている
    assert isinstance(data["params"]["seed"], int)

    image_id = data["image_id"]
    r2 = client.get(f"/api/images/{image_id}.webp")
    assert r2.status_code == 200
    assert r2.headers["content-type"] == "image/webp"


def test_generate_rejects_bad_image_id() -> None:
    app = create_app()
    with TestClient(app) as c:
        r = c.get("/api/images/not-a-uuid.webp")
        assert r.status_code == 400


def test_generate_rejects_missing_image_id() -> None:
    app = create_app()
    with TestClient(app) as c:
        r = c.get("/api/images/00000000-0000-0000-0000-000000000000.webp")
        assert r.status_code == 404


def test_generate_rejects_empty_instruction(client: TestClient) -> None:
    r = client.post("/api/generate", json={"instruction_ja": ""})
    assert r.status_code == 422


def test_generate_honors_explicit_req_seed_override(client: TestClient) -> None:
    """`POST /generate` の dev 用 `seed` 指定は seed_action を飛び越えて採用される。"""
    r = client.post(
        "/api/generate",
        json={"instruction_ja": "テスト指定 seed", "seed": 999},
    )
    assert r.status_code == 200, r.text
    assert r.json()["params"]["seed"] == 999


def test_generate_base_mode_passes_turbo_false(client: TestClient) -> None:
    """conftest が IMAGE_TURBO_LORA="" を強制するので base モード → turbo=False が渡る。"""
    r = client.post("/api/generate", json={"instruction_ja": "base mode"})
    assert r.status_code == 200, r.text
    last = client.app.state.image_gen.calls[-1]  # type: ignore[attr-defined]
    assert last["turbo"] is False
    assert last["steps"] == 32
    assert last["cfg"] == 4.0


@contextmanager
def _turbo_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, FakeImageGen]]:
    """IMAGE_TURBO_LORA を設定した turbo 有効アプリ。FakeImageGen が turbo フラグを記録する。"""
    monkeypatch.setenv("IMAGES_DIR", str(tmp_path / "images"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "models"))
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("STARTUP_PRELOAD", "false")
    monkeypatch.setenv("IMAGE_TURBO_LORA", _TURBO_URN)
    get_settings.cache_clear()
    app = create_app()
    fake = FakeImageGen()
    with TestClient(app) as c:
        app.state.llm = FakeLlm()
        app.state.image_gen = fake
        yield c, fake
    get_settings.cache_clear()


def test_generate_turbo_mode_passes_turbo_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """turbo 有効設定なら turbo=True と Turbo 既定 (10step/CFG1) が渡る。"""
    with _turbo_app(tmp_path, monkeypatch) as (c, fake):
        r = c.post("/api/generate", json={"instruction_ja": "turbo mode"})
        assert r.status_code == 200, r.text
    last = fake.calls[-1]
    assert last["turbo"] is True
    assert last["steps"] == 10
    assert last["cfg"] == 1.0


def test_generate_cfg_override_forces_turbo_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """turbo 有効でも cfg を手動上書きしたら base 整合させ、turbo=False にする（LoRA 破綻回避）。"""
    with _turbo_app(tmp_path, monkeypatch) as (c, fake):
        r = c.post("/api/generate", json={"instruction_ja": "manual cfg", "cfg": 4.0})
        assert r.status_code == 200, r.text
    last = fake.calls[-1]
    assert last["turbo"] is False
    assert last["cfg"] == 4.0
    # steps は未指定なので base 既定に整合する。
    assert last["steps"] == 32


def test_generate_explicit_turbo_false_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """明示 turbo=false は cfg/steps 有無に関わらず最優先される。"""
    with _turbo_app(tmp_path, monkeypatch) as (c, fake):
        r = c.post("/api/generate", json={"instruction_ja": "explicit off", "turbo": False})
        assert r.status_code == 200, r.text
    assert fake.calls[-1]["turbo"] is False
