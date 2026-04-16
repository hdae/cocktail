from __future__ import annotations

import json
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
                "score_7, masterpiece, best quality, safe, newest, 1girl, fake tags for chat test"
            ),
            negative="worst quality, low quality, score_1, score_2, score_3, artist name",
            rationale="fake rationale",
        )

    def unload(self) -> None:
        return None


class FakeImageGen:
    async def generate(self, **kwargs: Any) -> PILImage.Image:
        return PILImage.new("RGB", (kwargs["width"], kwargs["height"]), (0, 128, 255))

    def unload(self) -> None:
        return None


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("IMAGES_DIR", str(tmp_path / "images"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "models"))
    get_settings.cache_clear()

    app = create_app()
    with TestClient(app) as c:
        app.state.llm = FakeLlm()
        app.state.image_gen = FakeImageGen()
        yield c
    get_settings.cache_clear()


def _parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    parsed: list[tuple[str, dict[str, Any]]] = []
    for frame in text.split("\n\n"):
        if not frame.strip():
            continue
        event_name = ""
        data_line = ""
        for line in frame.split("\n"):
            if line.startswith("event: "):
                event_name = line[len("event: ") :]
            elif line.startswith("data: "):
                data_line = line[len("data: ") :]
        parsed.append((event_name, json.loads(data_line)))
    return parsed


def test_chat_new_conversation_emits_full_sequence(client: TestClient) -> None:
    r = client.post(
        "/chat",
        json={"parts": [{"type": "text", "text": "ピンクの猫耳少女"}]},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(r.text)
    names = [name for name, _ in events]
    assert names == [
        "conversation",
        "user_saved",
        "assistant_start",
        "tool_call_start",
        "image_ready",
        "tool_call_end",
        "assistant_end",
        "done",
    ]

    conversation_id = events[0][1]["conversation_id"]
    user_msg = events[1][1]["message"]
    assert user_msg["conversation_id"] == conversation_id
    assert user_msg["role"] == "user"
    assert user_msg["parts"][0]["type"] == "text"

    image_ready = events[4][1]
    assert image_ready["mime"] == "image/webp"
    assert image_ready["image_url"].startswith("/images/")
    assert image_ready["image_url"].endswith(".webp")

    tool_end = events[5][1]
    assert tool_end["status"] == "done"
    assert "prompt" in tool_end["data"]
    assert tool_end["data"]["prompt"].startswith("score_7")

    assistant_msg = events[6][1]["message"]
    assert assistant_msg["role"] == "assistant"
    assert [p["type"] for p in assistant_msg["parts"]] == [
        "tool_call",
        "tool_result",
        "image",
    ]
    assert assistant_msg["parts"][2]["image_id"] == image_ready["image_id"]


def test_chat_continues_existing_conversation(client: TestClient) -> None:
    first = client.post(
        "/chat",
        json={"parts": [{"type": "text", "text": "最初のリクエスト"}]},
    )
    assert first.status_code == 200
    first_events = _parse_sse(first.text)
    conversation_id = first_events[0][1]["conversation_id"]

    second = client.post(
        "/chat",
        json={
            "conversation_id": conversation_id,
            "parts": [{"type": "text", "text": "続けて描いて"}],
        },
    )
    assert second.status_code == 200
    second_events = _parse_sse(second.text)
    names = [n for n, _ in second_events]
    # 既存会話なので conversation イベントは飛ばない
    assert "conversation" not in names
    assert names[0] == "user_saved"
    assert names[-1] == "done"


def test_chat_rejects_unknown_conversation(client: TestClient) -> None:
    r = client.post(
        "/chat",
        json={
            "conversation_id": "00000000-0000-0000-0000-000000000000",
            "parts": [{"type": "text", "text": "hi"}],
        },
    )
    assert r.status_code == 200  # SSE は 200 で開き、本文で error を流す
    events = _parse_sse(r.text)
    names = [n for n, _ in events]
    assert names == ["error", "done"]
    assert events[0][1]["code"] == "conversation_not_found"


def test_chat_rejects_empty_parts(client: TestClient) -> None:
    r = client.post("/chat", json={"parts": []})
    assert r.status_code == 422


def test_chat_rejects_parts_without_text(client: TestClient) -> None:
    r = client.post(
        "/chat",
        json={
            "parts": [
                {
                    "type": "image",
                    "image_id": "00000000-0000-0000-0000-000000000000",
                    "mime": "image/webp",
                }
            ],
        },
    )
    assert r.status_code == 422


def test_chat_generated_image_is_served(client: TestClient) -> None:
    r = client.post(
        "/chat",
        json={"parts": [{"type": "text", "text": "draw something"}]},
    )
    events = _parse_sse(r.text)
    image_url = next(data["image_url"] for name, data in events if name == "image_ready")

    r2 = client.get(image_url)
    assert r2.status_code == 200
    assert r2.headers["content-type"] == "image/webp"


def test_chat_echoes_reference_images_in_tool_args(client: TestClient) -> None:
    image_ref = "11111111-1111-1111-1111-111111111111"
    r = client.post(
        "/chat",
        json={
            "parts": [
                {"type": "text", "text": "この絵を参考に"},
                {"type": "image", "image_id": image_ref, "mime": "image/webp"},
            ],
        },
    )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    tool_start = next(data for name, data in events if name == "tool_call_start")
    assert tool_start["name"] == "generate_image"
    assert tool_start["args"]["reference_images"] == [image_ref]
