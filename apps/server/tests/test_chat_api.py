from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from cocktail_server.config import get_settings
from cocktail_server.main import create_app
from cocktail_server.schemas.generate import GenerateImageCall, LlmTurnSpec
from cocktail_server.schemas.messages import Message
from cocktail_server.services.llm import LlmStreamChunk, LlmTextDelta, LlmTurnComplete
from cocktail_server.services.prompt_builder import NEGATIVE_DEFAULT
from fastapi.testclient import TestClient
from PIL import Image as PILImage


def _make_spec(
    *,
    reasoning: str = "テスト用の応答です。生成しますね。",
    tool_calls: list[GenerateImageCall] | None = None,
) -> LlmTurnSpec:
    if tool_calls is None:
        tool_calls = [
            GenerateImageCall(
                name="generate_image",
                positive=(
                    "score_7, masterpiece, best quality, safe, newest, 1girl, "
                    "fake tags for chat test"
                ),
                negative=NEGATIVE_DEFAULT,
                aspect_ratio="portrait",
                cfg_preset="standard",
                seed_action="new",
                rationale="fake rationale",
            )
        ]
    return LlmTurnSpec(reasoning=reasoning, tool_calls=tool_calls)


class FakeLlm:
    """`run_turn` を AsyncIterator で返すフェイク。reasoning を 2 つの text_delta に分割して返す。

    受け取った履歴は `received_histories` に記録し、multi-turn テストから検証できるようにする。
    `specs` を与えるとターンごとに順番に消費する（ターン数に足りなければ最後の spec を繰り返す）。
    """

    def __init__(
        self,
        spec: LlmTurnSpec | None = None,
        specs: list[LlmTurnSpec] | None = None,
    ) -> None:
        if specs is not None:
            self._specs = list(specs)
        else:
            self._specs = [spec if spec is not None else _make_spec()]
        self._turn = 0
        self.received_histories: list[list[Message]] = []

    async def run_turn(self, history: list[Message]) -> AsyncIterator[LlmStreamChunk]:
        self.received_histories.append(list(history))
        idx = min(self._turn, len(self._specs) - 1)
        self._turn += 1
        spec = self._specs[idx]
        reasoning = spec.reasoning
        if reasoning:
            mid = len(reasoning) // 2
            if mid > 0:
                yield LlmTextDelta(delta=reasoning[:mid])
            yield LlmTextDelta(delta=reasoning[mid:])
        yield LlmTurnComplete(spec=spec)

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
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("STARTUP_PRELOAD", "false")
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
    # text_delta は 0 個以上流れるので除外して比較
    names_no_delta = [n for n in names if n != "text_delta"]
    assert names_no_delta == [
        "conversation",
        "user_saved",
        "assistant_start",
        "tool_call_start",
        "image_ready",
        "tool_call_end",
        "assistant_end",
        "done",
    ]

    # text_delta は assistant_start と tool_call_start の間にだけ現れる
    start_idx = names.index("assistant_start")
    tool_start_idx = names.index("tool_call_start")
    for i in range(start_idx + 1, tool_start_idx):
        assert names[i] == "text_delta"
    assert tool_start_idx > start_idx + 1  # 少なくとも 1 つ流れた

    conversation_id = events[0][1]["conversation_id"]
    user_msg = events[1][1]["message"]
    assert user_msg["conversation_id"] == conversation_id
    assert user_msg["role"] == "user"

    tool_start = next(data for name, data in events if name == "tool_call_start")
    assert tool_start["name"] == "generate_image"
    assert tool_start["args"]["aspect_ratio"] == "portrait"
    assert tool_start["args"]["cfg_preset"] == "standard"
    assert tool_start["args"]["width"] == 896
    assert tool_start["args"]["height"] == 1152
    assert tool_start["args"]["positive"].startswith("score_7")

    image_ready = next(data for name, data in events if name == "image_ready")
    assert image_ready["mime"] == "image/webp"
    assert image_ready["image_url"].endswith(".webp")

    tool_end = next(data for name, data in events if name == "tool_call_end")
    assert tool_end["status"] == "done"
    assert tool_end["data"]["prompt"].startswith("score_7")

    assistant_msg = next(data for name, data in events if name == "assistant_end")["message"]
    assert assistant_msg["role"] == "assistant"
    assert [p["type"] for p in assistant_msg["parts"]] == [
        "text",
        "tool_call",
        "tool_result",
        "image",
    ]
    assert assistant_msg["parts"][3]["image_id"] == image_ready["image_id"]


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
    assert r.status_code == 200
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


def test_chat_without_tool_call_emits_text_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gemma が `tool_calls=[]` で返したら、画像生成をスキップして text のみで閉じる。"""
    monkeypatch.setenv("IMAGES_DIR", str(tmp_path / "images"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "models"))
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("STARTUP_PRELOAD", "false")
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as c:
        app.state.llm = FakeLlm(spec=_make_spec(reasoning="ありがとうございます！", tool_calls=[]))
        app.state.image_gen = FakeImageGen()
        r = c.post("/chat", json={"parts": [{"type": "text", "text": "ありがとう"}]})
    get_settings.cache_clear()

    assert r.status_code == 200
    events = _parse_sse(r.text)
    names = [n for n, _ in events]
    names_no_delta = [n for n in names if n != "text_delta"]
    assert names_no_delta == [
        "conversation",
        "user_saved",
        "assistant_start",
        "assistant_end",
        "done",
    ]
    assistant_msg = next(data for name, data in events if name == "assistant_end")["message"]
    assert [p["type"] for p in assistant_msg["parts"]] == ["text"]
    assert assistant_msg["parts"][0]["text"] == "ありがとうございます！"


def test_chat_second_turn_receives_previous_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2 ターン目の Gemma 呼び出しは、過去のユーザ発話と assistant ターンを含む履歴を受け取る。"""
    monkeypatch.setenv("IMAGES_DIR", str(tmp_path / "images"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "models"))
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("STARTUP_PRELOAD", "false")
    get_settings.cache_clear()
    app = create_app()
    fake_llm = FakeLlm()
    with TestClient(app) as c:
        app.state.llm = fake_llm
        app.state.image_gen = FakeImageGen()

        first = c.post(
            "/chat",
            json={"parts": [{"type": "text", "text": "初音ミクを描いて"}]},
        )
        assert first.status_code == 200
        conversation_id = _parse_sse(first.text)[0][1]["conversation_id"]

        second = c.post(
            "/chat",
            json={
                "conversation_id": conversation_id,
                "parts": [{"type": "text", "text": "もっと笑顔に"}],
            },
        )
        assert second.status_code == 200
    get_settings.cache_clear()

    # 1 ターン目は user 1 件、2 ターン目は user/assistant/user の 3 件が渡る
    assert len(fake_llm.received_histories) == 2
    first_history = fake_llm.received_histories[0]
    assert [m.role for m in first_history] == ["user"]

    second_history = fake_llm.received_histories[1]
    assert [m.role for m in second_history] == ["user", "assistant", "user"]
    first_user_text = "".join(
        p.text
        for p in second_history[0].parts
        if p.type == "text"  # type: ignore[union-attr]
    )
    assert "初音ミク" in first_user_text
    latest_user_text = "".join(
        p.text
        for p in second_history[-1].parts
        if p.type == "text"  # type: ignore[union-attr]
    )
    assert "もっと笑顔に" in latest_user_text


def test_chat_landscape_aspect_ratio_resolves_to_correct_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gemma が landscape を選んだら tool_call_start.args.width/height が 1152x896 になる。"""
    monkeypatch.setenv("IMAGES_DIR", str(tmp_path / "images"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "models"))
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("STARTUP_PRELOAD", "false")
    get_settings.cache_clear()
    app = create_app()
    landscape_call = GenerateImageCall(
        name="generate_image",
        positive="score_7, masterpiece, best quality, safe, newest, 1girl, landscape",
        negative=NEGATIVE_DEFAULT,
        aspect_ratio="landscape",
        cfg_preset="crisp",
        seed_action="new",
        rationale="wide shot",
    )
    with TestClient(app) as c:
        app.state.llm = FakeLlm(spec=_make_spec(tool_calls=[landscape_call]))
        app.state.image_gen = FakeImageGen()
        r = c.post("/chat", json={"parts": [{"type": "text", "text": "横長で風景を"}]})
    get_settings.cache_clear()

    assert r.status_code == 200
    events = _parse_sse(r.text)
    tool_start = next(data for name, data in events if name == "tool_call_start")
    assert tool_start["args"]["aspect_ratio"] == "landscape"
    assert tool_start["args"]["width"] == 1152
    assert tool_start["args"]["height"] == 896
    assert tool_start["args"]["cfg_preset"] == "crisp"
    assert tool_start["args"]["cfg"] == 4.5
