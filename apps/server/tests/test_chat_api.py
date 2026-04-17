from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from cocktail_server.config import get_settings
from cocktail_server.main import create_app
from cocktail_server.schemas.generate import GenerateImageCall, LlmTurnSpec
from cocktail_server.schemas.messages import Message
from cocktail_server.services.llm import LlmStreamChunk, LlmTextDelta, LlmTurnComplete
from cocktail_server.services.prompt_builder import NEGATIVE_DEFAULT
from cocktail_server.services.turn_registry import TurnRegistry
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


def _start_chat(client: TestClient, payload: dict[str, Any]) -> tuple[str, str]:
    """`POST /api/chat` を呼んで `(conversation_id, turn_id)` を返すヘルパ。"""
    r = client.post("/api/chat", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    return body["conversation_id"], body["turn_id"]


def _chat(
    client: TestClient, payload: dict[str, Any]
) -> tuple[str, str, list[tuple[str, dict[str, Any]]]]:
    """POST → subscribe → 全 SSE を読みきるヘルパ。`(conversation_id, turn_id, events)` を返す。"""
    conv_id, turn_id = _start_chat(client, payload)
    r = client.get(f"/api/chat/turns/{turn_id}/events")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    return conv_id, turn_id, _parse_sse(r.text)


def test_start_chat_returns_conversation_and_turn_ids(client: TestClient) -> None:
    conversation_id, turn_id = _start_chat(
        client,
        {"parts": [{"type": "text", "text": "hi"}]},
    )
    assert conversation_id
    assert turn_id
    assert conversation_id != turn_id


def test_subscribe_emits_full_sequence(client: TestClient) -> None:
    conversation_id, _turn_id, events = _chat(
        client,
        {"parts": [{"type": "text", "text": "ピンクの猫耳少女"}]},
    )

    names = [name for name, _ in events]
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

    start_idx = names.index("assistant_start")
    tool_start_idx = names.index("tool_call_start")
    for i in range(start_idx + 1, tool_start_idx):
        assert names[i] == "text_delta"
    assert tool_start_idx > start_idx + 1

    assert events[0][1]["conversation_id"] == conversation_id
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


def test_existing_conversation_skips_conversation_event(client: TestClient) -> None:
    first_conv, _, _ = _chat(
        client,
        {"parts": [{"type": "text", "text": "最初のリクエスト"}]},
    )

    _, _, events = _chat(
        client,
        {
            "conversation_id": first_conv,
            "parts": [{"type": "text", "text": "続けて描いて"}],
        },
    )
    names = [n for n, _ in events]
    assert "conversation" not in names
    assert names[0] == "user_saved"
    assert names[-1] == "done"


def test_unknown_conversation_id_returns_404_on_post(client: TestClient) -> None:
    r = client.post(
        "/api/chat",
        json={
            "conversation_id": "00000000-0000-0000-0000-000000000000",
            "parts": [{"type": "text", "text": "hi"}],
        },
    )
    assert r.status_code == 404


def test_subscribe_unknown_turn_returns_404(client: TestClient) -> None:
    r = client.get("/api/chat/turns/00000000-0000-0000-0000-000000000000/events")
    assert r.status_code == 404


def test_subscribe_after_completion_replays_events(client: TestClient) -> None:
    """完走済みターンを購読すると、replay バッファの全イベントが届いてから close する。"""
    _, turn_id = _start_chat(client, {"parts": [{"type": "text", "text": "完走確認"}]})

    # 1 回目の購読で完走を待つ
    first = client.get(f"/api/chat/turns/{turn_id}/events")
    assert first.status_code == 200
    first_events = _parse_sse(first.text)
    assert first_events[-1][0] == "done"

    # 2 回目（retention 内）を購読すると replay で同じ順序で全部流れる
    second = client.get(f"/api/chat/turns/{turn_id}/events")
    assert second.status_code == 200
    second_events = _parse_sse(second.text)
    assert [n for n, _ in second_events] == [n for n, _ in first_events]


def test_chat_rejects_empty_parts(client: TestClient) -> None:
    r = client.post("/api/chat", json={"parts": []})
    assert r.status_code == 422


def test_chat_rejects_parts_without_text(client: TestClient) -> None:
    r = client.post(
        "/api/chat",
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
    _, _, events = _chat(
        client,
        {"parts": [{"type": "text", "text": "draw something"}]},
    )
    image_url = next(data["image_url"] for name, data in events if name == "image_ready")

    r2 = client.get(image_url)
    assert r2.status_code == 200
    assert r2.headers["content-type"] == "image/webp"


def test_chat_echoes_reference_images_in_tool_args(client: TestClient) -> None:
    image_ref = "11111111-1111-1111-1111-111111111111"
    _, _, events = _chat(
        client,
        {
            "parts": [
                {"type": "text", "text": "この絵を参考に"},
                {"type": "image", "image_id": image_ref, "mime": "image/webp"},
            ],
        },
    )
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
        _, _, events = _chat(c, {"parts": [{"type": "text", "text": "ありがとう"}]})
    get_settings.cache_clear()

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

        conv_id, _, _ = _chat(c, {"parts": [{"type": "text", "text": "初音ミクを描いて"}]})
        _chat(
            c,
            {
                "conversation_id": conv_id,
                "parts": [{"type": "text", "text": "もっと笑顔に"}],
            },
        )
    get_settings.cache_clear()

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


def test_chat_seed_action_keep_reuses_previous_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2 ターン目に `seed_action="keep"` を出せば、直前画像と同じ seed で生成される。"""
    monkeypatch.setenv("IMAGES_DIR", str(tmp_path / "images"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "models"))
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("STARTUP_PRELOAD", "false")
    get_settings.cache_clear()
    app = create_app()
    first_call = GenerateImageCall(
        name="generate_image",
        positive="score_7, masterpiece, best quality, safe, newest, 1girl, first",
        negative=NEGATIVE_DEFAULT,
        aspect_ratio="portrait",
        cfg_preset="standard",
        seed_action="new",
        rationale="first",
    )
    keep_call = GenerateImageCall(
        name="generate_image",
        positive="score_7, masterpiece, best quality, safe, newest, 1girl, tweak",
        negative=NEGATIVE_DEFAULT,
        aspect_ratio="portrait",
        cfg_preset="standard",
        seed_action="keep",
        rationale="tweak",
    )
    with TestClient(app) as c:
        app.state.llm = FakeLlm(
            specs=[_make_spec(tool_calls=[first_call]), _make_spec(tool_calls=[keep_call])]
        )
        app.state.image_gen = FakeImageGen()

        conv_id, _, first_events = _chat(c, {"parts": [{"type": "text", "text": "1 枚目"}]})
        first_seed = next(
            data["args"]["seed"] for name, data in first_events if name == "tool_call_start"
        )

        _, _, second_events = _chat(
            c,
            {
                "conversation_id": conv_id,
                "parts": [{"type": "text", "text": "色味だけ調整"}],
            },
        )
        second_seed = next(
            data["args"]["seed"] for name, data in second_events if name == "tool_call_start"
        )
    get_settings.cache_clear()

    assert first_seed == second_seed
    assert (
        next(
            data["args"]["seed_action"] for name, data in second_events if name == "tool_call_start"
        )
        == "keep"
    )


def test_chat_generated_image_appears_in_list_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IMAGES_DIR", str(tmp_path / "images"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "models"))
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("STARTUP_PRELOAD", "false")
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as c:
        app.state.llm = FakeLlm()
        app.state.image_gen = FakeImageGen()
        _, _, events = _chat(c, {"parts": [{"type": "text", "text": "初回"}]})
        image_ready = next(data for name, data in events if name == "image_ready")
        listing = c.get("/api/images").json()
    get_settings.cache_clear()

    image_ids = [img["image_id"] for img in listing["images"]]
    assert image_ready["image_id"] in image_ids
    found = next(img for img in listing["images"] if img["image_id"] == image_ready["image_id"])
    assert found["aspect_ratio"] == "portrait"
    assert found["cfg_preset"] == "standard"
    assert found["width"] == 896
    assert found["height"] == 1152
    assert found["prompt"].startswith("score_7")


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
        _, _, events = _chat(c, {"parts": [{"type": "text", "text": "横長で風景を"}]})
    get_settings.cache_clear()

    tool_start = next(data for name, data in events if name == "tool_call_start")
    assert tool_start["args"]["aspect_ratio"] == "landscape"
    assert tool_start["args"]["width"] == 1152
    assert tool_start["args"]["height"] == 896
    assert tool_start["args"]["cfg_preset"] == "crisp"
    assert tool_start["args"]["cfg"] == 4.5


def test_retention_gc_removes_turn_after_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """retention 経過後の turn は 404 を返す。"""
    monkeypatch.setenv("IMAGES_DIR", str(tmp_path / "images"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "models"))
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("STARTUP_PRELOAD", "false")
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as c:
        app.state.llm = FakeLlm()
        app.state.image_gen = FakeImageGen()
        # retention を極めて短くした registry に差し替える
        app.state.turn_registry = TurnRegistry(retention_seconds=0.05)

        _, turn_id, _ = _chat(c, {"parts": [{"type": "text", "text": "retention"}]})
        # 完走待ちと GC 待ちを少しだけ寝かせる
        import time

        time.sleep(0.3)

        r = c.get(f"/api/chat/turns/{turn_id}/events")
        assert r.status_code == 404
    get_settings.cache_clear()


async def test_turn_registry_concurrent_subscribers_receive_same_events() -> None:
    """2 subscriber が同じターンを観測し、同じ順序で全イベントを受け取る。"""
    from cocktail_server.schemas.events import (
        AssistantStartEvent,
        DoneEvent,
        UserSavedEvent,
    )
    from cocktail_server.schemas.messages import Message, TextPart

    registry = TurnRegistry(retention_seconds=1.0)
    turn = registry.register("conv-1")

    msg = Message(
        id="u1",
        conversation_id="conv-1",
        role="user",
        parts=[TextPart(text="hi")],
        created_at=datetime.now(UTC),
    )

    collected: list[list[str]] = [[], []]

    async def collector(idx: int) -> None:
        async with registry.subscribe(turn.turn_id) as iterator:
            async for ev in iterator:
                collected[idx].append(ev.type)

    t1 = asyncio.create_task(collector(0))
    t2 = asyncio.create_task(collector(1))
    await asyncio.sleep(0)  # subscribe の登録を先に走らせる

    registry.publish(turn.turn_id, UserSavedEvent(message=msg))
    registry.publish(turn.turn_id, AssistantStartEvent(message_id="a1"))
    registry.publish(turn.turn_id, DoneEvent())
    registry.finish(turn.turn_id)

    await asyncio.wait_for(asyncio.gather(t1, t2), timeout=1.0)
    assert collected[0] == ["user_saved", "assistant_start", "done"]
    assert collected[1] == ["user_saved", "assistant_start", "done"]
