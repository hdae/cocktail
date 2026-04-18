from __future__ import annotations

from datetime import UTC, datetime

import pytest
from cocktail_server.schemas.messages import (
    ImagePart,
    Message,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from cocktail_server.services.llm import _build_chat_messages


def _user(text: str, mid: str = "u1") -> Message:
    return Message(
        id=mid,
        conversation_id="conv1",
        role="user",
        parts=[TextPart(text=text)],
        created_at=datetime.now(UTC),
    )


def _assistant_with_image(image_id: str, mid: str = "a1") -> Message:
    return Message(
        id=mid,
        conversation_id="conv1",
        role="assistant",
        parts=[
            TextPart(text="生成しました"),
            ToolCallPart(
                type="tool_call",
                id="call-1",
                name="generate_image",
                args={
                    "positive": "score_7, safe, 1girl",
                    "negative": "worst quality",
                    "aspect_ratio": "portrait",
                    "cfg_preset": "standard",
                    "seed_action": "new",
                    "width": 896,
                    "height": 1152,
                    "cfg": 4.0,
                    "steps": 32,
                    "seed": 42,
                },
                status="done",
            ),
            ToolResultPart(call_id="call-1", summary="done", data={}),
            ImagePart(image_id=image_id, mime="image/webp", width=896, height=1152),
        ],
        created_at=datetime.now(UTC),
    )


def test_text_only_path_returns_string_content() -> None:
    messages = _build_chat_messages([_user("hello")])
    assert len(messages) == 1
    assert isinstance(messages[0]["content"], str)
    assert "hello" in messages[0]["content"]


def test_history_with_assistant_image_still_text_only() -> None:
    image_id = "11111111-1111-1111-1111-111111111111"
    history = [_user("初回", mid="u1"), _assistant_with_image(image_id), _user("調整", mid="u2")]
    messages = _build_chat_messages(history)
    # 全ての content が str（tokenizer 経路のみ）
    assert all(isinstance(m["content"], str) for m in messages)


def test_empty_history_raises() -> None:
    with pytest.raises(ValueError, match="at least one message"):
        _build_chat_messages([])
