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
from cocktail_server.services.prompt_builder import build_user_message


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


def test_first_message_is_system_prompt() -> None:
    messages = _build_chat_messages([_user("hello")])
    assert messages[0]["role"] == "system"
    assert "generate_image" in messages[0]["content"]


def test_user_follows_system_and_is_labeled_current() -> None:
    messages = _build_chat_messages([_user("hello")])
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"] == build_user_message("hello", turn_index=1, is_current=True)
    assert "hello" in messages[1]["content"]


def test_multi_turn_user_labels_are_sequential_and_last_is_current() -> None:
    image_id = "11111111-1111-1111-1111-111111111111"
    history = [
        _user("一枚目お願い", mid="u1"),
        _assistant_with_image(image_id, mid="a1"),
        _user("色違いで", mid="u2"),
    ]
    messages = _build_chat_messages(history)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == build_user_message(
        "一枚目お願い", turn_index=1, is_current=False
    )
    assert messages[3]["content"] == build_user_message("色違いで", turn_index=2, is_current=True)


def test_assistant_is_reconstructed_as_plain_text_with_positive_note() -> None:
    image_id = "11111111-1111-1111-1111-111111111111"
    history = [_user("初回", mid="u1"), _assistant_with_image(image_id), _user("調整", mid="u2")]
    messages = _build_chat_messages(history)
    assistant = messages[2]
    assert assistant["role"] == "assistant"
    content = assistant["content"]
    assert isinstance(content, str)
    assert "生成しました" in content
    # 「n個前」参照のため過去 positive タグを見せる。native 特殊トークンは注入しない。
    assert "score_7, safe, 1girl" in content
    assert "<|tool_call>" not in content
    # 全 content が str（tokenizer 経路のみ）
    assert all(isinstance(m["content"], str) for m in messages)


def test_pure_chat_turn_still_counts_as_a_turn() -> None:
    # 純チャット応答も 1 ターンとしてカウントする（user/assistant ペア単位）
    chat_assistant = Message(
        id="a1",
        conversation_id="conv1",
        role="assistant",
        parts=[TextPart(text="ありがとうございます！")],
        created_at=datetime.now(UTC),
    )
    history = [
        _user("最初のお願い", mid="u1"),
        chat_assistant,
        _user("次のお願い", mid="u2"),
    ]
    messages = _build_chat_messages(history)
    assert messages[1]["content"] == build_user_message(
        "最初のお願い", turn_index=1, is_current=False
    )
    assert messages[3]["content"] == build_user_message("次のお願い", turn_index=2, is_current=True)


def test_empty_history_raises() -> None:
    with pytest.raises(ValueError, match="at least one message"):
        _build_chat_messages([])


def test_history_starting_with_assistant_raises() -> None:
    image_id = "11111111-1111-1111-1111-111111111111"
    with pytest.raises(ValueError, match="begin with a user message"):
        _build_chat_messages([_assistant_with_image(image_id)])
