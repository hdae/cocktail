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


def test_single_user_is_labeled_turn_1_current() -> None:
    messages = _build_chat_messages([_user("hello")])
    content = messages[0]["content"]
    assert isinstance(content, str)
    # 最初の user は system prompt が前置されるため、build_user_message の結果が
    # 末尾に埋まっていることで検証する（system prompt 本文中にラベル文字列そのものが
    # 登場するので単純な `in` チェックだと過剰マッチする）
    expected = build_user_message("hello", turn_index=1, is_current=True)
    assert content.endswith(expected)


def test_multi_turn_user_labels_are_sequential_and_last_is_current() -> None:
    image_id = "11111111-1111-1111-1111-111111111111"
    history = [
        _user("一枚目お願い", mid="u1"),
        _assistant_with_image(image_id, mid="a1"),
        _user("色違いで", mid="u2"),
    ]
    messages = _build_chat_messages(history)
    assert len(messages) == 3

    first_user = messages[0]["content"]
    assert isinstance(first_user, str)
    # Turn 1 の user 本文は is_current=False で再現できる
    assert first_user.endswith(build_user_message("一枚目お願い", turn_index=1, is_current=False))

    # assistant はラベルなしの JSON 文字列のまま
    assert messages[1]["role"] == "assistant"
    assert isinstance(messages[1]["content"], str)
    assert "reasoning" in messages[1]["content"]

    last_user = messages[2]["content"]
    assert last_user == build_user_message("色違いで", turn_index=2, is_current=True)


def test_pure_chat_turn_still_counts_as_a_turn() -> None:
    # 純チャット応答も 1 ターンとしてカウントする（案A: user/assistant ペア単位）
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
    assert messages[0]["content"].endswith(
        build_user_message("最初のお願い", turn_index=1, is_current=False)
    )
    assert messages[2]["content"] == build_user_message("次のお願い", turn_index=2, is_current=True)


def test_history_with_assistant_image_still_text_only() -> None:
    image_id = "11111111-1111-1111-1111-111111111111"
    history = [_user("初回", mid="u1"), _assistant_with_image(image_id), _user("調整", mid="u2")]
    messages = _build_chat_messages(history)
    # 全ての content が str（tokenizer 経路のみ）
    assert all(isinstance(m["content"], str) for m in messages)


def test_empty_history_raises() -> None:
    with pytest.raises(ValueError, match="at least one message"):
        _build_chat_messages([])


def test_history_starting_with_assistant_raises() -> None:
    image_id = "11111111-1111-1111-1111-111111111111"
    with pytest.raises(ValueError, match="begin with a user message"):
        _build_chat_messages([_assistant_with_image(image_id)])
