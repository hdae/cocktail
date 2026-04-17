from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from cocktail_server.schemas.messages import (
    ImagePart,
    Message,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from cocktail_server.services.llm import _build_chat_messages
from PIL import Image as PILImage


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


def test_no_images_dir_means_no_image_attachment() -> None:
    image_id = "11111111-1111-1111-1111-111111111111"
    history = [_user("初回", mid="u1"), _assistant_with_image(image_id), _user("調整", mid="u2")]
    messages = _build_chat_messages(history)
    # 全ての content が str（tokenizer 経路）
    assert all(isinstance(m["content"], str) for m in messages)


def test_images_dir_with_prior_assistant_image_attaches_pil(tmp_path: Path) -> None:
    image_id = "22222222-2222-2222-2222-222222222222"
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    PILImage.new("RGB", (32, 32), (255, 0, 0)).save(images_dir / f"{image_id}.webp", format="WEBP")
    history = [_user("初回", mid="u1"), _assistant_with_image(image_id), _user("調整", mid="u2")]
    messages = _build_chat_messages(history, images_dir=images_dir)

    # 最終 user メッセージは image + text の list-content
    assert isinstance(messages[-1]["content"], list)
    parts = messages[-1]["content"]
    assert len(parts) == 2
    assert parts[0]["type"] == "image"
    assert isinstance(parts[0]["image"], PILImage.Image)
    assert parts[1]["type"] == "text"
    # 画像を添付するターンは processor 経路に乗るため、過去メッセージも
    # 全て list-content 形式に統一される（apply_chat_template が混在を許さない）
    assert all(isinstance(m["content"], list) for m in messages)
    assert messages[0]["content"][0]["type"] == "text"


def test_images_dir_but_missing_file_falls_back_to_text_only(tmp_path: Path) -> None:
    image_id = "33333333-3333-3333-3333-333333333333"
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    # ファイルを置かない
    history = [_user("初回", mid="u1"), _assistant_with_image(image_id), _user("調整", mid="u2")]
    messages = _build_chat_messages(history, images_dir=images_dir)
    # ImagePart ファイルが無いときは全ての content が str のまま
    assert all(isinstance(m["content"], str) for m in messages)


def test_no_prior_assistant_image_means_no_attachment(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    # 初回ユーザのみ → 添付対象なし
    messages = _build_chat_messages([_user("初回")], images_dir=images_dir)
    assert isinstance(messages[0]["content"], str)


def test_empty_history_raises() -> None:
    with pytest.raises(ValueError, match="at least one message"):
        _build_chat_messages([])
