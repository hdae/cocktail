from __future__ import annotations

from datetime import UTC, datetime

import pytest
from cocktail_server.schemas.messages import Message, TextPart, ToolCallPart
from cocktail_server.services.llm import (
    _build_result,
    _reconstruct_assistant_turn,
    parse_gguf_ref,
)
from cocktail_server.services.native_tools import ParsedToolCall, ParsedTurn
from pydantic import ValidationError

# --- parse_gguf_ref --------------------------------------------------------------


def test_parse_gguf_ref_repo_file() -> None:
    assert parse_gguf_ref("org/repo:model.gguf") == ("org/repo", "model.gguf")


def test_parse_gguf_ref_non_gguf_suffix_returns_none() -> None:
    assert parse_gguf_ref("org/repo:model.bin") is None


def test_parse_gguf_ref_plain_repo_returns_none() -> None:
    assert parse_gguf_ref("org/repo") is None


# --- _build_result: ParsedTurn -> 検証済み LlmTurnResult --------------------------


def _turn(*calls: ParsedToolCall, text: str = "", thought: str = "") -> ParsedTurn:
    return ParsedTurn(text=text, thought=thought, tool_calls=list(calls))


def test_build_result_validates_generate_image_call() -> None:
    result = _build_result(
        _turn(
            ParsedToolCall(
                name="generate_image",
                args={"positive": "1girl, solo", "aspect_ratio": "portrait"},
            ),
            text="出すよ",
            thought="内緒の思考",
        )
    )
    assert result.text == "出すよ"
    assert result.thought == "内緒の思考"
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.positive == "1girl, solo"
    assert call.negative_extra == ""  # 省略時は空
    assert call.seed_action == "new"  # 省略時は既定


def test_build_result_ignores_unknown_tool() -> None:
    # Phase 1 は generate_image のみ配線。未知ツールは無視する。
    result = _build_result(_turn(ParsedToolCall(name="search_tags", args={"q": "cat"})))
    assert result.tool_calls == []


def test_build_result_takes_only_first_generate_image() -> None:
    result = _build_result(
        _turn(
            ParsedToolCall(
                name="generate_image", args={"positive": "a", "aspect_ratio": "portrait"}
            ),
            ParsedToolCall(name="generate_image", args={"positive": "b", "aspect_ratio": "square"}),
        )
    )
    assert [c.positive for c in result.tool_calls] == ["a"]


def test_build_result_empty_positive_raises() -> None:
    with pytest.raises(ValidationError):
        _build_result(
            _turn(
                ParsedToolCall(
                    name="generate_image", args={"positive": "", "aspect_ratio": "portrait"}
                )
            )
        )


def test_build_result_invalid_aspect_ratio_raises() -> None:
    with pytest.raises(ValidationError):
        _build_result(
            _turn(
                ParsedToolCall(
                    name="generate_image", args={"positive": "1girl", "aspect_ratio": "wide"}
                )
            )
        )


# --- _reconstruct_assistant_turn -------------------------------------------------


def _assistant(parts: list) -> Message:  # type: ignore[type-arg]
    return Message(
        id="a1",
        conversation_id="c1",
        role="assistant",
        parts=parts,
        created_at=datetime.now(UTC),
    )


def test_reconstruct_replays_tool_call_in_native_format() -> None:
    msg = _assistant(
        [
            TextPart(text="出したよ"),
            ToolCallPart(
                id="c1",
                name="generate_image",
                args={
                    "positive": "1girl, red hair",
                    "aspect_ratio": "portrait",
                    "seed_action": "new",
                },
                status="done",
            ),
        ]
    )
    out = _reconstruct_assistant_turn(msg)
    assert "出したよ" in out
    assert "1girl, red hair" in out  # 「n個前」参照のため過去 positive を見せる
    # 記述注記ではなく native 形式で replay する（多ターンの形式ドリフトを断つため）
    assert "<|tool_call>call:generate_image{" in out
    assert '<|"|>portrait<|"|>' in out


def test_reconstruct_text_only_turn() -> None:
    assert _reconstruct_assistant_turn(_assistant([TextPart(text="ありがとう")])) == "ありがとう"
