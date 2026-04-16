from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"

Role = Literal["user", "assistant", "tool", "system"]
ToolCallStatus = Literal["pending", "running", "done", "error"]


class TextPart(BaseModel):
    """テキスト。ユーザ入力でも、アシスタント出力の自然言語でも使う。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["text"] = "text"
    text: str = Field(min_length=1, max_length=10_000)


class ImagePart(BaseModel):
    """画像参照。`image_id` は `POST /images` か `generate_image` ツールが発行した UUID。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["image"] = "image"
    image_id: str = Field(pattern=_UUID_PATTERN)
    mime: str = Field(min_length=1, max_length=64)
    width: int | None = Field(default=None, ge=1, le=8192)
    height: int | None = Field(default=None, ge=1, le=8192)


class ToolCallPart(BaseModel):
    """アシスタントがツールを呼び出した記録。`args` は OpenAPI の付随ペイロード。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["tool_call"] = "tool_call"
    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=64)
    args: dict[str, Any]
    status: ToolCallStatus


class ToolResultPart(BaseModel):
    """ツールの結果。画像などの成果物は別途 `ImagePart` として同メッセージに同居させる。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["tool_result"] = "tool_result"
    call_id: str = Field(min_length=1, max_length=128)
    summary: str = Field(max_length=1000)
    data: dict[str, Any]


ContentPart = Annotated[
    TextPart | ImagePart | ToolCallPart | ToolResultPart,
    Field(discriminator="type"),
]
"""アシスタント/ツールメッセージで許容される全パート。"""

UserContentPart = Annotated[
    TextPart | ImagePart,
    Field(discriminator="type"),
]
"""ユーザ入力で許容されるパート（tool_call/result はサーバ生成のみ）。"""


class Message(BaseModel):
    """1 ターン 1 メッセージ。`parts` の並びが UI の表示順になる。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    conversation_id: str = Field(min_length=1, max_length=128)
    role: Role
    parts: list[ContentPart] = Field(min_length=1, max_length=32)
    created_at: datetime
    parent_id: str | None = Field(default=None, max_length=128)
