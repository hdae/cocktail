from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cocktail_server.schemas.messages import Message


class ConversationEvent(BaseModel):
    """新規会話が生成されたことを通知する（既存 `conversation_id` 指定時は飛ばない）。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["conversation"] = "conversation"
    conversation_id: str


class UserSavedEvent(BaseModel):
    """ユーザメッセージが保存された。UI は楽観的表示をサーバ側 id に置き換える。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["user_saved"] = "user_saved"
    message: Message


class AssistantStartEvent(BaseModel):
    """アシスタントのターン開始。以降の token/tool 系イベントはこの `message_id` に紐づく。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["assistant_start"] = "assistant_start"
    message_id: str


class ToolCallStartEvent(BaseModel):
    """ツール呼び出しの開始。UI はスピナーと args を即時表示する。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["tool_call_start"] = "tool_call_start"
    call_id: str
    name: str
    args: dict[str, Any]


class ToolCallEndEvent(BaseModel):
    """ツール呼び出しの終了。status='error' のときは `data.error` にメッセージを入れる。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["tool_call_end"] = "tool_call_end"
    call_id: str
    status: Literal["done", "error"]
    summary: str
    data: dict[str, Any]


class ImageReadyEvent(BaseModel):
    """画像がディスクに書かれて配信可能になった。便宜上 tool_call_end より前に飛ぶ。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["image_ready"] = "image_ready"
    call_id: str
    image_id: str
    image_url: str
    mime: str
    width: int
    height: int


class AssistantEndEvent(BaseModel):
    """アシスタントメッセージが完成。これが UI にとっての正規状態。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["assistant_end"] = "assistant_end"
    message: Message


class ErrorEvent(BaseModel):
    """致命的エラー。ストリームはこのあと `done` で閉じられる。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["error"] = "error"
    code: str
    message: str


class DoneEvent(BaseModel):
    """ストリーム完了。以降イベントは来ない。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["done"] = "done"


SseEvent = Annotated[
    ConversationEvent
    | UserSavedEvent
    | AssistantStartEvent
    | ToolCallStartEvent
    | ToolCallEndEvent
    | ImageReadyEvent
    | AssistantEndEvent
    | ErrorEvent
    | DoneEvent,
    Field(discriminator="type"),
]
"""`POST /chat` が SSE で流す全イベント。"""
