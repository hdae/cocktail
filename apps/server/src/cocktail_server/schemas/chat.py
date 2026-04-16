from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cocktail_server.schemas.messages import TextPart, UserContentPart


class ChatRequest(BaseModel):
    """`POST /chat` のリクエスト。

    `conversation_id=None` のときはサーバが新規会話を作り、SSE 先頭で `conversation` イベント
    を流す。`parts` には少なくとも 1 つのテキストパートが必要（LLM を駆動するため）。
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    conversation_id: str | None = Field(default=None, max_length=128)
    parts: list[UserContentPart] = Field(min_length=1, max_length=32)
    parent_id: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def _require_text(self) -> ChatRequest:
        if not any(isinstance(p, TextPart) for p in self.parts):
            raise ValueError("parts must contain at least one text part")
        return self
