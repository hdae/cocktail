from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from cocktail_server.schemas.images import GeneratedImageRef
from cocktail_server.schemas.messages import Message


class ConversationDetail(BaseModel):
    """`GET /conversations/{id}` の応答。URL 直開き / リロードで会話を復元するためのスナップショット。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    created_at: datetime
    updated_at: datetime
    messages: list[Message]
    generated_images: list[GeneratedImageRef]


class ConversationSummary(BaseModel):
    """`GET /conversations` の応答要素。左メニューの履歴一覧向けに軽量なメタのみ持つ。

    `title` は会話識別用の短いラベル（最初のユーザテキスト先頭 40 文字、なければ "(新規会話)"）。
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=80)
    created_at: datetime
    updated_at: datetime
    message_count: int = Field(ge=0)
