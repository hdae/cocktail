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
