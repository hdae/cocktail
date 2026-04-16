from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ImageUploadResponse(BaseModel):
    """`POST /images` の応答。返ってきた `image_id` を ImagePart に詰めて chat に送る。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    image_id: str
    image_url: str
    mime: str
    width: int
    height: int
