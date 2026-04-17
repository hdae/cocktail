from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from cocktail_server.schemas.generate import AspectRatio, CfgPreset


class ImageUploadResponse(BaseModel):
    """`POST /images` の応答。返ってきた `image_id` を ImagePart に詰めて chat に送る。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    image_id: str
    image_url: str
    mime: str
    width: int
    height: int


class GeneratedImageRef(BaseModel):
    """`generate_image` ツールで生成された画像のメタデータ。ギャラリー横断ビュー用。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    image_id: str
    image_url: str
    conversation_id: str
    created_at: datetime
    prompt: str
    seed: int
    aspect_ratio: AspectRatio
    cfg_preset: CfgPreset
    width: int
    height: int


class GeneratedImageList(BaseModel):
    """`GET /images` の応答。`next_before` を `?before=` に渡すと次ページが取れる。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    images: list[GeneratedImageRef]
    next_before: datetime | None = None
