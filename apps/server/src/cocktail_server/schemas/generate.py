from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AspectRatio = Literal["portrait", "landscape", "square"]
SeedAction = Literal["new", "keep"]

ASPECT_RATIO_RESOLUTIONS: dict[AspectRatio, tuple[int, int]] = {
    "portrait": (896, 1152),
    "landscape": (1152, 896),
    "square": (1024, 1024),
}


class GenerateImageCall(BaseModel):
    """Gemma が native tool 形式で出す `generate_image` 呼び出しの検証済み表現。

    `positive` は Danbooru タグ本文。`negative_extra` はこの画像固有の追加ネガのみで、
    固定ベース(`prompt_builder.NEGATIVE_DEFAULT`)はサーバが `compose_negative` で前置する
    （モデルに巨大な negative を自由生成させると temp 次第でリピートループに陥るため）。
    CFG / steps はモード(base/Turbo)でサーバが決め、seed 値も `seed_action` の意図だけ
    選ばせて実値はサーバが `seed_resolver.resolve_seed` で決める。
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    positive: str = Field(min_length=1)
    negative_extra: str = ""
    aspect_ratio: AspectRatio = "portrait"
    seed_action: SeedAction = "new"


class GenerateRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    instruction_ja: str = Field(min_length=1, max_length=2000)
    width: int | None = Field(default=None, ge=256, le=2048)
    height: int | None = Field(default=None, ge=256, le=2048)
    steps: int | None = Field(default=None, ge=1, le=100)
    cfg: float | None = Field(default=None, ge=0.0, le=20.0)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    # dev/デバッグ用の turbo 上書き。None なら settings 既定に従うが、cfg/steps を手動指定した
    # ときは base 整合のため turbo を切る（CFG≈1 蒸留 LoRA を高 CFG で回す破綻を防ぐ）。
    turbo: bool | None = Field(default=None)


class GenerateParams(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    width: int
    height: int
    steps: int
    cfg: float
    seed: int | None


class LatencyBreakdown(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    llm_ms: int
    image_gen_ms: int
    total_ms: int


class GenerateResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    image_id: str
    image_url: str
    prompt: str
    negative_prompt: str
    params: GenerateParams
    latency_ms: LatencyBreakdown
    rationale: str = ""
