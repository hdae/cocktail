from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AspectRatio = Literal["portrait", "landscape", "square"]
CfgPreset = Literal["soft", "standard", "crisp"]
SeedAction = Literal["new", "keep"]

ASPECT_RATIO_RESOLUTIONS: dict[AspectRatio, tuple[int, int]] = {
    "portrait": (896, 1152),
    "landscape": (1152, 896),
    "square": (1024, 1024),
}

CFG_PRESET_VALUES: dict[CfgPreset, float] = {
    "soft": 3.5,
    "standard": 4.0,
    "crisp": 4.5,
}


class PromptSpec(BaseModel):
    """Anima 向けポジ/ネガ本体。`GenerateImageCall` の内部形としても流用する。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    positive: str = Field(min_length=1)
    negative: str = Field(min_length=1)
    rationale: str = ""


class GenerateImageCall(BaseModel):
    """Gemma が選ぶ `generate_image` ツール呼び出し。

    `aspect_ratio` と `cfg_preset` はプリセット指定。seed 値は Gemma に扱わせず、
    `seed_action` ("new"=採番し直す / "keep"=前回と同じ seed を維持) の意図だけ
    選ばせる。実際の seed 値はサーバが `seed_resolver.resolve_seed` で決める。
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    name: Literal["generate_image"] = "generate_image"
    positive: str = Field(min_length=1)
    negative: str = Field(min_length=1)
    aspect_ratio: AspectRatio = "portrait"
    cfg_preset: CfgPreset = "standard"
    seed_action: SeedAction = "new"
    rationale: str = ""


class LlmTurnSpec(BaseModel):
    """Gemma の 1 ターン出力。日本語テキスト + 0 or 1 件のツール呼び出し。

    `reasoning` は UI にそのまま出すユーザ向け日本語テキスト。
    `tool_calls` が空ならツール実行せずに閉じる（「ありがとう」などへの返答）。
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    reasoning: str = ""
    tool_calls: list[GenerateImageCall] = Field(default_factory=list, max_length=1)


class GenerateRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    instruction_ja: str = Field(min_length=1, max_length=2000)
    width: int | None = Field(default=None, ge=256, le=2048)
    height: int | None = Field(default=None, ge=256, le=2048)
    steps: int | None = Field(default=None, ge=1, le=100)
    cfg: float | None = Field(default=None, ge=0.0, le=20.0)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)


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
