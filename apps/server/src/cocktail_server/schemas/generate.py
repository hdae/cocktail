from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PromptSpec(BaseModel):
    """Gemma が生成する Anima 向けプロンプト仕様（JSON 構造化出力の受け皿）"""

    model_config = ConfigDict(strict=True, extra="forbid")

    positive: str = Field(min_length=1)
    negative: str = Field(min_length=1)
    rationale: str = ""


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
