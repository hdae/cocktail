from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

ModelStatus = Literal["loaded", "loading", "idle", "error"]
ResidencyPolicy = Literal["swap", "coresident"]


class GpuInfo(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    memory_used_mb: int
    memory_total_mb: int
    vram_total_gb: float | None = None
    vram_free_gb: float | None = None


class ModelsStatus(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    llm: ModelStatus
    image: ModelStatus


class HealthResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    status: Literal["ok", "loading"]
    gpu: GpuInfo | None
    models: ModelsStatus
    queue_depth: int
    residency_policy: ResidencyPolicy
