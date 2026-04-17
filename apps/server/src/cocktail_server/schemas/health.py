from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

ModelStatus = Literal["loaded", "loading", "idle", "error"]
ResidencyPolicy = Literal["swap", "coresident"]
# 起動フェーズ: DL 中 → GPU ロード中 → 準備完了 → 失敗。
StartupState = Literal["downloading", "loading", "ready", "error"]


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


class StartupStatus(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    state: StartupState
    error: str | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    startup: StartupStatus
    gpu: GpuInfo | None
    models: ModelsStatus
    queue_depth: int
    residency_policy: ResidencyPolicy
