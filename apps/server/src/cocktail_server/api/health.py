from __future__ import annotations

import torch
from fastapi import APIRouter, Request

from cocktail_server.schemas.health import (
    GpuInfo,
    HealthResponse,
    ModelsStatus,
    StartupState,
    StartupStatus,
)

router = APIRouter()


def _gpu_info() -> GpuInfo | None:
    if not torch.cuda.is_available():
        return None
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    free, total = torch.cuda.mem_get_info(idx)
    gib = 1024**3
    return GpuInfo(
        name=name,
        memory_used_mb=(total - free) // (1024 * 1024),
        memory_total_mb=total // (1024 * 1024),
        vram_total_gb=round(total / gib, 2),
        vram_free_gb=round(free / gib, 2),
    )


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    manager = request.app.state.model_manager
    statuses = manager.snapshot_status()
    state: StartupState = getattr(request.app.state, "startup_state", "ready")
    error: str | None = getattr(request.app.state, "startup_error", None)
    return HealthResponse(
        startup=StartupStatus(state=state, error=error),
        gpu=_gpu_info(),
        models=ModelsStatus(llm=statuses["llm"], image=statuses["image"]),
        queue_depth=manager.queue_depth,
        residency_policy=manager.policy,
    )
