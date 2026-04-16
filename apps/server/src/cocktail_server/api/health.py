from __future__ import annotations

import torch
from fastapi import APIRouter, Request

from cocktail_server.schemas.health import GpuInfo, HealthResponse, ModelsStatus

router = APIRouter()


def _gpu_info() -> GpuInfo | None:
    if not torch.cuda.is_available():
        return None
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    free, total = torch.cuda.mem_get_info(idx)
    return GpuInfo(
        name=name,
        memory_used_mb=(total - free) // (1024 * 1024),
        memory_total_mb=total // (1024 * 1024),
    )


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    manager = request.app.state.model_manager
    statuses = manager.snapshot_status()
    return HealthResponse(
        status="ok",
        gpu=_gpu_info(),
        models=ModelsStatus(llm=statuses["llm"], image=statuses["image"]),
        queue_depth=manager.queue_depth,
    )
