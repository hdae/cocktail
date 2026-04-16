from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from PIL.Image import Image

from cocktail_server.schemas.generate import (
    GenerateParams,
    GenerateRequest,
    GenerateResponse,
    LatencyBreakdown,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _save_webp(img: Image, images_dir: Path) -> str:
    image_id = str(uuid.uuid4())
    path = images_dir / f"{image_id}.webp"
    img.save(path, format="WEBP", quality=92, method=6)
    return image_id


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, request: Request) -> GenerateResponse:
    settings = request.app.state.settings
    manager = request.app.state.model_manager
    llm = request.app.state.llm
    image_gen = request.app.state.image_gen

    width = req.width if req.width is not None else settings.default_width
    height = req.height if req.height is not None else settings.default_height
    steps = req.steps if req.steps is not None else settings.default_steps
    cfg = req.cfg if req.cfg is not None else settings.default_cfg
    seed = req.seed

    start_ns = time.perf_counter_ns()

    manager.set_status("llm", "loading")
    try:
        async with manager.acquire("llm"):
            manager.set_status("llm", "loaded")
            llm_start = time.perf_counter_ns()
            prompt_spec = await llm.build_anima_prompt(req.instruction_ja)
            llm_ms = (time.perf_counter_ns() - llm_start) // 1_000_000
    except HTTPException:
        raise
    except Exception as exc:
        manager.set_status("llm", "error")
        logger.exception("LLM prompt build failed")
        raise HTTPException(status_code=500, detail=f"LLM failed: {exc}") from exc

    manager.set_status("image", "loading")
    try:
        async with manager.acquire("image"):
            manager.set_status("image", "loaded")
            image_start = time.perf_counter_ns()
            img = await image_gen.generate(
                positive=prompt_spec.positive,
                negative=prompt_spec.negative,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                seed=seed,
            )
            image_ms = (time.perf_counter_ns() - image_start) // 1_000_000
    except HTTPException:
        raise
    except Exception as exc:
        manager.set_status("image", "error")
        logger.exception("Image generation failed")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {exc}") from exc

    image_id = _save_webp(img, settings.images_dir)
    total_ms = (time.perf_counter_ns() - start_ns) // 1_000_000

    return GenerateResponse(
        image_id=image_id,
        image_url=f"/images/{image_id}.webp",
        prompt=prompt_spec.positive,
        negative_prompt=prompt_spec.negative,
        params=GenerateParams(width=width, height=height, steps=steps, cfg=cfg, seed=seed),
        latency_ms=LatencyBreakdown(llm_ms=llm_ms, image_gen_ms=image_ms, total_ms=total_ms),
        rationale=prompt_spec.rationale,
    )
