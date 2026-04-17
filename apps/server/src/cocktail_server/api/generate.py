from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from PIL.Image import Image

from cocktail_server.schemas.generate import (
    ASPECT_RATIO_RESOLUTIONS,
    CFG_PRESET_VALUES,
    GenerateParams,
    GenerateRequest,
    GenerateResponse,
    LatencyBreakdown,
    LlmTurnSpec,
)
from cocktail_server.schemas.messages import Message, TextPart
from cocktail_server.services.llm import LlmTurnComplete
from cocktail_server.services.seed_resolver import resolve_seed

logger = logging.getLogger(__name__)

router = APIRouter()


def _save_webp(img: Image, images_dir: Path) -> str:
    image_id = str(uuid.uuid4())
    path = images_dir / f"{image_id}.webp"
    img.save(path, format="WEBP", quality=92, method=6)
    return image_id


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, request: Request) -> GenerateResponse:
    """同期版 `/generate`。M0 互換のスクリプト用 API。

    新実装では Gemma から `LlmTurnSpec` を受け取り、1 件目の `generate_image` ツール
    呼び出しを採用する。ツール呼び出しがないレスポンスは 500 で返す。
    """
    settings = request.app.state.settings
    manager = request.app.state.model_manager
    llm = request.app.state.llm
    image_gen = request.app.state.image_gen

    start_ns = time.perf_counter_ns()

    synthetic_history = [
        Message(
            id=str(uuid.uuid4()),
            conversation_id=str(uuid.uuid4()),
            role="user",
            parts=[TextPart(text=req.instruction_ja)],
            created_at=datetime.now(UTC),
        )
    ]

    manager.set_status("llm", "loading")
    spec: LlmTurnSpec | None = None
    try:
        async with manager.acquire("llm"):
            manager.set_status("llm", "loaded")
            llm_start = time.perf_counter_ns()
            async for chunk in llm.run_turn(synthetic_history):
                if isinstance(chunk, LlmTurnComplete):
                    spec = chunk.spec
            llm_ms = (time.perf_counter_ns() - llm_start) // 1_000_000
    except HTTPException:
        raise
    except Exception as exc:
        manager.set_status("llm", "error")
        logger.exception("LLM turn failed")
        raise HTTPException(status_code=500, detail=f"LLM failed: {exc}") from exc

    if spec is None or not spec.tool_calls:
        raise HTTPException(
            status_code=500,
            detail="Gemma did not request image generation for this instruction",
        )
    call = spec.tool_calls[0]

    preset_width, preset_height = ASPECT_RATIO_RESOLUTIONS[call.aspect_ratio]
    width = req.width if req.width is not None else preset_width
    height = req.height if req.height is not None else preset_height
    steps = req.steps if req.steps is not None else settings.default_steps
    cfg = req.cfg if req.cfg is not None else CFG_PRESET_VALUES[call.cfg_preset]
    # `/generate` は会話を持たないので last_image_seed は常に None
    # （`seed_action="keep"` を指定されたら内部で新規採番に縮退する）
    seed = resolve_seed(req_seed=req.seed, action=call.seed_action, last_image_seed=None)

    manager.set_status("image", "loading")
    try:
        async with manager.acquire("image"):
            manager.set_status("image", "loaded")
            image_start = time.perf_counter_ns()
            img = await image_gen.generate(
                positive=call.positive,
                negative=call.negative,
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
        prompt=call.positive,
        negative_prompt=call.negative,
        params=GenerateParams(width=width, height=height, steps=steps, cfg=cfg, seed=seed),
        latency_ms=LatencyBreakdown(llm_ms=llm_ms, image_gen_ms=image_ms, total_ms=total_ms),
        rationale=call.rationale,
    )
