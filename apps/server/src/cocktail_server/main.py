from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cocktail_server.api.chat import router as chat_router
from cocktail_server.api.generate import router as generate_router
from cocktail_server.api.health import router as health_router
from cocktail_server.api.images import make_images_router
from cocktail_server.config import Settings, get_settings
from cocktail_server.scripts import fetch_models
from cocktail_server.services.conversation_store import ConversationStore
from cocktail_server.services.image_gen import ImageGenService
from cocktail_server.services.llm import LlmService
from cocktail_server.services.model_manager import ModelManager, Policy

logger = logging.getLogger(__name__)


def _detect_vram_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    total = torch.cuda.get_device_properties(0).total_memory
    return round(total / (1024**3), 2)


def _resolve_residency_policy(settings: Settings) -> Policy:
    """`settings.residency_mode` が auto のときは VRAM で決定、それ以外はそのまま採用。"""
    if settings.residency_mode == "swap":
        return "swap"
    if settings.residency_mode == "coresident":
        return "coresident"
    vram = _detect_vram_gb()
    if vram is None:
        return "swap"
    return "coresident" if vram >= settings.residency_coresident_threshold_gb else "swap"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.ensure_dirs()
    os.environ.setdefault("HF_HOME", str(settings.hf_home.resolve()))

    image_model_path: str | None = settings.image_model_id
    if settings.startup_preload:
        # ディスクにモデルを揃える。AIR 経由の場合は解決済みローカルパスが返る。
        resolved = fetch_models.ensure_all(settings)
        if resolved is not None:
            image_model_path = str(resolved)

    if not image_model_path:
        # startup_preload=False のテスト用フォールバック（実際には app.state.image_gen が差し替えられる）。
        image_model_path = "unresolved"

    policy = _resolve_residency_policy(settings)
    logger.info(
        "residency policy resolved: %s (mode=%s, vram=%s GiB, threshold=%s GiB)",
        policy,
        settings.residency_mode,
        _detect_vram_gb(),
        settings.residency_coresident_threshold_gb,
    )

    llm = LlmService(settings.llm_model_id)
    image_gen = ImageGenService(image_model_path)
    manager = ModelManager(policy=policy)
    conversations = ConversationStore()

    async def _evict_llm() -> None:
        llm.unload()
        manager.set_status("llm", "idle")

    async def _evict_image() -> None:
        image_gen.unload()
        manager.set_status("image", "idle")

    manager.register_evictor("llm", _evict_llm)
    manager.register_evictor("image", _evict_image)

    app.state.settings = settings
    app.state.model_manager = manager
    app.state.llm = llm
    app.state.image_gen = image_gen
    app.state.conversations = conversations

    if settings.startup_preload:
        logger.info("preloading LLM…")
        async with manager.acquire("llm"):
            manager.set_status("llm", "loading")
            llm.load()
            manager.set_status("llm", "loaded")
        if policy == "coresident":
            logger.info("preloading Image pipeline (coresident)…")
            async with manager.acquire("image"):
                manager.set_status("image", "loading")
                image_gen.load()
                manager.set_status("image", "loaded")

    logger.info("Cocktail server ready (host=%s port=%d)", settings.host, settings.port)
    try:
        yield
    finally:
        logger.info("Cocktail server shutting down")
        llm.unload()
        image_gen.unload()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Cocktail", version="0.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(generate_router)
    app.include_router(chat_router)
    app.include_router(make_images_router(settings.images_dir))
    return app


app = create_app()


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "cocktail_server.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
