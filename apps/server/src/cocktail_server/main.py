from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cocktail_server.api.chat import router as chat_router
from cocktail_server.api.generate import router as generate_router
from cocktail_server.api.health import router as health_router
from cocktail_server.api.images import make_images_router
from cocktail_server.config import get_settings
from cocktail_server.services.conversation_store import ConversationStore
from cocktail_server.services.image_gen import ImageGenService
from cocktail_server.services.llm import LlmService
from cocktail_server.services.model_manager import ModelManager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.ensure_dirs()
    os.environ.setdefault("HF_HOME", str(settings.hf_home.resolve()))

    llm = LlmService(settings.llm_model_id)
    image_gen = ImageGenService(settings.image_model_id)
    manager = ModelManager()
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
