from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from cocktail_server.api.chat import router as chat_router
from cocktail_server.api.conversations import router as conversations_router
from cocktail_server.api.generate import router as generate_router
from cocktail_server.api.health import router as health_router
from cocktail_server.api.images import make_images_router
from cocktail_server.config import Settings, get_settings
from cocktail_server.schemas.health import StartupState
from cocktail_server.scripts import fetch_models
from cocktail_server.services.conversation_store import ConversationStore
from cocktail_server.services.image_gen import ImageGenService
from cocktail_server.services.llm import LlmService
from cocktail_server.services.model_manager import ModelManager, Policy
from cocktail_server.services.turn_registry import TurnRegistry

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    """cocktail_server 配下のロガーを stdout に流す。uvicorn の既定構成と衝突させない。"""
    lvl = getattr(logging, level.upper(), logging.INFO)
    pkg_logger = logging.getLogger("cocktail_server")
    pkg_logger.setLevel(lvl)
    if not any(isinstance(h, logging.StreamHandler) for h in pkg_logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        pkg_logger.addHandler(handler)
    pkg_logger.propagate = False


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


async def _run_preload(
    app: FastAPI,
    settings: Settings,
    manager: ModelManager,
    llm: LlmService,
    image_gen: ImageGenService,
    policy: Policy,
) -> None:
    """fetch_models → LLM ロード → (coresident なら) Image ロード を順に実行。

    進捗を `app.state.startup_state` に反映し、終了時に必ず `ready_event` を set する。
    例外時は `startup_error` にメッセージを残し、state を `error` に遷移させる。
    """
    ready_event: asyncio.Event = app.state.ready_event
    try:
        app.state.startup_state = "downloading"
        logger.info("ensuring models on disk…")
        resolved = await asyncio.to_thread(fetch_models.ensure_all, settings)
        if resolved is not None:
            image_gen.set_model_id(str(resolved))

        app.state.startup_state = "loading"
        logger.info("preloading LLM…")
        async with manager.acquire("llm"):
            manager.set_status("llm", "loading")
            await asyncio.to_thread(llm.load)
            manager.set_status("llm", "loaded")
            if policy == "swap":
                # 量子化コストをここで払い切り、以降の LLM 呼び出しを CPU→CUDA
                # memcpy のみで済ませる。スナップショットを作ってから VRAM を解放し、
                # 起動直後も常に swap（warm load）経路で動く状態に揃える。
                logger.info("snapshotting LLM to CPU (swap mode)")
                await asyncio.to_thread(llm.evict_to_cpu)
                manager.set_status("llm", "idle")

        if policy == "coresident":
            logger.info("preloading Image pipeline (coresident)…")
            async with manager.acquire("image"):
                manager.set_status("image", "loading")
                await asyncio.to_thread(image_gen.load)
                manager.set_status("image", "loaded")

        app.state.startup_state = "ready"
        logger.info("Cocktail server ready (host=%s port=%d)", settings.host, settings.port)
    except Exception as exc:
        logger.exception("startup preload failed")
        app.state.startup_state = "error"
        app.state.startup_error = str(exc)
    finally:
        ready_event.set()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.ensure_dirs()
    os.environ.setdefault("HF_HOME", str(settings.hf_home.resolve()))

    policy = _resolve_residency_policy(settings)
    logger.info(
        "residency policy resolved: %s (mode=%s, vram=%s GiB, threshold=%s GiB)",
        policy,
        settings.residency_mode,
        _detect_vram_gb(),
        settings.residency_coresident_threshold_gb,
    )

    llm = LlmService(
        settings.llm_model_id,
        weights_dir=settings.weights_dir,
    )
    image_gen = ImageGenService(settings.image_model_id)
    manager = ModelManager(policy=policy)
    conversations = ConversationStore()
    turn_registry = TurnRegistry()

    async def _evict_llm() -> None:
        # swap モードでは CPU 退避で再活性化を高速化する（再量子化を再実行しない）
        await asyncio.to_thread(llm.evict_to_cpu)
        manager.set_status("llm", "idle")

    async def _evict_image() -> None:
        # swap モードでは CPU 退避で再活性化を高速化する（from_pretrained を再実行しない）
        await asyncio.to_thread(image_gen.evict_to_cpu)
        manager.set_status("image", "idle")

    manager.register_evictor("llm", _evict_llm)
    manager.register_evictor("image", _evict_image)

    ready_event = asyncio.Event()
    initial_state: StartupState = "downloading" if settings.startup_preload else "ready"
    app.state.settings = settings
    app.state.model_manager = manager
    app.state.llm = llm
    app.state.image_gen = image_gen
    app.state.conversations = conversations
    app.state.turn_registry = turn_registry
    app.state.ready_event = ready_event
    app.state.startup_state = initial_state
    app.state.startup_error = None

    preload_task: asyncio.Task[None] | None = None
    if settings.startup_preload:
        # lifespan を即座に完了させ、ポートは bind されるがモデル準備はバックグラウンドで進める。
        preload_task = asyncio.create_task(
            _run_preload(app, settings, manager, llm, image_gen, policy)
        )
    else:
        ready_event.set()

    try:
        yield
    finally:
        logger.info("Cocktail server shutting down")
        if preload_task is not None and not preload_task.done():
            preload_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await preload_task
        await turn_registry.shutdown()
        llm.unload()
        image_gen.unload()


_READY_GATED_PATHS: frozenset[str] = frozenset({"/api/chat", "/api/generate"})
_READY_GATED_PREFIXES: tuple[str, ...] = ("/api/chat/",)


def _is_ready_gated(path: str) -> bool:
    if path in _READY_GATED_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in _READY_GATED_PREFIXES)


async def _await_ready_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """`/chat` / `/generate` は preload 完了まで待機させる。タイムアウトで 503 を返す。"""
    path = request.url.path
    if _is_ready_gated(path):
        ready_event: asyncio.Event | None = getattr(request.app.state, "ready_event", None)
        if ready_event is not None and not ready_event.is_set():
            try:
                await asyncio.wait_for(ready_event.wait(), timeout=900.0)
            except TimeoutError:
                return JSONResponse(
                    {"detail": "server still warming up; please retry"},
                    status_code=503,
                )
        state: StartupState = getattr(request.app.state, "startup_state", "ready")
        if state == "error":
            err = getattr(request.app.state, "startup_error", "") or "unknown error"
            return JSONResponse(
                {"detail": f"server failed to start: {err}"},
                status_code=503,
            )
    return await call_next(request)


def _register_spa(app: FastAPI, dist_dir: Path) -> None:
    """ビルド済み Vite dist があれば同一オリジンで配信する。API は `/api/*` 配下に
    固定しているため、catch-all は `api` 以外を捕まえて、実ファイルがあればそれを、
    なければ `index.html` を返して SPA ルーティングを成立させる。
    """
    dist = dist_dir.resolve()
    index_html = dist / "index.html"
    if not dist.is_dir() or not index_html.is_file():
        logger.info("client dist not found at %s; skipping SPA serving", dist)
        return

    logger.info("serving client SPA from %s", dist)

    @app.get("/{spa_path:path}", include_in_schema=False)
    async def _spa_fallback(spa_path: str) -> FileResponse:
        # `/api/*` に落ちてきた未定義ルートは JSON 404 で返す。index.html を返すと
        # クライアントの fetch が HTML を受け取ってパース失敗する事故が起きる。
        if spa_path == "api" or spa_path.startswith("api/"):
            raise HTTPException(status_code=404)

        # path traversal 対策: dist の外を指す path は全部 index.html に丸める。
        candidate = (dist / spa_path).resolve() if spa_path else index_html
        try:
            candidate.relative_to(dist)
        except ValueError:
            return FileResponse(index_html)
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index_html)


def create_app() -> FastAPI:
    settings = get_settings()
    _configure_logging(settings.log_level)
    app = FastAPI(title="Cocktail", version="0.0.0", lifespan=lifespan)

    app.middleware("http")(_await_ready_middleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API はすべて `/api` 配下に置く。Vite dev server は `/api` のみをプロキシし、
    # それ以外 (`/conversations/xxx` など) は SPA ルーティング用に index.html を返す
    # ため、リロード時や直接 URL を踏んだ時も SPA 側が担当する。
    app.include_router(health_router, prefix="/api")
    app.include_router(generate_router, prefix="/api")
    app.include_router(chat_router, prefix="/api")
    app.include_router(conversations_router, prefix="/api")
    app.include_router(make_images_router(settings.images_dir), prefix="/api")

    # SPA は API ルート登録の後に付ける。先に catch-all があるとルーティング順で
    # `/api/*` まで拾ってしまう。
    _register_spa(app, settings.client_dist_dir)
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
