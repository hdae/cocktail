from __future__ import annotations

import asyncio
import io
import re
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Annotated, Final

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image as PILImage
from PIL.Image import Image

from cocktail_server.schemas.images import (
    GeneratedImageList,
    ImageUploadResponse,
)
from cocktail_server.services.conversation_store import ConversationStore

# keepalive `: ping` を流す間隔（秒）。接続が途切れる前に送ってプロキシ側の idle 切断を防ぐ。
_SSE_PING_INTERVAL_S: Final[float] = 15.0

_UUID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)

# クライアントの「とりあえず添付」を前提に 10MB まで許容。意図的に控えめ。
_MAX_UPLOAD_BYTES: Final[int] = 10 * 1024 * 1024
_ALLOWED_MIMES: Final[frozenset[str]] = frozenset(
    {"image/png", "image/jpeg", "image/jpg", "image/webp"}
)


def _normalize_and_save(raw: bytes, images_dir: Path) -> ImageUploadResponse:
    """受け取ったバイト列を PIL で開き直し、常に webp 正規化して UUID で保存する。

    クライアントから来る MIME は信用しない（偽装される）。PIL でデコードできれば通す。
    EXIF 向きは `PIL.ImageOps.exif_transpose` 相当が PIL 側で勝手に走るので任せる。
    """
    try:
        src: Image = PILImage.open(io.BytesIO(raw))
        src.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unsupported image: {exc}") from exc

    if src.mode not in ("RGB", "RGBA"):
        src = src.convert("RGBA" if "A" in src.mode else "RGB")

    image_id = str(uuid.uuid4())
    path = images_dir / f"{image_id}.webp"
    src.save(path, format="WEBP", quality=92, method=6)

    return ImageUploadResponse(
        image_id=image_id,
        image_url=f"/api/images/{image_id}.webp",
        mime="image/webp",
        width=src.width,
        height=src.height,
    )


def make_images_router(images_dir: Path) -> APIRouter:
    router = APIRouter()

    # NOTE: ルート順序重要。`/images` のフラット一覧および `/images/events` の SSE は
    # `/images/{image_id}.webp` より先に宣言する（FastAPI は宣言順にマッチする）。
    @router.get("/images/events")
    async def stream_image_events(request: Request) -> StreamingResponse:
        """生成画像の broadcast SSE。誰かが `record_generated_image` を呼ぶと全接続者に流れる。

        event 名は `image_created`、payload は `GeneratedImageRef` の JSON。
        15 秒おきに `: ping` コメントを流してプロキシの idle 切断を回避する。
        """
        store: ConversationStore = request.app.state.conversations

        async def event_source() -> AsyncIterator[bytes]:
            async with store.subscribe_images() as queue:
                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        ref = await asyncio.wait_for(queue.get(), timeout=_SSE_PING_INTERVAL_S)
                    except TimeoutError:
                        yield b": ping\n\n"
                        continue
                    payload = ref.model_dump_json()
                    yield f"event: image_created\ndata: {payload}\n\n".encode()

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @router.get("/images", response_model=GeneratedImageList)
    async def list_generated_images(
        request: Request,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        before: Annotated[datetime | None, Query()] = None,
    ) -> GeneratedImageList:
        store: ConversationStore = request.app.state.conversations
        items, next_before = await store.list_all_generated_images(limit=limit, before=before)
        return GeneratedImageList(images=items, next_before=next_before)

    @router.post("/images", response_model=ImageUploadResponse)
    async def upload_image(
        file: Annotated[UploadFile, File(...)],
    ) -> ImageUploadResponse:
        if file.content_type not in _ALLOWED_MIMES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported content type: {file.content_type}",
            )
        raw = await file.read()
        if len(raw) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        if len(raw) > _MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File too large")
        return _normalize_and_save(raw, images_dir)

    @router.get("/images/{image_id}.webp")
    def get_image(image_id: str) -> FileResponse:
        if not _UUID_RE.match(image_id):
            raise HTTPException(status_code=400, detail="Invalid image id")
        path = images_dir / f"{image_id}.webp"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(
            path,
            media_type="image/webp",
            headers={"Cache-Control": "public, immutable, max-age=31536000"},
        )

    return router
