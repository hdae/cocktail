from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

_UUID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def make_images_router(images_dir: Path) -> APIRouter:
    router = APIRouter()

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
