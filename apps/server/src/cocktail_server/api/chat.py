from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from cocktail_server.schemas.chat import ChatRequest
from cocktail_server.schemas.events import SseEvent
from cocktail_server.services.orchestrator import ChatOrchestrator

router = APIRouter()


def _format_sse(event: BaseModel) -> bytes:
    """pydantic モデルを SSE フレームにエンコード。

    `event: <type>` は SSE の EventSource API の `event` フィールドを立てるため、
    `data:` には同じ JSON を丸ごと入れて discriminator でパースできるようにしておく。
    """
    payload = event.model_dump(mode="json")
    event_name = payload.get("type", "message")
    body = json.dumps(payload, ensure_ascii=False)
    return f"event: {event_name}\ndata: {body}\n\n".encode()


async def _stream(events: AsyncIterator[SseEvent]) -> AsyncIterator[bytes]:
    async for ev in events:
        yield _format_sse(ev)


@router.post(
    "/chat",
    responses={
        200: {
            "description": "Server-Sent Events stream of chat events.",
            "content": {"text/event-stream": {}},
        }
    },
)
async def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    # テストで llm/image_gen を差し替えた後も拾えるよう、毎リクエスト構築する。
    orchestrator = ChatOrchestrator(
        llm=request.app.state.llm,
        image_gen=request.app.state.image_gen,
        manager=request.app.state.model_manager,
        store=request.app.state.conversations,
        settings=request.app.state.settings,
    )
    return StreamingResponse(
        _stream(orchestrator.run(req)),
        media_type="text/event-stream",
        headers={
            # SSE はバッファ禁止。リバースプロキシ越しでも流れるように nginx ヒントを付ける。
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
