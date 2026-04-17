from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from cocktail_server.schemas.chat import ChatRequest, ChatStartResponse
from cocktail_server.schemas.events import ErrorEvent, SseEvent
from cocktail_server.services.orchestrator import ChatOrchestrator
from cocktail_server.services.turn_registry import ChatTurn, TurnRegistry

logger = logging.getLogger(__name__)

router = APIRouter()

# pump タスクは fire-and-forget だが、参照を失うと GC で任意に止まり得るため
# 完走するまで強参照を持ち続ける。完了後は done コールバックで自動的に外れる。
_PUMP_TASKS: set[asyncio.Task[None]] = set()


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


async def _pump(registry: TurnRegistry, turn: ChatTurn, events: AsyncIterator[SseEvent]) -> None:
    """オーケストレータ出力を registry に流し込むバックグラウンドタスク。

    例外が飛んだら `ErrorEvent` で包んで publish してから finish する。finish を
    finally で呼ばないと subscriber が sentinel を受け取れず永遠に待つので必ず呼ぶ。
    """
    try:
        async for ev in events:
            registry.publish(turn.turn_id, ev)
    except Exception as exc:
        logger.exception("Chat turn pump failed")
        registry.publish(
            turn.turn_id,
            ErrorEvent(code="internal_error", message=str(exc)),
        )
    finally:
        registry.finish(turn.turn_id)


@router.post("/chat", response_model=ChatStartResponse)
async def start_chat(req: ChatRequest, request: Request) -> ChatStartResponse:
    """チャットターンを開始し、`conversation_id` と `turn_id` を JSON で返す。

    SSE はこの応答には含めず、クライアントが別接続で
    `GET /chat/turns/{turn_id}/events` を購読して受け取る構造。
    これにより瞬間瞬断や複数タブ観測、navigate 中の stream 維持が自然に扱える。
    """
    orchestrator = ChatOrchestrator(
        llm=request.app.state.llm,
        image_gen=request.app.state.image_gen,
        manager=request.app.state.model_manager,
        store=request.app.state.conversations,
        settings=request.app.state.settings,
    )
    registry: TurnRegistry = request.app.state.turn_registry

    conversation_id = await orchestrator.resolve_conversation_id(req)
    if conversation_id is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown conversation_id: {req.conversation_id}",
        )

    turn = registry.register(conversation_id)
    # pump はバックグラウンドで走らせっぱなし。subscriber の有無に関わらず
    # registry がイベントを replay バッファに蓄積するので先に走らせて OK。
    task = asyncio.create_task(_pump(registry, turn, orchestrator.run_stream(req, conversation_id)))
    _PUMP_TASKS.add(task)
    task.add_done_callback(_PUMP_TASKS.discard)

    return ChatStartResponse(conversation_id=conversation_id, turn_id=turn.turn_id)


@router.get(
    "/chat/turns/{turn_id}/events",
    responses={
        200: {
            "description": "Server-Sent Events stream of chat events for the turn.",
            "content": {"text/event-stream": {}},
        },
        404: {"description": "Unknown turn_id (expired retention or never created)."},
    },
)
async def subscribe_turn(turn_id: str, request: Request) -> StreamingResponse:
    """ターンの SSE を別接続で購読する。完走済みなら replay だけ流してすぐ閉じる。"""
    registry: TurnRegistry = request.app.state.turn_registry
    if registry.get(turn_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown turn_id: {turn_id}")

    async def _events() -> AsyncIterator[bytes]:
        async with registry.subscribe(turn_id) as iterator:
            async for ev in iterator:
                yield _format_sse(ev)

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
