from __future__ import annotations

import logging
import secrets
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL.Image import Image

from cocktail_server.config import Settings
from cocktail_server.schemas.chat import ChatRequest
from cocktail_server.schemas.events import (
    AssistantEndEvent,
    AssistantStartEvent,
    ConversationEvent,
    DoneEvent,
    ErrorEvent,
    ImageReadyEvent,
    SseEvent,
    TextDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    UserSavedEvent,
)
from cocktail_server.schemas.generate import (
    ASPECT_RATIO_RESOLUTIONS,
    CFG_PRESET_VALUES,
    GenerateImageCall,
    LlmTurnSpec,
)
from cocktail_server.schemas.messages import (
    ContentPart,
    ImagePart,
    Message,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from cocktail_server.services.conversation_store import ConversationStore
from cocktail_server.services.image_gen import ImageGenService
from cocktail_server.services.llm import LlmService, LlmTextDelta, LlmTurnComplete
from cocktail_server.services.model_manager import ModelManager

logger = logging.getLogger(__name__)

GENERATE_IMAGE_TOOL = "generate_image"
_TOOL_SUCCESS_SUMMARY = "画像を生成しました"
_SEED_MAX = 2**63 - 1


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _save_webp(img: Image, images_dir: Path) -> str:
    image_id = str(uuid.uuid4())
    path = images_dir / f"{image_id}.webp"
    img.save(path, format="WEBP", quality=92, method=6)
    return image_id


class ChatOrchestrator:
    """`POST /chat` の本体。Gemma の `LlmTurnSpec` に従ってツール有無を分岐。"""

    def __init__(
        self,
        *,
        llm: LlmService,
        image_gen: ImageGenService,
        manager: ModelManager,
        store: ConversationStore,
        settings: Settings,
    ) -> None:
        self._llm = llm
        self._image_gen = image_gen
        self._manager = manager
        self._store = store
        self._settings = settings

    async def run(self, req: ChatRequest) -> AsyncIterator[SseEvent]:
        try:
            async for event in self._run_inner(req):
                yield event
        except Exception as exc:
            logger.exception("Chat orchestration failed")
            yield ErrorEvent(code="internal_error", message=str(exc))
        yield DoneEvent()

    async def _run_inner(self, req: ChatRequest) -> AsyncIterator[SseEvent]:
        conversation_id = await self._resolve_conversation(req)
        if conversation_id is None:
            yield ErrorEvent(
                code="conversation_not_found",
                message=f"Unknown conversation_id: {req.conversation_id}",
            )
            return
        if req.conversation_id is None:
            yield ConversationEvent(conversation_id=conversation_id)

        user_message = await self._save_user_message(conversation_id, req)
        yield UserSavedEvent(message=user_message)

        assistant_message_id = str(uuid.uuid4())
        yield AssistantStartEvent(message_id=assistant_message_id)

        reference_images = [p.image_id for p in req.parts if isinstance(p, ImagePart)]
        history = await self._store.list_messages(conversation_id)

        # --- 1) Gemma ターン: reasoning を逐次流し、最後に LlmTurnSpec を受け取る ---
        self._manager.set_status("llm", "loading")
        spec: LlmTurnSpec | None = None
        try:
            async with self._manager.acquire("llm"):
                self._manager.set_status("llm", "loaded")
                async for chunk in self._llm.run_turn(history):
                    if isinstance(chunk, LlmTextDelta):
                        if chunk.delta:
                            yield TextDeltaEvent(
                                message_id=assistant_message_id,
                                delta=chunk.delta,
                            )
                    elif isinstance(chunk, LlmTurnComplete):
                        spec = chunk.spec
        except Exception:
            self._manager.set_status("llm", "error")
            raise
        assert spec is not None, "LLM stream ended without LlmTurnComplete"

        # --- 2) ツール呼び出しが無ければここで閉じる（純粋な会話応答） ---
        if not spec.tool_calls:
            assistant_message = self._build_chat_only_message(
                message_id=assistant_message_id,
                conversation_id=conversation_id,
                parent_id=user_message.id,
                reasoning=spec.reasoning,
            )
            await self._store.append(conversation_id, assistant_message)
            yield AssistantEndEvent(message=assistant_message)
            return

        # --- 3) generate_image ツール実行 ---
        tool_call = spec.tool_calls[0]
        call_id = str(uuid.uuid4())
        resolved_seed = (
            tool_call.seed if tool_call.seed is not None else secrets.randbelow(_SEED_MAX)
        )
        tool_args = self._build_tool_args(tool_call, reference_images, seed=resolved_seed)
        yield ToolCallStartEvent(call_id=call_id, name=GENERATE_IMAGE_TOOL, args=tool_args)

        try:
            result = await self._run_generate_image_tool(tool_call, seed=resolved_seed)
        except Exception as exc:
            logger.exception("generate_image tool failed")
            yield ToolCallEndEvent(
                call_id=call_id,
                status="error",
                summary="画像生成に失敗しました",
                data={"error": str(exc)},
            )
            assistant_message = self._build_error_message(
                message_id=assistant_message_id,
                conversation_id=conversation_id,
                parent_id=user_message.id,
                reasoning=spec.reasoning,
                call_id=call_id,
                tool_args=tool_args,
                error=str(exc),
            )
            await self._store.append(conversation_id, assistant_message)
            yield AssistantEndEvent(message=assistant_message)
            return

        yield ImageReadyEvent(
            call_id=call_id,
            image_id=result["image_id"],
            image_url=result["image_url"],
            mime="image/webp",
            width=result["params"]["width"],
            height=result["params"]["height"],
        )
        yield ToolCallEndEvent(
            call_id=call_id,
            status="done",
            summary=_TOOL_SUCCESS_SUMMARY,
            data=result,
        )

        assistant_message = self._build_assistant_message(
            message_id=assistant_message_id,
            conversation_id=conversation_id,
            parent_id=user_message.id,
            reasoning=spec.reasoning,
            call_id=call_id,
            tool_args=tool_args,
            result=result,
        )
        await self._store.append(conversation_id, assistant_message)
        yield AssistantEndEvent(message=assistant_message)

    async def _resolve_conversation(self, req: ChatRequest) -> str | None:
        if req.conversation_id is None:
            return await self._store.create()
        if not await self._store.exists(req.conversation_id):
            return None
        return req.conversation_id

    async def _save_user_message(self, conversation_id: str, req: ChatRequest) -> Message:
        parts: list[ContentPart] = []
        for p in req.parts:
            parts.append(p)
        msg = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role="user",
            parts=parts,
            created_at=_utcnow(),
            parent_id=req.parent_id,
        )
        await self._store.append(conversation_id, msg)
        return msg

    def _build_tool_args(
        self,
        call: GenerateImageCall,
        reference_images: list[str],
        *,
        seed: int,
    ) -> dict[str, Any]:
        width, height = ASPECT_RATIO_RESOLUTIONS[call.aspect_ratio]
        args: dict[str, Any] = {
            "positive": call.positive,
            "negative": call.negative,
            "aspect_ratio": call.aspect_ratio,
            "cfg_preset": call.cfg_preset,
            "width": width,
            "height": height,
            "cfg": CFG_PRESET_VALUES[call.cfg_preset],
            "steps": self._settings.default_steps,
            "seed": seed,
        }
        if reference_images:
            args["reference_images"] = reference_images
        return args

    async def _run_generate_image_tool(
        self, call: GenerateImageCall, *, seed: int
    ) -> dict[str, Any]:
        width, height = ASPECT_RATIO_RESOLUTIONS[call.aspect_ratio]
        cfg = CFG_PRESET_VALUES[call.cfg_preset]
        steps = self._settings.default_steps

        start_ns = time.perf_counter_ns()

        self._manager.set_status("image", "loading")
        try:
            async with self._manager.acquire("image"):
                self._manager.set_status("image", "loaded")
                image_start = time.perf_counter_ns()
                img = await self._image_gen.generate(
                    positive=call.positive,
                    negative=call.negative,
                    width=width,
                    height=height,
                    steps=steps,
                    cfg=cfg,
                    seed=seed,
                )
                image_ms = (time.perf_counter_ns() - image_start) // 1_000_000
        except Exception:
            self._manager.set_status("image", "error")
            raise

        image_id = _save_webp(img, self._settings.images_dir)
        total_ms = (time.perf_counter_ns() - start_ns) // 1_000_000

        return {
            "image_id": image_id,
            "image_url": f"/images/{image_id}.webp",
            "prompt": call.positive,
            "negative_prompt": call.negative,
            "aspect_ratio": call.aspect_ratio,
            "cfg_preset": call.cfg_preset,
            "rationale": call.rationale,
            "params": {
                "width": width,
                "height": height,
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
            },
            "latency_ms": {
                "image_gen_ms": image_ms,
                "total_ms": total_ms,
            },
        }

    def _build_assistant_message(
        self,
        *,
        message_id: str,
        conversation_id: str,
        parent_id: str,
        reasoning: str,
        call_id: str,
        tool_args: dict[str, Any],
        result: dict[str, Any],
    ) -> Message:
        parts: list[ContentPart] = []
        if reasoning:
            parts.append(TextPart(text=reasoning))
        parts.extend(
            [
                ToolCallPart(
                    id=call_id,
                    name=GENERATE_IMAGE_TOOL,
                    args=tool_args,
                    status="done",
                ),
                ToolResultPart(
                    call_id=call_id,
                    summary=_TOOL_SUCCESS_SUMMARY,
                    data=result,
                ),
                ImagePart(
                    image_id=result["image_id"],
                    mime="image/webp",
                    width=result["params"]["width"],
                    height=result["params"]["height"],
                ),
            ]
        )
        return Message(
            id=message_id,
            conversation_id=conversation_id,
            role="assistant",
            parts=parts,
            created_at=_utcnow(),
            parent_id=parent_id,
        )

    def _build_chat_only_message(
        self,
        *,
        message_id: str,
        conversation_id: str,
        parent_id: str,
        reasoning: str,
    ) -> Message:
        text = reasoning or "（返答なし）"
        return Message(
            id=message_id,
            conversation_id=conversation_id,
            role="assistant",
            parts=[TextPart(text=text)],
            created_at=_utcnow(),
            parent_id=parent_id,
        )

    def _build_error_message(
        self,
        *,
        message_id: str,
        conversation_id: str,
        parent_id: str,
        reasoning: str,
        call_id: str,
        tool_args: dict[str, Any],
        error: str,
    ) -> Message:
        parts: list[ContentPart] = []
        if reasoning:
            parts.append(TextPart(text=reasoning))
        parts.extend(
            [
                ToolCallPart(
                    id=call_id,
                    name=GENERATE_IMAGE_TOOL,
                    args=tool_args,
                    status="error",
                ),
                ToolResultPart(
                    call_id=call_id,
                    summary="画像生成に失敗しました",
                    data={"error": error},
                ),
            ]
        )
        return Message(
            id=message_id,
            conversation_id=conversation_id,
            role="assistant",
            parts=parts,
            created_at=_utcnow(),
            parent_id=parent_id,
        )
