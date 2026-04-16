from __future__ import annotations

import logging
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
    ToolCallEndEvent,
    ToolCallStartEvent,
    UserSavedEvent,
)
from cocktail_server.schemas.messages import (
    ContentPart,
    ImagePart,
    Message,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    UserContentPart,
)
from cocktail_server.services.conversation_store import ConversationStore
from cocktail_server.services.image_gen import ImageGenService
from cocktail_server.services.llm import LlmService
from cocktail_server.services.model_manager import ModelManager

logger = logging.getLogger(__name__)

GENERATE_IMAGE_TOOL = "generate_image"
_TOOL_SUCCESS_SUMMARY = "画像を生成しました"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _save_webp(img: Image, images_dir: Path) -> str:
    image_id = str(uuid.uuid4())
    path = images_dir / f"{image_id}.webp"
    img.save(path, format="WEBP", quality=92, method=6)
    return image_id


def _extract_instruction(parts: list[UserContentPart]) -> str:
    """ユーザの全 text part を 2 行区切りで結合する。ImagePart は M1 では LLM に渡さない。"""
    return "\n\n".join(p.text for p in parts if isinstance(p, TextPart))


class ChatOrchestrator:
    """`POST /chat` の本体。M1 は `generate_image` 固定呼び出しでシンプルに通す。"""

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

        call_id = str(uuid.uuid4())
        instruction_ja = _extract_instruction(req.parts)
        reference_images = [p.image_id for p in req.parts if isinstance(p, ImagePart)]
        tool_args: dict[str, Any] = {"instruction_ja": instruction_ja}
        if reference_images:
            tool_args["reference_images"] = reference_images

        yield ToolCallStartEvent(call_id=call_id, name=GENERATE_IMAGE_TOOL, args=tool_args)

        result = await self._run_generate_image_tool(instruction_ja)

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

    async def _run_generate_image_tool(self, instruction_ja: str) -> dict[str, Any]:
        width = self._settings.default_width
        height = self._settings.default_height
        steps = self._settings.default_steps
        cfg = self._settings.default_cfg
        seed: int | None = None

        start_ns = time.perf_counter_ns()

        self._manager.set_status("llm", "loading")
        try:
            async with self._manager.acquire("llm"):
                self._manager.set_status("llm", "loaded")
                llm_start = time.perf_counter_ns()
                prompt_spec = await self._llm.build_anima_prompt(instruction_ja)
                llm_ms = (time.perf_counter_ns() - llm_start) // 1_000_000
        except Exception:
            self._manager.set_status("llm", "error")
            raise

        self._manager.set_status("image", "loading")
        try:
            async with self._manager.acquire("image"):
                self._manager.set_status("image", "loaded")
                image_start = time.perf_counter_ns()
                img = await self._image_gen.generate(
                    positive=prompt_spec.positive,
                    negative=prompt_spec.negative,
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
            "prompt": prompt_spec.positive,
            "negative_prompt": prompt_spec.negative,
            "rationale": prompt_spec.rationale,
            "params": {
                "width": width,
                "height": height,
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
            },
            "latency_ms": {
                "llm_ms": llm_ms,
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
        call_id: str,
        tool_args: dict[str, Any],
        result: dict[str, Any],
    ) -> Message:
        parts: list[ContentPart] = [
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
        return Message(
            id=message_id,
            conversation_id=conversation_id,
            role="assistant",
            parts=parts,
            created_at=_utcnow(),
            parent_id=parent_id,
        )
