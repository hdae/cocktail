from __future__ import annotations

import asyncio
import gc
import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from threading import Thread
from typing import Any, Final

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from cocktail_server.schemas.generate import LlmTurnSpec
from cocktail_server.schemas.messages import (
    Message,
    TextPart,
    ToolCallPart,
)
from cocktail_server.services.prompt_builder import (
    NEGATIVE_DEFAULT,
    build_system_prompt,
    build_user_message,
)

logger = logging.getLogger(__name__)

_JSON_OBJECT_RE: Final[re.Pattern[str]] = re.compile(r"\{[\s\S]*\}")
_REASONING_START_RE: Final[re.Pattern[str]] = re.compile(r'"reasoning"\s*:\s*"')


@dataclass(frozen=True)
class LlmTextDelta:
    """`reasoning` フィールドから逐次抽出したユーザ向けテキスト差分。"""

    delta: str


@dataclass(frozen=True)
class LlmTurnComplete:
    """LLM ターン完了。`spec.reasoning` は全文、`spec.tool_calls` は確定済み。"""

    spec: LlmTurnSpec


LlmStreamChunk = LlmTextDelta | LlmTurnComplete


def _decode_partial_reasoning(all_text: str) -> tuple[str, bool]:
    """蓄積された生テキストから `reasoning` 文字列値を部分デコードする。

    Returns:
        (decoded_so_far, ended): `ended=True` なら閉じクォートまで到達。
        途中の不完全エスケープ（末尾 `\\` や `\\uXX`）は保留して次回呼出で再解決。
    """
    m = _REASONING_START_RE.search(all_text)
    if not m:
        return ("", False)
    start = m.end()
    out: list[str] = []
    i = start
    n = len(all_text)
    while i < n:
        c = all_text[i]
        if c == "\\":
            if i + 1 >= n:
                break
            nc = all_text[i + 1]
            if nc == "n":
                out.append("\n")
                i += 2
            elif nc == "t":
                out.append("\t")
                i += 2
            elif nc == "r":
                out.append("\r")
                i += 2
            elif nc == '"':
                out.append('"')
                i += 2
            elif nc == "\\":
                out.append("\\")
                i += 2
            elif nc == "/":
                out.append("/")
                i += 2
            elif nc == "b":
                out.append("\b")
                i += 2
            elif nc == "f":
                out.append("\f")
                i += 2
            elif nc == "u":
                if i + 5 >= n:
                    break
                hexchars = all_text[i + 2 : i + 6]
                try:
                    out.append(chr(int(hexchars, 16)))
                except ValueError:
                    out.append("\\u" + hexchars)
                i += 6
            else:
                out.append(nc)
                i += 2
        elif c == '"':
            return ("".join(out), True)
        else:
            out.append(c)
            i += 1
    return ("".join(out), False)


def _extract_user_text(msg: Message) -> str:
    parts = [p.text for p in msg.parts if isinstance(p, TextPart)]
    return "\n\n".join(parts)


def _reconstruct_assistant_spec(msg: Message) -> str:
    """保存済み assistant Message から、Gemma が前ターンに出した JSON 相当を復元する。

    `TextPart` が reasoning、`ToolCallPart(name=generate_image, status=done)` が tool_calls。
    `status=error` や tool 未呼び出しターン（TextPart のみ）も素直にシリアライズする。
    """
    reasoning = ""
    tool_calls: list[dict[str, Any]] = []
    for p in msg.parts:
        if isinstance(p, TextPart):
            if not reasoning:
                reasoning = p.text
        elif isinstance(p, ToolCallPart):
            if p.name != "generate_image" or p.status != "done":
                continue
            args = p.args
            call: dict[str, Any] = {
                "name": "generate_image",
                "positive": args.get("positive", ""),
                "negative": args.get("negative", NEGATIVE_DEFAULT),
                "aspect_ratio": args.get("aspect_ratio", "portrait"),
                "cfg_preset": args.get("cfg_preset", "standard"),
                "rationale": "",
            }
            if "seed" in args and args["seed"] is not None:
                call["seed"] = args["seed"]
            tool_calls.append(call)
    return json.dumps(
        {"reasoning": reasoning, "tool_calls": tool_calls},
        ensure_ascii=False,
    )


def _build_chat_messages(history: list[Message]) -> list[dict[str, str]]:
    """会話履歴を Gemma の chat_template 入力に変換する。

    最初のユーザターンにだけシステムプロンプトを埋め込む（Gemma の template は
    system ロールを受け付けないので user メッセージに前置する）。
    """
    if not history:
        raise ValueError("history must contain at least one message")

    messages: list[dict[str, str]] = []
    first_user_seen = False
    system_prompt = build_system_prompt()
    for msg in history:
        if msg.role == "user":
            text = _extract_user_text(msg) or "(no text)"
            user_body = build_user_message(text)
            if not first_user_seen:
                content = f"{system_prompt}\n\n{user_body}"
                first_user_seen = True
            else:
                content = user_body
            messages.append({"role": "user", "content": content})
        elif msg.role == "assistant":
            messages.append({"role": "assistant", "content": _reconstruct_assistant_spec(msg)})
        # tool / system ロールのメッセージは現状発行していないので無視
    return messages


def _parse_turn_spec(text: str) -> LlmTurnSpec:
    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        raise ValueError(f"No JSON object found in model output: {text!r}")
    data = json.loads(match.group(0))
    return LlmTurnSpec.model_validate(data)


class LlmService:
    """Gemma 4bit 量子化モデルで日本語指示から `LlmTurnSpec` をストリーム生成する。"""

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._model: Any = None
        self._tokenizer: Any = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading Gemma model: %s", self._model_id)
        quant_config = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )

    def unload(self) -> None:
        # bnb 4bit は .to("cpu") で安定しないため、退避は完全アンロード
        if self._model is None:
            return
        logger.info("Unloading Gemma model")
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def run_turn(self, history: list[Message]) -> AsyncIterator[LlmStreamChunk]:
        """1 ターン分の応答を生成。`reasoning` を `LlmTextDelta` で逐次流し、最後に `LlmTurnComplete`。

        `history` は会話の全メッセージ（末尾が今回のユーザ発話）。過去ターンの
        アシスタント応答は保存済み parts から `LlmTurnSpec` JSON を復元して渡す。

        失敗時は温度 0.3 でリプレイ（同期、text_delta は流さない）。ただし attempt 0 で
        ユーザに見えたテキストは `spec.reasoning` に上書きして整合を取る。
        """
        if self._model is None or self._tokenizer is None:
            await asyncio.to_thread(self.load)

        streamed_reasoning = ""
        try:
            async for chunk in self._stream_once(history, temperature=0.0):
                if isinstance(chunk, LlmTextDelta):
                    streamed_reasoning += chunk.delta
                yield chunk
            return
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("LlmTurnSpec streaming parse failed, retrying: %s", exc)

        spec = await asyncio.to_thread(self._sync_turn, history, 0.3)
        if streamed_reasoning:
            spec = spec.model_copy(update={"reasoning": streamed_reasoning})
        yield LlmTurnComplete(spec=spec)

    async def _stream_once(
        self, history: list[Message], *, temperature: float
    ) -> AsyncIterator[LlmStreamChunk]:
        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None

        messages = _build_chat_messages(history)
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60.0,
        )
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": 1024,
            "pad_token_id": tokenizer.eos_token_id,
            "streamer": streamer,
        }
        if temperature > 0:
            gen_kwargs |= {"do_sample": True, "temperature": temperature, "top_p": 0.9}
        else:
            gen_kwargs["do_sample"] = False

        gen_error: list[BaseException] = []

        def _run_generate() -> None:
            try:
                with torch.inference_mode():
                    model.generate(**inputs, **gen_kwargs)
            except BaseException as exc:
                gen_error.append(exc)
                # 失敗時もストリーマを閉じないとコンシューマがハングする
                streamer.end()  # type: ignore[no-untyped-call]

        thread = Thread(target=_run_generate, daemon=True)
        thread.start()

        all_text = ""
        emitted = ""
        loop = asyncio.get_running_loop()

        try:
            while True:
                chunk = await loop.run_in_executor(None, _safe_next, streamer)
                if chunk is _STOP:
                    break
                assert isinstance(chunk, str)
                all_text += chunk
                decoded, ended = _decode_partial_reasoning(all_text)
                if len(decoded) > len(emitted):
                    yield LlmTextDelta(delta=decoded[len(emitted) :])
                    emitted = decoded
                if ended:
                    # 以降は tool_calls のパース用に蓄積するだけ
                    while True:
                        tail = await loop.run_in_executor(None, _safe_next, streamer)
                        if tail is _STOP:
                            break
                        assert isinstance(tail, str)
                        all_text += tail
                    break
        finally:
            thread.join(timeout=10.0)

        if gen_error:
            raise RuntimeError("Gemma generate failed") from gen_error[0]

        spec = _parse_turn_spec(all_text)
        yield LlmTurnComplete(spec=spec)

    def _sync_turn(self, history: list[Message], temperature: float) -> LlmTurnSpec:
        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None

        messages = _build_chat_messages(history)
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": 1024,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs |= {"do_sample": True, "temperature": temperature, "top_p": 0.9}
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        raw_ids = out[0][prompt_len:]
        text = tokenizer.decode(raw_ids, skip_special_tokens=True)
        return _parse_turn_spec(text)


_STOP: Final[object] = object()


def _safe_next(it: Any) -> Any:
    try:
        return next(it)
    except StopIteration:
        return _STOP
