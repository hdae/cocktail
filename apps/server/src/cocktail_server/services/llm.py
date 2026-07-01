from __future__ import annotations

import asyncio
import ctypes
import gc
import json
import logging
import re
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Final

from json_repair import repair_json
from pydantic import ValidationError

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

# GGUF 推論パラメータ。max_tokens は JSON 1 ターン分。n_ctx はシステムプロンプト
# (~3.4k tok) + 数ターンの履歴を見込んだ余裕。
_MAX_TOKENS: Final[int] = 1024
_N_CTX: Final[int] = 8192


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
                    code = int(hexchars, 16)
                except ValueError:
                    out.append("\\u" + hexchars)
                    i += 6
                else:
                    if 0xD800 <= code <= 0xDBFF:
                        # 上位サロゲート。非 BMP 文字(絵文字・稀な漢字)は \uXXXX\uYYYY の
                        # ペアで来るので、続く \uDCxx と結合して 1 コードポイントにする。
                        # 単独で emit すると lone surrogate になり SSE の UTF-8 encode で落ちる。
                        if i + 11 >= n or all_text[i + 6 : i + 8] != "\\u":
                            break  # 下位サロゲートがまだ/不完全 → 保留して次回再解決
                        try:
                            low = int(all_text[i + 8 : i + 12], 16)
                        except ValueError:
                            low = 0
                        if 0xDC00 <= low <= 0xDFFF:
                            out.append(chr(0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)))
                            i += 12
                        else:
                            out.append(chr(code))
                            i += 6
                    else:
                        out.append(chr(code))
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
                "seed_action": args.get("seed_action", "new"),
                "rationale": "",
            }
            tool_calls.append(call)
    return json.dumps(
        {"reasoning": reasoning, "tool_calls": tool_calls},
        ensure_ascii=False,
    )


def _build_chat_messages(history: list[Message]) -> list[dict[str, Any]]:
    """会話履歴を Gemma の chat_template 入力に変換する。

    各 user メッセージに `[Turn N]` ラベルを埋め、末尾 user には
    `[Turn N / current]` を付けて「今回応答すべきターン」を明示する。turn は
    user 発話の出現順で 1 起点。純チャット応答のみだったターンも 1 としてカウント
    する（case A: user/assistant ペア単位）。

    最初のユーザターンにだけシステムプロンプトを埋め込む（Gemma の template は
    system ロールを受け付けないので user メッセージに前置する）。
    """
    if not history:
        raise ValueError("history must contain at least one message")
    if history[0].role != "user":
        raise ValueError("history must begin with a user message")

    last_user_pos = max(
        (i for i, m in enumerate(history) if m.role == "user"),
        default=-1,
    )

    messages: list[dict[str, Any]] = []
    first_user_seen = False
    system_prompt = build_system_prompt()
    turn_index = 0
    for i, msg in enumerate(history):
        if msg.role == "user":
            turn_index += 1
            text = _extract_user_text(msg) or "(no text)"
            user_body = build_user_message(
                text, turn_index=turn_index, is_current=(i == last_user_pos)
            )
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
    # Gemma の生出力はデバッグに必須（trailing comma や片側クォートなど破綻パターンの同定用）
    logger.info("Gemma raw output: %r", text)
    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        raise ValueError(f"No JSON object found in model output: {text!r}")
    raw = match.group(0)
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        repaired, repair_log = repair_json(raw, return_objects=True, logging=True)
        logger.warning(
            "json.loads failed (%s); json_repair applied %d fix(es): %s",
            exc,
            len(repair_log),
            repair_log,
        )
        if not isinstance(repaired, dict):
            raise ValueError(
                f"json_repair did not yield an object (got {type(repaired).__name__}): {raw!r}"
            ) from exc
        data = repaired
    return LlmTurnSpec.model_validate(data)


def parse_gguf_ref(model_id: str) -> tuple[str, str] | None:
    """`LLM_MODEL_ID` を GGUF 参照として解釈する。

    `"org/repo:weights.gguf"` 形式なら `(repo_id, filename)` を返す。ローカルの
    `.gguf` パスやその他の形式は `None`（呼び出し側がローカルパスとして扱う）。
    """
    if Path(model_id).expanduser().is_file():
        return None
    if ":" in model_id:
        repo, _, filename = model_id.rpartition(":")
        if "/" in repo and filename.endswith(".gguf"):
            return (repo, filename)
    return None


def _preload_cuda_runtime() -> None:
    """libllama.so が要求する CUDA ランタイム(cudart/cublas)を torch 同梱の nvidia
    パッケージから `RTLD_GLOBAL` で先読みする。

    こうしておくと llama.cpp の CUDA バックエンドが、別途 `LD_LIBRARY_PATH` を
    通さなくても依存シンボルを解決できる（cuBLAS は cuBLASLt に依存するので順序厳守）。
    ベストエフォート: 見つからなければ何もしない（ビルドが静的 or LD パスが既に通っている等）。
    """
    try:
        import nvidia
    except ImportError:
        return
    bases = [Path(p) for p in getattr(nvidia, "__path__", [])]
    for sub, name in (
        ("cublas", "libcublasLt.so.12"),
        ("cublas", "libcublas.so.12"),
        ("cuda_runtime", "libcudart.so.12"),
    ):
        for base in bases:
            lib = base / sub / "lib" / name
            if lib.is_file():
                with suppress(OSError):
                    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                break


class LlmService:
    """GGUF 量子化 Gemma を llama.cpp(llama-cpp-python) で動かし、日本語指示から
    `LlmTurnSpec` をストリーム生成する。

    全層 GPU offload(`n_gpu_layers=-1`)で常駐させ、swap は CPU 退避ではなく
    モデル破棄→ディスク再ロードで行う（llama.cpp は CPU パークの口を持たないが、
    `del` で VRAM をドライバに即返すため Anima との 1 プロセス同居が成立する）。
    """

    def __init__(self, model_id: str, *, hf_home: Path) -> None:
        self._model_id = model_id
        self._hf_home = hf_home
        self._llm: Any = None  # llama_cpp.Llama | None
        self._n_ctx = _N_CTX

    def is_loaded(self) -> bool:
        return self._llm is not None

    def load(self) -> None:
        """GGUF を全層 GPU に載せる。既にロード済みなら何もしない。"""
        if self._llm is not None:
            return
        path = self._resolve_model_path()
        _preload_cuda_runtime()
        from llama_cpp import Llama

        logger.info("Loading GGUF LLM: %s (n_ctx=%d)", Path(path).name, self._n_ctx)
        t0 = perf_counter_ns()
        self._llm = Llama(
            model_path=path,
            n_gpu_layers=-1,
            n_ctx=self._n_ctx,
            verbose=False,
        )
        logger.info("load llm (gguf): %.0f ms", (perf_counter_ns() - t0) / 1_000_000)

    def _resolve_model_path(self) -> str:
        """`model_id` をローカル GGUF パスへ解決する。

        `repo:filename` 形式は HuggingFace から取得（`fetch_models` が起動時に
        ダウンロード済みなのでキャッシュ命中で即返る）。ローカルパスはそのまま使う。
        """
        ref = parse_gguf_ref(self._model_id)
        if ref is None:
            p = Path(self._model_id).expanduser()
            if not p.is_file():
                raise FileNotFoundError(f"GGUF が見つかりません: {p}")
            return str(p)
        repo, filename = ref
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=repo, filename=filename, cache_dir=str(self._hf_home))

    def evict_to_cpu(self) -> None:
        """VRAM を解放する。llama.cpp は CPU パークを持たないためモデルを破棄し、
        次の `load` でディスクから再構築する（再ロードは数秒）。"""
        self._free()

    def unload(self) -> None:
        """モデルを完全破棄する。プロセス終了時・障害復旧用。"""
        self._free()

    def _free(self) -> None:
        if self._llm is None:
            return
        logger.info("Freeing GGUF LLM (VRAM release)")
        t0 = perf_counter_ns()
        with suppress(Exception):
            self._llm.close()
        self._llm = None
        gc.collect()
        logger.info("free llm: %.0f ms", (perf_counter_ns() - t0) / 1_000_000)

    async def run_turn(self, history: list[Message]) -> AsyncIterator[LlmStreamChunk]:
        """1 ターン分の応答を生成。`reasoning` を `LlmTextDelta` で逐次流し、最後に `LlmTurnComplete`。

        `history` は会話の全メッセージ（末尾が今回のユーザ発話）。過去ターンの
        アシスタント応答は保存済み parts から `LlmTurnSpec` JSON を復元して渡す。

        失敗時は温度 0.3 でリプレイ（同期、text_delta は流さない）。ただし attempt 0 で
        ユーザに見えたテキストは `spec.reasoning` に上書きして整合を取る。
        """
        if self._llm is None:
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

        try:
            spec = await asyncio.to_thread(self._sync_turn, history, 0.3)
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            # リトライも失敗（出力截断 / スキーマ不一致 / 文脈長超過の ValueError 等）。
            # 例外を上げるとターンが内部エラーで落ちるので、チャットのみ応答に degrade する。
            logger.warning("LLM retry failed (%s); degrading to a chat-only reply", exc)
            fallback = streamed_reasoning or "うまく生成できませんでした。もう一度試してください。"
            yield LlmTurnComplete(spec=LlmTurnSpec(reasoning=fallback))
            return
        if streamed_reasoning:
            spec = spec.model_copy(update={"reasoning": streamed_reasoning})
        yield LlmTurnComplete(spec=spec)

    async def _stream_once(
        self, history: list[Message], *, temperature: float
    ) -> AsyncIterator[LlmStreamChunk]:
        llm = self._llm
        assert llm is not None

        messages = _build_chat_messages(history)
        loop = asyncio.get_running_loop()

        def _open_stream() -> Any:
            return llm.create_chat_completion(
                messages=messages,
                max_tokens=_MAX_TOKENS,
                temperature=temperature,
                stream=True,
            )

        # 生成本体（prompt eval + decode）はブロッキングなので executor で回す。
        stream = await loop.run_in_executor(None, _open_stream)

        all_text = ""
        emitted = ""
        while True:
            chunk = await loop.run_in_executor(None, _safe_next, stream)
            if chunk is _STOP:
                break
            piece = chunk["choices"][0]["delta"].get("content")
            if not piece:
                continue
            all_text += piece
            decoded, _ended = _decode_partial_reasoning(all_text)
            if len(decoded) > len(emitted):
                yield LlmTextDelta(delta=decoded[len(emitted) :])
                emitted = decoded

        spec = _parse_turn_spec(all_text)
        yield LlmTurnComplete(spec=spec)

    def _sync_turn(self, history: list[Message], temperature: float) -> LlmTurnSpec:
        llm = self._llm
        assert llm is not None

        messages = _build_chat_messages(history)
        out = llm.create_chat_completion(
            messages=messages,
            max_tokens=_MAX_TOKENS,
            temperature=temperature,
        )
        text = out["choices"][0]["message"].get("content") or ""
        return _parse_turn_spec(text)


_STOP: Final[object] = object()


def _safe_next(it: Any) -> Any:
    try:
        return next(it)
    except StopIteration:
        return _STOP
