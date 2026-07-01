from __future__ import annotations

import asyncio
import ctypes
import gc
import logging
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Final

from pydantic import ValidationError

from cocktail_server.schemas.generate import GenerateImageCall
from cocktail_server.schemas.messages import (
    Message,
    TextPart,
    ToolCallPart,
)
from cocktail_server.services.native_tools import (
    NativeToolStream,
    ParsedTurn,
    parse_native_output,
    render_tool_call,
)
from cocktail_server.services.prompt_builder import (
    GENERATE_IMAGE_TOOL,
    build_system_prompt,
    build_user_message,
)

logger = logging.getLogger(__name__)

# GGUF 推論パラメータ。max_tokens は会話 + 1 ツール分。n_ctx はシステムプロンプト
# + 数ターンの履歴を見込んだ余裕。
_MAX_TOKENS: Final[int] = 1024
_N_CTX: Final[int] = 8192


@dataclass(frozen=True)
class LlmTextDelta:
    """ユーザ向け会話テキストのストリーム差分（native の会話領域から逐次抽出）。"""

    delta: str


@dataclass(frozen=True)
class LlmTurnResult:
    """1 ターンの確定結果。`text`=会話テキスト、`thought`=非表示の思考、
    `tool_calls`=検証済み呼び出し（Phase 1 は 0 or 1 件の generate_image）。"""

    text: str
    thought: str
    tool_calls: list[GenerateImageCall]


@dataclass(frozen=True)
class LlmTurnComplete:
    """LLM ターン完了。`result` は確定済み。"""

    result: LlmTurnResult


LlmStreamChunk = LlmTextDelta | LlmTurnComplete


def _extract_user_text(msg: Message) -> str:
    parts = [p.text for p in msg.parts if isinstance(p, TextPart)]
    return "\n\n".join(parts)


def _reconstruct_assistant_turn(msg: Message) -> str:
    """保存済み assistant Message を、Gemma に replay する形式へ復元する。

    会話テキスト(TextPart) に加え、generate_image を出したターンは native 形式
    (`<|tool_call>call:generate_image{...}<tool_call|>`) で replay する。過去ツール呼び出しを
    記述的な注記で見せると、モデルがそれを真似て tool を呼ばなくなる／ASCII クオートに
    崩れることが実機で確認されたため、モデル自身が出す形式に一致させて多ターンの形式
    ドリフトを断つ。positive も含めるので「n個前の絵」参照・再調整に使える。
    """
    text = ""
    tool_repr = ""
    for p in msg.parts:
        if isinstance(p, TextPart):
            if not text:
                text = p.text
        elif isinstance(p, ToolCallPart):
            if p.name != "generate_image" or p.status != "done":
                continue
            tool_repr = render_tool_call(
                "generate_image",
                {
                    "aspect_ratio": str(p.args.get("aspect_ratio", "portrait")),
                    "positive": str(p.args.get("positive", "")),
                    "seed_action": str(p.args.get("seed_action", "new")),
                },
            )
    segments = [s for s in (text, tool_repr) if s]
    return "\n\n".join(segments) or "(no response)"


def _build_chat_messages(history: list[Message]) -> list[dict[str, Any]]:
    """会話履歴を Gemma の chat_completion 入力に変換する。

    先頭に system ロールでペルソナ + タグ規約を置く（埋込 Gemma 4 テンプレは system を
    受け付ける）。各 user に `[Turn N]` ラベルを埋め、末尾 user に `[Turn N / current]`。
    純チャット応答のみだったターンも 1 としてカウントする（user/assistant ペア単位）。
    """
    if not history:
        raise ValueError("history must contain at least one message")
    if history[0].role != "user":
        raise ValueError("history must begin with a user message")

    last_user_pos = max(
        (i for i, m in enumerate(history) if m.role == "user"),
        default=-1,
    )

    messages: list[dict[str, Any]] = [{"role": "system", "content": build_system_prompt()}]
    turn_index = 0
    for i, msg in enumerate(history):
        if msg.role == "user":
            turn_index += 1
            text = _extract_user_text(msg) or "(no text)"
            messages.append(
                {
                    "role": "user",
                    "content": build_user_message(
                        text, turn_index=turn_index, is_current=(i == last_user_pos)
                    ),
                }
            )
        elif msg.role == "assistant":
            messages.append({"role": "assistant", "content": _reconstruct_assistant_turn(msg)})
        # tool / system ロールのメッセージは現状発行していないので無視
    return messages


def _build_result(parsed: ParsedTurn, *, text: str | None = None) -> LlmTurnResult:
    """パース済み native 出力を検証済み `LlmTurnResult` に変換する。

    ツール引数は `GenerateImageCall` で検証する（不正 aspect_ratio や空 positive は
    ValidationError となり、呼び出し側でリトライ/縮退に回る）。Phase 1 は generate_image
    のみ配線し、1 件目だけ採用する。

    `text` を渡すとその値を会話テキストにする（ストリーミング経路で「実際に流した
    テキスト」を渡し、永続化テキストと逐次表示の乖離を構造的に無くすため）。None なら
    `parsed.text`（非ストリームのリトライ経路で使う）。
    """
    calls: list[GenerateImageCall] = []
    for tc in parsed.tool_calls:
        if tc.name != "generate_image":
            continue
        calls.append(
            GenerateImageCall.model_validate(
                {
                    "positive": tc.args.get("positive", ""),
                    "negative_extra": tc.args.get("negative_extra", ""),
                    "aspect_ratio": tc.args.get("aspect_ratio", "portrait"),
                    "seed_action": tc.args.get("seed_action", "new"),
                }
            )
        )
        break
    return LlmTurnResult(
        text=parsed.text if text is None else text,
        thought=parsed.thought,
        tool_calls=calls,
    )


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
    """GGUF 量子化 Gemma 4 を llama.cpp(llama-cpp-python) で動かし、日本語指示から
    会話テキスト + `generate_image` ツール呼び出しをストリーム生成する。

    Gemma 4 の native tool 形式(`<|tool_call>...`)は binding が構造化パースできない
    (issue #2227)ため、`native_tools` でサーバ側パースする（一時措置）。

    全層 GPU offload(`n_gpu_layers=-1`)で常駐させ、swap は CPU 退避ではなく
    モデル破棄→ディスク再ロードで行う（llama.cpp は CPU パークの口を持たないが、
    `del` で VRAM をドライバに即返すため Anima との 1 プロセス同居が成立する）。
    """

    def __init__(
        self,
        model_id: str,
        *,
        hf_home: Path,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
    ) -> None:
        self._model_id = model_id
        self._hf_home = hf_home
        self._llm: Any = None  # llama_cpp.Llama | None
        self._n_ctx = _N_CTX
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._repeat_penalty = repeat_penalty

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
        """1 ターン分を生成。会話テキストを `LlmTextDelta` で逐次流し、最後に `LlmTurnComplete`。

        `history` は会話の全メッセージ（末尾が今回のユーザ発話）。過去ターンの
        アシスタント応答は保存済み parts からプレーンテキストで復元して渡す。

        ストリーム中の native パースに失敗したら、同期でリプレイ（text_delta は流さない）。
        attempt 0 でユーザに見えたテキストは `result.text` に上書きして整合を取る。
        リトライも失敗したらチャットのみ応答へ縮退する（ターンを内部エラーで落とさない）。
        """
        if self._llm is None:
            await asyncio.to_thread(self.load)

        streamed_text = ""
        try:
            async for chunk in self._stream_once(history, temperature=self._temperature):
                if isinstance(chunk, LlmTextDelta):
                    streamed_text += chunk.delta
                yield chunk
            return
        except (ValueError, ValidationError) as exc:
            logger.warning("native turn streaming parse failed, retrying: %s", exc)

        try:
            result = await asyncio.to_thread(self._sync_turn, history)
        except (ValueError, ValidationError) as exc:
            logger.warning("LLM retry failed (%s); degrading to a chat-only reply", exc)
            fallback = streamed_text or "うまく応答できませんでした。もう一度試してください。"
            yield LlmTurnComplete(result=LlmTurnResult(text=fallback, thought="", tool_calls=[]))
            return
        if streamed_text:
            result = replace(result, text=streamed_text)
        yield LlmTurnComplete(result=result)

    def _create_kwargs(self, history: list[Message], *, temperature: float) -> dict[str, Any]:
        return {
            "messages": _build_chat_messages(history),
            "tools": [GENERATE_IMAGE_TOOL],
            "tool_choice": "auto",
            "max_tokens": _MAX_TOKENS,
            "temperature": temperature,
            "top_p": self._top_p,
            "top_k": self._top_k,
            "repeat_penalty": self._repeat_penalty,
        }

    async def _stream_once(
        self, history: list[Message], *, temperature: float
    ) -> AsyncIterator[LlmStreamChunk]:
        llm = self._llm
        assert llm is not None

        kwargs = self._create_kwargs(history, temperature=temperature)
        loop = asyncio.get_running_loop()

        def _open_stream() -> Any:
            return llm.create_chat_completion(stream=True, **kwargs)

        # 生成本体（prompt eval + decode）はブロッキングなので executor で回す。
        stream = await loop.run_in_executor(None, _open_stream)

        parser = NativeToolStream()
        streamed = ""
        while True:
            chunk = await loop.run_in_executor(None, _safe_next, stream)
            if chunk is _STOP:
                break
            piece = chunk["choices"][0]["delta"].get("content")
            if not piece:
                continue
            new_text = parser.feed(piece)
            if new_text:
                streamed += new_text
                yield LlmTextDelta(delta=new_text)
        # マーカーが来ないまま終わった場合に保留していた末尾を確定して流す。
        tail = parser.flush()
        if tail:
            streamed += tail
            yield LlmTextDelta(delta=tail)

        parsed = parse_native_output(parser.raw)
        # 生出力は破綻パターン同定に必須。thought は UI 非表示なのでログにだけ残す。
        logger.info("Gemma raw output: %r", parser.raw)
        if parsed.thought:
            logger.info("Gemma thought (hidden): %s", parsed.thought)
        # 永続化テキストは「実際に流したテキスト」に一致させ、逐次表示との乖離を排除する。
        # thought / tool_calls のみ parse から取る。
        yield LlmTurnComplete(result=_build_result(parsed, text=streamed.strip()))

    def _sync_turn(self, history: list[Message]) -> LlmTurnResult:
        llm = self._llm
        assert llm is not None

        out = llm.create_chat_completion(
            **self._create_kwargs(history, temperature=self._temperature)
        )
        text = out["choices"][0]["message"].get("content") or ""
        parsed = parse_native_output(text)
        logger.info("Gemma raw output (retry): %r", text)
        if parsed.thought:
            logger.info("Gemma thought (hidden, retry): %s", parsed.thought)
        return _build_result(parsed)


_STOP: Final[object] = object()


def _safe_next(it: Any) -> Any:
    try:
        return next(it)
    except StopIteration:
        return _STOP
