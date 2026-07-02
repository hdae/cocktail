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

from cocktail_server.schemas.generate import GenerateImageCall, SearchTagsCall
from cocktail_server.schemas.messages import (
    Message,
    TextPart,
    ToolCallPart,
)
from cocktail_server.schemas.tags import TagSuggestion
from cocktail_server.services.native_tools import (
    NativeToolStream,
    ParsedToolCall,
    ParsedTurn,
    parse_native_output,
    render_tool_call,
)
from cocktail_server.services.prompt_builder import (
    GENERATE_IMAGE_TOOL,
    SEARCH_TAGS_TOOL,
    build_system_prompt,
    build_user_message,
    display_tag,
)
from cocktail_server.services.tags import TagService

logger = logging.getLogger(__name__)

# GGUF 推論パラメータ。max_tokens は会話 + 1 ツール分。n_ctx / KV 型 / flash_attn / swa_full
# は Settings から注入する（既定は fp16 KV + iSWA コンパクトキャッシュで 32768。
# docs/decisions/0003-kv-cache-fp16-iswa.md）。
_MAX_TOKENS: Final[int] = 1024

# エージェントループの上限ホップ数（検索→検索→生成の最大 3 ホップ）。無限ループと n_ctx
# 膨張を構造的に封じる。実機計測で system+tools≈2036tok / n_ctx=8192 のため 3 ホップは十分安全
# （中間検索結果は上位 N 件の {tag, ja} 要約のみ戻すので 1 ホップ数百 token に収まる）。
_MAX_ITERS: Final[int] = 3
# モデルへ戻す検索候補の件数（n_ctx 予算のため要約する。多すぎると positive を埋没させる）。
_SEARCH_TOP_N: Final[int] = 8


@dataclass(frozen=True)
class LlmTextDelta:
    """ユーザ向け会話テキストのストリーム差分（native の会話領域から逐次抽出）。"""

    delta: str


@dataclass(frozen=True)
class LlmTurnResult:
    """1 ターンの確定結果。`text`=会話テキスト、`thought`=非表示の思考、
    `tool_calls`=検証済み呼び出し（0 or 1 件の generate_image）。

    `search_tags` は `run_turn` のエージェントループ内で解決され外へは出ないため、
    ここに現れるツールは常に generate_image のみ（消費側の契約は Phase 1 から不変）。"""

    text: str
    thought: str
    tool_calls: list[GenerateImageCall]


@dataclass(frozen=True)
class LlmTurnComplete:
    """LLM ターン完了。`result` は確定済み。"""

    result: LlmTurnResult


LlmStreamChunk = LlmTextDelta | LlmTurnComplete


@dataclass(frozen=True)
class _HopResult:
    """エージェントループ 1 ホップの内部確定結果（`run_turn` 内でのみ使う）。"""

    parsed: ParsedTurn


def _summarize_tags(results: list[TagSuggestion]) -> str:
    """検索結果を `role:tool` へ戻す最小要約に整形する（`tag [ja]` を並べる）。

    n_ctx 予算のため post_count / 全 alias は落とし、モデルが positive に使う tag 名と
    当たりを確認する日本語読みだけを渡す。空なら「見つからなかった」と明示する。
    tag は `display_tag` でスペース区切りに正規化して見せる（アンダースコア形のまま
    返すとモデルが positive へそのまま写し、プロンプト規約と衝突する）。
    """
    if not results:
        return "(no matching tags found)"
    return "; ".join(
        f"{display_tag(t.tag)} [{t.ja}]" if t.ja else display_tag(t.tag) for t in results
    )


def _search_args(call: SearchTagsCall) -> dict[str, Any]:
    """検索呼び出しを assistant `tool_calls` へ replay するための引数 mapping。

    Gemma 4 テンプレは mapping の arguments を native 形式へ整形する。`category` は
    指定時のみ含める（None は絞り込み無効なので省く）。
    """
    args: dict[str, Any] = {"query": call.query}
    if call.category is not None:
        args["category"] = call.category
    return args


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
        tags: TagService,
        n_ctx: int = 32768,
        kv_cache_type: str = "f16",
        flash_attn: bool = True,
        swa_full: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
    ) -> None:
        self._model_id = model_id
        self._hf_home = hf_home
        self._tags = tags
        self._llm: Any = None  # llama_cpp.Llama | None
        self._n_ctx = n_ctx
        self._kv_cache_type = kv_cache_type
        self._flash_attn = flash_attn
        self._swa_full = swa_full
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
        import llama_cpp
        from llama_cpp import Llama

        # KV 量子化型 → ggml 型コード。量子化 KV(特に V)は flash_attn が前提なので、
        # 明示 False でも量子化時は強制 True にして warning で知らせる（黙って壊さない）。
        kv_type = {
            "f16": llama_cpp.GGML_TYPE_F16,
            "q8_0": llama_cpp.GGML_TYPE_Q8_0,
            "q4_0": llama_cpp.GGML_TYPE_Q4_0,
        }[self._kv_cache_type]
        flash_attn = self._flash_attn
        if self._kv_cache_type != "f16" and not flash_attn:
            logger.warning(
                "KV cache type %s requires flash_attn; forcing it on", self._kv_cache_type
            )
            flash_attn = True

        logger.info(
            "Loading GGUF LLM: %s (n_ctx=%d, kv=%s, flash_attn=%s, swa_full=%s)",
            Path(path).name,
            self._n_ctx,
            self._kv_cache_type,
            flash_attn,
            self._swa_full,
        )
        t0 = perf_counter_ns()
        self._llm = Llama(
            model_path=path,
            n_gpu_layers=-1,
            n_ctx=self._n_ctx,
            flash_attn=flash_attn,
            type_k=kv_type,
            type_v=kv_type,
            swa_full=self._swa_full,
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
        """1 ターン分を「検索→(必要なだけ)検索→生成」のエージェントループで生成する。

        既定は自由文の会話。モデルが `search_tags` を native 呼び出ししたら、その候補を
        サーバ側(`TagService`)で引いて `role:tool` で戻し、次ホップを回す。`generate_image`
        か会話確定（ツール無し）か `_MAX_ITERS` 到達で確定する。検索の中間往復は会話履歴・
        SSE・永続化に一切出さず、この関数内で完結する（消費側の契約は不変）。

        会話テキストは各ホップの `LlmTextDelta` を逐次流し、確定 `result.text` は「実際に
        流したテキスト」に一致させる（逐次表示との乖離を排除）。native パースに失敗したら
        非ストリームで単発リトライ→チャットのみ縮退（ターンを内部エラーで落とさない）。
        """
        if self._llm is None:
            await asyncio.to_thread(self.load)

        messages = _build_chat_messages(history)
        streamed_text = ""
        try:
            for hop in range(_MAX_ITERS):
                parsed: ParsedTurn | None = None
                async for chunk in self._stream_hop(messages):
                    if isinstance(chunk, LlmTextDelta):
                        streamed_text += chunk.delta
                        yield chunk
                    else:
                        parsed = chunk.parsed
                assert parsed is not None, "hop ended without a _HopResult"

                gen_present = any(c.name == "generate_image" for c in parsed.tool_calls)
                search_calls = [c for c in parsed.tool_calls if c.name == "search_tags"]
                is_last_hop = hop == _MAX_ITERS - 1
                # generate_image / 会話確定 / 上限到達 なら確定。search のみなら往復して次ホップ。
                if gen_present or not search_calls or is_last_hop:
                    yield LlmTurnComplete(result=_build_result(parsed, text=streamed_text.strip()))
                    return
                appended = await asyncio.to_thread(
                    self._append_search_roundtrip, messages, search_calls
                )
                if not appended:
                    # 有効な検索が無かった（全て不正引数）ら、そのホップで確定して抜ける。
                    yield LlmTurnComplete(result=_build_result(parsed, text=streamed_text.strip()))
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

    def _hop_kwargs(self, messages: list[dict[str, Any]], *, temperature: float) -> dict[str, Any]:
        return {
            "messages": messages,
            "tools": [GENERATE_IMAGE_TOOL, SEARCH_TAGS_TOOL],
            "tool_choice": "auto",
            "max_tokens": _MAX_TOKENS,
            "temperature": temperature,
            "top_p": self._top_p,
            "top_k": self._top_k,
            "repeat_penalty": self._repeat_penalty,
        }

    def _append_search_roundtrip(
        self,
        messages: list[dict[str, Any]],
        search_calls: list[ParsedToolCall],
    ) -> bool:
        """検索ホップを実行し、結果を次ホップの `messages` へ追記する（ブロッキング）。

        Gemma 4 テンプレの OpenAI 互換経路に合わせ、モデルの検索要求を構造化 `tool_calls`
        を持つ assistant メッセージとして、結果を `tool_call_id` 一致の `role:tool` として
        積む（テンプレが `<|tool_call>`/`<|tool_response>` へ整形する）。引数不正な検索は
        黙って落とさずログに残してスキップし、有効な検索が 1 件も無ければ False を返す
        （呼び出し側がそのホップで確定する）。

        assistant の `content` は空にする（MUST）。テンプレは content をツール応答の後に
        描画し、非空だと `<turn|>` でモデルターンを閉じてしまい、次ホップの生成開始点が
        消える。content 空ならターンは開いたままで、モデルは `<|tool_response>…` の直後
        から続きを書ける（テンプレの cascade 設計）。ホップの会話テキストは既にストリーム
        済みで最終 `result.text` にも累積されるため、ここで捨てても失われない。
        """
        valid: list[tuple[str, SearchTagsCall, str]] = []
        for i, pc in enumerate(search_calls):
            try:
                call = SearchTagsCall.from_native(pc.args)
            except ValidationError as exc:
                logger.warning("skipping malformed search_tags call %r: %s", pc.args, exc)
                continue
            results = self._tags.search(call.query, limit=_SEARCH_TOP_N, category=call.category)
            summary = _summarize_tags(results)
            logger.info("search_tags %r (category=%s) -> %s", call.query, call.category, summary)
            valid.append((f"search_{len(messages)}_{i}", call, summary))

        if not valid:
            return False

        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": cid,
                        "type": "function",
                        "function": {"name": "search_tags", "arguments": _search_args(call)},
                    }
                    for cid, call, _summary in valid
                ],
            }
        )
        for cid, _call, summary in valid:
            messages.append({"role": "tool", "tool_call_id": cid, "content": summary})
        return True

    async def _stream_hop(
        self, messages: list[dict[str, Any]]
    ) -> AsyncIterator[LlmTextDelta | _HopResult]:
        """1 ホップをストリーミングし、会話テキストを `LlmTextDelta` で流して最後に
        `_HopResult`（パース済み）を 1 度だけ yield する（`run_turn` 内部専用）。"""
        llm = self._llm
        assert llm is not None

        kwargs = self._hop_kwargs(messages, temperature=self._temperature)
        loop = asyncio.get_running_loop()

        def _open_stream() -> Any:
            return llm.create_chat_completion(stream=True, **kwargs)

        # 生成本体（prompt eval + decode）はブロッキングなので executor で回す。
        stream = await loop.run_in_executor(None, _open_stream)

        parser = NativeToolStream()
        while True:
            chunk = await loop.run_in_executor(None, _safe_next, stream)
            if chunk is _STOP:
                break
            piece = chunk["choices"][0]["delta"].get("content")
            if not piece:
                continue
            new_text = parser.feed(piece)
            if new_text:
                yield LlmTextDelta(delta=new_text)
        # マーカーが来ないまま終わった場合に保留していた末尾を確定して流す。
        tail = parser.flush()
        if tail:
            yield LlmTextDelta(delta=tail)

        parsed = parse_native_output(parser.raw)
        # 生出力は破綻パターン同定に必須。thought は UI 非表示なのでログにだけ残す。
        logger.info("Gemma raw output: %r", parser.raw)
        if parsed.thought:
            logger.info("Gemma thought (hidden): %s", parsed.thought)
        yield _HopResult(parsed=parsed)

    def _sync_turn(self, history: list[Message]) -> LlmTurnResult:
        llm = self._llm
        assert llm is not None

        messages = _build_chat_messages(history)
        out = llm.create_chat_completion(
            **self._hop_kwargs(messages, temperature=self._temperature)
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
