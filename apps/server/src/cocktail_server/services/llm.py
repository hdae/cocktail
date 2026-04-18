from __future__ import annotations

import asyncio
import gc
import json
import logging
import re
import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from time import perf_counter_ns
from typing import Any, Final

import torch
from json_repair import repair_json
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


class LlmService:
    """Gemma 4bit 量子化モデルで日本語指示から `LlmTurnSpec` をストリーム生成する。"""

    def __init__(self, model_id: str, weights_dir: Path) -> None:
        self._model_id = model_id
        self._weights_dir = weights_dir
        self._model: Any = None
        self._tokenizer: Any = None
        self._on_cuda: bool = False
        # 量子化済み weight の CPU RAM 退避バッファ。`evict_to_cpu` で params/buffers の
        # 生バイトを格納し、`load` (warm) で CUDA にコピーして復帰する。bnb の
        # Linear4bit は `.to("cpu")` が再量子化相当の挙動になるため、.data 直接
        # 書き換えで量子化コストを完全に迂回する。
        self._cpu_snapshot: dict[str, torch.Tensor] | None = None

    def is_loaded(self) -> bool:
        return self._model is not None and self._on_cuda

    def load(self) -> None:
        """モデルを GPU に載せる。初回は量子化キャッシュ（あれば）から直接、無ければ
        bf16 ロード + on-the-fly bnb NF4 量子化 + キャッシュ書き出し。2 回目以降は
        CPU スナップショット → CUDA の memcpy のみで再量子化も再読み出しも走らない。"""
        if self._model is not None and self._on_cuda:
            return
        if self._model is None:
            self._cold_load()
            self._on_cuda = True
            return
        # Warm 復帰: CPU snapshot から CUDA に memcpy。
        assert self._cpu_snapshot is not None, "warm load without snapshot"
        t0 = perf_counter_ns()
        self._restore_from_snapshot()
        t1 = perf_counter_ns()
        self._on_cuda = True
        logger.info("load llm (warm): restore=%.0f ms", (t1 - t0) / 1_000_000)

    def _cold_load(self) -> None:
        """ディスクから weight を取得して GPU に配置する。キャッシュ優先、失敗時は
        HF hub から再量子化してキャッシュを作り直す。"""
        cache_dir = self._quantized_cache_dir()
        if self._cache_valid(cache_dir):
            try:
                self._load_from_source(str(cache_dir), from_cache=True)
                return
            except Exception as exc:
                logger.warning(
                    "quantized cache load failed (%s); purging %s and falling back to hub",
                    exc,
                    cache_dir,
                )
                shutil.rmtree(cache_dir, ignore_errors=True)
                # from_pretrained が途中で落ちた場合の部分状態を一掃
                self._tokenizer = None
                self._model = None
        self._load_from_source(self._model_id, from_cache=False)
        self._save_quantized_cache(cache_dir)

    def _load_from_source(self, source: str, *, from_cache: bool) -> None:
        """`source` はキャッシュディレクトリ or HF モデル ID。キャッシュからは既に
        量子化済みの weight をそのまま読み込むため、BitsAndBytesConfig は渡さない。
        （saved config.json の `quantization_config` を transformers 側が再利用する）"""
        # NF4 は正規分布前提の非一様コードブック + 二重量子化で、構造トークン
        # （JSON 区切り等の稀頻度トークン）を壊しにくい。小型 Gemma の tail
        # 誤差を抑えるため int4 線形グリッドではなく NF4 を採用。
        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "device_map": {"": 0},
        }
        if not from_cache:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        logger.info(
            "Loading Gemma model from %s (%s)",
            source,
            "pre-quantized cache" if from_cache else "HF hub + on-the-fly NF4 quantization",
        )
        t0 = perf_counter_ns()
        self._tokenizer = AutoTokenizer.from_pretrained(source)
        t1 = perf_counter_ns()
        self._model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)
        t2 = perf_counter_ns()
        logger.info(
            "load llm (cold, %s): tokenizer=%.0f ms, from_pretrained=%.0f ms, total=%.0f ms",
            "cache" if from_cache else "hub",
            (t1 - t0) / 1_000_000,
            (t2 - t1) / 1_000_000,
            (t2 - t0) / 1_000_000,
        )

    def _quantized_cache_dir(self) -> Path:
        """`weights_dir/llm-nf4/<slug>` を返す。slug はモデル ID のスラッシュ置換。
        モデル ID が変われば別ディレクトリになるので古いキャッシュと競合しない。"""
        slug = self._model_id.replace("/", "--").replace(":", "--")
        return self._weights_dir / "llm-nf4" / slug

    @staticmethod
    def _cache_valid(cache_dir: Path) -> bool:
        """config.json と少なくとも 1 つの safetensors シャードがあれば有効とみなす。
        壊れている場合は `from_pretrained` 側で例外になり、`_cold_load` が掃除する。"""
        if not cache_dir.is_dir():
            return False
        if not (cache_dir / "config.json").is_file():
            return False
        return any(cache_dir.glob("*.safetensors"))

    def _save_quantized_cache(self, cache_dir: Path) -> None:
        """tokenizer + 量子化済みモデルをローカルに保存する。
        transformers は bnb Params4bit を含めて save_pretrained でき、config.json に
        `quantization_config` を埋め込むので、次回ロードは再量子化なしで済む。
        書き出し失敗は警告に留め、推論継続を妨げない。"""
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            assert self._tokenizer is not None and self._model is not None
            t0 = perf_counter_ns()
            self._tokenizer.save_pretrained(str(cache_dir))
            self._model.save_pretrained(str(cache_dir), safe_serialization=True)
            t1 = perf_counter_ns()
            logger.info(
                "saved quantized cache to %s: %.0f ms",
                cache_dir,
                (t1 - t0) / 1_000_000,
            )
        except Exception as exc:
            logger.warning("failed to save quantized cache to %s: %s", cache_dir, exc)

    def evict_to_cpu(self) -> None:
        """量子化済み weight を CPU RAM に常駐させ、VRAM だけを解放する。
        初回は CPU へ完全コピー（snapshot 作成）、以降は GPU storage の解放のみ。
        推論では重みが更新されないため、snapshot は作り直さず再利用して
        毎 swap の GPU→CPU memcpy を省く。bnb Params4bit の `.to("cpu")` は
        再量子化相当のコストが走るため使わず、`.data` を空テンソルに差し替える。"""
        if self._model is None or not self._on_cuda:
            return
        t0 = perf_counter_ns()
        if self._cpu_snapshot is None:
            logger.info("Snapshotting Gemma model to CPU (first evict)")
            self._snapshot_to_cpu()
            phase = "snapshot"
        else:
            logger.info("Releasing Gemma GPU storage (snapshot retained)")
            self._release_gpu_storage()
            phase = "release"
        t1 = perf_counter_ns()
        gc.collect()
        t2 = perf_counter_ns()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t3 = perf_counter_ns()
        self._on_cuda = False
        logger.info(
            "evict llm: %s=%.0f ms, gc=%.0f ms, empty_cache=%.0f ms, total=%.0f ms",
            phase,
            (t1 - t0) / 1_000_000,
            (t2 - t1) / 1_000_000,
            (t3 - t2) / 1_000_000,
            (t3 - t0) / 1_000_000,
        )

    def _snapshot_to_cpu(self) -> None:
        """全 parameter / buffer の生バイトを CPU tensor として保持し、元の GPU
        storage は `torch.empty(0)` に差し替えて解放する。Params4bit は `.data`
        が packed uint8 のためそのまま CPU copy すれば復帰時に再量子化不要。
        `quant_state` は bnb 側の `.to("cpu")` を使って CPU に移す。"""
        assert self._model is not None
        snapshot: dict[str, torch.Tensor] = {}
        for name, param in self._model.named_parameters():
            snapshot[f"p:{name}"] = param.data.detach().to("cpu", copy=True)
            qs = getattr(param, "quant_state", None)
            if qs is not None:
                qs.to("cpu")  # QuantState は in-place で全フィールドを移動
            param.data = torch.empty(0, dtype=param.data.dtype, device="cuda")
        for name, buf in self._model.named_buffers():
            snapshot[f"b:{name}"] = buf.detach().to("cpu", copy=True)
            buf.data = torch.empty(0, dtype=buf.data.dtype, device="cuda")
        self._cpu_snapshot = snapshot

    def _release_gpu_storage(self) -> None:
        """既に CPU snapshot がある状態で、GPU 側の storage だけを解放する。
        重みは inference-only で不変なので CPU コピーは不要。"""
        assert self._model is not None and self._cpu_snapshot is not None
        for param in self._model.parameters():
            qs = getattr(param, "quant_state", None)
            if qs is not None:
                qs.to("cpu")
            param.data = torch.empty(0, dtype=param.data.dtype, device="cuda")
        for buf in self._model.buffers():
            buf.data = torch.empty(0, dtype=buf.data.dtype, device="cuda")

    def _restore_from_snapshot(self) -> None:
        """`_snapshot_to_cpu` で退避した生バイトを CUDA に戻す。`quant_state` も
        同じ順序で `.to("cuda")` して bnb カーネルから参照可能な状態に戻す。
        snapshot 自体は保持し、次回 evict を「GPU 解放のみ」で終わらせる。"""
        assert self._model is not None and self._cpu_snapshot is not None
        snapshot = self._cpu_snapshot
        for name, param in self._model.named_parameters():
            cpu_data = snapshot[f"p:{name}"]
            param.data = cpu_data.to("cuda", non_blocking=True)
            qs = getattr(param, "quant_state", None)
            if qs is not None:
                qs.to("cuda")
        for name, buf in self._model.named_buffers():
            buf.data = snapshot[f"b:{name}"].to("cuda", non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def unload(self) -> None:
        """モデルを完全破棄する。プロセス終了時・障害復旧用。"""
        if self._model is None:
            return
        logger.info("Unloading Gemma model")
        t0 = perf_counter_ns()
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._on_cuda = False
        self._cpu_snapshot = None
        gc.collect()
        t1 = perf_counter_ns()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t2 = perf_counter_ns()
        logger.info(
            "unload llm: del=%.0f ms, empty_cache=%.0f ms, total=%.0f ms",
            (t1 - t0) / 1_000_000,
            (t2 - t1) / 1_000_000,
            (t2 - t0) / 1_000_000,
        )

    async def run_turn(self, history: list[Message]) -> AsyncIterator[LlmStreamChunk]:
        """1 ターン分の応答を生成。`reasoning` を `LlmTextDelta` で逐次流し、最後に `LlmTurnComplete`。

        `history` は会話の全メッセージ（末尾が今回のユーザ発話）。過去ターンの
        アシスタント応答は保存済み parts から `LlmTurnSpec` JSON を復元して渡す。

        失敗時は温度 0.3 でリプレイ（同期、text_delta は流さない）。ただし attempt 0 で
        ユーザに見えたテキストは `spec.reasoning` に上書きして整合を取る。
        """
        if not self._on_cuda or self._model is None or self._tokenizer is None:
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

    def _prepare_inputs(self, history: list[Message]) -> Any:
        """履歴から chat template + tokenize 済みテンソルを作る。"""
        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None

        messages = _build_chat_messages(history)
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

    async def _stream_once(
        self, history: list[Message], *, temperature: float
    ) -> AsyncIterator[LlmStreamChunk]:
        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None

        inputs = self._prepare_inputs(history)

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

        inputs = self._prepare_inputs(history)
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
