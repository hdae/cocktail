from __future__ import annotations

import asyncio
import gc
import logging
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import torch
from PIL.Image import Image

logger = logging.getLogger(__name__)

# 注入する Turbo LoRA アダプタ名。単一固定なのでこの1個だけを載せる。
_TURBO_ADAPTER = "turbo"


class ImageGenService:
    """公式 diffusers の Anima modular pipeline を呼出時ロードで運用するラッパ。

    Cosmos-Predict2 派生のため **bfloat16** 固定（fp16 は数値不安定の報告あり）。
    ベース(diffusers 形式リポ)の VAE / Qwen3 text encoder / tokenizer / scheduler を
    共有し、派生(WAI-Anima 等)の単体チェックポイントは DiT(+任意の LLM アダプタ)だけを
    `anima_convert` でメモリ内変換して `update_components` で差し込む。公式 Anima には
    `from_single_file` が無いため、この変換が単体ロードの代替になる。
    """

    def __init__(
        self,
        base_model_id: str,
        dit_path: str | None = None,
        *,
        turbo_lora_path: str | None = None,
        turbo_lora_strength: float = 1.0,
    ) -> None:
        self._base_model_id = base_model_id
        self._dit_path = dit_path
        self._turbo_lora_path = turbo_lora_path
        self._turbo_lora_strength = turbo_lora_strength
        # Turbo LoRA が実際に DiT へ注入されたか。cold load で確定する。
        self._turbo_loaded = False
        self._pipe: Any = None
        self._on_cuda: bool = False
        # 現在 guider に載っている guidance_scale。cfg 変更時のみ guider を作り直す。
        self._guider_cfg: float | None = None

    def set_dit_path(self, dit_path: str | None) -> None:
        """ロード前に派生 DiT の単体チェックポイントパスを差し替える。

        `None` のときはベースの transformer をそのまま使う（派生なし）。
        """
        if self._pipe is not None:
            raise RuntimeError("cannot change dit_path after the pipeline is loaded")
        self._dit_path = dit_path

    def set_turbo_lora(self, path: str | None, strength: float) -> None:
        """ロード前に Turbo LoRA(単体 .safetensors)のパス/強度を差し替える。

        `None`/空文字 のときは Turbo 無効(base 品質)。cold load 時に注入される。
        """
        if self._pipe is not None:
            raise RuntimeError("cannot change turbo lora after the pipeline is loaded")
        self._turbo_lora_path = path
        self._turbo_lora_strength = strength

    def is_loaded(self) -> bool:
        return self._pipe is not None and self._on_cuda

    def load(self) -> None:
        """pipe を GPU に載せる。初回は from_pretrained + DiT 変換、2 回目以降は CPU→CUDA。"""
        if self._pipe is not None and self._on_cuda:
            return
        if self._pipe is None:
            self._cold_load()
            return
        t0 = perf_counter_ns()
        self._pipe.to("cuda")
        self._on_cuda = True
        logger.info("load image (warm): to_cuda=%.0f ms", (perf_counter_ns() - t0) / 1_000_000)

    def _cold_load(self) -> None:
        """ベースを ModularPipeline でロードし、派生 DiT を変換して差し込み、CUDA に載せる。"""
        from diffusers import ModularPipeline

        logger.info(
            "Loading Anima modular pipeline (base=%s, dit=%s)",
            self._base_model_id,
            self._dit_path,
        )
        t0 = perf_counter_ns()
        pipe = ModularPipeline.from_pretrained(self._base_model_id)
        pipe.load_components(torch_dtype=torch.bfloat16)
        t1 = perf_counter_ns()
        if self._dit_path:
            from cocktail_server.services.anima_convert import load_single_file_components

            transformer, text_conditioner = load_single_file_components(
                self._dit_path, torch.bfloat16
            )
            # DiT のみの派生は text_conditioner=None → ベースの LLM アダプタを流用する。
            if text_conditioner is not None:
                pipe.update_components(transformer=transformer, text_conditioner=text_conditioner)
            else:
                pipe.update_components(transformer=transformer)
        t2 = perf_counter_ns()
        pipe.to("cuda")
        t3 = perf_counter_ns()
        self._pipe = pipe
        self._on_cuda = True
        self._guider_cfg = None
        logger.info(
            "load image (cold): base=%.0f ms, convert_dit=%.0f ms, to_cuda=%.0f ms, total=%.0f ms",
            (t1 - t0) / 1_000_000,
            (t2 - t1) / 1_000_000,
            (t3 - t2) / 1_000_000,
            (t3 - t0) / 1_000_000,
        )
        self._inject_turbo_lora()

    def _inject_turbo_lora(self) -> None:
        """Turbo LoRA を PEFT アダプタとして DiT に注入する（cold load 時に一度だけ）。

        変換済み state dict を `load_lora_weights` で注入し、実際に載ったかを
        `get_list_adapters` で確認する（キー不一致だと 0 件ロードでも例外は出ないため、
        silent no-op を fail-loud に変える）。以後は生成毎の `enable/disable_lora` で
        無再ロードにトグルする。
        """
        if not self._turbo_lora_path:
            return
        from cocktail_server.services.lora_convert import lora_state_dict_for

        t0 = perf_counter_ns()
        state = lora_state_dict_for(Path(self._turbo_lora_path))
        self._pipe.load_lora_weights(state, adapter_name=_TURBO_ADAPTER)
        injected = {a for names in self._pipe.get_list_adapters().values() for a in names}
        if _TURBO_ADAPTER not in injected:
            raise RuntimeError(f"Turbo LoRA を適用できませんでした: {self._turbo_lora_path}")
        self._pipe.set_adapters([_TURBO_ADAPTER], [self._turbo_lora_strength])
        self._turbo_loaded = True
        logger.info(
            "inject turbo lora: %s (strength=%.2f) %.0f ms",
            self._turbo_lora_path,
            self._turbo_lora_strength,
            (perf_counter_ns() - t0) / 1_000_000,
        )

    def evict_to_cpu(self) -> None:
        """pipe を CPU RAM に退避させ VRAM を解放する。次の load は warm 経路になる。

        ModularPipeline 全体が `.to("cpu")` をサポートしており、全コンポーネントが
        まとめて CPU へ移る（bnb のようなカーネル再初期化依存が無い bf16 なので安全）。
        """
        if self._pipe is None or not self._on_cuda:
            return
        logger.info("Evicting Anima pipeline to CPU")
        t0 = perf_counter_ns()
        self._pipe.to("cpu")
        t1 = perf_counter_ns()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t2 = perf_counter_ns()
        self._on_cuda = False
        logger.info(
            "evict image: to_cpu=%.0f ms, cleanup=%.0f ms, total=%.0f ms",
            (t1 - t0) / 1_000_000,
            (t2 - t1) / 1_000_000,
            (t2 - t0) / 1_000_000,
        )

    def unload(self) -> None:
        """pipe を完全破棄する。プロセス終了時・障害復旧用。"""
        if self._pipe is None:
            return
        logger.info("Unloading Anima pipeline")
        t0 = perf_counter_ns()
        del self._pipe
        self._pipe = None
        self._on_cuda = False
        self._guider_cfg = None
        self._turbo_loaded = False
        gc.collect()
        t1 = perf_counter_ns()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t2 = perf_counter_ns()
        logger.info(
            "unload image: del=%.0f ms, empty_cache=%.0f ms, total=%.0f ms",
            (t1 - t0) / 1_000_000,
            (t2 - t1) / 1_000_000,
            (t2 - t0) / 1_000_000,
        )

    async def generate(
        self,
        *,
        positive: str,
        negative: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int | None,
        turbo: bool = False,
    ) -> Image:
        return await asyncio.to_thread(
            self._generate_sync,
            positive=positive,
            negative=negative,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed,
            turbo=turbo,
        )

    def _apply_cfg(self, cfg: float) -> None:
        """guidance_scale が変わったときだけ guider コンポーネントを作り直す。

        公式 Anima の CFG は `__call__` 引数ではなく `ClassifierFreeGuidance` guider
        コンポーネントで決まるため、属性代入ではなく `update_components` で差し替える。
        """
        if self._guider_cfg == cfg:
            return
        from diffusers import ClassifierFreeGuidance

        self._pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=cfg))
        self._guider_cfg = cfg

    def _apply_turbo(self, turbo: bool) -> None:
        """Turbo LoRA の有効/無効を生成ごとに切り替える（PEFT アダプタの enable/disable）。

        LoRA 未注入で Turbo を要求されたら fail-loud（base モデルを Turbo の 10 step/CFG1 で
        回すと破綻するため、静かに劣化させない）。
        """
        if self._turbo_loaded:
            if turbo:
                self._pipe.enable_lora()
            else:
                self._pipe.disable_lora()
        elif turbo:
            raise RuntimeError("Turbo が要求されましたが Turbo LoRA が読み込まれていません")

    def _generate_sync(
        self,
        *,
        positive: str,
        negative: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int | None,
        turbo: bool,
    ) -> Image:
        if not self._on_cuda:
            self.load()

        self._apply_turbo(turbo)
        self._apply_cfg(cfg)

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        with torch.inference_mode():
            result = self._pipe(
                prompt=positive,
                negative_prompt=negative,
                width=width,
                height=height,
                num_inference_steps=steps,
                num_images_per_prompt=1,
                generator=generator,
            )
        image: Image = result.images[0]
        return image
