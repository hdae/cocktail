from __future__ import annotations

import asyncio
import gc
import logging
from time import perf_counter_ns
from typing import Any

import torch
from PIL.Image import Image

logger = logging.getLogger(__name__)


class ImageGenService:
    """AnimaPipeline を呼出時ロードで運用するラッパ。

    Cosmos-Predict2 派生のため **bfloat16** 固定。fp16 だと数値不安定の報告あり。
    """

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._pipe: Any = None
        self._on_cuda: bool = False

    def set_model_id(self, model_id: str) -> None:
        """ロード前に model_id を差し替える（AIR を解決したローカルパスへ置換する用途）。"""
        if self._pipe is not None:
            raise RuntimeError("cannot change model_id after the pipeline is loaded")
        self._model_id = model_id

    def is_loaded(self) -> bool:
        return self._pipe is not None and self._on_cuda

    def load(self) -> None:
        """pipe を GPU に載せる。初回は from_pretrained、2 回目以降は CPU→CUDA 転送のみ。"""
        if self._pipe is not None and self._on_cuda:
            return
        if self._pipe is None:
            logger.info("Loading Anima pipeline: %s", self._model_id)
            from diffusers_anima import AnimaPipeline

            t0 = perf_counter_ns()
            if self._model_id.endswith((".safetensors", ".ckpt")):
                pipe = AnimaPipeline.from_single_file(
                    self._model_id,
                    torch_dtype=torch.bfloat16,
                )
            else:
                pipe = AnimaPipeline.from_pretrained(
                    self._model_id,
                    torch_dtype=torch.bfloat16,
                )
            t1 = perf_counter_ns()
            pipe.to("cuda")
            t2 = perf_counter_ns()
            self._pipe = pipe
            self._on_cuda = True
            logger.info(
                "load image (cold): from_pretrained=%.0f ms, to_cuda=%.0f ms, total=%.0f ms",
                (t1 - t0) / 1_000_000,
                (t2 - t1) / 1_000_000,
                (t2 - t0) / 1_000_000,
            )
            return
        t0 = perf_counter_ns()
        self._pipe.to("cuda")
        t1 = perf_counter_ns()
        self._on_cuda = True
        logger.info("load image (warm): to_cuda=%.0f ms", (t1 - t0) / 1_000_000)

    def evict_to_cpu(self) -> None:
        """pipe を CPU RAM に退避させ、VRAM を解放する。次の load は warm 経路になる。

        diffusers の `.to("cpu")` は bf16 pipeline 全体に対して安全に動作する
        （bnb のようなカーネル再初期化依存が無い）。
        """
        if self._pipe is None or not self._on_cuda:
            return
        logger.info("Evicting Anima pipeline to CPU")
        t0 = perf_counter_ns()
        self._pipe.to("cpu")
        t1 = perf_counter_ns()
        gc.collect()
        t2 = perf_counter_ns()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t3 = perf_counter_ns()
        self._on_cuda = False
        logger.info(
            "evict image: to_cpu=%.0f ms, gc=%.0f ms, empty_cache=%.0f ms, total=%.0f ms",
            (t1 - t0) / 1_000_000,
            (t2 - t1) / 1_000_000,
            (t3 - t2) / 1_000_000,
            (t3 - t0) / 1_000_000,
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
        )

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
    ) -> Image:
        if not self._on_cuda:
            self.load()

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
                guidance_scale=cfg,
                generator=generator,
            )
        image: Image = result.images[0]
        return image
