from __future__ import annotations

import asyncio
import gc
import json
import logging
import re
from typing import Any, Final

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from cocktail_server.schemas.generate import PromptSpec
from cocktail_server.services.prompt_builder import build_system_prompt, build_user_message

logger = logging.getLogger(__name__)

_JSON_OBJECT_RE: Final[re.Pattern[str]] = re.compile(r"\{[\s\S]*\}")


class LlmService:
    """Gemma 4bit 量子化モデルで日本語指示から Anima 向け PromptSpec を生成する。"""

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

    async def build_anima_prompt(self, instruction_ja: str) -> PromptSpec:
        return await asyncio.to_thread(self._build_sync, instruction_ja)

    def _build_sync(self, instruction_ja: str) -> PromptSpec:
        if self._model is None or self._tokenizer is None:
            self.load()

        combined = f"{build_system_prompt()}\n\n{build_user_message(instruction_ja)}"
        messages = [{"role": "user", "content": combined}]
        # Gemma 4 の apply_chat_template は BatchEncoding (dict) を返すため return_dict=True で受け取り、
        # generate() には **inputs で展開する。
        inputs = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self._model.device)
        prompt_len = inputs["input_ids"].shape[1]

        last_error: Exception | None = None
        for attempt in range(2):
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": 512,
                "pad_token_id": self._tokenizer.eos_token_id,
            }
            if attempt == 1:
                gen_kwargs |= {"do_sample": True, "temperature": 0.3, "top_p": 0.9}
            else:
                gen_kwargs["do_sample"] = False
            with torch.inference_mode():
                out = self._model.generate(**inputs, **gen_kwargs)
            raw_ids = out[0][prompt_len:]
            text = self._tokenizer.decode(raw_ids, skip_special_tokens=True)
            try:
                return self._parse_json(text)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning("PromptSpec JSON parse failed (attempt %d): %s", attempt + 1, exc)

        assert last_error is not None
        raise RuntimeError(f"Failed to obtain PromptSpec JSON after retries: {last_error}")

    @staticmethod
    def _parse_json(text: str) -> PromptSpec:
        match = _JSON_OBJECT_RE.search(text)
        if match is None:
            raise ValueError(f"No JSON object found in model output: {text!r}")
        data = json.loads(match.group(0))
        return PromptSpec.model_validate(data)
