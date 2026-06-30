"""単体(ComfyUI 形式)の Anima 派生チェックポイントを native diffusers に変換ロードする。

Anima 用の `from_single_file` 相当。公式 diffusers は Anima を **modular pipeline
専用**で提供しており単体ファイルローダが無い。一方、実在する派生（WAI-Anima など）は
DiT と LLM アダプタだけを内包した 1 ファイルの ``.safetensors`` で配布される
(Qwen3 テキストエンコーダと Qwen-Image VAE はベースモデルと共有)。そこでその 2
コンポーネントだけを変換し、エンコーダ/VAE/tokenizer はベースパイプラインのものを
呼び出し側で再利用する。

下のキー対応表と config は公式コンバータ（``scripts/convert_cosmos_to_diffusers.py``
と ``scripts/convert_anima_to_diffusers.py``）を Anima が構築されている Cosmos-2.0
2B Text2Image アーキテクチャ向けに切り詰めた写し。変更すると重みのロードに失敗する
ため、アーキテクチャ定数として厳密に保持する。
"""

from __future__ import annotations

from typing import Any

import torch
from accelerate import init_empty_weights
from diffusers import AnimaTextConditioner, CosmosTransformer3DModel
from safetensors.torch import load_file

COMFY_PREFIX = "model.diffusion_model."
NET_PREFIX = "net."
ADAPTER_PREFIX = "net.llm_adapter."


def _remove_key(key: str, state_dict: dict[str, Any]) -> None:
    state_dict.pop(key)


def _rename_key(state_dict: dict[str, Any], old_key: str, new_key: str) -> None:
    state_dict[new_key] = state_dict.pop(old_key)


# Cosmos-2.0 DiT のキー改名。出現順に部分文字列置換として順次適用する。
TRANSFORMER_KEYS_RENAME_DICT = {
    "t_embedder.1": "time_embed.t_embedder",
    "t_embedding_norm": "time_embed.norm",
    "blocks": "transformer_blocks",
    "adaln_modulation_self_attn.1": "norm1.linear_1",
    "adaln_modulation_self_attn.2": "norm1.linear_2",
    "adaln_modulation_cross_attn.1": "norm2.linear_1",
    "adaln_modulation_cross_attn.2": "norm2.linear_2",
    "adaln_modulation_mlp.1": "norm3.linear_1",
    "adaln_modulation_mlp.2": "norm3.linear_2",
    "self_attn": "attn1",
    "cross_attn": "attn2",
    "q_proj": "to_q",
    "k_proj": "to_k",
    "v_proj": "to_v",
    "output_proj": "to_out.0",
    "q_norm": "norm_q",
    "k_norm": "norm_k",
    "mlp.layer1": "ff.net.0.proj",
    "mlp.layer2": "ff.net.2",
    "x_embedder.proj.1": "patch_embed.proj",
    "final_layer.adaln_modulation.1": "norm_out.linear_1",
    "final_layer.adaln_modulation.2": "norm_out.linear_2",
    "final_layer.linear": "proj_out",
}

# チェックポイントに在るが diffusers モデルには無いキー。捨てる。
TRANSFORMER_SPECIAL_KEYS_REMAP = {
    "accum_video_sample_counter": _remove_key,
    "accum_image_sample_counter": _remove_key,
    "accum_iteration": _remove_key,
    "accum_train_in_hours": _remove_key,
    "pos_embedder.seq": _remove_key,
    "pos_embedder.dim_spatial_range": _remove_key,
    "pos_embedder.dim_temporal_range": _remove_key,
    "_extra_state": _remove_key,
}

COSMOS_2B_T2I_CONFIG = {
    "in_channels": 16,
    "out_channels": 16,
    "num_attention_heads": 16,
    "attention_head_dim": 128,
    "num_layers": 28,
    "mlp_ratio": 4.0,
    "text_embed_dim": 1024,
    "adaln_lora_dim": 256,
    "max_size": (128, 240, 240),
    "patch_size": (1, 2, 2),
    "rope_scale": (1.0, 4.0, 4.0),
    "concat_padding_mask": True,
    "extra_pos_embed_type": None,
}


class AnimaConversionError(ValueError):
    """チェックポイントのキーが想定の Anima アーキテクチャと一致しないとき送出する。"""


def _split_dit_and_adapter(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    dit: dict[str, torch.Tensor] = {}
    adapter: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(ADAPTER_PREFIX):
            adapter[key.removeprefix(ADAPTER_PREFIX)] = value
        else:
            dit[key] = value
    return dit, adapter


def _convert_dit(state_dict: dict[str, torch.Tensor]) -> CosmosTransformer3DModel:
    with init_empty_weights():
        transformer = CosmosTransformer3DModel(**COSMOS_2B_T2I_CONFIG)

    for key in list(state_dict.keys()):
        new_key = key.removeprefix(NET_PREFIX) if key.startswith(NET_PREFIX) else key
        for old, new in TRANSFORMER_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(old, new)
        _rename_key(state_dict, key, new_key)

    for key in list(state_dict.keys()):
        for marker, handler in TRANSFORMER_SPECIAL_KEYS_REMAP.items():
            if marker in key:
                handler(key, state_dict)

    _assert_keys_match(transformer, state_dict, "transformer")
    transformer.load_state_dict(state_dict, strict=True, assign=True)
    return transformer


def _infer_text_conditioner_config(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    model_dim = state_dict["blocks.0.self_attn.q_proj.weight"].shape[0]
    source_dim = state_dict["blocks.0.cross_attn.k_proj.weight"].shape[1]
    target_vocab_size, target_dim = state_dict["embed.weight"].shape
    attention_head_dim = state_dict["blocks.0.self_attn.q_norm.weight"].shape[0]
    num_layers = 1 + max(int(k.split(".")[1]) for k in state_dict if k.startswith("blocks."))
    return {
        "source_dim": source_dim,
        "target_dim": target_dim,
        "model_dim": model_dim,
        "num_layers": num_layers,
        "num_attention_heads": model_dim // attention_head_dim,
        "target_vocab_size": target_vocab_size,
    }


def _convert_text_conditioner(state_dict: dict[str, torch.Tensor]) -> AnimaTextConditioner:
    config = _infer_text_conditioner_config(state_dict)
    with init_empty_weights():
        text_conditioner = AnimaTextConditioner(**config)
    _assert_keys_match(text_conditioner, state_dict, "text_conditioner")
    text_conditioner.load_state_dict(state_dict, strict=True, assign=True)
    return text_conditioner


def _assert_keys_match(module: torch.nn.Module, state_dict: dict[str, Any], what: str) -> None:
    expected = set(module.state_dict().keys())
    got = set(state_dict.keys())
    missing = expected - got
    unexpected = got - expected
    if missing or unexpected:
        raise AnimaConversionError(
            f"{what} key mismatch: {len(missing)} missing, {len(unexpected)} unexpected. "
            f"missing sample={sorted(missing)[:3]} unexpected sample={sorted(unexpected)[:3]}"
        )


def load_single_file_components(
    path: str, dtype: torch.dtype = torch.bfloat16
) -> tuple[CosmosTransformer3DModel, AnimaTextConditioner | None]:
    """単体 Anima 派生を diffusers の transformer(+アダプタ)に変換して返す。

    transformer と、チェックポイントが独自の LLM アダプタを内包していれば対応する
    text_conditioner を返す。DiT のみのチェックポイントなら adapter は ``None`` で、
    呼び出し側はベースモデルの text_conditioner を流用する。
    """
    raw = load_file(path, device="cpu")
    renamed = {
        (NET_PREFIX + k[len(COMFY_PREFIX) :]) if k.startswith(COMFY_PREFIX) else k: v
        for k, v in raw.items()
    }
    dit_sd, adapter_sd = _split_dit_and_adapter(renamed)

    transformer = _convert_dit(dit_sd).to(dtype=dtype)
    text_conditioner = _convert_text_conditioner(adapter_sd).to(dtype=dtype) if adapter_sd else None
    return transformer, text_conditioner
