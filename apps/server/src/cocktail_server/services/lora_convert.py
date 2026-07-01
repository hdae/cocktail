"""単体(ComfyUI / kohya / diffusers 形式)の Anima LoRA を diffusers-loadable な
state dict に変換する境界コンバータ。

`load_lora_weights` に渡せる形へ正規化するのが役目。実際の DiT モジュールパス変換
（`diffusion_model.*` → `transformer.*`）は diffusers の in-tree コンバータ
`_convert_non_diffusers_anima_lora_to_diffusers` が `load_lora_weights` 内部で行うため、
ここでは (1) Anima 由来かの判定 (2) kohya の平坦化復元 (3) alpha 畳み込み・DoRA 破棄
(4) `lora_down`/`lora_up` → PEFT `lora_A`/`lora_B` の改名 までを担う。

Anima LoRA が実際に配布される 3 レイアウトを受け付ける:
- **ComfyUI / diffusion-pipe**（`diffusion_model.` prefix）— Civitai / HF の native 形式。
- **diffusers**（`transformer.` / `text_conditioner.` prefix）— そのまま素通し。
- **kohya / sd-scripts**（`lora_unet_` DiT / `lora_te_` Qwen3 のアンダースコア平坦化）—
  既知の leaf 語彙で `diffusion_model.*` へ復元。`lora_te_`(Qwen3 encoder)は Anima
  パイプラインで LoRA 非対応なので警告して破棄する。

Anima(Cosmos)以外は positive allowlist で loud に reject する。`"lora"` は
`lora_unet_` の部分文字列でもあるため、当て推量で素通しすると diffusers 側の妥当性
チェックが発火せず「何も読まない」事故になる。参考実装: vImagen `services/loras.py`。
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

LoraFormat = Literal["diffusion_pipe", "diffusers", "kohya", "unknown"]

# Cosmos/Anima アーキの positive マーカー（on-disk の ComfyUI 命名と diffusers 命名の両方）。
# 判定にはこれらのいずれかを含み、かつ reject マーカーを含まないことを要求する。
_ANIMA_MARKERS = (
    "self_attn",
    "cross_attn",
    "adaln_modulation",
    "x_embedder",
    "t_embedder",
    "mlp.layer",
    "llm_adapter",
    # diffusers 側の leaf
    "transformer_blocks",
    "attn1.to_",
    "attn2.to_",
    "patch_embed",
    "time_embed",
    "norm_out.linear",
)
# Flux は Anima と同じ `diffusion_model.`/`lora_unet_` prefix を持つが別アーキ → reject。
_FLUX_MARKERS = (
    "double_blocks",
    "single_blocks",
    "img_attn",
    "txt_attn",
)
# SD 系 UNet ブロック名（SGM=kohya と diffusers 綴りの両方）。SD1.5/SD2/SDXL 全てが共有。
# cocktail は Anima 専用なので、これらに当たった LoRA は SD 系として loud に reject する。
_SD_UNET_MARKERS = (
    "input_blocks",
    "middle_block",
    "output_blocks",
    "down_blocks",
    "mid_block",
    "up_blocks",
)


class LoraIncompatibleError(ValueError):
    """LoRA ファイルが Anima/Cosmos 以外のアーキを対象にしているとき送出する。"""


# Cosmos DiT のブロック内 LoRA 対象 leaf（dotted）。kohya のアンダースコア平坦化を
# 決定的に逆変換するための語彙。実在の sd-scripts Anima LoRA で検証済み。
_KOHYA_DIT_BLOCK_LEAVES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.output_proj",
    "cross_attn.q_proj",
    "cross_attn.k_proj",
    "cross_attn.v_proj",
    "cross_attn.output_proj",
    "mlp.layer1",
    "mlp.layer2",
    "adaln_modulation_self_attn.1",
    "adaln_modulation_self_attn.2",
    "adaln_modulation_cross_attn.1",
    "adaln_modulation_cross_attn.2",
    "adaln_modulation_mlp.1",
    "adaln_modulation_mlp.2",
)
_KOHYA_DIT_TOPLEVEL_LEAVES = (
    "x_embedder.proj.1",
    "t_embedder.1",
    "final_layer.linear",
    "final_layer.adaln_modulation.1",
    "final_layer.adaln_modulation.2",
)
_KOHYA_BLOCK_LEAF = {leaf.replace(".", "_"): leaf for leaf in _KOHYA_DIT_BLOCK_LEAVES}
_KOHYA_TOPLEVEL_LEAF = {leaf.replace(".", "_"): leaf for leaf in _KOHYA_DIT_TOPLEVEL_LEAVES}
_KOHYA_BLOCK_RE = re.compile(r"^blocks_(\d+)_(.+)$")


def _has_prefix(keys: list[str], *prefixes: str) -> bool:
    return any(k.startswith(p) for k in keys for p in prefixes)


def _has_marker(keys: list[str], markers: tuple[str, ...]) -> bool:
    return any(marker in k for k in keys for marker in markers)


def detect_anima_format(header: dict[str, list[int]]) -> tuple[LoraFormat, str | None]:
    """テンソル名から LoRA 形式を判定する。純関数。

    戻り値 `(format, detail)`。`format=="unknown"` は Anima 非互換で、`detail` に理由。
    真実源は Civitai のラベルではなくテンソル名。既定は reject（positive に Anima/Cosmos
    と識別できたものだけ非 unknown を返す）。
    """
    keys = list(header)
    if not keys:
        return "unknown", "空のチェックポイントです"

    if _has_marker(keys, _FLUX_MARKERS):
        return "unknown", "Flux LoRA です（未対応）"

    if _has_marker(keys, _SD_UNET_MARKERS) or _has_prefix(keys, "lora_te1_", "lora_te2_"):
        return "unknown", "SD系(SDXL/SD1.5 等)UNet LoRA です（未対応。Anima のみ対応）"

    if _has_prefix(keys, "lora_unet_", "lora_te_"):
        # kohya/sd-scripts 平坦化。SD 系 kohya は上の SD UNet ブロック名で既に弾かれている。
        if _has_marker(keys, _ANIMA_MARKERS):
            return "kohya", None
        return "unknown", "kohya形式だがAnima(Cosmos)構造に一致しません"

    if _has_prefix(keys, "transformer.", "text_conditioner."):
        if _has_marker(keys, _ANIMA_MARKERS):
            return "diffusers", None
        return "unknown", "diffusers形式だがAnima構造に一致しません"

    if _has_prefix(keys, "diffusion_model."):
        if _has_marker(keys, _ANIMA_MARKERS):
            return "diffusion_pipe", None
        return "unknown", "diffusion_model形式だがAnima構造に一致しません"

    return "unknown", "未知のLoRA形式です"


def _unflatten_dit_module(module_flat: str) -> str | None:
    """kohya の ``.``→``_`` 平坦化を 1 モジュール分だけ逆変換する。未知なら None。"""
    block = _KOHYA_BLOCK_RE.match(module_flat)
    if block:
        leaf = _KOHYA_BLOCK_LEAF.get(block.group(2))
        return f"blocks.{block.group(1)}.{leaf}" if leaf else None
    return _KOHYA_TOPLEVEL_LEAF.get(module_flat)


def _convert_kohya_to_diffusion_model(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """kohya の ``lora_unet_*`` キーを ``diffusion_model.*``(Cosmos dotted パス)へ書き換える。

    以降は ComfyUI 経路が変換を仕上げる。``lora_te_*``(Qwen3 encoder)は Anima パイプラインで
    LoRA 非対応なので破棄し、既知 leaf 語彙外の DiT モジュールも当て推量せず落とす（警告する）。
    ``.alpha`` は保持し、後段の :func:`_fold_alpha` で畳み込む。
    """
    converted: dict[str, torch.Tensor] = {}
    skipped_te = 0
    skipped_unknown: list[str] = []
    for key, value in state_dict.items():
        if key.startswith("lora_te_"):
            skipped_te += 1
            continue
        if not key.startswith("lora_unet_"):
            skipped_unknown.append(key)
            continue
        module_flat, _, suffix = key[len("lora_unet_") :].partition(".")
        dotted = _unflatten_dit_module(module_flat)
        if dotted is None:
            skipped_unknown.append(key)
            continue
        converted[f"diffusion_model.{dotted}.{suffix}"] = value

    if skipped_te:
        logger.warning(
            "kohya LoRA: Qwen3 text-encoder キーを %d 件破棄（Anima に読み込めない）", skipped_te
        )
    if skipped_unknown:
        logger.warning(
            "kohya LoRA: 未知モジュールキーを %d 件破棄、例 %s",
            len(skipped_unknown),
            skipped_unknown[:3],
        )
    if not converted:
        raise LoraIncompatibleError("kohya LoRA に変換可能なDiT(lora_unet_)キーがありません")
    return converted


def _lora_pair_keys(
    state_dict: dict[str, torch.Tensor], module: str
) -> tuple[str | None, str | None]:
    """あるモジュールの (down, up) 重みキーを、どちらの命名規約でも返す。"""
    for down_suffix, up_suffix in (
        (".lora_down.weight", ".lora_up.weight"),
        (".lora_A.weight", ".lora_B.weight"),
    ):
        down, up = module + down_suffix, module + up_suffix
        if down in state_dict and up in state_dict:
            return down, up
    return None, None


def _fold_alpha(state_dict: dict[str, torch.Tensor]) -> None:
    """モジュール毎の ``.alpha`` スケールを LoRA 重みへ畳み込み、alpha テンソルを落とす。

    kohya/ComfyUI は module 毎に ``.alpha`` スカラを持ち、実効スケールは ``alpha/rank``。
    diffusers の Anima ローダは ``network_alphas=None`` 固定で alpha を適用せず、生の
    ``.alpha`` キーは妥当性チェック(``all("lora" in key)``)を壊すため、ここで焼き込む。
    オーバーフロー回避のため係数を down/up(A/B)へ ``sqrt(scale)`` で分割する。
    """
    alpha_keys = [k for k in state_dict if k.endswith(".alpha")]
    for alpha_key in alpha_keys:
        alpha = float(state_dict.pop(alpha_key).item())
        module = alpha_key[: -len(".alpha")]
        down_key, up_key = _lora_pair_keys(state_dict, module)
        if down_key is None or up_key is None:
            logger.warning("LoRA alpha %s に対応する重みが無い; 無視", alpha_key)
            continue
        rank = state_dict[down_key].shape[0]
        if rank == 0:
            continue
        factor = math.sqrt(alpha / rank)
        state_dict[down_key] = state_dict[down_key] * factor
        state_dict[up_key] = state_dict[up_key] * factor


def _drop_dora(state_dict: dict[str, torch.Tensor]) -> None:
    """DoRA テンソル(未対応)を破棄する。magnitude が失われる旨を警告。"""
    dora_keys = [k for k in state_dict if "dora_scale" in k or "lora_magnitude_vector" in k]
    if dora_keys:
        logger.warning(
            "DoRA LoRA を検出; magnitude ベクトルを破棄し plain LoRA として読み込む: %d 件",
            len(dora_keys),
        )
        for key in dora_keys:
            del state_dict[key]


def _to_peft_lora_naming(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``lora_down``/``lora_up`` → PEFT ``lora_A``/``lora_B`` に改名（既に A/B なら no-op）。

    diffusers の Anima ローダの UNET→PEFT フォールバックは dotted ``.lora.down`` 綴りしか
    書き換えないため、kohya/diffusion-pipe の ``lora_down``/``lora_up`` を放置すると
    ``lora_B`` キーが見つからず rank 辞書が空になり ``IndexError`` を出す。
    """
    return {
        key.replace(".lora_down.", ".lora_A.").replace(".lora_up.", ".lora_B."): value
        for key, value in state_dict.items()
    }


def lora_state_dict_for(path: Path) -> dict[str, torch.Tensor]:
    """LoRA ファイルを読み ``load_lora_weights`` に渡せる state dict へ変換する。

    非 Anima / 未認識チェックポイントは :class:`LoraIncompatibleError`。
    """
    if not path.is_file():
        raise FileNotFoundError(f"LoRA not found: {path}")

    state_dict = load_file(str(path), device="cpu")
    fmt, detail = detect_anima_format({k: list(v.shape) for k, v in state_dict.items()})
    if fmt == "unknown":
        raise LoraIncompatibleError(detail or "このLoRAはAnimaと互換性がありません")

    if fmt == "kohya":
        state_dict = _convert_kohya_to_diffusion_model(state_dict)
    _drop_dora(state_dict)
    _fold_alpha(state_dict)
    return _to_peft_lora_naming(state_dict)
