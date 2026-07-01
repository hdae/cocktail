"""lora_convert の純関数テスト（GPU 不要）。

形式判定・kohya アンフラット・alpha/dora 処理を、形式ごとの小さな合成 safetensors で検証する。
実パイプラインへの `load_lora_weights` 注入は GPU 限定なので別途スモークで確認する。
参考実装: vImagen `tests/test_loras.py`。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from cocktail_server.services.lora_convert import (
    LoraIncompatibleError,
    _convert_kohya_to_diffusion_model,
    _fold_alpha,
    detect_anima_format,
    lora_state_dict_for,
)
from safetensors.torch import save_file


def _down_up(rank: int = 4, in_dim: int = 8, out_dim: int = 8) -> dict[str, torch.Tensor]:
    return {"lora_down": torch.ones(rank, in_dim), "lora_up": torch.ones(out_dim, rank)}


def _header(sd: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    return {key: list(value.shape) for key, value in sd.items()}


def _comfy_sd() -> dict[str, torch.Tensor]:
    pair = _down_up()
    return {
        "diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight": pair["lora_down"],
        "diffusion_model.blocks.0.self_attn.q_proj.lora_up.weight": pair["lora_up"],
    }


def _comfy_ab_sd() -> dict[str, torch.Tensor]:
    """現行 Civitai(diffusion-pipe)エクスポート: 既に lora_A/lora_B・alpha 無し。

    各キーは独立テンソル（safetensors は共有メモリを保存できない）。
    """
    return {
        "diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight": torch.ones(4, 8),
        "diffusion_model.blocks.0.self_attn.q_proj.lora_B.weight": torch.ones(8, 4),
        "diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.lora_A.weight": torch.ones(4, 8),
        "diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.lora_B.weight": torch.ones(8, 4),
    }


def _diffusers_sd() -> dict[str, torch.Tensor]:
    pair = _down_up()
    return {
        "transformer.transformer_blocks.0.attn1.to_q.lora_A.weight": pair["lora_down"],
        "transformer.transformer_blocks.0.attn1.to_q.lora_B.weight": pair["lora_up"],
    }


def _kohya_sd() -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for module in (
        "blocks_0_self_attn_q_proj",
        "blocks_0_mlp_layer1",
        "blocks_1_cross_attn_v_proj",
    ):
        sd[f"lora_unet_{module}.lora_down.weight"] = torch.ones(8, 16)
        sd[f"lora_unet_{module}.lora_up.weight"] = torch.ones(16, 8)
        sd[f"lora_unet_{module}.alpha"] = torch.tensor(4.0)
    # Qwen3 text-encoder モジュール — 破棄されるべき。
    sd["lora_te_layers_0_self_attn_q_proj.lora_down.weight"] = torch.ones(8, 16)
    sd["lora_te_layers_0_self_attn_q_proj.lora_up.weight"] = torch.ones(16, 8)
    sd["lora_te_layers_0_self_attn_q_proj.alpha"] = torch.tensor(4.0)
    return sd


def _sdxl_sd() -> dict[str, torch.Tensor]:
    attn = "lora_unet_input_blocks_4_1_transformer_blocks_0_attn2_to_k"
    return {
        f"{attn}.lora_down.weight": torch.ones(4, 2048),
        f"{attn}.lora_up.weight": torch.ones(640, 4),
    }


class TestDetectAnimaFormat:
    def test_comfy_diffusion_model_is_anima(self) -> None:
        assert detect_anima_format(_header(_comfy_sd())) == ("diffusion_pipe", None)

    def test_comfy_lora_a_b_is_anima(self) -> None:
        # 我々の実際の Turbo LoRA の形（lora_A/B・alpha 無し）。
        assert detect_anima_format(_header(_comfy_ab_sd())) == ("diffusion_pipe", None)

    def test_diffusers_prefixed_is_anima(self) -> None:
        assert detect_anima_format(_header(_diffusers_sd()))[0] == "diffusers"

    def test_anima_kohya_is_kohya(self) -> None:
        assert detect_anima_format(_header(_kohya_sd()))[0] == "kohya"

    def test_sdxl_is_rejected(self) -> None:
        fmt, detail = detect_anima_format(_header(_sdxl_sd()))
        assert fmt == "unknown"
        assert detail is not None

    def test_flux_is_rejected(self) -> None:
        keys = [
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_down.weight",
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_up.weight",
        ]
        assert detect_anima_format({k: [4, 8] for k in keys})[0] == "unknown"

    def test_diffusion_model_without_anima_leaves_is_rejected(self) -> None:
        assert detect_anima_format({"diffusion_model.foo.lora_down.weight": [4, 8]})[0] == "unknown"

    def test_empty_is_rejected(self) -> None:
        assert detect_anima_format({})[0] == "unknown"


class TestFoldAlpha:
    def test_scale_is_split_sqrt_across_the_pair(self) -> None:
        rank, alpha = 4, 2.0
        sd = {
            "m.lora_down.weight": torch.ones(rank, 8),
            "m.lora_up.weight": torch.ones(8, rank),
            "m.alpha": torch.tensor(alpha),
        }
        _fold_alpha(sd)
        factor = math.sqrt(alpha / rank)
        assert "m.alpha" not in sd
        assert torch.allclose(sd["m.lora_down.weight"], torch.full((rank, 8), factor))
        assert torch.allclose(sd["m.lora_up.weight"], torch.full((8, rank), factor))

    def test_product_preserves_alpha_over_rank(self) -> None:
        rank, alpha = 4, 1.0  # scale 0.25
        sd = {
            "m.lora_down.weight": torch.ones(rank, 2),
            "m.lora_up.weight": torch.ones(3, rank),
            "m.alpha": torch.tensor(alpha),
        }
        _fold_alpha(sd)
        product = sd["m.lora_up.weight"] @ sd["m.lora_down.weight"]
        expected = torch.full((3, 2), rank * (alpha / rank))
        assert torch.allclose(product, expected)

    def test_lora_a_b_naming_is_also_folded(self) -> None:
        rank, alpha = 2, 1.0
        sd = {
            "m.lora_A.weight": torch.ones(rank, 4),
            "m.lora_B.weight": torch.ones(4, rank),
            "m.alpha": torch.tensor(alpha),
        }
        _fold_alpha(sd)
        assert "m.alpha" not in sd
        assert torch.allclose(sd["m.lora_A.weight"], torch.full((rank, 4), math.sqrt(0.5)))

    def test_alpha_without_a_matching_pair_is_dropped(self) -> None:
        sd = {"m.alpha": torch.tensor(1.0), "other.lora_down.weight": torch.ones(2, 2)}
        _fold_alpha(sd)
        assert "m.alpha" not in sd
        assert torch.allclose(sd["other.lora_down.weight"], torch.ones(2, 2))


class TestKohyaConversion:
    def test_unflattens_dit_modules_to_diffusion_model(self) -> None:
        out = _convert_kohya_to_diffusion_model(_kohya_sd())
        assert "diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight" in out
        assert "diffusion_model.blocks.0.self_attn.q_proj.alpha" in out
        assert "diffusion_model.blocks.0.mlp.layer1.lora_up.weight" in out
        assert "diffusion_model.blocks.1.cross_attn.v_proj.lora_down.weight" in out

    def test_drops_text_encoder_keys(self) -> None:
        out = _convert_kohya_to_diffusion_model(_kohya_sd())
        assert not any("lora_te" in k or "text_enc" in k for k in out)
        assert len(out) == 9  # 3 DiT modules × (down, up, alpha)

    def test_skips_unknown_dit_modules_rather_than_guessing(self) -> None:
        sd = {
            "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight": torch.ones(8, 16),
            "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight": torch.ones(16, 8),
            "lora_unet_blocks_0_mystery_layer.lora_down.weight": torch.ones(8, 16),
            "lora_unet_blocks_0_mystery_layer.lora_up.weight": torch.ones(16, 8),
        }
        out = _convert_kohya_to_diffusion_model(sd)
        assert "diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight" in out
        assert not any("mystery" in k for k in out)

    def test_no_convertible_keys_raises(self) -> None:
        with pytest.raises(LoraIncompatibleError):
            _convert_kohya_to_diffusion_model(
                {"lora_te_layers_0_self_attn_q_proj.lora_down.weight": torch.ones(8, 16)}
            )


class TestLoraStateDictFor:
    def test_comfy_file_loads_and_strips_alpha(self, tmp_path: Path) -> None:
        sd = _comfy_sd()
        sd["diffusion_model.blocks.0.self_attn.q_proj.alpha"] = torch.tensor(2.0)
        path = tmp_path / "x.safetensors"
        save_file(sd, str(path))

        out = lora_state_dict_for(path)
        assert not any(k.endswith(".alpha") for k in out)
        assert all(k.startswith("diffusion_model.") for k in out)
        factor = math.sqrt(2.0 / 4)
        assert torch.allclose(
            out["diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight"],
            torch.full((4, 8), factor),
        )

    def test_our_turbo_lora_shape_is_near_passthrough(self, tmp_path: Path) -> None:
        # lora_A/B・alpha 無しの diffusion_pipe 形式 = 実際の Turbo LoRA。素通しに近い。
        path = tmp_path / "turbo.safetensors"
        save_file(_comfy_ab_sd(), str(path))
        out = lora_state_dict_for(path)
        assert set(out) == set(_comfy_ab_sd())  # キーは不変
        assert all(k.startswith("diffusion_model.") for k in out)

    def test_diffusers_file_passes_through(self, tmp_path: Path) -> None:
        path = tmp_path / "d.safetensors"
        save_file(_diffusers_sd(), str(path))
        out = lora_state_dict_for(path)
        assert all(k.startswith("transformer.") for k in out)

    def test_kohya_file_unflattens_and_drops_text_encoder(self, tmp_path: Path) -> None:
        path = tmp_path / "k.safetensors"
        save_file(_kohya_sd(), str(path))
        out = lora_state_dict_for(path)
        assert out and all(k.startswith("diffusion_model.") for k in out)
        assert not any(k.endswith(".alpha") for k in out)
        assert not any(k.startswith("lora_te_") for k in out)
        assert "diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight" in out
        assert "diffusion_model.blocks.0.mlp.layer1.lora_B.weight" in out

    def test_output_uses_peft_lora_a_b_naming(self, tmp_path: Path) -> None:
        for name, sd in (("comfy", _comfy_sd()), ("kohya", _kohya_sd())):
            path = tmp_path / f"{name}.safetensors"
            save_file(sd, str(path))
            out = lora_state_dict_for(path)
            assert not any(".lora_down." in k or ".lora_up." in k for k in out)
            assert any(".lora_A." in k for k in out)
            assert any(".lora_B." in k for k in out)

    def test_sdxl_raises_incompatible(self, tmp_path: Path) -> None:
        path = tmp_path / "s.safetensors"
        save_file(_sdxl_sd(), str(path))
        with pytest.raises(LoraIncompatibleError):
            lora_state_dict_for(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            lora_state_dict_for(tmp_path / "nope.safetensors")
