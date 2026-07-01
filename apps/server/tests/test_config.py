from __future__ import annotations

from cocktail_server.config import Settings

_TURBO_URN = "urn:air:anima:lora:civitai:2560840@2979642"


def test_turbo_disabled_when_lora_empty() -> None:
    s = Settings(image_turbo_lora="")
    assert s.turbo_enabled is False
    # base モードでは default_steps / default_cfg を返す。
    assert s.image_steps_cfg() == (s.default_steps, s.default_cfg)


def test_turbo_enabled_uses_turbo_params() -> None:
    s = Settings(
        image_turbo_lora=_TURBO_URN,
        image_turbo_steps=8,
        image_turbo_cfg=1.0,
    )
    assert s.turbo_enabled is True
    # Turbo モードでは steps/cfg を蒸留値に切り替える（CFG プリセットは通らない）。
    assert s.image_steps_cfg() == (8, 1.0)


def test_local_lora_path_also_enables_turbo() -> None:
    # AIR に限らずローカルパス指定でも Turbo は有効。
    s = Settings(image_turbo_lora="/data/loras/anima-turbo.safetensors")
    assert s.turbo_enabled is True
