from __future__ import annotations

from pathlib import Path

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


def test_image_steps_cfg_explicit_turbo_override() -> None:
    """`turbo` を明示すると turbo_enabled を上書きしてモードを固定できる（/generate の base 整合用）。"""
    s = Settings(image_turbo_lora=_TURBO_URN, image_turbo_steps=8, image_turbo_cfg=1.0)
    # turbo_enabled=True でも明示 False で base 値を返す。
    assert s.image_steps_cfg(turbo=False) == (s.default_steps, s.default_cfg)
    # 明示 True は Turbo 値。
    assert s.image_steps_cfg(turbo=True) == (8, 1.0)
    # None(既定) は turbo_enabled に従う。
    assert s.image_steps_cfg() == (8, 1.0)


def test_tags_csv_is_derived_from_tags_dir() -> None:
    # 索引元 CSV パスは tags_dir から一意に導出する（別フィールドで二重管理しない）。
    s = Settings(tags_dir=Path("/data/tags"))
    assert s.tags_csv == Path("/data/tags/danbooru.csv")


def test_tags_auto_download_defaults_off() -> None:
    # WSL2 オフライン運用を既定にするため自動 DL は既定 OFF（事前配置前提）。
    assert Settings().tags_auto_download is False


def test_context_length_defaults_are_long_with_kv_quant() -> None:
    # 16GB に収めるため長文脈は KV 量子化(q8_0)+flash_attn 前提が既定。
    s = Settings()
    assert s.llm_n_ctx == 16384
    assert s.llm_kv_cache_type == "q8_0"
    assert s.llm_flash_attn is True
