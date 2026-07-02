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


def test_context_length_defaults_fp16_kv_with_compact_swa() -> None:
    # 退行ガード: 旧既定(q8_0+16384)は実運用の多ターンで劣化が報告され撤回した。
    # 既定は fp16 KV + iSWA コンパクトキャッシュ(swa_full=False) + 32768
    # （実測と経緯は docs/decisions/0003-kv-cache-fp16-iswa.md）。
    s = Settings()
    assert s.llm_n_ctx == 32768
    assert s.llm_kv_cache_type == "f16"
    assert s.llm_flash_attn is True
    assert s.llm_swa_full is False


def test_tag_hints_default_on() -> None:
    # 事前検索(タグ候補注入)は既定 ON。OFF は A/B・切り分け用の kill-switch
    # （docs/decisions/0005-tag-hints-presearch.md）。
    assert Settings().llm_tag_hints is True
