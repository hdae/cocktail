from __future__ import annotations

from typing import Any

import pytest
from cocktail_server.config import Settings
from cocktail_server.main import _turbo_without_preload_warning
from cocktail_server.services.image_gen import ImageGenService

_TURBO_URN = "urn:air:anima:lora:civitai:2560840@2979642"


class _FakePipe:
    """PEFT トグルと注入だけを観測する最小の pipe スパイ（GPU 非依存）。"""

    def __init__(self, adapters_by_component: dict[str, list[str]]) -> None:
        self._adapters_by_component = adapters_by_component
        self.loaded: list[tuple[dict[str, Any], str]] = []
        self.set_adapters_calls: list[tuple[list[str], list[float]]] = []
        self.enabled: bool | None = None

    def load_lora_weights(self, state: dict[str, Any], adapter_name: str) -> None:
        self.loaded.append((state, adapter_name))

    def get_list_adapters(self) -> dict[str, list[str]]:
        return self._adapters_by_component

    def set_adapters(self, names: list[str], strengths: list[float]) -> None:
        self.set_adapters_calls.append((names, strengths))

    def enable_lora(self) -> None:
        self.enabled = True

    def disable_lora(self) -> None:
        self.enabled = False


# --- _apply_turbo: 生成毎のトグルと fail-loud -------------------------------------


def test_apply_turbo_fail_loud_when_requested_but_not_loaded() -> None:
    svc = ImageGenService("base")
    # _turbo_loaded は既定 False。turbo を要求されたら static に落とす（静かに base 劣化させない）。
    with pytest.raises(RuntimeError, match="読み込まれていません"):
        svc._apply_turbo(True)


def test_apply_turbo_base_is_noop_when_not_loaded() -> None:
    svc = ImageGenService("base")
    # turbo=False なら LoRA 未ロードでも base として素通りする（例外を投げない）。
    svc._apply_turbo(False)


def test_apply_turbo_toggles_enable_disable_when_loaded() -> None:
    svc = ImageGenService("base")
    pipe = _FakePipe({"transformer": ["turbo"]})
    svc._pipe = pipe
    svc._turbo_loaded = True

    svc._apply_turbo(True)
    assert pipe.enabled is True
    svc._apply_turbo(False)
    assert pipe.enabled is False
    # 現フラグのみで無条件に適用する（前ターンの状態に依存しない）。
    svc._apply_turbo(True)
    assert pipe.enabled is True


# --- _inject_turbo_lora: 注入と DiT 適用の検証（D） -------------------------------


def _inject_with(
    monkeypatch: pytest.MonkeyPatch, pipe: _FakePipe, strength: float = 1.0
) -> ImageGenService:
    # _inject_turbo_lora は関数内で lora_convert.lora_state_dict_for を import するので、
    # そのモジュール属性を差し替えればディスク I/O 無しで注入経路を通せる。
    from cocktail_server.services import lora_convert

    monkeypatch.setattr(lora_convert, "lora_state_dict_for", lambda _p: {"k": 1})
    svc = ImageGenService(
        "base", turbo_lora_path="/tmp/turbo.safetensors", turbo_lora_strength=strength
    )
    svc._pipe = pipe
    svc._inject_turbo_lora()
    return svc


def test_inject_turbo_lora_success_sets_strength_and_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = _FakePipe({"transformer": ["turbo"], "text_conditioner": ["turbo"]})
    svc = _inject_with(monkeypatch, pipe, strength=0.7)
    assert svc._turbo_loaded is True
    assert pipe.set_adapters_calls == [(["turbo"], [0.7])]


def test_inject_turbo_lora_fail_loud_when_dit_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from cocktail_server.services import lora_convert

    monkeypatch.setattr(lora_convert, "lora_state_dict_for", lambda _p: {"k": 1})
    # text_conditioner だけに載って DiT(transformer) が落ちた退行を捕捉する。
    pipe = _FakePipe({"text_conditioner": ["turbo"]})
    svc = ImageGenService("base", turbo_lora_path="/tmp/turbo.safetensors")
    svc._pipe = pipe
    with pytest.raises(RuntimeError, match="transformer"):
        svc._inject_turbo_lora()
    assert svc._turbo_loaded is False


def test_inject_turbo_lora_fail_loud_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    from cocktail_server.services import lora_convert

    monkeypatch.setattr(lora_convert, "lora_state_dict_for", lambda _p: {"k": 1})
    # アダプタ名が一切登録されない（0 件ロード）ケースも fail-loud。
    pipe = _FakePipe({"transformer": []})
    svc = ImageGenService("base", turbo_lora_path="/tmp/turbo.safetensors")
    svc._pipe = pipe
    with pytest.raises(RuntimeError):
        svc._inject_turbo_lora()
    assert svc._turbo_loaded is False


def test_inject_turbo_lora_noop_when_path_empty() -> None:
    # Turbo 無効(パス空) なら注入せず _turbo_loaded=False のまま。
    svc = ImageGenService("base")
    svc._pipe = _FakePipe({})
    svc._inject_turbo_lora()
    assert svc._turbo_loaded is False


# --- set_turbo_lora: ロード後の差し替え禁止 --------------------------------------


def test_set_turbo_lora_rejected_after_pipe_loaded() -> None:
    svc = ImageGenService("base")
    svc._pipe = _FakePipe({})
    with pytest.raises(RuntimeError, match="cannot change turbo lora"):
        svc.set_turbo_lora("/tmp/turbo.safetensors", 1.0)


# --- 起動ガード（A）: turbo 有効 × preload 無効の loud 検出 ----------------------


def test_turbo_without_preload_warns_only_on_conflicting_config() -> None:
    # turbo 有効 × preload 無効 → 警告文（生成が fail-loud する誤設定）。
    assert (
        _turbo_without_preload_warning(Settings(image_turbo_lora=_TURBO_URN, startup_preload=False))
        is not None
    )
    # turbo 無効なら preload 無効でも問題なし。
    assert (
        _turbo_without_preload_warning(Settings(image_turbo_lora="", startup_preload=False)) is None
    )
    # preload 有効なら turbo は正しく注入されるので問題なし。
    assert (
        _turbo_without_preload_warning(Settings(image_turbo_lora=_TURBO_URN, startup_preload=True))
        is None
    )
