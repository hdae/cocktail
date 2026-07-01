from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from cocktail_server.config import Settings
from cocktail_server.scripts import fetch_models
from cocktail_server.scripts.fetch_models import (
    FetchError,
    _select_primary_file,
    _slugify,
    ensure_all,
    parse_air,
)

# autouse フィクスチャで差し替えるため、実関数の参照を import 時に確保しておく。
_REAL_ENSURE_IMAGE_BASE = fetch_models._ensure_image_base
_REAL_ENSURE_LLM = fetch_models._ensure_llm


@pytest.fixture(autouse=True)
def _stub_image_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """ensure_all が呼ぶ base リポの実ダウンロードを既定で抑止する。"""
    monkeypatch.setattr(fetch_models, "_ensure_image_base", lambda _: None)


def _settings(tmp_path: Path) -> Settings:
    (tmp_path / "models").mkdir()
    (tmp_path / "images").mkdir()
    (tmp_path / "weights").mkdir()
    return Settings(
        hf_home=tmp_path / "models",
        images_dir=tmp_path / "images",
        weights_dir=tmp_path / "weights",
        llm_model_id="igorls/gemma-4-12B-it-heretic-GGUF:gemma-4-12B-it-heretic-Q4_K_M.gguf",
        image_model_id="urn:air:anima:checkpoint:civitai:2544636@2983680",
        image_turbo_lora="",
        civitai_token=None,
    )


def test_parse_air_accepts_civitai_urn() -> None:
    air = parse_air("urn:air:anima:checkpoint:civitai:2544636@2859702")
    assert air.ecosystem == "anima"
    assert air.type == "checkpoint"
    assert air.source == "civitai"
    assert air.model_id == 2544636
    assert air.version_id == 2859702


def test_parse_air_rejects_bad_format() -> None:
    with pytest.raises(FetchError):
        parse_air("not-a-urn")


def test_parse_air_rejects_unknown_source() -> None:
    with pytest.raises(FetchError):
        parse_air("urn:air:anima:checkpoint:huggingface:1@2")


def test_slugify_lowercases_and_strips_extension() -> None:
    assert _slugify("waiANIMA_v10.safetensors") == "waianima-v10"
    assert _slugify("Some Model.ckpt") == "some-model"


def test_select_primary_file_prefers_primary_model() -> None:
    payload: dict[str, Any] = {
        "files": [
            {"type": "VAE", "primary": False, "name": "vae.safetensors"},
            {"type": "Model", "primary": False, "name": "old.safetensors"},
            {"type": "Model", "primary": True, "name": "new.safetensors"},
        ]
    }
    chosen = _select_primary_file(payload)
    assert chosen["name"] == "new.safetensors"


def test_select_primary_file_falls_back_to_first_model() -> None:
    payload: dict[str, Any] = {
        "files": [
            {"type": "VAE", "primary": False, "name": "vae.safetensors"},
            {"type": "Model", "primary": False, "name": "only.safetensors"},
        ]
    }
    chosen = _select_primary_file(payload)
    assert chosen["name"] == "only.safetensors"


def test_select_primary_file_raises_when_no_model() -> None:
    with pytest.raises(FetchError):
        _select_primary_file({"files": [{"type": "VAE", "name": "v.safetensors"}]})


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, json_data: Any | None = None) -> None:
        self.status_code = status_code
        self._json = json_data
        self.text = str(json_data) if json_data is not None else ""

    def json(self) -> Any:
        return self._json


class _FakeStreamResponse:
    def __init__(self, *, status_code: int, chunks: list[bytes], content_length: int) -> None:
        self.status_code = status_code
        self._chunks = chunks
        self.headers = {"content-length": str(content_length)}

    def iter_bytes(self, _size: int) -> Iterator[bytes]:
        yield from self._chunks


class _FakeClient:
    def __init__(
        self,
        *,
        get_response: _FakeResponse,
        stream_response: _FakeStreamResponse,
    ) -> None:
        self._get_response = get_response
        self._stream_response = stream_response
        self.get_calls: list[tuple[str, dict[str, str]]] = []
        self.stream_calls: list[tuple[str, str, dict[str, str]]] = []

    def get(self, url: str, headers: dict[str, str]) -> _FakeResponse:
        self.get_calls.append((url, dict(headers)))
        return self._get_response

    @contextmanager
    def stream(
        self, method: str, url: str, headers: dict[str, str]
    ) -> Iterator[_FakeStreamResponse]:
        self.stream_calls.append((method, url, dict(headers)))
        yield self._stream_response

    def close(self) -> None:
        pass


def _prepare_client(
    *, body: bytes, sha_override: str | None = None
) -> tuple[_FakeClient, str, str]:
    sha = sha_override or hashlib.sha256(body).hexdigest()
    name = "waiANIMA_v10.safetensors"
    download_url = "https://civitai.com/api/download/models/2859702"
    get_response = _FakeResponse(
        status_code=200,
        json_data={
            "files": [
                {
                    "type": "Model",
                    "primary": True,
                    "name": name,
                    "downloadUrl": download_url,
                    "hashes": {"SHA256": sha},
                }
            ]
        },
    )
    stream_response = _FakeStreamResponse(status_code=200, chunks=[body], content_length=len(body))
    return (
        _FakeClient(get_response=get_response, stream_response=stream_response),
        sha,
        download_url,
    )


def test_ensure_all_downloads_civitai_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    body = b"fake-weights-payload"
    client, sha, _ = _prepare_client(body=body)

    monkeypatch.setattr(fetch_models, "_http_client", lambda timeout=30.0: client)
    called_llm: list[str] = []

    def fake_ensure_llm(s: Settings) -> None:
        called_llm.append(s.llm_model_id)

    monkeypatch.setattr(fetch_models, "_ensure_llm", fake_ensure_llm)

    resolved = ensure_all(settings)
    dit = resolved.image_dit_path
    assert dit is not None
    expected = settings.weights_dir / "civitai" / f"waianima-v10-{sha[:12]}.safetensors"
    assert dit == expected
    assert dit.read_bytes() == body
    assert resolved.turbo_lora_path is None
    assert called_llm == [settings.llm_model_id]


def test_ensure_all_skips_when_sha_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    body = b"pre-existing"
    sha = hashlib.sha256(body).hexdigest()
    target_dir = settings.weights_dir / "civitai"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"waianima-v10-{sha[:12]}.safetensors"
    target.write_bytes(body)

    client, _, _ = _prepare_client(body=body, sha_override=sha)
    monkeypatch.setattr(fetch_models, "_http_client", lambda timeout=30.0: client)
    monkeypatch.setattr(fetch_models, "_ensure_llm", lambda _: None)

    resolved = ensure_all(settings)
    assert resolved.image_dit_path == target
    assert client.stream_calls == []  # stream は呼ばれない


def test_ensure_all_raises_on_sha_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    body = b"ok-body"
    # Civitai が返す期待 SHA と、実際の body が一致しないケース
    client, _, _ = _prepare_client(body=body, sha_override="0" * 64)
    monkeypatch.setattr(fetch_models, "_http_client", lambda timeout=30.0: client)
    monkeypatch.setattr(fetch_models, "_ensure_llm", lambda _: None)

    with pytest.raises(FetchError, match="SHA256"):
        ensure_all(settings)

    # tmp ファイルが残っていない
    tmp_dir = settings.weights_dir / "civitai" / ".tmp"
    assert not tmp_dir.exists() or not any(tmp_dir.iterdir())


def test_ensure_all_accepts_local_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    local = tmp_path / "custom.safetensors"
    local.write_bytes(b"payload")
    settings = settings.model_copy(update={"image_model_id": str(local)})

    monkeypatch.setattr(fetch_models, "_ensure_llm", lambda _: None)

    resolved = ensure_all(settings)
    assert resolved.image_dit_path == local


def test_ensure_all_raises_when_local_path_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    settings = settings.model_copy(update={"image_model_id": str(tmp_path / "nope.safetensors")})
    monkeypatch.setattr(fetch_models, "_ensure_llm", lambda _: None)
    with pytest.raises(FetchError, match="存在しません"):
        ensure_all(settings)


def test_ensure_all_rejects_hf_repo_as_image_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # diffusers 形式リポは派生(IMAGE_MODEL_ID)ではなくベースに置く運用へ変えたため拒否する。
    settings = _settings(tmp_path)
    settings = settings.model_copy(update={"image_model_id": "someorg/some-anima-diffusers"})
    monkeypatch.setattr(fetch_models, "_ensure_llm", lambda _: None)
    with pytest.raises(FetchError, match="IMAGE_BASE_MODEL_ID"):
        ensure_all(settings)


def test_ensure_all_returns_none_without_derivative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 派生未指定(空)ならベースだけを使うため None を返す。
    settings = _settings(tmp_path)
    settings = settings.model_copy(update={"image_model_id": ""})
    monkeypatch.setattr(fetch_models, "_ensure_llm", lambda _: None)
    assert ensure_all(settings).image_dit_path is None


def test_ensure_image_base_downloads_base_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    calls: list[dict[str, Any]] = []

    def fake_snapshot(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(tmp_path / "hf-cache")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    _REAL_ENSURE_IMAGE_BASE(settings)
    assert calls and calls[0]["repo_id"] == settings.image_base_model_id


def test_civitai_401_hints_at_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    unauth = _FakeResponse(status_code=401, json_data={"error": "gated"})

    class _Client:
        def get(self, url: str, headers: dict[str, str]) -> _FakeResponse:
            return unauth

        def close(self) -> None:
            pass

    monkeypatch.setattr(fetch_models, "_http_client", lambda timeout=30.0: _Client())
    monkeypatch.setattr(fetch_models, "_ensure_llm", lambda _: None)

    with pytest.raises(FetchError, match="CIVITAI_TOKEN"):
        ensure_all(settings)


def test_parse_gguf_ref_forms(tmp_path: Path) -> None:
    from cocktail_server.services.llm import parse_gguf_ref

    # repo:filename 形式 → 分解
    assert parse_gguf_ref("org/repo:model-Q4_K_M.gguf") == ("org/repo", "model-Q4_K_M.gguf")
    # filename 無しの素のリポ ID → None（ローカル扱い）
    assert parse_gguf_ref("org/repo") is None
    # 実在するローカル .gguf パス → None（パスとしてそのまま使う）
    local = tmp_path / "m.gguf"
    local.write_bytes(b"x")
    assert parse_gguf_ref(str(local)) is None


def test_ensure_llm_downloads_gguf_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    calls: list[dict[str, Any]] = []

    def fake_hf_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(tmp_path / "x.gguf")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_download)
    _REAL_ENSURE_LLM(settings)
    assert calls and calls[0]["repo_id"] == "igorls/gemma-4-12B-it-heretic-GGUF"
    assert calls[0]["filename"] == "gemma-4-12B-it-heretic-Q4_K_M.gguf"


def test_ensure_llm_rejects_non_gguf(tmp_path: Path) -> None:
    # GGUF でも実在ローカルパスでもない指定は拒否する。
    settings = _settings(tmp_path).model_copy(update={"llm_model_id": "google/gemma-4-E4B-it"})
    with pytest.raises(FetchError, match="gguf"):
        _REAL_ENSURE_LLM(settings)


def test_ensure_turbo_lora_empty_returns_none(tmp_path: Path) -> None:
    # 空文字は Turbo 無効(base 品質) → None。
    assert fetch_models._ensure_turbo_lora(_settings(tmp_path)) is None


def test_ensure_turbo_lora_accepts_local_path(tmp_path: Path) -> None:
    local = tmp_path / "turbo.safetensors"
    local.write_bytes(b"lora")
    settings = _settings(tmp_path).model_copy(update={"image_turbo_lora": str(local)})
    assert fetch_models._ensure_turbo_lora(settings) == local


def test_ensure_turbo_lora_raises_when_local_missing(tmp_path: Path) -> None:
    settings = _settings(tmp_path).model_copy(
        update={"image_turbo_lora": str(tmp_path / "nope.safetensors")}
    )
    with pytest.raises(FetchError, match="存在しません"):
        fetch_models._ensure_turbo_lora(settings)


def test_ensure_turbo_lora_downloads_from_civitai(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"turbo-lora-bytes"
    client, _, _ = _prepare_client(body=body)
    monkeypatch.setattr(fetch_models, "_http_client", lambda timeout=30.0: client)
    settings = _settings(tmp_path).model_copy(
        update={"image_turbo_lora": "urn:air:anima:lora:civitai:2560840@2979642"}
    )
    resolved = fetch_models._ensure_turbo_lora(settings)
    assert resolved is not None
    assert resolved.read_bytes() == body
    assert resolved.parent == settings.weights_dir / "civitai"


def test_ensure_all_resolves_both_dit_and_turbo_lora(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # DiT 派生(image_model_id)と Turbo LoRA を両方 Civitai から解決する。
    body = b"shared-fake-weights"
    client, _, _ = _prepare_client(body=body)
    monkeypatch.setattr(fetch_models, "_http_client", lambda timeout=30.0: client)
    monkeypatch.setattr(fetch_models, "_ensure_llm", lambda _: None)
    settings = _settings(tmp_path).model_copy(
        update={"image_turbo_lora": "urn:air:anima:lora:civitai:2560840@2979642"}
    )
    resolved = ensure_all(settings)
    assert resolved.image_dit_path is not None
    assert resolved.turbo_lora_path is not None
    assert resolved.turbo_lora_path.read_bytes() == body
