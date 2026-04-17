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


def _settings(tmp_path: Path) -> Settings:
    (tmp_path / "models").mkdir()
    (tmp_path / "images").mkdir()
    (tmp_path / "weights").mkdir()
    return Settings(
        hf_home=tmp_path / "models",
        images_dir=tmp_path / "images",
        weights_dir=tmp_path / "weights",
        llm_model_id="google/gemma-4-E4B-it",
        image_model_id="urn:air:anima:checkpoint:civitai:2544636@2859702",
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
    assert resolved is not None
    expected = settings.weights_dir / "civitai" / f"waianima-v10-{sha[:12]}.safetensors"
    assert resolved == expected
    assert resolved.read_bytes() == body
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
    assert resolved == target
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
    assert resolved == local


def test_ensure_all_raises_when_local_path_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    settings = settings.model_copy(update={"image_model_id": str(tmp_path / "nope.safetensors")})
    monkeypatch.setattr(fetch_models, "_ensure_llm", lambda _: None)
    with pytest.raises(FetchError, match="存在しません"):
        ensure_all(settings)


def test_ensure_all_calls_snapshot_for_hf_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    settings = settings.model_copy(update={"image_model_id": "hdae/diffusers-anima-preview"})
    monkeypatch.setattr(fetch_models, "_ensure_llm", lambda _: None)

    calls: list[dict[str, Any]] = []

    def fake_snapshot(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(tmp_path / "hf-cache" / "hdae--diffusers-anima-preview")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)

    resolved = ensure_all(settings)
    assert resolved == Path(str(tmp_path / "hf-cache" / "hdae--diffusers-anima-preview"))
    assert calls and calls[0]["repo_id"] == "hdae/diffusers-anima-preview"


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
