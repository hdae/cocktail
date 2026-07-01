"""必要なモデルウェイトがディスクに揃っているか確認し、足りなければ取得する。

呼び出し口は `ensure_all(settings) -> EnsuredModels`（DiT 派生と Turbo LoRA の
ローカル解決パスをまとめて返す）。

処理方針:
- LLM: HuggingFace リポ ID のみサポート。`snapshot_download` で取得（既存キャッシュがあれば即座に返る）。
- Image: `settings.image_model_id` を値のプレフィックスで振り分ける。
  1. `urn:air:...` で始まる → AIR(URN) を Civitai API で解決し
     `{weights_dir}/civitai/{slug}-{sha256[:12]}.{ext}` に配置
  2. `xxx/yyy` 形式 → HF リポ ID として `snapshot_download`
  3. 上記以外（`.safetensors` / `.ckpt` 等の明示ローカルパス）→ 存在確認のみ

Civitai AIR(URN) 例:
    urn:air:anima:checkpoint:civitai:2544636@2859702
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from cocktail_server.config import Settings

logger = logging.getLogger(__name__)

_AIR_RE = re.compile(
    r"^urn:air:(?P<ecosystem>[^:]+):(?P<type>[^:]+):(?P<source>[^:]+):"
    r"(?P<model_id>\d+)@(?P<version_id>\d+)$"
)
_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][\w\-.]*/[\w\-.]+$")
_SAFE_SLUG_RE = re.compile(r"[^a-z0-9]+")
_CIVITAI_API_BASE = "https://civitai.com/api/v1"
_DL_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class CivitaiAir:
    """AIR(URN) を分解した識別子。"""

    ecosystem: str
    type: str
    source: str
    model_id: int
    version_id: int


class FetchError(RuntimeError):
    """fetch_models が起動中断したい失敗をまとめて上げる例外。"""


def parse_air(urn: str) -> CivitaiAir:
    m = _AIR_RE.match(urn.strip())
    if not m:
        raise FetchError(f"AIR の書式が不正です: {urn!r}")
    source = m.group("source")
    if source != "civitai":
        raise FetchError(f"未対応の AIR ソース: {source!r}（現在は civitai のみ対応）")
    return CivitaiAir(
        ecosystem=m.group("ecosystem"),
        type=m.group("type"),
        source=source,
        model_id=int(m.group("model_id")),
        version_id=int(m.group("version_id")),
    )


def _looks_like_hf_repo(value: str) -> bool:
    return (
        "/" in value
        and not value.endswith((".safetensors", ".ckpt"))
        and bool(_HF_REPO_RE.match(value))
    )


def _slugify(name: str) -> str:
    stem = name.rsplit(".", 1)[0].lower()
    slug = _SAFE_SLUG_RE.sub("-", stem).strip("-")
    return slug or "model"


def _extension(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".safetensors"):
        return "safetensors"
    if lower.endswith(".ckpt"):
        return "ckpt"
    raise FetchError(f"Civitai ファイル拡張子が未対応: {name!r}")


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(_DL_CHUNK_BYTES)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _civitai_headers(settings: Settings) -> dict[str, str]:
    headers = {"User-Agent": "cocktail-server/0.0.0"}
    if settings.civitai_token:
        headers["Authorization"] = f"Bearer {settings.civitai_token}"
    return headers


def _select_primary_file(payload: dict[str, Any]) -> dict[str, Any]:
    files = payload.get("files") or []
    if not isinstance(files, list):
        raise FetchError("Civitai API レスポンスの files が配列ではありません")
    primary = [f for f in files if f.get("type") == "Model" and bool(f.get("primary"))]
    if primary:
        return primary[0]  # type: ignore[no-any-return]
    model_files = [f for f in files if f.get("type") == "Model"]
    if model_files:
        return model_files[0]  # type: ignore[no-any-return]
    raise FetchError("Civitai API レスポンスに Model 種別のファイルが見つかりません")


def _http_client(timeout: float = 30.0) -> httpx.Client:
    return httpx.Client(timeout=timeout, follow_redirects=True)


def _fetch_version_metadata(
    air: CivitaiAir, settings: Settings, *, client: httpx.Client
) -> dict[str, Any]:
    url = f"{_CIVITAI_API_BASE}/model-versions/{air.version_id}"
    r = client.get(url, headers=_civitai_headers(settings))
    if r.status_code == 401 or r.status_code == 403:
        raise FetchError(
            f"Civitai API が {r.status_code} を返しました。"
            "gated モデルの場合は CIVITAI_TOKEN を設定してください。"
        )
    if r.status_code >= 400:
        raise FetchError(
            f"Civitai API エラー {r.status_code}: {r.text[:200]}（version={air.version_id}）"
        )
    data = r.json()
    if not isinstance(data, dict):
        raise FetchError("Civitai API レスポンスが JSON オブジェクトではありません")
    return data


def _download_to(
    url: str,
    dest: Path,
    *,
    expected_sha256: str,
    settings: Settings,
    client: httpx.Client,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = dest.parent / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / f"{dest.stem}-{secrets.token_hex(4)}.part"

    try:
        headers = _civitai_headers(settings)
        h = hashlib.sha256()
        with client.stream("GET", url, headers=headers) as resp:
            if resp.status_code == 401 or resp.status_code == 403:
                raise FetchError(
                    f"Civitai ダウンロードが {resp.status_code} を返しました。"
                    "gated モデルの場合は CIVITAI_TOKEN を設定してください。"
                )
            if resp.status_code >= 400:
                raise FetchError(f"Civitai ダウンロードエラー {resp.status_code}: url={url}")
            total = int(resp.headers.get("content-length") or 0)
            done = 0
            next_log = 0
            with tmp.open("wb") as fp:
                for chunk in resp.iter_bytes(_DL_CHUNK_BYTES):
                    if not chunk:
                        continue
                    fp.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    if total and done >= next_log:
                        pct = done * 100 // total
                        logger.info(
                            "civitai download %s: %d%% (%.2f / %.2f GiB)",
                            dest.name,
                            pct,
                            done / (1024**3),
                            total / (1024**3),
                        )
                        next_log = done + max(total // 20, _DL_CHUNK_BYTES * 16)

        actual = h.hexdigest()
        if actual.lower() != expected_sha256.lower():
            raise FetchError(
                f"SHA256 不一致: expected={expected_sha256}, actual={actual}, url={url}"
            )

        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                logger.warning("一時ファイルの削除に失敗: %s", tmp)


def _resolve_civitai(
    air: CivitaiAir, settings: Settings, *, client: httpx.Client | None = None
) -> Path:
    owned_client = client is None
    cli = client or _http_client()
    try:
        meta = _fetch_version_metadata(air, settings, client=cli)
        file_meta = _select_primary_file(meta)
        name = str(file_meta.get("name") or "")
        download_url = str(file_meta.get("downloadUrl") or "")
        sha = str((file_meta.get("hashes") or {}).get("SHA256") or "")
        if not name or not download_url or not sha:
            raise FetchError(
                f"Civitai ファイルメタデータが不足しています: name={name!r}, "
                f"url={bool(download_url)}, sha={bool(sha)}"
            )

        slug = _slugify(name)
        ext = _extension(name)
        target = settings.weights_dir / "civitai" / f"{slug}-{sha.lower()[:12]}.{ext}"

        if target.exists():
            existing_sha = _sha256_of(target)
            if existing_sha.lower() == sha.lower():
                logger.info("civitai model already present (sha match): %s", target)
                return target
            logger.warning("既存ファイルの sha256 が一致しないため再取得します: %s", target)
            target.unlink()

        logger.info("fetching civitai model %s → %s", name, target)
        _download_to(download_url, target, expected_sha256=sha, settings=settings, client=cli)
        return target
    finally:
        if owned_client:
            cli.close()


def _ensure_llm(settings: Settings) -> None:
    """LLM(GGUF)をローカルに用意する。

    ``repo:filename.gguf`` 形式なら HF からそのファイルだけ取得し、ローカル
    ``.gguf`` パスなら存在確認のみ。LLM は llama.cpp(GGUF) で動かすため、
    diffusers 形式リポや transformers モデルは受け付けない。
    """
    from cocktail_server.services.llm import parse_gguf_ref

    ref = parse_gguf_ref(settings.llm_model_id)
    if ref is None:
        p = Path(settings.llm_model_id).expanduser()
        if not p.is_file():
            raise FetchError(
                "LLM_MODEL_ID は 'repo:filename.gguf' 形式かローカル .gguf パスで"
                f"指定してください: {settings.llm_model_id!r}"
            )
        return
    repo, filename = ref
    from huggingface_hub import hf_hub_download

    logger.info("ensuring GGUF LLM: %s / %s", repo, filename)
    hf_hub_download(repo_id=repo, filename=filename, cache_dir=str(settings.hf_home))


def _ensure_image_base(settings: Settings) -> None:
    """Anima のベース(diffusers 形式リポ)を取得する。

    VAE / Qwen3 text encoder / tokenizer / scheduler / modular pipeline 定義はここから
    読む。派生(WAI-Anima 等)は DiT 単体だけを差し替えるため、ベースは常に必要。
    """
    from huggingface_hub import snapshot_download

    repo = settings.image_base_model_id
    if not _looks_like_hf_repo(repo):
        raise FetchError(
            f"IMAGE_BASE_MODEL_ID は HuggingFace リポ ID である必要があります: {repo!r}"
        )
    logger.info("ensuring Anima base snapshot: %s", repo)
    snapshot_download(repo_id=repo, cache_dir=str(settings.hf_home))


def _ensure_image(settings: Settings) -> Path | None:
    """派生(DiT 単体チェックポイント)を解決する。

    - 空文字 → 派生なし(ベースだけを使う)を意味し ``None`` を返す。
    - ``urn:air:...`` → Civitai から DiT 単体ファイルを取得。
    - それ以外 → 単体 ``.safetensors`` のローカルパスとして存在確認のみ。

    diffusers 形式リポは派生ではなくベースなので IMAGE_BASE_MODEL_ID に指定する。
    """
    value = settings.image_model_id
    if not value:
        return None

    if value.startswith("urn:air:"):
        air = parse_air(value)
        return _resolve_civitai(air, settings)

    if _looks_like_hf_repo(value):
        raise FetchError(
            "IMAGE_MODEL_ID は DiT 単体チェックポイント(Civitai AIR か .safetensors パス)を"
            f"指定してください。diffusers 形式リポは IMAGE_BASE_MODEL_ID へ: {value!r}"
        )

    path = Path(value).expanduser()
    if not path.exists():
        raise FetchError(f"IMAGE_MODEL_ID に指定されたローカルパスが存在しません: {path}")
    return path


def _ensure_turbo_lora(settings: Settings) -> Path | None:
    """Turbo LoRA(単体 .safetensors)をディスクに用意して解決する。

    - 空文字 → Turbo 無効(base 品質)を意味し ``None`` を返す。
    - ``urn:air:...`` → Civitai から LoRA 単体ファイルを取得。
    - それ以外 → ローカル ``.safetensors`` パスとして存在確認のみ。

    LoRA は単体ファイルなので diffusers 形式リポ(HF)は受け付けない。
    """
    value = settings.image_turbo_lora
    if not value:
        return None

    if value.startswith("urn:air:"):
        air = parse_air(value)
        return _resolve_civitai(air, settings)

    if _looks_like_hf_repo(value):
        raise FetchError(
            "IMAGE_TURBO_LORA は LoRA 単体(Civitai AIR か .safetensors パス)で"
            f"指定してください: {value!r}"
        )

    path = Path(value).expanduser()
    if not path.exists():
        raise FetchError(f"IMAGE_TURBO_LORA に指定されたローカルパスが存在しません: {path}")
    return path


@dataclass(frozen=True)
class EnsuredModels:
    """`ensure_all` がディスクに用意し、ローカル解決したモデルのパス群。

    ``None`` は「その要素は未指定/リポ解決でパス固定不要」を意味する。
    """

    image_dit_path: Path | None
    turbo_lora_path: Path | None


def ensure_all(settings: Settings) -> EnsuredModels:
    """モデルをディスクに揃え、ローカルファイルで解決された派生/LoRA のパスを返す。

    DiT 派生・Turbo LoRA が特定のローカルファイルで解決された場合はそのパスを、
    そうでなければ ``None`` を返す（呼び出し側はベース/リポ解決に委ねる）。
    """
    settings.ensure_dirs()
    os.environ.setdefault("HF_HOME", str(settings.hf_home.resolve()))

    _ensure_llm(settings)
    _ensure_image_base(settings)
    return EnsuredModels(
        image_dit_path=_ensure_image(settings),
        turbo_lora_path=_ensure_turbo_lora(settings),
    )
