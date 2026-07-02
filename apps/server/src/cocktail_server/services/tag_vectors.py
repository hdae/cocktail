"""Danbooru タグのベクトル索引 — 事前検索 Stage 2（意味検索）の実体。

Stage 1（`tags.match_in_text`、辞書完全一致）が構造的に拾えない層 — 日本語 alias を
持たない英語 general タグ（実測 ~62%）への言い換え・情景語クエリ — を、日→英
クロスリンガル埋め込みで橋渡しする（docs/decisions/0006-vector-tag-hints.md）。

構成:

- 文書 = general(category 0) タグのみの 1 タグ 1 文書。英語タグ名を意味アンカーの主軸に
  置き、alias（日本語読み含む）を連結する。固有名詞（character/artist）はベクトルに意味が
  乗らずノイズ源になるため索引対象外（Stage 1 の担当）。
- 埋め込み = llama.cpp（llama-cpp-python）の第 2 ハンドル（`embedding=True`）。pooling は
  指定せず GGUF 既定に任せる（EmbeddingGemma は MEAN が正で、CLS 強制は既知の精度乖離
  issue の原因）。モデル規定のクエリ/文書プロンプト接頭辞は `EmbeddingSpec` が持つ
  （付け忘れは無言の精度劣化になるため、素のテキストを直接埋め込む口は出さない）。
- 索引 = 正規化済み float32 行列 + タグ名列（.npz）とメタ（.json）。索引元 CSV の
  sha256 をメタに焼き、ロード時に不一致なら fail-loud（黙って古い索引で検索しない）。
- 検索 = numpy の内積全探索（正規化済みなので cosine と等価）。35k 行 × 256 次元で
  サブ ms、ANN は不要。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from cocktail_server.services.tags import TagRow, _normalize

logger = logging.getLogger(__name__)

_VECTORS_FILENAME: Final[str] = "tag_vectors.npz"
_META_FILENAME: Final[str] = "tag_vectors.json"

# 文書に連結する alias の上限。同義 alias の冗長票と長文化（メモ的な長い alias）を抑える。
_DOC_MAX_ALIASES: Final[int] = 8

# ベクトル索引の対象カテゴリ。DECIDED: general(0) のみ。character/copyright は Stage 1 の
# 日本語 alias 辞書が既に強く（72-81% 収録）、artist 68,440 行は情景クエリへの名前ノイズに
# しかならない（docs/decisions/0006-vector-tag-hints.md。Stage 1 の「フィルタ無し」とは
# 方針を分ける — 辞書完全一致は誤爆が少なく、ベクトルは緩く当たるため）。
_INDEX_CATEGORY: Final[int] = 0


@dataclass(frozen=True)
class EmbeddingSpec:
    """モデル別の埋め込み仕様。

    `query_prefix` / `document_prefix` はモデルカード規定の接頭辞（非対称）。
    `truncate_dim` は Matryoshka 切詰め次元（切詰め後の再正規化は `_renormalize` が MUST）。
    """

    query_prefix: str
    document_prefix: str
    truncate_dim: int | None = None


# 対応モデルの仕様プリセット。キーは Settings で選ぶ。
EMBEDDING_SPECS: Final[dict[str, EmbeddingSpec]] = {
    # EmbeddingGemma-300m。接頭辞はモデルカード規定（retrieval タスク）。768→256 Matryoshka。
    "embeddinggemma": EmbeddingSpec(
        query_prefix="task: search result | query: ",
        document_prefix="title: none | text: ",
        truncate_dim=256,
    ),
    # multilingual-e5 系（非 instruct）。query:/passage: の非対称接頭辞が規定。
    "e5": EmbeddingSpec(query_prefix="query: ", document_prefix="passage: "),
    # 接頭辞を要求しないモデル（bge-m3 等）。
    "plain": EmbeddingSpec(query_prefix="", document_prefix=""),
}


_TRAILING_QUALIFIER = re.compile(r"^(?P<main>.+?)\s*\((?P<qualifier>[^()]+)\)$")


def tag_document(row: TagRow) -> str:
    """タグ 1 件を埋め込み文書テキストへ整形する。

    タグ名は `_`→空白で英語句に戻し、末尾の括弧限定子（`(action)` 等）は主名の後ろへ
    回して、主名の埋め込みが限定子に支配されないようにする。alias は正規化重複を除いて
    連結（日本語読みが入る行は日→日の単言語ショートカットにもなる）。
    """
    name = row.tag.replace("_", " ")
    qualifier: str | None = None
    m = _TRAILING_QUALIFIER.match(name)
    if m:
        name, qualifier = m.group("main"), m.group("qualifier")
    parts = [name]
    seen = {_normalize(name)}
    for alias in row.aliases:
        norm = _normalize(alias)
        if norm in seen:
            continue
        seen.add(norm)
        parts.append(alias)
        if len(parts) >= 1 + _DOC_MAX_ALIASES:
            break
    if qualifier:
        parts.append(qualifier)
    return ", ".join(parts)


def build_tag_documents(rows: list[TagRow]) -> tuple[list[str], list[str]]:
    """索引対象（general のみ）の (タグ名列, 文書列) を作る。並びは対応が保証される。"""
    tags: list[str] = []
    docs: list[str] = []
    for row in rows:
        if row.category != _INDEX_CATEGORY:
            continue
        tags.append(row.tag)
        docs.append(tag_document(row))
    return tags, docs


def csv_sha256(path: Path) -> str:
    """索引元 CSV の同一性キー。索引メタに焼き、ロード時に照合する。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _renormalize(matrix: np.ndarray) -> np.ndarray:
    """行ごとに L2 正規化する。Matryoshka 切詰め後は必須（切って正規化しないと
    cosine が崩れる）。ゼロベクトルは 0 のまま返す（ゼロ除算を避ける）。"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    result: np.ndarray = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    return result


@dataclass(frozen=True)
class VectorIndexMeta:
    """索引の由来と整合キー。`spec` は `EMBEDDING_SPECS` のキー。"""

    model_file: str
    spec: str
    dims: int
    rows: int
    csv_sha256: str


class VectorIndexError(RuntimeError):
    """索引アーティファクトの不整合（CSV ハッシュ不一致・形状不整合・破損）。"""


def save_index(out_dir: Path, vectors: np.ndarray, tags: list[str], meta: VectorIndexMeta) -> None:
    """索引を `out_dir` へ永続化する（行列+タグ名は npz、メタは人が読める json）。"""
    if vectors.shape[0] != len(tags) or vectors.shape != (meta.rows, meta.dims):
        raise VectorIndexError(
            f"index shape mismatch: vectors={vectors.shape} tags={len(tags)} meta={meta}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / _VECTORS_FILENAME,
        vectors=vectors.astype(np.float32),
        tags=np.array(tags, dtype=np.str_),
    )
    (out_dir / _META_FILENAME).write_text(
        json.dumps(asdict(meta), ensure_ascii=False, indent=1), encoding="utf-8"
    )


class VectorTagIndex:
    """ロード済みのベクトル索引。`search` はタグ名と cosine スコアを返す。"""

    def __init__(self, vectors: np.ndarray, tags: list[str], meta: VectorIndexMeta) -> None:
        self._vectors = vectors
        self._tags = tags
        self.meta = meta

    @property
    def size(self) -> int:
        return len(self._tags)

    @classmethod
    def load(cls, index_dir: Path, *, expect_csv_sha256: str) -> VectorTagIndex:
        """索引をロードする。CSV ハッシュ不一致・形状不整合は fail-loud
        （未リリース段階の規約: 古い/壊れたデータで黙って検索しない）。"""
        meta_path = index_dir / _META_FILENAME
        vec_path = index_dir / _VECTORS_FILENAME
        if not meta_path.is_file() or not vec_path.is_file():
            raise VectorIndexError(f"vector index not found in {index_dir}")
        meta = VectorIndexMeta(**json.loads(meta_path.read_text(encoding="utf-8")))
        if meta.csv_sha256 != expect_csv_sha256:
            raise VectorIndexError(
                "vector index is stale: CSV sha256 mismatch "
                f"(index={meta.csv_sha256[:12]}…, csv={expect_csv_sha256[:12]}…); "
                "rebuild with scripts/build_tag_vectors"
            )
        data = np.load(vec_path)
        vectors = data["vectors"]
        tags = [str(t) for t in data["tags"]]
        if vectors.shape != (meta.rows, meta.dims) or len(tags) != meta.rows:
            raise VectorIndexError(
                f"vector index corrupt: vectors={vectors.shape} tags={len(tags)} meta={meta}"
            )
        logger.info(
            "tag vector index ready: %d tags × %d dims (model=%s)",
            meta.rows,
            meta.dims,
            meta.model_file,
        )
        return cls(vectors, tags, meta)

    def search(self, query_vec: np.ndarray, limit: int) -> list[tuple[str, float]]:
        """正規化済みクエリベクトルで cosine 上位 `limit` 件を返す（スコア降順）。"""
        if self.size == 0 or limit <= 0:
            return []
        scores = self._vectors @ query_vec
        limit = min(limit, self.size)
        top = np.argpartition(scores, -limit)[-limit:]
        order = top[np.argsort(scores[top])[::-1]]
        return [(self._tags[i], float(scores[i])) for i in order]


class TagEmbedder:
    """llama.cpp の埋め込みハンドル（`embedding=True`）で文書/クエリをベクトル化する。

    pooling は指定せず GGUF 既定に任せ、L2 正規化は llama 側で行う。Matryoshka 切詰めが
    ある spec は切詰め後に再正規化する。チャット用 `LlmService` とは独立の第 2 ハンドル
    （llama_backend_init はプロセス共有・冪等）。NOTE: 2 ハンドル同居は実装挙動に基づく
    確認で仕様保証ではない（docs/decisions/0006-vector-tag-hints.md の留保）。
    """

    def __init__(
        self,
        model_path: str,
        spec_name: str,
        *,
        n_gpu_layers: int = -1,
        n_ctx: int = 2048,
        n_batch: int = 512,
    ) -> None:
        self._model_path = model_path
        self._spec = EMBEDDING_SPECS[spec_name]
        self._n_gpu_layers = n_gpu_layers
        self._n_ctx = n_ctx
        self._n_batch = n_batch
        self._llm: Any = None  # llama_cpp.Llama | None

    def is_loaded(self) -> bool:
        return self._llm is not None

    def load(self) -> None:
        if self._llm is not None:
            return
        from llama_cpp import Llama

        logger.info("Loading embedding GGUF: %s", Path(self._model_path).name)
        self._llm = Llama(
            model_path=self._model_path,
            embedding=True,
            n_gpu_layers=self._n_gpu_layers,
            n_ctx=self._n_ctx,
            n_batch=self._n_batch,
            verbose=False,
        )

    def close(self) -> None:
        if self._llm is None:
            return
        self._llm.close()
        self._llm = None

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, self._spec.document_prefix)

    def embed_query(self, text: str) -> np.ndarray:
        vec: np.ndarray = self._embed([text], self._spec.query_prefix)[0]
        return vec

    def _embed(self, texts: list[str], prefix: str) -> np.ndarray:
        if self._llm is None:
            self.load()
        assert self._llm is not None
        raw = self._llm.embed([prefix + t for t in texts], normalize=True, truncate=True)
        matrix: np.ndarray = np.asarray(raw, dtype=np.float32)
        dim = self._spec.truncate_dim
        if dim is not None and matrix.shape[1] > dim:
            matrix = _renormalize(matrix[:, :dim])
        return matrix
