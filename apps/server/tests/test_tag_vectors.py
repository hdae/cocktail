"""タグベクトル索引(services/tag_vectors.py)の振る舞いテスト。

埋め込みモデル(GPU/GGUF)は untestable なので、文書構築・アーティファクト整合・
cosine 検索の純粋部分を合成ベクトルで検証する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from cocktail_server.services.tag_vectors import (
    VectorIndexError,
    VectorIndexMeta,
    VectorTagIndex,
    _renormalize,
    build_tag_documents,
    csv_sha256,
    save_index,
    tag_document,
)
from cocktail_server.services.tags import _make_row

# --- tag_document: 文書構築 --------------------------------------------------------


def test_tag_document_normalizes_underscores_and_joins_aliases() -> None:
    row = _make_row("looking_at_viewer", 0, 4_000_000, ["カメラ目線", "視聴者を見る"])
    assert tag_document(row) == "looking at viewer, カメラ目線, 視聴者を見る"


def test_tag_document_moves_trailing_qualifier_after_aliases() -> None:
    # 括弧限定子は主名の後ろへ回し、主名の埋め込みが限定子に支配されないようにする。
    row = _make_row("crying_(action)", 0, 100, ["号泣"])
    assert tag_document(row) == "crying, 号泣, action"


def test_tag_document_dedupes_and_caps_aliases() -> None:
    # タグ名と正規化一致する alias は落とし、上限(8)で打ち切る。
    aliases = ["Wet", *[f"alias{i}" for i in range(12)]]
    row = _make_row("wet", 0, 100, aliases)
    doc = tag_document(row)
    assert doc.startswith("wet, alias0")
    assert "Wet," not in doc
    assert len(doc.split(", ")) == 9  # 主名 + alias 8


def test_build_tag_documents_indexes_general_only() -> None:
    rows = [
        _make_row("sunset", 0, 50_000, ["夕日"]),
        _make_row("texas_(arknights)", 4, 8_125, ["テキサス"]),
        _make_row("some_artist", 1, 10, []),
    ]
    tags, docs = build_tag_documents(rows)
    assert tags == ["sunset"]
    assert docs == ["sunset, 夕日"]


# --- _renormalize ------------------------------------------------------------------


def test_renormalize_unit_norm_and_zero_safe() -> None:
    m = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    out = _renormalize(m)
    assert np.allclose(np.linalg.norm(out[0]), 1.0)
    assert np.allclose(out[1], 0.0)  # ゼロベクトルはゼロ除算せず 0 のまま


# --- アーティファクト整合(fail-loud) ----------------------------------------------


def _meta(sha: str, rows: int = 2, dims: int = 3) -> VectorIndexMeta:
    return VectorIndexMeta(
        model_file="test.gguf", spec="plain", dims=dims, rows=rows, csv_sha256=sha
    )


def _save_small_index(tmp_path: Path, sha: str) -> np.ndarray:
    vectors = _renormalize(np.array([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0]], dtype=np.float32))
    save_index(tmp_path, vectors, ["sunset", "street"], _meta(sha))
    return vectors


def test_index_roundtrip_and_search_ranking(tmp_path: Path) -> None:
    _save_small_index(tmp_path, "sha-abc")
    index = VectorTagIndex.load(tmp_path, expect_csv_sha256="sha-abc")
    assert index.size == 2
    # [1,0,0] に最も近いのは sunset。limit は索引サイズでクランプされる。
    results = index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), limit=10)
    assert [t for t, _ in results] == ["sunset", "street"]
    assert results[0][1] > results[1][1]


def test_index_load_rejects_stale_csv_hash(tmp_path: Path) -> None:
    # CSV が差し替わった索引は fail-loud（黙って古い索引で検索しない）。
    _save_small_index(tmp_path, "sha-old")
    with pytest.raises(VectorIndexError, match="stale"):
        VectorTagIndex.load(tmp_path, expect_csv_sha256="sha-new")


def test_index_load_rejects_missing_artifacts(tmp_path: Path) -> None:
    with pytest.raises(VectorIndexError, match="not found"):
        VectorTagIndex.load(tmp_path, expect_csv_sha256="sha-abc")


def test_save_index_rejects_shape_mismatch(tmp_path: Path) -> None:
    vectors = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(VectorIndexError, match="shape mismatch"):
        save_index(tmp_path, vectors, ["only-one-tag"], _meta("sha", rows=2, dims=3))


def test_csv_sha256_changes_with_content(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    a.write_text("1girl,0,1,x\n")
    b = tmp_path / "b.csv"
    b.write_text("1girl,0,2,x\n")
    assert csv_sha256(a) != csv_sha256(b)
