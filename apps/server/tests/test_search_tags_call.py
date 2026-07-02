"""SearchTagsCall — native DSL の文字列 args を検証済み呼び出しへ変換する境界の検証。"""

from __future__ import annotations

import pytest
from cocktail_server.schemas.generate import SearchTagsCall
from pydantic import ValidationError


def test_from_native_minimal_query_only() -> None:
    call = SearchTagsCall.from_native({"query": "cat ears"})
    assert call.query == "cat ears"
    assert call.category is None


def test_from_native_coerces_known_category_string_to_int() -> None:
    # native の値は文字列で届く。既知カテゴリは整数へコアーションする。
    assert SearchTagsCall.from_native({"query": "hatsune miku", "category": "4"}).category == 4


def test_from_native_drops_unknown_category_to_none() -> None:
    # 未知カテゴリ(2 は Danbooru に無い)・非整数はモデルの誤値として絞り込み無効(None)に落とす。
    assert SearchTagsCall.from_native({"query": "x", "category": "2"}).category is None
    assert SearchTagsCall.from_native({"query": "x", "category": "character"}).category is None


def test_from_native_empty_query_raises() -> None:
    with pytest.raises(ValidationError):
        SearchTagsCall.from_native({"query": ""})


def test_from_native_missing_query_raises() -> None:
    with pytest.raises(ValidationError):
        SearchTagsCall.from_native({})


def test_overlong_query_raises() -> None:
    with pytest.raises(ValidationError):
        SearchTagsCall(query="a" * 101)
