from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TagSuggestion(BaseModel):
    """`search_tags` が返す 1 件の Danbooru タグ候補。

    `category` は Danbooru ネイティブ整数(0 general / 1 artist / 3 copyright /
    4 character / 5 meta)。`ja` は最初の日本語読み（グロス表示用）、`matched` は
    クエリに当たった元エイリアス（タグ名の直接ヒット時は None）。モデルはこれを見て
    「どの読みで当たったか」を確認し、`tag` を positive に採用する。
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    tag: str
    category: int
    post_count: int
    ja: str | None = None
    matched: str | None = None
