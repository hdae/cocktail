from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AspectRatio = Literal["portrait", "landscape", "square"]
SeedAction = Literal["new", "keep"]

ASPECT_RATIO_RESOLUTIONS: dict[AspectRatio, tuple[int, int]] = {
    "portrait": (896, 1152),
    "landscape": (1152, 896),
    "square": (1024, 1024),
}


# Danbooru ネイティブのカテゴリ整数。これ以外(モデルの誤値)は絞り込み無効として None に落とす。
_KNOWN_TAG_CATEGORIES: frozenset[int] = frozenset({0, 1, 3, 4, 5})


def _coerce_tag_category(raw: str | None) -> int | None:
    """native DSL の文字列カテゴリを検証済み整数へ。既知カテゴリ以外・非整数は None（絞り込み無効）。"""
    if raw is None:
        return None
    try:
        value = int(raw.strip())
    except (ValueError, AttributeError):
        return None
    return value if value in _KNOWN_TAG_CATEGORIES else None


class SearchTagsCall(BaseModel):
    """Gemma が native tool 形式で出す `search_tags` 呼び出しの検証済み表現。

    正規 Danbooru タグ/キャラ表記に確信が持てないとき、概念(英語 or 日本語)から候補を
    引くための検索クエリ。`category` を与えると Danbooru カテゴリ(0 general / 1 artist /
    3 copyright / 4 character / 5 meta)で絞る。返す件数はサーバが n_ctx 予算で決めるため、
    モデルには limit を持たせない。
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    query: str = Field(min_length=1, max_length=100)
    category: int | None = None

    @classmethod
    def from_native(cls, args: dict[str, str]) -> SearchTagsCall:
        """native DSL の文字列 args を検証済み呼び出しへ。

        native の値は全て文字列なので、strict モデルへ渡す前に `category` を明示コアーション
        する（未知カテゴリ・非整数は絞り込み無効として None）。`query` の空/超過は
        ValidationError となり、呼び出し側で握って検索をスキップできる。
        """
        return cls(
            query=args.get("query", ""),
            category=_coerce_tag_category(args.get("category")),
        )


class GenerateImageCall(BaseModel):
    """Gemma が native tool 形式で出す `generate_image` 呼び出しの検証済み表現。

    `positive` は Danbooru タグ本文。`negative_extra` はこの画像固有の追加ネガのみで、
    固定ベース(`prompt_builder.NEGATIVE_DEFAULT`)はサーバが `compose_negative` で前置する
    （モデルに巨大な negative を自由生成させると temp 次第でリピートループに陥るため）。
    CFG / steps はモード(base/Turbo)でサーバが決め、seed 値も `seed_action` の意図だけ
    選ばせて実値はサーバが `seed_resolver.resolve_seed` で決める。
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    positive: str = Field(min_length=1)
    negative_extra: str = ""
    aspect_ratio: AspectRatio = "portrait"
    seed_action: SeedAction = "new"


class GenerateRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    instruction_ja: str = Field(min_length=1, max_length=2000)
    width: int | None = Field(default=None, ge=256, le=2048)
    height: int | None = Field(default=None, ge=256, le=2048)
    steps: int | None = Field(default=None, ge=1, le=100)
    cfg: float | None = Field(default=None, ge=0.0, le=20.0)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    # dev/デバッグ用の turbo 上書き。None なら settings 既定に従うが、cfg/steps を手動指定した
    # ときは base 整合のため turbo を切る（CFG≈1 蒸留 LoRA を高 CFG で回す破綻を防ぐ）。
    turbo: bool | None = Field(default=None)


class GenerateParams(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    width: int
    height: int
    steps: int
    cfg: float
    seed: int | None


class LatencyBreakdown(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    llm_ms: int
    image_gen_ms: int
    total_ms: int


class GenerateResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    image_id: str
    image_url: str
    prompt: str
    negative_prompt: str
    params: GenerateParams
    latency_ms: LatencyBreakdown
    rationale: str = ""
