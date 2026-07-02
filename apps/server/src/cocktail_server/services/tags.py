"""Danbooru タグの語彙検索 — インメモリ索引で prefix + 日本語エイリアス + fuzzy 照合。

`search_tags` ツールの実体。モデルが正規 Danbooru タグ/キャラ表記に確信を持てないとき、
概念(英語 or 日本語)から候補タグを引き、返った `tag` を画像プロンプトの positive に使う。

データ源: hdae/danbooru-tagcomplete-extra の prebuilt CSV(headerless 4列)::

    tag,category,post_count,"alias1,alias2,..."

alias 列に英語読み・日本語読み・中国語表記が混在するため、日本語クエリはこの列経由で
英語正規タグに解決する（専用の対訳辞書は不要）。`category` は Danbooru ネイティブ整数
(0 general / 1 artist / 3 copyright / 4 character / 5 meta)。

検索は 3 段(best-first)で、上位段が limit を満たしたら以降の段は走らせない:

1. tag 名・全 alias への prefix(+ exact)照合(`bisect` でソート済みキー走査)、
2. tag/alias への substring 照合(日本語中間一致「目線」→「カメラ目線」を拾う)、
3. tag 名へのタイポ許容 fuzzy 照合(`rapidfuzz`)。

ランキングは `(tier, post_count)` 降順。tier 基底は 1000 刻みで離してあり、
sort 時に足す post_count が tier 境界を跨がない（人気は同 tier 内でのみ効く）。

`match_in_text` は逆方向の照合（発話文→その中に現れる tag/alias）で、事前検索の
タグ候補注入（tag hints）の実体。sliding window で最左最長一致を貪欲に取り、
post_count 降順で返す（docs/decisions/0005-tag-hints-presearch.md）。

NOTE: vImagen `apps/api/src/vimagen_api/services/tags.py` からの移植。paste 補完用の
`lookup`/REST 面は Phase2(エージェントループ)では不要なので移植していない。
"""

from __future__ import annotations

import csv
import logging
import shutil
import urllib.request
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz, process

from cocktail_server.config import Settings
from cocktail_server.schemas.tags import TagSuggestion

logger = logging.getLogger(__name__)

# Tier 基底 — sort 時に足す post_count が tier を跨がないよう十分離す。
_TIER_TAG_EXACT = 6000
_TIER_TAG_PREFIX = 5000
_TIER_ALIAS_EXACT = 4000
_TIER_ALIAS_PREFIX = 3000
_TIER_TAG_SUBSTRING = 2000
_TIER_ALIAS_SUBSTRING = 1000
# fuzzy は rapidfuzz の生スコア(0-100)をそのまま使うので、上のどの tier よりも常に下。
# `fuzz.ratio`(全体 Levenshtein)を使う: prefix は上で処理済みで、`WRatio` の部分一致成分は
# 短いタグを過大評価する(query「mastrpiece」で「mast」が「masterpiece」を上回る)。
_FUZZY_CUTOFF = 75.0

# 逆引き(match_in_text)の最短キー長。1 文字キー(「絵」「顔」等)は日本語文で無差別に
# 当たるノイズ源なので照合対象にしない。
_MATCH_MIN_KEY = 2


def _normalize(text: str) -> str:
    """小文字化し、`_` と空白を等価に扱う（danbooru タグとフリー入力が一致する:
    `looking at` ↔ `looking_at_viewer`）。日本語には影響しない。"""
    return text.strip().lower().replace("_", " ")


def _is_ascii_word_char(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def _ascii_word_boundary(text: str, start: int, end: int) -> bool:
    """ASCII 英数語の途中で始まる/終わる照合を弾く（"miku" を "mikura" に当てない）。

    CJK には単語境界が無いので、境界チェックは両隣が ASCII 英数のときだけ効かせる。
    """
    if start > 0 and _is_ascii_word_char(text[start]) and _is_ascii_word_char(text[start - 1]):
        return False
    if end < len(text) and _is_ascii_word_char(text[end - 1]) and _is_ascii_word_char(text[end]):
        return False
    return True


def _has_japanese(text: str) -> bool:
    return any(
        "぀" <= ch <= "ヿ"  # hiragana + katakana
        or "一" <= ch <= "鿿"  # CJK unified ideographs
        or "ｦ" <= ch <= "ﾝ"  # halfwidth katakana
        for ch in text
    )


@dataclass(slots=True)
class TagRow:
    tag: str
    category: int
    post_count: int
    aliases: tuple[str, ...]
    norm_tag: str
    norm_aliases: tuple[str, ...]
    ja: str | None


def _make_row(tag: str, category: int, post_count: int, aliases: list[str]) -> TagRow:
    return TagRow(
        tag=tag,
        category=category,
        post_count=post_count,
        aliases=tuple(aliases),
        norm_tag=_normalize(tag),
        norm_aliases=tuple(_normalize(a) for a in aliases),
        ja=next((a for a in aliases if _has_japanese(a)), None),
    )


def parse_csv(path: Path) -> list[TagRow]:
    """headerless 4 列 CSV を行へパースする。

    category/post_count が整数変換できない行は、黙って 0 埋めせず warning を出して
    その行だけスキップする（索引全体は壊さない）。空タグ/列不足の行は空行として静かに
    飛ばす。DECIDED: 未リリース段階の fail-loud 規約に沿い、壊れたデータを黙って通さない
    （docs/decisions/0002-phase2-agent-loop.md）。
    """
    rows: list[TagRow] = []
    malformed = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for lineno, record in enumerate(csv.reader(handle), start=1):
            if len(record) < 3 or not record[0].strip():
                continue
            tag = record[0].strip()
            try:
                category = int(record[1])
                post_count = int(record[2])
            except ValueError:
                logger.warning("skipping malformed tag row %d: %r", lineno, record[:3])
                malformed += 1
                continue
            alias_field = record[3] if len(record) > 3 else ""
            aliases = [a.strip() for a in alias_field.split(",") if a.strip()]
            rows.append(_make_row(tag, category, post_count, aliases))
    if malformed:
        logger.warning("tag CSV: skipped %d malformed rows", malformed)
    return rows


class TagIndex:
    """パース済みタグ行に対する、クエリ時のみ動く読み取り専用の検索索引。"""

    def __init__(self, rows: list[TagRow]) -> None:
        self._rows = rows
        tag_pairs = sorted((row.norm_tag, i) for i, row in enumerate(rows))
        self._tag_norms = [norm for norm, _ in tag_pairs]
        self._tag_idx = [idx for _, idx in tag_pairs]

        alias_triples = sorted(
            (norm, i, original)
            for i, row in enumerate(rows)
            for norm, original in zip(row.norm_aliases, row.aliases, strict=True)
        )
        self._alias_norms = [norm for norm, _, _ in alias_triples]
        self._alias_idx = [idx for _, idx, _ in alias_triples]
        self._alias_orig = [original for _, _, original in alias_triples]

        # 逆引き(match_in_text)用の完全一致辞書: 正規化キー → 行 index 群。キー文字列は
        # 既存の norm フィールドを共有するため、追加メモリは dict 構造ぶんのみ。
        match_keys: dict[str, list[int]] = {}
        max_key_len = 0
        for i, row in enumerate(rows):
            for norm in (row.norm_tag, *row.norm_aliases):
                if len(norm) < _MATCH_MIN_KEY:
                    continue
                entries = match_keys.setdefault(norm, [])
                if not entries or entries[-1] != i:
                    entries.append(i)
                max_key_len = max(max_key_len, len(norm))
        self._match_keys = match_keys
        self._max_key_len = max_key_len

    @property
    def size(self) -> int:
        return len(self._rows)

    def search(
        self, query: str, limit: int = 15, category: int | None = None
    ) -> list[TagSuggestion]:
        normalized = _normalize(query)
        if not normalized:
            return []

        best: dict[int, tuple[int, str | None]] = {}

        def consider(idx: int, score: int, matched: str | None = None) -> None:
            if category is not None and self._rows[idx].category != category:
                return
            current = best.get(idx)
            if current is None:
                best[idx] = (score, matched)
            elif score > current[0]:
                best[idx] = (score, matched if matched is not None else current[1])

        # Tier 1 — tag 名の prefix(+ exact)、次に alias の prefix(+ exact)。
        for pos in self._prefix_positions(self._tag_norms, normalized):
            exact = self._tag_norms[pos] == normalized
            consider(self._tag_idx[pos], _TIER_TAG_EXACT if exact else _TIER_TAG_PREFIX)
        for pos in self._prefix_positions(self._alias_norms, normalized):
            exact = self._alias_norms[pos] == normalized
            tier = _TIER_ALIAS_EXACT if exact else _TIER_ALIAS_PREFIX
            consider(self._alias_idx[pos], tier, self._alias_orig[pos])

        # Tier 2 — substring。中間一致（例: カメラ目線 の中の 目線）を拾う。
        if len(best) < limit and len(normalized) >= 2:
            for pos, norm in enumerate(self._tag_norms):
                if normalized in norm:
                    consider(self._tag_idx[pos], _TIER_TAG_SUBSTRING)
            for pos, norm in enumerate(self._alias_norms):
                if normalized in norm:
                    consider(self._alias_idx[pos], _TIER_ALIAS_SUBSTRING, self._alias_orig[pos])

        # Tier 3 — tag 名へのタイポ許容 fuzzy(有界コーパス)。
        if len(best) < limit and len(normalized) >= 3:
            matches = process.extract(
                normalized,
                self._tag_norms,
                scorer=fuzz.ratio,
                processor=None,
                score_cutoff=_FUZZY_CUTOFF,
                limit=limit,
            )
            for _norm, score, pos in matches:
                idx = self._tag_idx[pos]
                if idx not in best:
                    consider(idx, int(score))

        ranked = sorted(
            best.items(),
            key=lambda item: (item[1][0], self._rows[item[0]].post_count),
            reverse=True,
        )[:limit]
        return [self._suggestion(idx, matched) for idx, (_score, matched) in ranked]

    def match_in_text(self, text: str, limit: int = 8) -> list[TagSuggestion]:
        """発話文の中に現れる tag/alias を逆引きし、post_count 降順で返す。

        `search` がクエリ→タグの前方照合なのに対し、こちらは文→タグの完全一致照合
        （事前検索のタグ候補注入の実体）。sliding window で各位置の最長キーを貪欲に
        取る（最左最長一致）: 「花火大会」は aerial_fireworks だけに当たり、内側の
        「花火」を重ねて拾わない。同一キーが複数行の alias のとき（「花火」=
        fireworks と sparkle）は全行を返し、文脈での選択はモデルに委ねる。

        `matched` はヒットしたキー（正規化形。日本語 alias は原形と一致する）、
        tag 名の直接ヒットは `search` と同じく None。カテゴリでの絞り込み・優遇は
        しない（post_count 順のみ。docs/decisions/0005-tag-hints-presearch.md）。
        計算量は O(len(text) × 最長キー長) のハッシュ照合で、会話文ならサブ ms。
        """
        normalized = _normalize(text)
        n = len(normalized)
        found: dict[int, str] = {}  # 行 index → ヒットキー（最初の=最左のものを保持）
        i = 0
        while i < n:
            hit = ""
            for length in range(min(self._max_key_len, n - i), _MATCH_MIN_KEY - 1, -1):
                key = normalized[i : i + length]
                if key in self._match_keys and _ascii_word_boundary(normalized, i, i + length):
                    hit = key
                    break
            if not hit:
                i += 1
                continue
            for idx in self._match_keys[hit]:
                found.setdefault(idx, hit)
            i += len(hit)

        ranked = sorted(
            found.items(), key=lambda item: self._rows[item[0]].post_count, reverse=True
        )
        return [
            self._suggestion(idx, None if key == self._rows[idx].norm_tag else key)
            for idx, key in ranked[:limit]
        ]

    @staticmethod
    def _prefix_positions(sorted_norms: list[str], query: str) -> range:
        lo = bisect_left(sorted_norms, query)
        hi = lo
        total = len(sorted_norms)
        while hi < total and sorted_norms[hi].startswith(query):
            hi += 1
        return range(lo, hi)

    def _suggestion(self, idx: int, matched: str | None) -> TagSuggestion:
        row = self._rows[idx]
        return TagSuggestion(
            tag=row.tag,
            category=row.category,
            post_count=row.post_count,
            ja=row.ja,
            matched=matched,
        )


class TagService:
    """タグ索引のライフサイクルを持つ。CSV を用意し、リクエスト経路の外(起動時の
    バックグラウンド `to_thread` ロード)で索引を構築し、準備完了後に検索を提供する。

    索引未構築の間は `search` が空を返す（graceful degradation）。CSV 不在や破損では
    索引を空のままにしてアプリを落とさない（`_load` が warning/exception をログに残す）。
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._index: TagIndex | None = None

    @property
    def ready(self) -> bool:
        return self._index is not None

    @property
    def size(self) -> int:
        index = self._index
        return index.size if index is not None else 0

    def load(self) -> None:
        """CSV を用意し索引を構築する（ブロッキング。lifespan から
        `asyncio.to_thread` で呼び、イベントループを塞がない）。"""
        try:
            path = self._ensure_csv()
            if path is None:
                logger.warning(
                    "tag CSV not found at %s (auto-download disabled); search_tags "
                    "will return no results until the CSV is placed there",
                    self._settings.tags_csv,
                )
                return
            index = TagIndex(parse_csv(path))
            # 参照の差し替えは GIL 下で原子的。ロードスレッドが書き、検索側が読む。
            self._index = index
            logger.info("tag index ready: %d tags", index.size)
        except Exception:  # 壊れた CSV でアプリを落とさない
            logger.exception("failed to build tag index")

    def search(
        self, query: str, limit: int = 15, category: int | None = None
    ) -> list[TagSuggestion]:
        index = self._index
        if index is None:
            return []
        return index.search(query, limit, category)

    def match_in_text(self, text: str, limit: int = 8) -> list[TagSuggestion]:
        index = self._index
        if index is None:
            return []
        return index.match_in_text(text, limit)

    def _ensure_csv(self) -> Path | None:
        path = self._settings.tags_csv
        if path.exists() and path.stat().st_size > 0:
            return path
        if not self._settings.tags_auto_download:
            return None
        return self._download(path)

    def _download(self, path: Path) -> Path | None:
        url = self._settings.tags_csv_url
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        logger.info("downloading tag CSV from %s", url)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "cocktail"})
            with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as out:
                shutil.copyfileobj(response, out)
            tmp.replace(path)
            return path
        except Exception:
            logger.exception("tag CSV download failed")
            tmp.unlink(missing_ok=True)
            return None
