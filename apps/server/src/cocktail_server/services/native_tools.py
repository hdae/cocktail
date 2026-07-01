"""Gemma 4 の native チャネル/ツールコール形式をサーバ側でパースする（一時措置）。

TEMPORARY WORKAROUND
--------------------
Gemma 4 は tools を渡すと、以下の native トークン形式で応答する:

    <会話テキスト><|channel>thought<channel|><思考><|tool_call>call:NAME{key:<|"|>val<|"|>,...}<tool_call|>

会話のみのターンではマーカーを一切出さず素のテキストを返す。ところが
llama-cpp-python 0.3.32 はこの native トークンを構造化 `tool_calls` にパースできず、
生トークンが `message.content` に漏れ `tool_calls=None` になる
(abetlen/llama-cpp-python#2227、未マージの修正 PR #2232)。

そこで binding が直るまでは自前でこの形式をパースする。DECIDED: #2227 が解決したら
本モジュールを撤去し `message.tool_calls` を直接読む方式へ載せ替える
(docs/decisions/0001-llm-tool-harness.md)。

構造は「マーカーより前＝会話テキスト（ストリーム表示）／`thought` チャネル＝内部思考
（UI 非表示）／`<|tool_call>`＝ツール呼び出し」。会話テキストのみ逐次ストリームし、
最初のマーカー以降は buffer して `parse_native_output` で確定する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CHANNEL_OPEN = "<|channel>"
CHANNEL_CLOSE = "<channel|>"
TOOL_OPEN = "<|tool_call>"
TOOL_CLOSE = "<tool_call|>"
STRING_WRAP = '<|"|>'
_THOUGHT_CHANNEL = "thought"
_CALL_PREFIX = "call:"

# 会話テキストの終端になり得るマーカー（thought と tool_call の開始）。
_FIRST_MARKERS: tuple[str, ...] = (CHANNEL_OPEN, TOOL_OPEN)

# 非ラップの素値（enum 等）の終端検出。次の "key:" 開始を境界にする（カンマ単体では
# 切らない）。文字列値は STRING_WRAP で囲まれるので本パターンは素値にしか効かない。
_NEXT_KEY_RE = re.compile(r",\s*[A-Za-z_][A-Za-z0-9_]*\s*:")


@dataclass(frozen=True)
class ParsedToolCall:
    """native DSL から取り出した 1 件のツール呼び出し。値は全て未検証の文字列。"""

    name: str
    args: dict[str, str]


@dataclass(frozen=True)
class ParsedTurn:
    """1 ターンのパース結果。`text` は会話テキスト、`thought` は非表示の思考。"""

    text: str
    thought: str
    tool_calls: list[ParsedToolCall]


def _first_marker_index(text: str) -> int | None:
    """`text` 中で最初に現れる channel/tool マーカーの位置。無ければ None。"""
    found = [i for i in (text.find(CHANNEL_OPEN), text.find(TOOL_OPEN)) if i != -1]
    return min(found) if found else None


def _prefix_holdback(text: str, markers: tuple[str, ...]) -> int:
    """`text` 末尾のうち、マーカーの途中（分割到着した先頭）になり得る文字数を返す。

    ストリーム中に `<|tool_call>` が `<|tool` と `_call>` に分かれて届いても、前半を
    会話テキストとして誤って emit しないよう、マーカーの接頭辞になり得る末尾を保留する。
    """
    hold = 0
    for marker in markers:
        max_k = min(len(text), len(marker) - 1)
        for k in range(max_k, 0, -1):
            if text[-k:] == marker[:k]:
                hold = max(hold, k)
                break
    return hold


class NativeToolStream:
    """生ストリームから会話テキストだけを逐次抽出するステートフルスキャナ。

    最初の channel/tool マーカーより前が会話テキスト（部分マーカーは保留して emit）。
    マーカー以降は buffer するだけで、確定は `parse_native_output(self.raw)` に委ねる。
    """

    def __init__(self) -> None:
        self._raw = ""
        self._emitted = 0
        self._structured = False

    @property
    def raw(self) -> str:
        return self._raw

    def feed(self, piece: str) -> str:
        """デルタを 1 つ食わせ、新たに確定した会話テキスト差分を返す（無ければ空文字）。"""
        self._raw += piece
        if self._structured:
            return ""
        idx = _first_marker_index(self._raw)
        if idx is not None:
            self._structured = True
            safe_len = idx
        else:
            safe_len = len(self._raw) - _prefix_holdback(self._raw, _FIRST_MARKERS)
        if safe_len <= self._emitted:
            return ""
        new = self._raw[self._emitted : safe_len]
        self._emitted = safe_len
        return new

    def flush(self) -> str:
        """ストリーム終了時に呼ぶ。保留していた末尾マーカー接頭辞を確定して吐き出す。

        マーカーを一度も見ていない（＝会話のみ）状態でストリームが終わったら、`feed` が
        `_prefix_holdback` で握っていた末尾（結局マーカーにならなかった `<` 等）は普通の
        テキストなので emit する。マーカーを見た後は保留は無いので空を返す。これで
        ストリーム済みテキストと `parse_native_output(raw)` の乖離を無くす。
        """
        if self._structured:
            return ""
        new = self._raw[self._emitted :]
        self._emitted = len(self._raw)
        return new


def _unquote(value: str) -> str:
    """値の前後の空白と、1 層だけの ASCII クオート（"..." / '...'）を剥がす。

    多ターンで形式がドリフトし、native の `<|"|>...<|"|>` の代わりに `"portrait"` のような
    ASCII クオートで enum を包むことがある（履歴の見せ方が主因だが保険で吸収する）。前後が
    同じクオートで挟まれている時だけ剥がすので、キャプション途中の引用符は温存される。
    """
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1].strip()
    return value


def _parse_tool_dsl(inner: str) -> ParsedToolCall | None:
    """`call:NAME{key:<|"|>val<|"|>,key2:bare,...}` の中身を dict へ。名前無しなら None。

    文字列値は `<|"|>...<|"|>` で包まれ、内部のカンマ/コロン/波括弧は値の一部として扱う。
    それ以外（enum や数値）はカンマ/末尾までの素トークン。
    """
    inner = inner.strip()
    if inner.startswith(_CALL_PREFIX):
        inner = inner[len(_CALL_PREFIX) :]
    brace = inner.find("{")
    if brace == -1:
        name = inner.strip()
        return ParsedToolCall(name=name, args={}) if name else None
    name = inner[:brace].strip()
    if not name:
        return None
    body = inner[brace + 1 :].strip()
    if body.endswith("}"):
        body = body[:-1]

    args: dict[str, str] = {}
    i = 0
    n = len(body)
    while i < n:
        while i < n and body[i] in " ,\n\t":
            i += 1
        if i >= n:
            break
        colon = body.find(":", i)
        if colon == -1:
            break
        key = body[i:colon].strip()
        i = colon + 1
        while i < n and body[i] in " \n\t":
            i += 1
        if body.startswith(STRING_WRAP, i):
            vstart = i + len(STRING_WRAP)
            vend = body.find(STRING_WRAP, vstart)
            if vend == -1:
                args[key] = _unquote(body[vstart:])
                break
            args[key] = _unquote(body[vstart:vend])
            i = vend + len(STRING_WRAP)
        else:
            # 素値（enum 等）。次の "key:" 開始までを値にする。カンマ単体では切らないので、
            # モデルが文字列を STRING_WRAP で包まずカンマ入りで吐いても silent truncation
            # にならない（次のキーが来なければ末尾まで）。ASCII クオート付きの enum も剥がす。
            m = _NEXT_KEY_RE.search(body, i)
            end = m.start() if m else n
            args[key] = _unquote(body[i:end])
            i = end
    return ParsedToolCall(name=name, args=args)


def render_tool_call(name: str, args: dict[str, str]) -> str:
    """`args` を native DSL のツール呼び出し文字列へ整形する（履歴 replay 用）。

    `parse_native_output` と対。過去ターンをモデルが実際に出す native 形式で見せることで、
    多ターンでの形式ドリフト（記述的注記を真似て tool を呼ばなくなる／ASCII クオートに崩れる）
    を断つ。値は `<|"|>...<|"|>` で包む。
    """
    body = ",".join(f"{k}:{STRING_WRAP}{v}{STRING_WRAP}" for k, v in args.items())
    return f"{TOOL_OPEN}call:{name}{{{body}}}{TOOL_CLOSE}"


def parse_native_output(raw: str) -> ParsedTurn:
    """モデルの生出力全体を会話テキスト / 思考 / ツール呼び出しに分解する。

    マーカーで囲まれない領域＝会話テキスト、`thought` チャネル＝思考、それ以外の
    チャネル（final 等）＝会話テキスト側に寄せる（取りこぼし防止）。切断された
    チャネルヘッダは無視する。
    """
    text_parts: list[str] = []
    thought_parts: list[str] = []
    tool_calls: list[ParsedToolCall] = []
    i = 0
    n = len(raw)
    while i < n:
        rel = _first_marker_index(raw[i:])
        if rel is None:
            text_parts.append(raw[i:])
            break
        j = i + rel
        if j > i:
            text_parts.append(raw[i:j])
        if raw.startswith(TOOL_OPEN, j):
            close = raw.find(TOOL_CLOSE, j)
            inner_end = close if close != -1 else n
            inner = raw[j + len(TOOL_OPEN) : inner_end]
            parsed = _parse_tool_dsl(inner)
            if parsed is not None:
                tool_calls.append(parsed)
            i = inner_end + (len(TOOL_CLOSE) if close != -1 else 0)
        else:  # CHANNEL_OPEN
            name_close = raw.find(CHANNEL_CLOSE, j)
            if name_close == -1:
                break  # 切断されたチャネルヘッダ。以降に使える内容は無い。
            name = raw[j + len(CHANNEL_OPEN) : name_close].strip()
            content_start = name_close + len(CHANNEL_CLOSE)
            rel_next = _first_marker_index(raw[content_start:])
            content_end = content_start + rel_next if rel_next is not None else n
            content = raw[content_start:content_end]
            if name == _THOUGHT_CHANNEL:
                thought_parts.append(content)
            else:
                text_parts.append(content)
            i = content_end
    return ParsedTurn(
        text="".join(text_parts).strip(),
        thought="".join(thought_parts).strip(),
        tool_calls=tool_calls,
    )
