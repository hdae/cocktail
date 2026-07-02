"""Gemma 4 の native チャネル/ツールコール形式をサーバ側でパースする（一時措置）。

TEMPORARY WORKAROUND
--------------------
Gemma 4 は tools を渡すと native トークン形式で応答する。正文法は次のとおり
（GGUF 埋込テンプレの strip_thinking / llama.cpp peg-gemma4 パーサ /
ai.google.dev prompt-formatting-gemma4 の 3 系統で確証。
docs/decisions/0004-native-channel-grammar.md）:

    <|channel>thought\n<思考><channel|>   ← span。`<|channel>` が開き `<channel|>` が閉じ
    <可視テキスト>                         ← span の外側すべて（"final" チャネルは存在しない）
    <|tool_call>call:NAME{key:<|"|>val<|"|>,...}<tool_call|>

生成プロンプトは thinking 無効時（本プロジェクトの既定）、空 thought
`<|channel>thought\n<channel|>` をプリフィルして「本文から書け」と促すが、モデルは出力中にも
ghost の空 thought や孤立 `<channel|>` を混ぜることがある（既知のモデル癖。llama.cpp の文法も
これらを許容する）。したがってパーサは「span の中だけが思考、外はすべて本文」を MUST で厳守
する。span の後ろを思考扱いすると、ghost 空 thought の直後に来る本文を丸ごと隠してしまう。

llama-cpp-python 0.3.32 は native トークンを構造化 `tool_calls` にパースできず、生トークンが
`message.content` に漏れる (abetlen/llama-cpp-python#2227、修正 PR #2232 未マージ)。
DECIDED: #2227 が解決したら本モジュールを撤去し `message.tool_calls` を直接読む方式へ
載せ替える (docs/decisions/0001-llm-tool-harness.md)。
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

# 可視テキスト領域で意味を持つマーカー。孤立 `<channel|>`（開き無しの閉じ）も制御トークン
# なので本文から落とす対象に含める。
_MARKERS: tuple[str, ...] = (CHANNEL_OPEN, TOOL_OPEN, CHANNEL_CLOSE)

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


def _find_earliest(raw: str, start: int) -> tuple[int, str] | None:
    """`start` 以降で最初に現れるマーカーの (位置, マーカー) を返す。無ければ None。"""
    best: tuple[int, str] | None = None
    for marker in _MARKERS:
        k = raw.find(marker, start)
        if k != -1 and (best is None or k < best[0]):
            best = (k, marker)
    return best


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


def _channel_body(span: str) -> str:
    """チャネル span（`<|channel>`〜`<channel|>` の中身）から思考本文を取り出す。

    テンプレの書込形は `<|channel>thought\\n<思考><channel|>` なので、先頭行（チャネル名）を
    落とした残りが本文。改行が無い場合はチャネル名だけの空 span（ghost）か非標準形で、
    いずれも本文としては扱わない（span 内は表示対象外なので取りこぼしても本文は壊れない）。
    """
    nl = span.find("\n")
    if nl != -1:
        return span[nl + 1 :]
    if span.startswith(_THOUGHT_CHANNEL):
        return span[len(_THOUGHT_CHANNEL) :].strip()
    return ""


class NativeToolStream:
    """生ストリームから会話テキストだけを逐次抽出するステートフルスキャナ。

    span 規則に従い、チャネル span / ツール span の中は隠し、span が閉じたら可視テキストの
    emit を「再開」する（本文は span の外側すべて。最初のマーカーで打ち切らない）。部分
    マーカーがチャンク境界で割れて届いても、接頭辞になり得る末尾は保留して漏らさない。
    確定は `parse_native_output(self.raw)` に委ね、可視領域の判定規則は両者で一致させる。
    """

    def __init__(self) -> None:
        self._raw = ""
        self._pos = 0  # スキャン確定位置（この手前は emit/隠蔽が確定済み）
        self._state = "visible"  # visible | channel | tool

    @property
    def raw(self) -> str:
        return self._raw

    def feed(self, piece: str) -> str:
        """デルタを 1 つ食わせ、新たに確定した会話テキスト差分を返す（無ければ空文字）。"""
        self._raw += piece
        return self._scan(final=False)

    def flush(self) -> str:
        """ストリーム終了時に呼ぶ。保留していた末尾（結局マーカーにならなかった接頭辞）を
        確定して吐き出す。span 内で終わった場合（切断）は何も出さない。"""
        return self._scan(final=True)

    def _scan(self, *, final: bool) -> str:
        out: list[str] = []
        while True:
            if self._state == "visible":
                found = _find_earliest(self._raw, self._pos)
                if found is None:
                    if final:
                        safe = len(self._raw)
                    else:
                        safe = len(self._raw) - _prefix_holdback(self._raw[self._pos :], _MARKERS)
                    if safe > self._pos:
                        out.append(self._raw[self._pos : safe])
                        self._pos = safe
                    break
                j, marker = found
                if j > self._pos:
                    out.append(self._raw[self._pos : j])
                if marker == CHANNEL_OPEN:
                    self._state = "channel"
                    self._pos = j + len(CHANNEL_OPEN)
                elif marker == TOOL_OPEN:
                    self._state = "tool"
                    self._pos = j + len(TOOL_OPEN)
                else:  # 孤立 <channel|>: 制御トークンだけ落とし、前後の本文は温存する
                    self._pos = j + len(CHANNEL_CLOSE)
            elif self._state == "channel":
                close = self._raw.find(CHANNEL_CLOSE, self._pos)
                if close == -1:
                    break  # 閉じ待ち（切断時はここで終わり、span 内は出さない）
                self._pos = close + len(CHANNEL_CLOSE)
                self._state = "visible"
            else:  # tool
                close = self._raw.find(TOOL_CLOSE, self._pos)
                if close == -1:
                    break
                self._pos = close + len(TOOL_CLOSE)
                self._state = "visible"
        return "".join(out)


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

    span 規則: `<|channel>…<channel|>` の中＝思考（チャネル名は問わず非表示。テンプレの
    strip_thinking も名前を見ずに span を除去する）、`<|tool_call>…<tool_call|>` の中＝
    ツール呼び出し、それ以外の外側すべて＝会話テキスト。切断された span（閉じが来ない）は
    末尾まで span 扱い。孤立 `<channel|>` は制御トークンとして落とし前後の本文は残す。
    可視領域の判定は `NativeToolStream` と一致する（逐次表示と確定テキストの乖離を防ぐ）。
    """
    text_parts: list[str] = []
    thought_parts: list[str] = []
    tool_calls: list[ParsedToolCall] = []
    i = 0
    n = len(raw)
    while i < n:
        found = _find_earliest(raw, i)
        if found is None:
            text_parts.append(raw[i:])
            break
        j, marker = found
        if j > i:
            text_parts.append(raw[i:j])
        if marker == CHANNEL_OPEN:
            close = raw.find(CHANNEL_CLOSE, j + len(CHANNEL_OPEN))
            span_end = close if close != -1 else n
            body = _channel_body(raw[j + len(CHANNEL_OPEN) : span_end])
            if body.strip():
                thought_parts.append(body.strip())
            i = span_end + (len(CHANNEL_CLOSE) if close != -1 else 0)
        elif marker == TOOL_OPEN:
            close = raw.find(TOOL_CLOSE, j)
            inner_end = close if close != -1 else n
            inner = raw[j + len(TOOL_OPEN) : inner_end]
            parsed = _parse_tool_dsl(inner)
            if parsed is not None:
                tool_calls.append(parsed)
            i = inner_end + (len(TOOL_CLOSE) if close != -1 else 0)
        else:  # 孤立 <channel|>
            i = j + len(CHANNEL_CLOSE)
    return ParsedTurn(
        text="".join(text_parts).strip(),
        thought="\n".join(thought_parts).strip(),
        tool_calls=tool_calls,
    )
