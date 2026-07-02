"""native_tools のパーサ検証。

fixture は実機（gemma-4-12B heretic GGUF）で観測した native 出力をそのまま使う。

文法の前提（正文法。GGUF テンプレ strip_thinking / llama.cpp peg-gemma4 /
ai.google.dev の 3 系統で確証）: `<|channel>` が開き `<channel|>` が閉じ、span の中だけが
思考。**本文は span の外側すべて**（ghost の空 thought の後ろに本文が来る）。旧実装は
`<|channel>thought<channel|>` をヘッダ扱いして「後ろ〜次マーカー」を思考と誤分類しており、
本文が丸ごと隠れていた（本文欠落バグの根因。docs/decisions/0004-native-channel-grammar.md）。
"""

from __future__ import annotations

from cocktail_server.services.native_tools import (
    NativeToolStream,
    parse_native_output,
    render_tool_call,
)

# 実機観測: ペルソナ + tools=auto + 画像要求（会話前置き → ghost 空 thought → tool_call）
_REAL_IMAGE_OUTPUT = (
    "いいじゃん、それ。星空とピンク髪の猫耳っ娘か、王道だけど外さない組み合わせだね。\n\n"
    "さっそく出してみるよ。\n\n"
    "<|channel>thought\n<channel|>"
    '<|tool_call>call:generate_image{aspect_ratio:<|"|>portrait<|"|>,'
    'positive:<|"|>1girl, solo, cat ears, pink hair, long hair, smiling, '
    'starry sky, night, looking at viewer, masterpiece, highres<|"|>}<tool_call|>'
)

# 実機観測（本文欠落バグの回帰ケース）: ghost 空 thought に挟まれた本文 + tool_call。
# 旧文法はこの本文を「思考」と誤分類して隠していた。
_REAL_BODY_BETWEEN_GHOST_THOUGHTS = (
    "<|channel>thought\n<channel|>"
    "いいね、そのシチュエーション。しっかり攻めていこうか。\n\n"
    "そちらの要望に合わせて、大胆な構図で仕上げるね。\n\n"
    "<|channel>thought\n<channel|>"
    '<|tool_call>call:generate_image{aspect_ratio:<|"|>portrait<|"|>,'
    'positive:<|"|>1girl, solo<|"|>}<tool_call|>'
)


# --- parse_native_output: 会話のみ ------------------------------------------------


def test_plain_chat_has_no_markers_is_all_text() -> None:
    raw = "夢？……正直に言うなら、AIには無いと思うよ。君はどう思う？"
    parsed = parse_native_output(raw)
    assert parsed.text == raw
    assert parsed.thought == ""
    assert parsed.tool_calls == []


# --- parse_native_output: span 文法（本文はチャネルの外） --------------------------


def test_image_request_splits_text_thought_and_tool_call() -> None:
    parsed = parse_native_output(_REAL_IMAGE_OUTPUT)
    # 会話前置きは text に、ghost 空 thought は無内容、tool_call が 1 件
    assert parsed.text.startswith("いいじゃん、それ。")
    assert "<|channel>" not in parsed.text
    assert "<|tool_call>" not in parsed.text
    assert parsed.thought == ""
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.name == "generate_image"
    assert call.args["aspect_ratio"] == "portrait"
    assert call.args["positive"].startswith("1girl, solo, cat ears")
    # 文字列値の内部カンマは値として保持される
    assert "pink hair" in call.args["positive"]


def test_body_between_ghost_thoughts_is_visible_text() -> None:
    # 回帰ガード: ghost 空 thought の直後に来る本文は「思考」ではなく可視テキスト。
    parsed = parse_native_output(_REAL_BODY_BETWEEN_GHOST_THOUGHTS)
    assert parsed.text.startswith("いいね、そのシチュエーション。")
    assert parsed.text.endswith("大胆な構図で仕上げるね。")
    assert parsed.thought == ""
    assert parsed.tool_calls[0].name == "generate_image"


def test_thinking_span_content_is_hidden_and_answer_after_close_is_text() -> None:
    # 正規の thinking 形: <|channel>thought\n<思考><channel|><本文>。
    raw = "<|channel>thought\n構図と光を検討する。\n<channel|>できたよ、こんな感じでどう？"
    parsed = parse_native_output(raw)
    assert parsed.thought == "構図と光を検討する。"
    assert parsed.text == "できたよ、こんな感じでどう？"


def test_text_before_and_after_thinking_span_both_visible() -> None:
    raw = "まず前置き。<|channel>thought\n内緒の思考\n<channel|>そして続き。"
    parsed = parse_native_output(raw)
    assert parsed.text == "まず前置き。そして続き。"
    assert parsed.thought == "内緒の思考"


def test_channel_without_newline_name_is_empty_ghost() -> None:
    # 改行無しの `<|channel>thought<channel|>` はチャネル名だけの ghost（無内容）。
    raw = "<|channel>thought<channel|>本文です。"
    parsed = parse_native_output(raw)
    assert parsed.text == "本文です。"
    assert parsed.thought == ""


def test_orphan_channel_close_is_dropped_from_text() -> None:
    # 孤立 <channel|>（開き無しの閉じ）は制御トークンとして落とし、前後の本文は温存する
    # （llama.cpp peg-gemma4 も裾の孤立閉じを許容する）。
    raw = "本文の前半<channel|>と後半。"
    parsed = parse_native_output(raw)
    assert parsed.text == "本文の前半と後半。"


def test_unclosed_channel_span_hides_to_end() -> None:
    # 閉じが来ないまま切断された span は末尾まで span 扱い（本文へ漏らさない）。
    raw = "本文。<|channel>thought\n切断された思考"
    parsed = parse_native_output(raw)
    assert parsed.text == "本文。"
    assert parsed.thought == "切断された思考"


# --- parse_native_output: ツール引数 DSL ------------------------------------------


def test_string_value_may_contain_weight_syntax_colons_and_commas() -> None:
    raw = (
        "<|tool_call>call:generate_image{"
        'positive:<|"|>1girl, (vibrant red hair:1.3), [anime style], smile<|"|>,'
        'aspect_ratio:<|"|>square<|"|>}<tool_call|>'
    )
    call = parse_native_output(raw).tool_calls[0]
    # `:` や `,` を含む文字列値も STRING_WRAP 境界で正しく閉じる
    assert call.args["positive"] == "1girl, (vibrant red hair:1.3), [anime style], smile"
    assert call.args["aspect_ratio"] == "square"


def test_bare_enum_values_are_parsed() -> None:
    raw = (
        '<|tool_call>call:generate_image{positive:<|"|>1girl<|"|>,'
        'seed_action:keep,aspect_ratio:<|"|>portrait<|"|>}<tool_call|>'
    )
    call = parse_native_output(raw).tool_calls[0]
    assert call.args["seed_action"] == "keep"


def test_truncated_tool_call_without_close_still_parses_args() -> None:
    # max_tokens 到達などで閉じトークンが来ないケースでも args を拾う（後段で検証に回す）。
    raw = '<|tool_call>call:generate_image{positive:<|"|>1girl, solo<|"|>'
    parsed = parse_native_output(raw)
    assert parsed.tool_calls[0].args["positive"] == "1girl, solo"


def test_unwrapped_comma_value_is_not_truncated_at_first_comma() -> None:
    # モデルが文字列を <|"|> で包まずカンマ入りで吐いても、最初のカンマで silent truncation しない。
    raw = (
        "<|tool_call>call:generate_image{positive:1girl, solo, cat ears,"
        "aspect_ratio:portrait}<tool_call|>"
    )
    call = parse_native_output(raw).tool_calls[0]
    assert call.args["positive"] == "1girl, solo, cat ears"
    assert call.args["aspect_ratio"] == "portrait"


def test_unwrapped_last_value_scans_to_end() -> None:
    raw = (
        "<|tool_call>call:generate_image{aspect_ratio:portrait,"
        "positive:1girl, solo, smile}<tool_call|>"
    )
    call = parse_native_output(raw).tool_calls[0]
    assert call.args["aspect_ratio"] == "portrait"
    assert call.args["positive"] == "1girl, solo, smile"


def test_ascii_quoted_values_are_unwrapped() -> None:
    # 多ターンで崩れる `aspect_ratio: "portrait"`（ASCII クオート + colon 後空白）を吸収する。
    raw = (
        '<|tool_call>call:generate_image{aspect_ratio: "portrait", '
        'positive: "1girl, solo", seed_action: "keep"}<tool_call|>'
    )
    call = parse_native_output(raw).tool_calls[0]
    assert call.args["aspect_ratio"] == "portrait"
    assert call.args["seed_action"] == "keep"
    assert call.args["positive"] == "1girl, solo"


def test_double_wrapped_ascii_quotes_inside_native_wrap_are_stripped() -> None:
    raw = '<|tool_call>call:generate_image{aspect_ratio:<|"|>"portrait"<|"|>}<tool_call|>'
    assert parse_native_output(raw).tool_calls[0].args["aspect_ratio"] == "portrait"


def test_caption_internal_quotes_are_preserved() -> None:
    # 前後が同じクオートで挟まれている時だけ剥がす。キャプション途中の引用符は温存。
    raw = '<|tool_call>call:generate_image{positive:<|"|>1girl, a sign says "hi"<|"|>}<tool_call|>'
    assert parse_native_output(raw).tool_calls[0].args["positive"] == '1girl, a sign says "hi"'


def test_render_tool_call_round_trips_through_parse() -> None:
    # render_tool_call と parse_native_output は対（履歴 replay がそのまま読み戻せる）。
    args = {"aspect_ratio": "landscape", "positive": "1girl, solo, smile", "seed_action": "keep"}
    rendered = render_tool_call("generate_image", args)
    parsed = parse_native_output(rendered).tool_calls[0]
    assert parsed.name == "generate_image"
    assert parsed.args == args


# --- NativeToolStream: 逐次ストリーミング -----------------------------------------


def test_stream_emits_plain_chat_incrementally() -> None:
    stream = NativeToolStream()
    out = ""
    for piece in ["こんに", "ちは！", "元気？"]:
        out += stream.feed(piece)
    assert out == "こんにちは！元気？"
    assert parse_native_output(stream.raw).tool_calls == []


def test_stream_holds_back_partial_marker_split_across_chunks() -> None:
    stream = NativeToolStream()
    emitted = ""
    # "<|tool_call>" がチャンク境界で割れて届く。前半を会話テキストに漏らさない。
    for piece in ["出すよ", "<|tool", "_call>call:generate_image{"]:
        emitted += stream.feed(piece)
    assert emitted == "出すよ"
    assert "<|tool" not in emitted


def test_stream_resumes_text_after_channel_span() -> None:
    # span が閉じたら本文の emit を再開する（最初のマーカーで打ち切らない）。
    stream = NativeToolStream()
    emitted = ""
    for piece in [
        "作るね。",
        "<|channel>thought\n内緒の思考",
        "\n<channel|>",
        "できたよ。",
        '<|tool_call>call:generate_image{positive:<|"|>1girl<|"|>}<tool_call|>',
    ]:
        emitted += stream.feed(piece)
    assert emitted == "作るね。できたよ。"
    parsed = parse_native_output(stream.raw)
    assert parsed.text == "作るね。できたよ。"
    assert parsed.thought == "内緒の思考"
    assert parsed.tool_calls[0].name == "generate_image"


def test_stream_emits_body_between_ghost_thoughts() -> None:
    # 回帰ガード（本文欠落バグ）: ghost 空 thought に挟まれた本文が逐次表示される。
    for chunk_size in (1, 3, 7, 4096):
        stream = NativeToolStream()
        emitted = ""
        raw = _REAL_BODY_BETWEEN_GHOST_THOUGHTS
        for i in range(0, len(raw), chunk_size):
            emitted += stream.feed(raw[i : i + chunk_size])
        emitted += stream.flush()
        # 逐次ストリームした会話テキストと parse の会話テキストが一致する（乖離しない）
        assert emitted.strip() == parse_native_output(stream.raw).text
        assert emitted.strip().startswith("いいね、そのシチュエーション。")


def test_stream_does_not_hold_lone_angle_bracket_forever() -> None:
    # 会話中の "<" は marker 接頭辞になり得るため一瞬保留されるが、続く非マーカー文字で解放される。
    stream = NativeToolStream()
    emitted = ""
    for piece in ["a < ", "b です"]:
        emitted += stream.feed(piece)
    assert emitted == "a < b です"


def test_stream_flushes_trailing_marker_prefix_on_end() -> None:
    # ストリームが marker 接頭辞（結局 marker にならない末尾 "<"）で終わったら flush で確定する。
    stream = NativeToolStream()
    emitted = ""
    for piece in ["比較すると a ", "< b の方が良い", "<"]:
        emitted += stream.feed(piece)
    emitted += stream.flush()
    assert emitted == "比較すると a < b の方が良い<"
    assert emitted == parse_native_output(stream.raw).text


def test_stream_flush_is_empty_inside_unclosed_span() -> None:
    # span 内で切断されたら flush は何も出さない（思考/ツールの中身を本文へ漏らさない）。
    stream = NativeToolStream()
    emitted = stream.feed("本文。<|channel>thought\n切断された思考")
    assert emitted == "本文。"
    assert stream.flush() == ""


def test_stream_flush_is_empty_after_a_complete_tool_call() -> None:
    # ツール span が閉じて終わったら保留は無く flush は空。
    stream = NativeToolStream()
    for piece in [
        "作るね。",
        '<|tool_call>call:generate_image{positive:<|"|>1girl<|"|>}<tool_call|>',
    ]:
        stream.feed(piece)
    assert stream.flush() == ""
